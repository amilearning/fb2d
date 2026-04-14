"""FDWS sweep: tau=5.0, sensitivity=replay

Naive sequential FB on q123 (Q1 -> Q2 -> Q3), 60k updates/stage, z_dim=32.

  - z-sampling : FDWS (Fisher-weighted), temperature tau = 5.0
  - Sensitivity: computed through M = F * B^T on replay data
  - EMA        : 0.0 (only most recent stage's sensitivity, no accumulation)
  - Distillation: piGram(replay) + FB_l2(replay), alpha = 1.0
  - Eval       : final only (after all 3 stages)
"""
import argparse, os, sys, time
from datetime import datetime
import numpy as np, torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ReplayBuffer
from env_quaddyn import Nav2DQuadDyn
from env_quadrant import QUADRANT_BOUNDS
from fb_agent import FBAgent
from distill_lib import (make_teacher, sample_z, joint_update,
                          get_F, get_B, get_pi,
                          loss_l2, loss_gram,
                          SAReplayBuffer,
                          maybe_init_wandb, wandb_log, wandb_finish)
from z_sampling_lib import compute_z_sensitivity, fdws_sample_z

QS = ["Q1", "Q2", "Q3"]
device = torch.device("cuda")


def fill_buffer(npz, q):
    o = npz[f"{q}_obs"]; a = npz[f"{q}_act"]; n = npz[f"{q}_next"]
    buf = ReplayBuffer(o.shape[0] + 100, 2, 2)
    for i in range(o.shape[0]):
        buf.add(o[i], a[i], n[i])
    return buf


def in_q(rng, q, m=0.10):
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
    return np.array([rng.uniform(xlo + m, xhi - m), rng.uniform(ylo + m, yhi - m)],
                    dtype=np.float32)


def eval_in(agent, eq, n_goals=40, n_starts=4, n_steps=200, seed=0):
    rng = np.random.RandomState(seed)
    sample_obs = rng.uniform(-1, 1, size=(4096, 2)).astype(np.float32)
    env = Nav2DQuadDyn(max_steps=n_steps)
    finals = []
    for _ in range(n_goals):
        g = in_q(rng, eq)
        d = np.linalg.norm(sample_obs - g[None, :], axis=1)
        rew = (d < 0.07).astype(np.float32)
        if rew.sum() == 0:
            rew = np.exp(-25.0 * d).astype(np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rew)
        for _ in range(n_starts):
            s0 = in_q(rng, eq)
            obs = env.reset(state=s0)
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs, _, done, _ = env.step(a)
                if done:
                    break
            finals.append(float(np.linalg.norm(obs - g)))
    return np.array(finals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="checkpoints_offline_quaddyn/sub_perq.npz")
    p.add_argument("--updates_per_stage", type=int, default=60_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha_distill", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints_fdws_sweep/tau50_sensReplay")
    p.add_argument("--wandb_project", default=None,
                   help="if set, log to this W&B project (no-op otherwise)")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--log_every", type=int, default=200,
                   help="how often (in steps) to push training metrics to W&B")
    args = p.parse_args()

    FDWS_TAU = 5.0

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    npz = np.load(args.data)
    agent = FBAgent(2, 2, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                     lr=args.lr, device=device)

    # W&B init
    wandb_run = maybe_init_wandb(args, "fdws_tau50", "piGram+FBl2", "sensReplay")

    # Save original sample_z for restoration
    original_sample_z = agent.sample_z

    # z-sampling state
    sensitivity = None  # computed after each stage

    teacher  = None
    cum_step = 0
    sa_replay = SAReplayBuffer(max_size=300_000, obs_dim=2, action_dim=2,
                                device=device)

    for stage_idx in range(3):
        cur_q = QS[stage_idx]
        buf = fill_buffer(npz, cur_q)
        print(f"\n=== STAGE {stage_idx + 1}: train ONLY on {cur_q} (buf={buf.size}), "
              f"teacher={'YES' if teacher is not None else 'NO'}, "
              f"fdws={'ON' if sensitivity is not None else 'OFF (naive)'} ===", flush=True)
        t0 = time.time()

        # Monkey-patch sample_z for FDWS
        if sensitivity is not None:
            _sens = sensitivity  # capture for closure
            agent.sample_z = lambda n, _s=_sens: fdws_sample_z(
                n, args.z_dim, _s, temperature=FDWS_TAU, device=device)
        else:
            agent.sample_z = original_sample_z  # stage 1: naive

        for step in range(1, args.updates_per_stage + 1):
            batch = buf.sample(args.batch_size, device=device)

            if teacher is None:
                metrics = joint_update(agent, batch)
            else:
                def fb_extra_fn(ctx):
                    s_r, a_r = sa_replay.sample(args.batch_size)
                    z_r      = sample_z(s_r.size(0), agent.z_dim, device)
                    Fs1, Fs2 = get_F(agent,   s_r, a_r, z_r)
                    Bs       = get_B(agent,   s_r)
                    with torch.no_grad():
                        Ft1, Ft2 = get_F(teacher, s_r, a_r, z_r)
                        Bt       = get_B(teacher, s_r)
                    student = (Fs1, Fs2, Bs)
                    target  = (Ft1, Ft2, Bt)
                    return args.alpha_distill * loss_l2(student, target)

                def pi_extra_fn(ctx):
                    s_r, _ = sa_replay.sample(args.batch_size)
                    z_r    = sample_z(s_r.size(0), agent.z_dim, device)
                    student = get_pi(agent,   s_r, z_r)
                    with torch.no_grad():
                        target = get_pi(teacher, s_r, z_r)
                    return args.alpha_distill * loss_gram(student, target)

                metrics = joint_update(agent, batch,
                                       fb_extra_fn=fb_extra_fn,
                                       pi_extra_fn=pi_extra_fn)

            cum_step += 1
            if wandb_run is not None and (cum_step % args.log_every == 0):
                wandb_log(wandb_run, {
                    "train/stage":      stage_idx + 1,
                    "train/fb_loss":    metrics.get("fb_loss",    0.0),
                    "train/actor_loss": metrics.get("actor_loss", 0.0),
                    "train/distill_fb": metrics.get("distill_fb", 0.0),
                    "train/distill_pi": metrics.get("distill_pi", 0.0),
                }, step=cum_step)

            if step % 5000 == 0:
                print(f"  step {step:6d}/{args.updates_per_stage} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        # --- End of stage: snapshot teacher, fill replay, compute sensitivity ---
        teacher = make_teacher(agent)

        # Add current stage's (s, a) to replay buffer
        o_np = npz[f"{cur_q}_obs"]; a_np = npz[f"{cur_q}_act"]
        sa_replay.add_batch(torch.from_numpy(o_np).float().to(device),
                             torch.from_numpy(a_np).float().to(device))

        # Compute z-sensitivity on REPLAY buffer data (EMA=0.0 -> replace)
        if sa_replay.size > 0:
            s_sens, a_sens = sa_replay.obs[:sa_replay.size], sa_replay.act[:sa_replay.size]
            sensitivity = compute_z_sensitivity(agent, teacher, s_sens, a_sens,
                                                 source="FB", n_samples=1024)
            print(f"    sensitivity computed on replay buffer (size={sa_replay.size}), "
                  f"mean={sensitivity.mean():.4f}", flush=True)
        else:
            print("    no replay data yet, skipping sensitivity computation", flush=True)

        # Save checkpoint
        torch.save({
            "forward_net":  agent.forward_net.state_dict(),
            "backward_net": agent.backward_net.state_dict(),
            "actor":        agent.actor.state_dict(),
            "sensitivity":  sensitivity.cpu() if sensitivity is not None else None,
            "args":         vars(args),
            "stage":        stage_idx + 1,
            "trained_on":   cur_q,
            "fdws_tau":     FDWS_TAU,
            "sens_source":  "replay",
        }, os.path.join(save_dir, f"stage{stage_idx + 1}.pt"))

    # Restore original sample_z for evaluation
    agent.sample_z = original_sample_z

    # ----- Final evaluation (only after all stages) -----
    print("\n=== FINAL EVAL on Q1, Q2, Q3 ===", flush=True)
    finals_md  = np.zeros(3)
    finals_s10 = np.zeros(3)
    for j, eq in enumerate(QS):
        f = eval_in(agent, eq, seed=999 + j)
        finals_md[j]  = f.mean()
        finals_s10[j] = (f < 0.10).mean()
        print(f"  eval {eq}: mean_d={f.mean():.3f}  s@0.10={(f < 0.10).mean():.2f}",
              flush=True)
        if wandb_run is not None:
            wandb_log(wandb_run, {
                f"eval/{eq}/mean_d": float(f.mean()),
                f"eval/{eq}/s10":    float((f < 0.10).mean()),
            }, step=cum_step)

    mats_md  = finals_md[None,  :]
    mats_s10 = finals_s10[None, :]
    np.save(os.path.join(save_dir, "mean_d.npy"), mats_md)
    np.save(os.path.join(save_dir, "s10.npy"),    mats_s10)

    wandb_finish(wandb_run, summary={
        "final_mean_d_avg": float(finals_md.mean()),
        "final_s10_avg":    float(finals_s10.mean()),
        "final_mean_d_Q1":  float(finals_md[0]),
        "final_mean_d_Q2":  float(finals_md[1]),
        "final_mean_d_Q3":  float(finals_md[2]),
        "final_s10_Q1":     float(finals_s10[0]),
        "final_s10_Q2":     float(finals_s10[1]),
        "final_s10_Q3":     float(finals_s10[2]),
        "fdws_tau":         FDWS_TAU,
        "sens_source":      "replay",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, mat, title, fmt, cmap in [
        (axes[0], mats_md,  "mean final distance",   "{:.3f}", "RdYlGn_r"),
        (axes[1], mats_s10, "success rate @ r=0.10", "{:.2f}", "RdYlGn"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(QS)
        ax.set_yticks([0]); ax.set_yticklabels(["after S3"])
        ax.set_xlabel("eval quadrant")
        for j in range(3):
            ax.text(j, 0, fmt.format(mat[0, j]), ha="center", va="center",
                    fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontweight="bold")
    plt.suptitle(f"FDWS tau=5.0 sens=replay | seed{args.seed}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "heatmap.png"), dpi=140, bbox_inches="tight")
    print(f"\nsaved {save_dir}/heatmap.png")


if __name__ == "__main__":
    main()
