#!/usr/bin/env python3
"""Generator for the 48 distill sweep scripts.

For every (B target, D loss, C input) tuple this writes one script:
  train_fb_distill_{B}_{D}_{C}.py

The canonical hand-written script (F_l2_current) is overwritten so it stays
in sync with the rest. Re-run any time the helper API changes.
"""
import os, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

B_OPTIONS = ["F", "B", "FB", "M", "Q", "pi"]
D_OPTIONS = ["l2", "cosine", "contrastive", "gram"]
C_OPTIONS = ["current", "replay"]

LOSS_FN = {"l2": "loss_l2", "cosine": "loss_cosine",
           "contrastive": "loss_contrastive", "gram": "loss_gram"}

# ---------------------------------------------------------------------------
# Closure body per (B, C). Loss fn name is plugged in by .replace("LOSS_FN", ...)
# Body lives inside `else:` of the training loop. Outer indentation = 16 spaces.
# ---------------------------------------------------------------------------

CLOSURES = {
    # ----- B target = F (forward features) ---------------------------------
    ("F", "current"): '''\
                def fb_extra_fn(ctx):
                    student = (ctx["F1"], ctx["F2"])           # reuse FB step
                    with torch.no_grad():
                        target = get_F(teacher,
                                       ctx["obs"], ctx["act"], ctx["z"])
                    return args.alpha_distill * LOSS_FN(student, target)
''',
    ("F", "replay"): '''\
                def fb_extra_fn(ctx):
                    s_r, a_r = sa_replay.sample(args.batch_size)
                    z_r      = sample_z(s_r.size(0), agent.z_dim, device)
                    student  = get_F(agent,   s_r, a_r, z_r)
                    with torch.no_grad():
                        target = get_F(teacher, s_r, a_r, z_r)
                    return args.alpha_distill * LOSS_FN(student, target)
''',

    # ----- B target = B (backward features) --------------------------------
    ("B", "current"): '''\
                def fb_extra_fn(ctx):
                    student = ctx["B"]                          # B(next_obs)
                    with torch.no_grad():
                        target = get_B(teacher, ctx["next_obs"])
                    return args.alpha_distill * LOSS_FN(student, target)
''',
    ("B", "replay"): '''\
                def fb_extra_fn(ctx):
                    s_r, _ = sa_replay.sample(args.batch_size)
                    student = get_B(agent,   s_r)
                    with torch.no_grad():
                        target = get_B(teacher, s_r)
                    return args.alpha_distill * LOSS_FN(student, target)
''',

    # ----- B target = F + B (joint) ----------------------------------------
    ("FB", "current"): '''\
                def fb_extra_fn(ctx):
                    with torch.no_grad():
                        Ft1, Ft2 = get_F(teacher,
                                         ctx["obs"], ctx["act"], ctx["z"])
                        Bt       = get_B(teacher, ctx["next_obs"])
                    student = (ctx["F1"], ctx["F2"], ctx["B"])
                    target  = (Ft1, Ft2, Bt)
                    return args.alpha_distill * LOSS_FN(student, target)
''',
    ("FB", "replay"): '''\
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
                    return args.alpha_distill * LOSS_FN(student, target)
''',

    # ----- B target = M (successor matrix F1 @ B^T) ------------------------
    ("M", "current"): '''\
                def fb_extra_fn(ctx):
                    student = ctx["F1"] @ ctx["B"].T            # head1 M
                    with torch.no_grad():
                        target = get_M(teacher,
                                       ctx["obs"], ctx["act"], ctx["z"])
                    return args.alpha_distill * LOSS_FN(student, target)
''',
    ("M", "replay"): '''\
                def fb_extra_fn(ctx):
                    s_r, a_r = sa_replay.sample(args.batch_size)
                    z_r      = sample_z(s_r.size(0), agent.z_dim, device)
                    Fs1, _   = get_F(agent,   s_r, a_r, z_r)
                    Bs       = get_B(agent,   s_r)
                    student  = Fs1 @ Bs.T
                    with torch.no_grad():
                        target = get_M(teacher, s_r, a_r, z_r)
                    return args.alpha_distill * LOSS_FN(student, target)
''',

    # ----- B target = Q (scalar Q = F·z) -----------------------------------
    ("Q", "current"): '''\
                def fb_extra_fn(ctx):
                    student = (ctx["F1"] * ctx["z"]).sum(-1)    # head1 Q
                    with torch.no_grad():
                        target = get_Q(teacher,
                                       ctx["obs"], ctx["act"], ctx["z"])
                    return args.alpha_distill * LOSS_FN(student, target)
''',
    ("Q", "replay"): '''\
                def fb_extra_fn(ctx):
                    s_r, a_r = sa_replay.sample(args.batch_size)
                    z_r      = sample_z(s_r.size(0), agent.z_dim, device)
                    student  = get_Q(agent,   s_r, a_r, z_r)
                    with torch.no_grad():
                        target = get_Q(teacher, s_r, a_r, z_r)
                    return args.alpha_distill * LOSS_FN(student, target)
''',

    # ----- B target = π (policy actions, additive on actor loss) -----------
    ("pi", "current"): '''\
                def pi_extra_fn(ctx):
                    student = ctx["action_pi"]                  # actor(obs, z_actor)
                    with torch.no_grad():
                        target = get_pi(teacher, ctx["obs"], ctx["z_actor"])
                    return args.alpha_distill * LOSS_FN(student, target)
''',
    ("pi", "replay"): '''\
                def pi_extra_fn(ctx):
                    s_r, _ = sa_replay.sample(args.batch_size)
                    z_r    = sample_z(s_r.size(0), agent.z_dim, device)
                    student = get_pi(agent,   s_r, z_r)
                    with torch.no_grad():
                        target = get_pi(teacher, s_r, z_r)
                    return args.alpha_distill * LOSS_FN(student, target)
''',
}


# ---------------------------------------------------------------------------
# Whole-script template
# ---------------------------------------------------------------------------

TEMPLATE = '''"""B={B}  D={D}  C={C}

Naive sequential FB on q123 (Q1 → Q2 → Q3) with **{B}-distillation** fused
into the {opt} loss inside `joint_update` (single combined backward pass).

  • Stage k trains ONLY on Qk's data (no replay buffer for FB updates).
  • Teacher = frozen snapshot of the agent at end of previous stage.
  • Distillation target  : {B}
  • Distillation loss    : {D}
  • Distillation input C : {C}
  • z is taken from the FB step's `ctx["z"]` (axis E = naive FB sampler).
  • Distill weight: --alpha_distill, default 1.0.
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
                          get_F, get_B, get_M, get_Q, get_pi,
                          loss_l2, loss_cosine, loss_contrastive, loss_gram,
                          SAReplayBuffer,
                          maybe_init_wandb, wandb_log, wandb_finish)

QS = ["Q1", "Q2", "Q3"]
device = torch.device("cuda")


def fill_buffer(npz, q):
    o = npz[f"{{q}}_obs"]; a = npz[f"{{q}}_act"]; n = npz[f"{{q}}_next"]
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
    p.add_argument("--save_dir", default="checkpoints_distill/{B}_{D}_{C}")
    p.add_argument("--wandb_project", default=None,
                   help="if set, log to this W&B project (no-op otherwise)")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--log_every", type=int, default=200,
                   help="how often (in steps) to push training metrics to W&B")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"seed{{args.seed}}_{{timestamp}}")
    os.makedirs(save_dir, exist_ok=True)

    npz = np.load(args.data)
    agent = FBAgent(2, 2, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                     lr=args.lr, device=device)

    wandb_run = maybe_init_wandb(args, "{B}", "{D}", "{C}")

    teacher  = None
    cum_step = 0
{REPLAY_INIT}
    mats_md  = np.zeros((3, 3))
    mats_s10 = np.zeros((3, 3))
    for stage_idx in range(3):
        cur_q = QS[stage_idx]
        buf = fill_buffer(npz, cur_q)
        print(f"\\n=== STAGE {{stage_idx + 1}}: train ONLY on {{cur_q}} (buf={{buf.size}}), "
              f"teacher={{'YES' if teacher is not None else 'NO'}} ===", flush=True)
        t0 = time.time()

        for step in range(1, args.updates_per_stage + 1):
            batch = buf.sample(args.batch_size, device=device)

            if teacher is None:
                metrics = agent.update(batch) or {{}}
                metrics.setdefault("distill_fb", 0.0)
                metrics.setdefault("distill_pi", 0.0)
            else:
{CLOSURE}\
                metrics = joint_update(agent, batch, {EXTRA_KW}={EXTRA_NAME})

            cum_step += 1
            if wandb_run is not None and (cum_step % args.log_every == 0):
                wandb_log(wandb_run, {{
                    "train/stage":      stage_idx + 1,
                    "train/fb_loss":    metrics.get("fb_loss",    0.0),
                    "train/actor_loss": metrics.get("actor_loss", 0.0),
                    "train/distill_fb": metrics.get("distill_fb", 0.0),
                    "train/distill_pi": metrics.get("distill_pi", 0.0),
                }}, step=cum_step)

            if step % 5000 == 0:
                print(f"  step {{step:6d}}/{{args.updates_per_stage}} "
                      f"({{time.time() - t0:.0f}}s)", flush=True)

        teacher = make_teacher(agent)
{REPLAY_ADD}
        torch.save({{
            "forward_net":  agent.forward_net.state_dict(),
            "backward_net": agent.backward_net.state_dict(),
            "actor":        agent.actor.state_dict(),
            "args":         vars(args),
            "stage":        stage_idx + 1,
            "trained_on":   cur_q,
        }}, os.path.join(save_dir, f"stage{{stage_idx + 1}}.pt"))

    # ----- Final evaluation (only after all stages) -----
    print("\\n=== FINAL EVAL on Q1, Q2, Q3 ===", flush=True)
    finals_md  = np.zeros(3)
    finals_s10 = np.zeros(3)
    for j, eq in enumerate(QS):
        f = eval_in(agent, eq, seed=999 + j)
        finals_md[j]  = f.mean()
        finals_s10[j] = (f < 0.10).mean()
        print(f"  eval {{eq}}: mean_d={{f.mean():.3f}}  s@0.10={{(f < 0.10).mean():.2f}}",
              flush=True)
        if wandb_run is not None:
            wandb_log(wandb_run, {{
                f"eval/{{eq}}/mean_d": float(f.mean()),
                f"eval/{{eq}}/s10":    float((f < 0.10).mean()),
            }}, step=cum_step)

    # save: shape-(1,3) row representing the final stage's evaluation
    mats_md  = finals_md[None,  :]
    mats_s10 = finals_s10[None, :]
    np.save(os.path.join(save_dir, "mean_d.npy"), mats_md)
    np.save(os.path.join(save_dir, "s10.npy"),    mats_s10)

    wandb_finish(wandb_run, summary={{
        "final_mean_d_avg": float(finals_md.mean()),
        "final_s10_avg":    float(finals_s10.mean()),
        "final_mean_d_Q1":  float(finals_md[0]),
        "final_mean_d_Q2":  float(finals_md[1]),
        "final_mean_d_Q3":  float(finals_md[2]),
        "final_s10_Q1":     float(finals_s10[0]),
        "final_s10_Q2":     float(finals_s10[1]),
        "final_s10_Q3":     float(finals_s10[2]),
    }})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, mat, title, fmt, cmap in [
        (axes[0], mats_md,  "mean final distance",   "{{:.3f}}", "RdYlGn_r"),
        (axes[1], mats_s10, "success rate @ r=0.10", "{{:.2f}}", "RdYlGn"),
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
    plt.suptitle(f"Distill {B}  |  {D}  |  {C}  —  seed{{args.seed}}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "heatmap.png"), dpi=140, bbox_inches="tight")
    print(f"\\nsaved {{save_dir}}/heatmap.png")


if __name__ == "__main__":
    main()
'''


REPLAY_INIT_BLOCK = '''\
    sa_replay = SAReplayBuffer(max_size=300_000, obs_dim=2, action_dim=2,
                                device=device)
'''

REPLAY_ADD_BLOCK = '''\
        # add this stage's (s, a) to the replay buffer for the next stage's
        # distillation step
        o_np = npz[f"{q}_obs"]; a_np = npz[f"{q}_act"]
        sa_replay.add_batch(torch.from_numpy(o_np).float().to(device),
                             torch.from_numpy(a_np).float().to(device))
'''.replace("{q}", "{cur_q}")  # safety: cur_q is the loop var; rename


def render(B, D, C):
    closure = CLOSURES[(B, C)].replace("LOSS_FN", LOSS_FN[D])
    is_pi   = (B == "pi")
    extra_kw   = "pi_extra_fn" if is_pi else "fb_extra_fn"
    extra_name = "pi_extra_fn" if is_pi else "fb_extra_fn"
    opt_label  = "actor" if is_pi else "FB"

    replay_init = REPLAY_INIT_BLOCK if C == "replay" else "\n"
    replay_add  = REPLAY_ADD_BLOCK  if C == "replay" else ""

    out = TEMPLATE
    # placeholders use {{ }} for things that survive into the rendered file
    # and { } for what we substitute now.
    repls = [
        ("{B}",        B),
        ("{D}",        D),
        ("{C}",        C),
        ("{opt}",      opt_label),
        ("{CLOSURE}",  closure),
        ("{EXTRA_KW}", extra_kw),
        ("{EXTRA_NAME}", extra_name),
        ("{REPLAY_INIT}", replay_init),
        ("{REPLAY_ADD}",  replay_add),
    ]
    for k, v in repls:
        out = out.replace(k, v)
    # finally, escape the literal {{ }} the f-string body needs
    out = out.replace("{{", "{").replace("}}", "}")
    return out


def main():
    n = 0
    for B in B_OPTIONS:
        for D in D_OPTIONS:
            for C in C_OPTIONS:
                src = render(B, D, C)
                fname = f"train_fb_distill_{B}_{D}_{C}.py"
                with open(os.path.join(HERE, fname), "w") as f:
                    f.write(src)
                n += 1
    print(f"wrote {n} scripts to {HERE}")


if __name__ == "__main__":
    main()
