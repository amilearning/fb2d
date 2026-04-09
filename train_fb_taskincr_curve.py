"""Task-incremental FB with eval-during-training. Stage k evaluates on Q_k only."""
import argparse, os, time
from datetime import datetime
import numpy as np, torch, matplotlib.pyplot as plt
from env import ReplayBuffer
from env_quaddyn import Nav2DQuadDyn
from env_quadrant import QUADRANT_BOUNDS
from fb_agent import FBAgent

QS = ["Q1","Q2","Q3","Q4"]
device = torch.device("cuda")


def fill_buffer(npz, qs):
    cap = sum(npz[f"{q}_obs"].shape[0] for q in qs) + 100
    buf = ReplayBuffer(cap, 2, 2)
    for q in qs:
        o = npz[f"{q}_obs"]; a = npz[f"{q}_act"]; n = npz[f"{q}_next"]
        for i in range(o.shape[0]):
            buf.add(o[i], a[i], n[i])
    return buf


def in_q(rng, q, m=0.10):
    (xlo,xhi),(ylo,yhi) = QUADRANT_BOUNDS[q]
    return np.array([rng.uniform(xlo+m,xhi-m), rng.uniform(ylo+m,yhi-m)], dtype=np.float32)


def eval_in(agent, eval_q, n_goals=20, n_starts=4, n_steps=200, seed=0):
    rng = np.random.RandomState(seed)
    sample_obs = rng.uniform(-1,1, size=(4096,2)).astype(np.float32)
    env = Nav2DQuadDyn(max_steps=n_steps)
    finals = []
    for _ in range(n_goals):
        g = in_q(rng, eval_q)
        dists = np.linalg.norm(sample_obs - g[None,:], axis=1)
        rewards = (dists < 0.07).astype(np.float32)
        if rewards.sum() == 0: rewards = np.exp(-25.0*dists).astype(np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rewards)
        for _ in range(n_starts):
            s0 = in_q(rng, eval_q)
            obs = env.reset(state=s0)
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs,_,done,_ = env.step(a)
                if done: break
            finals.append(float(np.linalg.norm(obs - g)))
    finals = np.array(finals)
    return float(finals.mean()), float((finals < 0.10).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="checkpoints_offline_quaddyn/sub_perq.npz")
    p.add_argument("--updates_per_stage", type=int, default=60_000)
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints_taskincr_curve")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    npz = np.load(args.data)
    agent = FBAgent(2, 2, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                     lr=args.lr, device=device)

    curves = {}  # stage_idx -> list of (cum_step, mean_d, s10)
    cum = 0
    for stage_idx in range(4):
        cum_qs = QS[:stage_idx+1]
        cur_q = QS[stage_idx]
        buf = fill_buffer(npz, cum_qs)
        print(f"\n=== STAGE {stage_idx+1}: train on {cum_qs}, eval on {cur_q} ===")
        curves[stage_idx] = []
        # initial eval
        md, s10 = eval_in(agent, cur_q, seed=stage_idx*100)
        curves[stage_idx].append((cum, md, s10))
        print(f"  init   cum_step={cum}  eval {cur_q}: mean_d={md:.3f}  s@0.10={s10:.2f}")

        t0 = time.time()
        for step in range(1, args.updates_per_stage + 1):
            batch = buf.sample(args.batch_size, device=device)
            agent.update(batch)
            cum += 1
            if step % args.eval_every == 0:
                md, s10 = eval_in(agent, cur_q, seed=stage_idx*100 + step)
                curves[stage_idx].append((cum, md, s10))
                print(f"  step {step:6d}/{args.updates_per_stage}  cum={cum}  "
                      f"eval {cur_q}: mean_d={md:.3f}  s@0.10={s10:.2f}  "
                      f"({time.time()-t0:.0f}s)")
        torch.save({"forward_net": agent.forward_net.state_dict(),
                    "backward_net": agent.backward_net.state_dict(),
                    "actor": agent.actor.state_dict(),
                    "args": vars(args), "stage": stage_idx+1, "trained_on": cum_qs},
                   os.path.join(save_dir, f"stage{stage_idx+1}.pt"))

    # save curves and plot
    np.savez(os.path.join(save_dir, "curves.npz"),
             **{f"stage{k+1}": np.array(v) for k, v in curves.items()})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["tab:blue","tab:orange","tab:green","tab:red"]
    for k, c in enumerate(colors):
        arr = np.array(curves[k])
        axes[0].plot(arr[:,0], arr[:,1], 'o-', color=c, label=f"S{k+1}: train Q1..{QS[k]} / eval {QS[k]}")
        axes[1].plot(arr[:,0], arr[:,2], 'o-', color=c, label=f"S{k+1}: eval {QS[k]}")
    for ax, lbl in [(axes[0], "mean final dist"), (axes[1], "success @ r=0.10")]:
        ax.set_xlabel("cumulative gradient steps")
        ax.set_ylabel(lbl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("Mean final distance — eval on current-stage quadrant")
    axes[1].set_title("Success @ r=0.10 — eval on current-stage quadrant")
    plt.suptitle(f"Task-incremental FB — eval during training (seed{args.seed})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves.png"), dpi=140, bbox_inches="tight")
    print(f"\nsaved {save_dir}/curves.png")


if __name__ == "__main__":
    main()
