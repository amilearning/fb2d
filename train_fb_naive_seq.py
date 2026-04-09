"""Naive sequential FB: stage k trains ONLY on Qk data (init from stage k-1).
After each stage, eval on every quadrant. Reports forgetting matrix.
"""
import argparse, os, time
from datetime import datetime
import numpy as np, torch
import matplotlib.pyplot as plt
from env import ReplayBuffer
from env_quaddyn import Nav2DQuadDyn
from env_quadrant import QUADRANT_BOUNDS
from fb_agent import FBAgent

QS = ["Q1","Q2","Q3","Q4"]
device = torch.device("cuda")


def fill_buffer(npz, q):
    o = npz[f"{q}_obs"]; a = npz[f"{q}_act"]; n = npz[f"{q}_next"]
    buf = ReplayBuffer(o.shape[0]+100, 2, 2)
    for i in range(o.shape[0]):
        buf.add(o[i], a[i], n[i])
    return buf


def in_q(rng, q, m=0.10):
    (xlo,xhi),(ylo,yhi) = QUADRANT_BOUNDS[q]
    return np.array([rng.uniform(xlo+m,xhi-m), rng.uniform(ylo+m,yhi-m)], dtype=np.float32)


def eval_in(agent, eq, n_goals=40, n_starts=4, n_steps=200, seed=0):
    rng = np.random.RandomState(seed)
    sample_obs = rng.uniform(-1,1, size=(4096,2)).astype(np.float32)
    env = Nav2DQuadDyn(max_steps=n_steps)
    finals = []
    for _ in range(n_goals):
        g = in_q(rng, eq)
        d = np.linalg.norm(sample_obs - g[None,:], axis=1)
        rew = (d < 0.07).astype(np.float32)
        if rew.sum() == 0: rew = np.exp(-25.0*d).astype(np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rew)
        for _ in range(n_starts):
            s0 = in_q(rng, eq)
            obs = env.reset(state=s0)
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs,_,done,_ = env.step(a)
                if done: break
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints_naive_seq")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    npz = np.load(args.data)
    agent = FBAgent(2, 2, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                     lr=args.lr, device=device)

    mats_md = np.zeros((4,4))
    mats_s10 = np.zeros((4,4))
    for stage_idx in range(4):
        cur_q = QS[stage_idx]
        buf = fill_buffer(npz, cur_q)
        print(f"\n=== STAGE {stage_idx+1}: train ONLY on {cur_q} (buf={buf.size}) ===", flush=True)
        t0 = time.time()
        for step in range(1, args.updates_per_stage + 1):
            batch = buf.sample(args.batch_size, device=device)
            m = agent.update(batch)
            if step % 5000 == 0:
                print(f"  step {step:6d}/{args.updates_per_stage}  fb={m['fb_loss']:.2f} "
                      f"actor={m['actor_loss']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"forward_net": agent.forward_net.state_dict(),
                    "backward_net": agent.backward_net.state_dict(),
                    "actor": agent.actor.state_dict(),
                    "args": vars(args), "stage": stage_idx+1, "trained_on": cur_q},
                   os.path.join(save_dir, f"stage{stage_idx+1}.pt"))
        for j, eq in enumerate(QS):
            f = eval_in(agent, eq, seed=stage_idx*100 + j)
            mats_md[stage_idx, j] = f.mean()
            mats_s10[stage_idx, j] = (f<0.10).mean()
            print(f"  eval {eq}: mean_d={f.mean():.3f}  s@0.10={(f<0.10).mean():.2f}", flush=True)

    np.save(os.path.join(save_dir, "mean_d.npy"), mats_md)
    np.save(os.path.join(save_dir, "s10.npy"), mats_s10)
    print("\n=== mean final distance (rows=stage, cols=eval quad) ===")
    print("        " + "".join(f"{q:>10}" for q in QS))
    for i,_ in enumerate(QS):
        print(f"  S{i+1}    " + "".join(f"{v:>10.3f}" for v in mats_md[i]))
    print("\n=== success @ r=0.10 ===")
    for i,_ in enumerate(QS):
        print(f"  S{i+1}    " + "".join(f"{v:>10.2f}" for v in mats_s10[i]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, mat, vmax, title, fmt, cmap in [
        (axes[0], mats_md, 1.0, "mean final distance", "{:.3f}", "RdYlGn_r"),
        (axes[1], mats_s10, 1.0, "success rate @ r=0.10", "{:.2f}", "RdYlGn"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(QS)
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"S{i+1} (only {q})" for i,q in enumerate(QS)])
        ax.set_xlabel("eval quadrant (start in same)")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, fmt.format(mat[i,j]), ha="center", va="center",
                        fontsize=11, fontweight="bold", color="black")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontweight="bold")
    plt.suptitle(f"Naive sequential FB (only Qk per stage) — seed{args.seed}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "naive_seq.png"), dpi=140, bbox_inches="tight")
    print(f"\nsaved {save_dir}/naive_seq.png")


if __name__ == "__main__":
    main()
