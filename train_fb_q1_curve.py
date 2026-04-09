"""Train FB on Q1 sub-dataset only, with eval-during-training to find convergence."""
import argparse, os, time
from datetime import datetime
import numpy as np, torch, matplotlib.pyplot as plt
from env import ReplayBuffer
from env_quaddyn import Nav2DQuadDyn
from env_quadrant import QUADRANT_BOUNDS
from fb_agent import FBAgent

device = torch.device("cuda")


def in_q(rng, q, m=0.10):
    (xlo,xhi),(ylo,yhi) = QUADRANT_BOUNDS[q]
    return np.array([rng.uniform(xlo+m,xhi-m), rng.uniform(ylo+m,yhi-m)], dtype=np.float32)


def eval_in(agent, q, n_goals=20, n_starts=4, n_steps=200, seed=0):
    rng = np.random.RandomState(seed)
    sample_obs = rng.uniform(-1,1, size=(4096,2)).astype(np.float32)
    env = Nav2DQuadDyn(max_steps=n_steps)
    finals = []
    for _ in range(n_goals):
        g = in_q(rng, q)
        d = np.linalg.norm(sample_obs - g[None,:], axis=1)
        rew = (d < 0.07).astype(np.float32)
        if rew.sum() == 0: rew = np.exp(-25.0*d).astype(np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rew)
        for _ in range(n_starts):
            s0 = in_q(rng, q)
            obs = env.reset(state=s0)
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs,_,done,_ = env.step(a)
                if done: break
            finals.append(float(np.linalg.norm(obs - g)))
    f = np.array(finals)
    return float(f.mean()), float((f<0.10).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="checkpoints_offline_quaddyn/sub_perq.npz")
    p.add_argument("--updates", type=int, default=150_000)
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints_q1_curve")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    npz = np.load(args.data)
    o = npz["Q1_obs"]; a = npz["Q1_act"]; n = npz["Q1_next"]
    print(f"Q1 transitions: {o.shape[0]}")
    buf = ReplayBuffer(o.shape[0]+100, 2, 2)
    for i in range(o.shape[0]):
        buf.add(o[i], a[i], n[i])

    agent = FBAgent(2, 2, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                     lr=args.lr, device=device)

    rows = []  # (step, mean_d, s10)
    md, s10 = eval_in(agent, "Q1", seed=0)
    rows.append((0, md, s10))
    print(f"  init  step=0  Q1: mean_d={md:.3f}  s@0.10={s10:.2f}")

    t0 = time.time()
    for step in range(1, args.updates + 1):
        batch = buf.sample(args.batch_size, device=device)
        agent.update(batch)
        if step % args.eval_every == 0:
            md, s10 = eval_in(agent, "Q1", seed=step)
            rows.append((step, md, s10))
            print(f"  step {step:6d}/{args.updates}  Q1: mean_d={md:.3f}  s@0.10={s10:.2f}  ({time.time()-t0:.0f}s)")

    arr = np.array(rows)
    np.save(os.path.join(save_dir, "curve.npy"), arr)
    torch.save({"forward_net": agent.forward_net.state_dict(),
                "backward_net": agent.backward_net.state_dict(),
                "actor": agent.actor.state_dict(),
                "args": vars(args)},
               os.path.join(save_dir, "fb_agent.pt"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(arr[:,0], arr[:,1], 'o-', color='tab:blue'); axes[0].set_xlabel("update step")
    axes[0].set_ylabel("mean final dist"); axes[0].grid(alpha=0.3)
    axes[0].set_title("Q1 training: mean final distance vs steps", fontweight="bold")
    axes[1].plot(arr[:,0], arr[:,2], 'o-', color='tab:green'); axes[1].set_xlabel("update step")
    axes[1].set_ylabel("success @ r=0.10"); axes[1].grid(alpha=0.3); axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Q1 training: success rate vs steps", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve.png"), dpi=140, bbox_inches="tight")
    print(f"\nsaved {save_dir}/")


if __name__ == "__main__":
    main()
