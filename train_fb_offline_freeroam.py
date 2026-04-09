"""Offline FB on the saved freeroam dataset (per-quadrant dynamics, no clipping)."""
import argparse, os, time
from datetime import datetime
import numpy as np, torch
from env import ReplayBuffer
from env_quadrant import QUADRANT_BOUNDS
from env_quaddyn import Nav2DQuadDyn
from fb_agent import FBAgent

QUADRANTS = ["Q1","Q2","Q3","Q4"]


def in_q(s, q):
    (xlo,xhi),(ylo,yhi) = QUADRANT_BOUNDS[q]
    return (s[0]>=xlo and s[0]<=xhi and s[1]>=ylo and s[1]<=yhi)


def evaluate(agent, n_starts=8, n_steps=200, n_z=4096, seed=0):
    rng = np.random.RandomState(seed)
    env = Nav2DQuadDyn(max_steps=n_steps)
    sample_obs = rng.uniform(-1, 1, size=(n_z, 2)).astype(np.float32)
    starts = []
    for q in QUADRANTS:
        (xlo,xhi),(ylo,yhi) = QUADRANT_BOUNDS[q]
        for _ in range(n_starts // 4):
            starts.append(np.array([rng.uniform(xlo+0.1, xhi-0.1),
                                     rng.uniform(ylo+0.1, yhi-0.1)], dtype=np.float32))
    out = np.zeros(4)
    for j, tq in enumerate(QUADRANTS):
        rewards = np.array([1.0 if in_q(s, tq) else 0.0 for s in sample_obs], dtype=np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rewards)
        fracs = []
        for s in starts:
            obs = env.reset(state=s); cnt = 0
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs,_,done,_ = env.step(a)
                if in_q(obs, tq): cnt += 1
                if done: break
            fracs.append(cnt/n_steps)
        out[j] = float(np.mean(fracs))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="checkpoints_offline_quaddyn/freeroam_states_v2.npz")
    p.add_argument("--updates", type=int, default=60_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints_offline_freeroam")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    d = np.load(args.data_path)
    obs, act, nxt = d["obs"], d["act"], d["next_obs"]
    N = obs.shape[0]
    print(f"Loaded {N} transitions from {args.data_path}")
    buf = ReplayBuffer(N + 100, 2, 2)
    for i in range(N):
        buf.add(obs[i], act[i], nxt[i])
    print(f"Buffer: {buf.size}")

    agent = FBAgent(obs_dim=2, action_dim=2, z_dim=args.z_dim,
                     hidden_dim=args.hidden_dim, lr=args.lr, device=device)
    t0 = time.time()
    for step in range(1, args.updates + 1):
        batch = buf.sample(args.batch_size, device=device)
        m = agent.update(batch)
        if step % 5000 == 0:
            print(f"  step {step:6d}/{args.updates}  fb={m['fb_loss']:.2f}  "
                  f"actor={m['actor_loss']:.2f}  ({time.time()-t0:.0f}s)")

    perf = evaluate(agent)
    print(f"\nEval (fraction in target quadrant):")
    for q,v in zip(QUADRANTS, perf):
        print(f"  {q}: {v:.3f}")
    print(f"  mean: {perf.mean():.3f}")
    np.save(os.path.join(save_dir, "perf.npy"), perf)
    torch.save({"forward_net": agent.forward_net.state_dict(),
                "backward_net": agent.backward_net.state_dict(),
                "actor": agent.actor.state_dict(),
                "args": vars(args), "perf": perf.tolist()},
               os.path.join(save_dir, "fb_agent.pt"))


if __name__ == "__main__":
    main()
