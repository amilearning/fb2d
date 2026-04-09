"""
train_fb_offline_quaddyn.py — Offline FB training on a fixed dataset
collected in the per-quadrant-dynamics env. Eval uses the full-box
Nav2DQuadDyn so the dynamics differ across quadrants at test time.
"""

import argparse, os, time
from datetime import datetime
import numpy as np
import torch
from env import ReplayBuffer
from env_quadrant import QUADRANT_BOUNDS
from env_quaddyn import Nav2DQuadDyn, Nav2DQuadDynRestricted, state_quadrant
from fb_agent import FBAgent

QUADRANTS = ["Q1","Q2","Q3","Q4"]


def collect_from_quadrant(buf, q, n_steps, seed):
    env = Nav2DQuadDynRestricted(quadrant=q, max_steps=200, seed=seed)
    obs = env.reset()
    for _ in range(n_steps):
        a = env.random_action()
        next_obs, _, done, _ = env.step(a)
        buf.add(obs, a, next_obs)
        obs = next_obs if not done else env.reset()


def in_q(s, q):
    (xlo,xhi),(ylo,yhi) = QUADRANT_BOUNDS[q]
    return (s[0]>=xlo and s[0]<=xhi and s[1]>=ylo and s[1]<=yhi)


def evaluate(agent, n_starts=8, n_steps=200, n_z=4096, seed=0):
    rng = np.random.RandomState(seed)
    full_env = Nav2DQuadDyn(max_steps=n_steps)  # eval uses full-box per-state dynamics
    sample_obs = rng.uniform(-1,1, size=(n_z,2)).astype(np.float32)
    starts = []
    for q in QUADRANTS:
        (xlo,xhi),(ylo,yhi) = QUADRANT_BOUNDS[q]
        for _ in range(n_starts//4):
            starts.append(np.array([rng.uniform(xlo+0.1,xhi-0.1),
                                     rng.uniform(ylo+0.1,yhi-0.1)], dtype=np.float32))
    out = np.zeros(4)
    for j, tq in enumerate(QUADRANTS):
        rewards = np.array([1.0 if in_q(s, tq) else 0.0 for s in sample_obs], dtype=np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rewards)
        fracs = []
        for s in starts:
            obs = full_env.reset(state=s); cnt = 0
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs,_,done,_ = full_env.step(a)
                if in_q(obs, tq): cnt += 1
                if done: break
            fracs.append(cnt/n_steps)
        out[j] = float(np.mean(fracs))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quadrants", type=str, required=True)
    p.add_argument("--steps_per_quadrant", type=int, default=15_000)
    p.add_argument("--updates", type=int, default=60_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--buffer_capacity", type=int, default=200_000)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints_offline_quaddyn")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    qs = args.quadrants.split(",")
    device = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    tag = "+".join(qs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{tag}_seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"quadrants={qs}  out={save_dir}")

    buf = ReplayBuffer(args.buffer_capacity, 2, 2)
    for i, q in enumerate(qs):
        collect_from_quadrant(buf, q, args.steps_per_quadrant,
                              seed=args.seed + 1000*i + hash(q)%1000)
    print(f"Buffer: {buf.size}")

    agent = FBAgent(obs_dim=2, action_dim=2, z_dim=args.z_dim,
                     hidden_dim=args.hidden_dim, lr=args.lr, device=device)
    t0 = time.time()
    for step in range(1, args.updates + 1):
        batch = buf.sample(args.batch_size, device=device)
        m = agent.update(batch)
        if step % 5000 == 0:
            print(f"  step {step:6d}  fb={m['fb_loss']:.2f}  actor={m['actor_loss']:.2f}  ({time.time()-t0:.0f}s)")

    perf = evaluate(agent)
    print(f"\nEval on {QUADRANTS}: {perf}")
    np.save(os.path.join(save_dir, "perf.npy"), perf)
    torch.save({"forward_net": agent.forward_net.state_dict(),
                "backward_net": agent.backward_net.state_dict(),
                "actor": agent.actor.state_dict(),
                "args": vars(args), "perf": perf.tolist()},
               os.path.join(save_dir, "fb_agent.pt"))


if __name__ == "__main__":
    main()
