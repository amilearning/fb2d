"""
train_continual_fb_replay.py — Sequential FB with a SINGLE persistent replay
buffer that stores transitions from all phases. Each phase appends to the same
buffer, and updates sample uniformly from everything seen so far.

Supports both setups:
  --setup disjoint   : Nav2DQuadrant (4 disjoint quadrants)
  --setup quaddisk   : Nav2DQuadDisk (each task = quadrant ∪ central disk)
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from env import Nav2D, ReplayBuffer
from env_quadrant import Nav2DQuadrant, QUADRANT_BOUNDS
from env_quaddisk import Nav2DQuadDisk, _in_quadrant, _in_disk
from fb_agent import FBAgent

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]


def make_env(setup, q, args):
    seed = args.seed + hash(q) % 10000
    if setup == "disjoint":
        return Nav2DQuadrant(quadrant=q, max_steps=args.max_steps_ep, seed=seed)
    else:
        return Nav2DQuadDisk(quadrant=q, disk_r=args.disk_r,
                              max_steps=args.max_steps_ep, seed=seed)


def train_one_phase(agent, buffer, q, args, device):
    print(f"\n{'='*60}\n  PHASE: training on {q} (buffer size at start = {buffer.size})\n{'='*60}")
    env = make_env(args.setup, q, args)
    obs = env.reset()
    z = agent.sample_z(1).squeeze(0)
    start = time.time()
    for step in range(1, args.steps_per_phase + 1):
        if buffer.size < args.warmup_steps:
            action = env.random_action()
        else:
            action = agent.act(obs, z, noise=True)
        next_obs, _, done, _ = env.step(action)
        buffer.add(obs, action, next_obs)
        obs = next_obs
        if done:
            obs = env.reset()
        if step % args.update_z_every == 0:
            z = agent.sample_z(1).squeeze(0)
        if buffer.size >= args.batch_size:
            batch = buffer.sample(args.batch_size, device=device)
            metrics = agent.update(batch)
            if step % 5000 == 0:
                print(f"    {q} step {step:5d}/{args.steps_per_phase}  "
                      f"fb={metrics['fb_loss']:.2f}  actor={metrics['actor_loss']:.2f}  "
                      f"buf={buffer.size}  ({time.time()-start:.0f}s)")


def evaluate(agent, args, device, n_starts=8, n_steps=200, n_z_samples=4096):
    rng = np.random.RandomState(0)
    full_env = Nav2D(max_steps=n_steps)
    sample_obs = rng.uniform(-1, 1, size=(n_z_samples, 2)).astype(np.float32)
    starts = []
    for q in QUADRANTS:
        (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
        for _ in range(n_starts // 4):
            starts.append(np.array([rng.uniform(xlo+0.1, xhi-0.1),
                                     rng.uniform(ylo+0.1, yhi-0.1)], dtype=np.float32))
    quad = np.zeros(4)
    for j, tq in enumerate(QUADRANTS):
        rewards = np.array([1.0 if _in_quadrant(s, tq) else 0.0
                            for s in sample_obs], dtype=np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rewards)
        fracs = []
        for s in starts:
            obs = full_env.reset(state=s); cnt = 0
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs, _, done, _ = full_env.step(a)
                if _in_quadrant(obs, tq): cnt += 1
                if done: break
            fracs.append(cnt / n_steps)
        quad[j] = float(np.mean(fracs))
    return quad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", choices=["disjoint", "quaddisk"], default="disjoint")
    p.add_argument("--steps_per_phase", type=int, default=15_000)
    p.add_argument("--warmup_steps", type=int, default=2_000)
    p.add_argument("--update_z_every", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=512)
    # buffer big enough to hold all 4 phases, never overwritten
    p.add_argument("--buffer_capacity", type=int, default=200_000)
    p.add_argument("--max_steps_ep", type=int, default=200)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--disk_r", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints_replay")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.setup}_seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"setup={args.setup}  seed={args.seed}  device={device}  out={save_dir}")

    agent = FBAgent(obs_dim=2, action_dim=2, z_dim=args.z_dim,
                     hidden_dim=args.hidden_dim, lr=args.lr, device=device)
    # ONE persistent buffer across all phases
    buffer = ReplayBuffer(args.buffer_capacity, 2, 2)

    perf = np.zeros((4, 4))
    for phase_idx, q in enumerate(QUADRANTS):
        train_one_phase(agent, buffer, q, args, device)
        row = evaluate(agent, args, device)
        perf[phase_idx] = row
        print(f"\n[Eval] After phase {phase_idx+1} ({q}): {row}")
        torch.save({"forward_net": agent.forward_net.state_dict(),
                    "backward_net": agent.backward_net.state_dict(),
                    "actor": agent.actor.state_dict(),
                    "args": vars(args), "phase": phase_idx, "trained_on": q},
                   os.path.join(save_dir, f"phase{phase_idx+1}_{q}.pt"))

    np.save(os.path.join(save_dir, "perf_matrix.npy"), perf)
    print(f"\n{'='*60}\n  REPLAY-BUFFER FB ({args.setup}) — perf matrix\n{'='*60}")
    print(f"  {'After phase':<15}" + "".join(f"{q:>10}" for q in QUADRANTS))
    for i, q in enumerate(QUADRANTS):
        print(f"  {f'P{i+1} ({q})':<15}" + "".join(f"{perf[i,j]:>10.3f}" for j in range(4)))
    print("\n  Forgetting:")
    for i in range(4):
        print(f"    {QUADRANTS[i]}: trained={perf[i,i]:.3f}  final={perf[3,i]:.3f}  "
              f"forgot={perf[i,i]-perf[3,i]:+.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(perf, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(QUADRANTS)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"After P{i+1} ({q})" for i,q in enumerate(QUADRANTS)])
    for i in range(4):
        for j in range(4):
            c = "white" if perf[i,j] < 0.5 else "black"
            ax.text(j, i, f"{perf[i,j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color=c)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"FB + full replay buffer  ({args.setup})  seed{args.seed}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "perf_matrix.png"), dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
