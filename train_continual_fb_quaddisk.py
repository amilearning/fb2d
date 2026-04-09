"""
train_continual_fb_quaddisk.py — Naive sequential FB across 4 tasks where each
task = (one quadrant) ∪ (shared central disk of radius disk_r).

Eval after each phase:
  Suite 1: quadrant-membership tasks (4)
  Suite 2: central-disk task (1)
  Suite 3: point-goal tasks (5 fixed goals: one in each outer-quadrant area + origin)

Mirrors train_continual_fb.py's structure but uses Nav2DQuadDisk and a richer eval.
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from env import Nav2D, ReplayBuffer
from env_quadrant import QUADRANT_BOUNDS
from env_quaddisk import Nav2DQuadDisk, in_quaddisk, _in_quadrant, _in_disk
from fb_agent import FBAgent


QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]


def train_one_phase(agent, quadrant, args, device):
    print(f"\n{'='*60}\n  PHASE: training on {quadrant} ∪ disk(r={args.disk_r})\n{'='*60}")
    env = Nav2DQuadDisk(quadrant=quadrant, disk_r=args.disk_r,
                         max_steps=args.max_steps_ep,
                         seed=args.seed + hash(quadrant) % 10000)
    buffer = ReplayBuffer(args.buffer_capacity, env.obs_dim, env.action_dim)

    obs = env.reset()
    z = agent.sample_z(1).squeeze(0)
    start = time.time()
    for step in range(1, args.steps_per_phase + 1):
        if step < args.warmup_steps:
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
        if step >= args.warmup_steps and buffer.size >= args.batch_size:
            batch = buffer.sample(args.batch_size, device=device)
            metrics = agent.update(batch)
            if step % 5000 == 0:
                elapsed = time.time() - start
                print(f"    {quadrant} step {step:5d}/{args.steps_per_phase}  "
                      f"fb={metrics['fb_loss']:.2f}  "
                      f"actor={metrics['actor_loss']:.2f}  ({elapsed:.0f}s)")


def _spread_starts(rng, n=8):
    starts = []
    for q in QUADRANTS:
        (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
        for _ in range(n // 4):
            starts.append(np.array([
                rng.uniform(xlo + 0.1, xhi - 0.1),
                rng.uniform(ylo + 0.1, yhi - 0.1),
            ], dtype=np.float32))
    return starts


def evaluate(agent, args, device, n_starts=8, n_steps=200, n_z_samples=4096):
    rng = np.random.RandomState(0)
    full_env = Nav2D(max_steps=n_steps)
    sample_obs = rng.uniform(-1, 1, size=(n_z_samples, 2)).astype(np.float32)
    starts = _spread_starts(rng, n_starts)

    # Suite 1: quadrant-membership
    quad_results = np.zeros(4)
    for j, target_q in enumerate(QUADRANTS):
        rewards = np.array([1.0 if _in_quadrant(s, target_q) else 0.0
                            for s in sample_obs], dtype=np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rewards)
        fracs = []
        for start in starts:
            obs = full_env.reset(state=start)
            in_target = 0
            for _ in range(n_steps):
                action = agent.act(obs, z, noise=False)
                obs, _, done, _ = full_env.step(action)
                if _in_quadrant(obs, target_q):
                    in_target += 1
                if done: break
            fracs.append(in_target / n_steps)
        quad_results[j] = float(np.mean(fracs))

    # Suite 2: central disk
    rewards = np.array([1.0 if _in_disk(s, args.disk_r) else 0.0
                        for s in sample_obs], dtype=np.float32)
    z = agent.infer_z_from_rewards(sample_obs, rewards)
    fracs = []
    for start in starts:
        obs = full_env.reset(state=start)
        in_target = 0
        for _ in range(n_steps):
            action = agent.act(obs, z, noise=False)
            obs, _, done, _ = full_env.step(action)
            if _in_disk(obs, args.disk_r):
                in_target += 1
            if done: break
        fracs.append(in_target / n_steps)
    disk_result = float(np.mean(fracs))

    # Suite 3: point-goal — final distance to goal
    goals = [
        np.array([0.7, 0.7], dtype=np.float32),    # outer Q1
        np.array([-0.7, 0.7], dtype=np.float32),   # outer Q2
        np.array([-0.7, -0.7], dtype=np.float32),  # outer Q3
        np.array([0.7, -0.7], dtype=np.float32),   # outer Q4
        np.array([0.0, 0.0], dtype=np.float32),    # center
    ]
    goal_dists = np.zeros(len(goals))
    for gi, g in enumerate(goals):
        # one-hot reward at the closest sample point
        dists = np.linalg.norm(sample_obs - g[None, :], axis=1)
        rewards = (dists < 0.05).astype(np.float32)
        if rewards.sum() == 0:
            # fallback: softmax-ish reward by negative distance
            rewards = np.exp(-20.0 * dists).astype(np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rewards)
        ds = []
        for start in starts:
            obs = full_env.reset(state=start)
            for _ in range(n_steps):
                action = agent.act(obs, z, noise=False)
                obs, _, done, _ = full_env.step(action)
                if done: break
            ds.append(float(np.linalg.norm(obs - g)))
        goal_dists[gi] = float(np.mean(ds))

    return {"quad": quad_results, "disk": disk_result, "goal_dists": goal_dists}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_phase", type=int, default=15_000)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--update_z_every", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--max_steps_ep", type=int, default=200)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disk_r", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints_quaddisk")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  disk_r={args.disk_r}  seed={args.seed}")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    agent = FBAgent(obs_dim=2, action_dim=2, z_dim=args.z_dim,
                     hidden_dim=args.hidden_dim, lr=args.lr, device=device)

    quad_perf = np.zeros((4, 4))   # rows = phase, cols = test quadrant
    disk_perf = np.zeros(4)        # disk task per phase
    goal_perf = np.zeros((4, 5))   # rows = phase, cols = goal index

    print("\n[Eval] Initial random model:")
    init = evaluate(agent, args, device)
    print(f"   quad: {init['quad']}\n   disk: {init['disk']:.3f}\n   goal_dists: {init['goal_dists']}")

    ckpts = {}
    for phase_idx, q in enumerate(QUADRANTS):
        train_one_phase(agent, q, args, device)
        print(f"\n[Eval] After phase {phase_idx+1} ({q}):")
        res = evaluate(agent, args, device)
        quad_perf[phase_idx] = res["quad"]
        disk_perf[phase_idx] = res["disk"]
        goal_perf[phase_idx] = res["goal_dists"]
        print(f"   quad: {res['quad']}")
        print(f"   disk: {res['disk']:.3f}")
        print(f"   goal_dists: {res['goal_dists']}")
        # save phase checkpoint
        ckpt = {
            "forward_net": agent.forward_net.state_dict(),
            "backward_net": agent.backward_net.state_dict(),
            "actor": agent.actor.state_dict(),
            "phase": phase_idx, "trained_on": q, "args": vars(args),
        }
        torch.save(ckpt, os.path.join(save_dir, f"phase{phase_idx+1}_{q}.pt"))
        ckpts[q] = ckpt

    # Save metrics
    np.savez(os.path.join(save_dir, "metrics.npz"),
             quad_perf=quad_perf, disk_perf=disk_perf, goal_perf=goal_perf)

    # Print summary
    print(f"\n{'='*60}\n  CONTINUAL FB QUAD+DISK — quadrant-membership matrix\n{'='*60}")
    print(f"  {'After phase':<15}" + "".join(f"{q:>10}" for q in QUADRANTS) + f"{'disk':>10}")
    for i, q in enumerate(QUADRANTS):
        vals = "".join(f"{quad_perf[i,j]:>10.3f}" for j in range(4))
        print(f"  {f'P{i+1} ({q})':<15}{vals}{disk_perf[i]:>10.3f}")
    print("\n  Forgetting (when each quadrant trained → after all 4 phases):")
    for i in range(4):
        print(f"    {QUADRANTS[i]}: trained={quad_perf[i,i]:.3f}  "
              f"final={quad_perf[3,i]:.3f}  forgot={quad_perf[i,i]-quad_perf[3,i]:.3f}")

    # Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(quad_perf, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(QUADRANTS)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"After P{i+1} ({q})" for i, q in enumerate(QUADRANTS)])
    for i in range(4):
        for j in range(4):
            color = "white" if quad_perf[i, j] < 0.5 else "black"
            ax.text(j, i, f"{quad_perf[i,j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="fraction in target")
    ax.set_title(f"Naive continual FB on quadrant∪disk(r={args.disk_r}) — seed {args.seed}",
                  fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quad_membership.png"), dpi=140, bbox_inches="tight")
    plt.close()

    # Retention curves (quad + disk + goal)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    phases = np.arange(1, 5)
    for j, q in enumerate(QUADRANTS):
        axes[0].plot(phases, quad_perf[:, j], 'o-', lw=2, ms=8, label=q)
    axes[0].set_title("Quadrant-membership retention"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xticks(phases); axes[0].set_xlabel("phase"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(phases, disk_perf, 'ks-', lw=2, ms=10)
    axes[1].set_title(f"Central disk (r={args.disk_r}) retention")
    axes[1].set_ylim(-0.05, 1.05); axes[1].set_xticks(phases); axes[1].grid(alpha=0.3)
    goal_names = ["Q1 outer", "Q2 outer", "Q3 outer", "Q4 outer", "origin"]
    for gi, gn in enumerate(goal_names):
        axes[2].plot(phases, goal_perf[:, gi], 'o-', lw=2, ms=8, label=gn)
    axes[2].set_title("Point-goal final distance (lower=better)")
    axes[2].set_xticks(phases); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "retention.png"), dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\nSaved to: {save_dir}/")


if __name__ == "__main__":
    main()
