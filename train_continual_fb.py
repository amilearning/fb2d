"""
train_continual_fb.py — Sequential (continual) FB training across quadrants.

Phase 1: train on Q1 only
Phase 2: train on Q2 only  (same agent, fresh buffer)
Phase 3: train on Q3 only
Phase 4: train on Q4 only

After each phase, evaluate the agent on all 4 quadrant-preference tasks.
This is the plain (no-replay, no-distillation) baseline to measure
catastrophic forgetting in FB representations.
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
from fb_agent import FBAgent


QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]


def in_quadrant(s, q):
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
    return (s[0] >= xlo) and (s[0] <= xhi) and (s[1] >= ylo) and (s[1] <= yhi)


def train_one_phase(agent, quadrant, args, device):
    """Train the agent on data from one quadrant only. Fresh buffer per phase."""
    print(f"\n{'='*60}\n  PHASE: training on {quadrant}\n{'='*60}")

    env = Nav2DQuadrant(quadrant=quadrant, max_steps=args.max_steps_ep,
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
                      f"ortho={metrics['ortho_loss']:.2f}  "
                      f"actor={metrics['actor_loss']:.2f}  "
                      f"({elapsed:.0f}s)")


def evaluate_on_all_quadrants(agent, args, device, n_starts=8, n_steps=200,
                                n_z_samples=4096):
    """
    For each target quadrant, infer z and roll out in the FULL env from
    starts spread across all 4 quadrants. Returns a length-4 array of
    mean fraction-of-time-in-target.
    """
    rng = np.random.RandomState(0)
    full_env = Nav2D(max_steps=n_steps)

    # Sample states uniformly from the full box for z inference
    sample_obs = rng.uniform(-1, 1, size=(n_z_samples, 2)).astype(np.float32)

    # Spread starts across all 4 quadrants
    starts = []
    for q in QUADRANTS:
        (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
        for _ in range(n_starts // 4):
            starts.append(np.array([
                rng.uniform(xlo + 0.1, xhi - 0.1),
                rng.uniform(ylo + 0.1, yhi - 0.1),
            ], dtype=np.float32))

    results = np.zeros(4)
    for j, target_q in enumerate(QUADRANTS):
        rewards = np.array([1.0 if in_quadrant(s, target_q) else 0.0
                            for s in sample_obs], dtype=np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rewards)

        fracs = []
        for start in starts:
            obs = full_env.reset(state=start)
            in_target = 0
            for _ in range(n_steps):
                action = agent.act(obs, z, noise=False)
                obs, _, done, _ = full_env.step(action)
                if in_quadrant(obs, target_q):
                    in_target += 1
                if done:
                    break
            fracs.append(in_target / n_steps)
        results[j] = float(np.mean(fracs))
    return results


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints_continual")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    # ONE persistent agent
    agent = FBAgent(
        obs_dim=2,
        action_dim=2,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device,
    )

    # 4x4 matrix: rows = phase done, cols = test quadrant
    perf = np.zeros((4, 4))

    # Initial eval (no training yet)
    print("\n[Eval] Initial random model:")
    init_perf = evaluate_on_all_quadrants(agent, args, device)
    for j, q in enumerate(QUADRANTS):
        print(f"    target {q}: {init_perf[j]:.3f}")

    for phase_idx, q in enumerate(QUADRANTS):
        train_one_phase(agent, q, args, device)
        print(f"\n[Eval] After phase {phase_idx+1} (just trained on {q}):")
        row = evaluate_on_all_quadrants(agent, args, device)
        perf[phase_idx] = row
        for j, target_q in enumerate(QUADRANTS):
            print(f"    target {target_q}: {row[j]:.3f}")

    # ---- Print final matrix ----
    print(f"\n{'='*60}")
    print(f"  CONTINUAL FB — PERFORMANCE MATRIX")
    print(f"  (rows = after phase k; cols = test target quadrant)")
    print(f"{'='*60}")
    print(f"  {'After phase':<15}" + "".join(f"{q:>10}" for q in QUADRANTS))
    print(f"  {'-'*55}")
    for i, q in enumerate(QUADRANTS):
        vals = "".join(f"{perf[i,j]:>10.3f}" for j in range(4))
        print(f"  {f'P{i+1} ({q})':<15}{vals}")

    # Forgetting analysis
    print(f"\n  FORGETTING (drop from when each quadrant was its training phase):")
    for i in range(4):
        # When quadrant i was just trained, value = perf[i, i]
        # After all 4 phases, value = perf[3, i]
        forgetting = perf[i, i] - perf[3, i]
        print(f"    {QUADRANTS[i]}: trained={perf[i,i]:.3f}  final={perf[3,i]:.3f}  "
              f"forgot={forgetting:.3f}")

    # ---- Save checkpoint and plots ----
    ckpt = {
        "forward_net": agent.forward_net.state_dict(),
        "backward_net": agent.backward_net.state_dict(),
        "actor": agent.actor.state_dict(),
        "args": vars(args),
        "perf_matrix": perf,
    }
    torch.save(ckpt, os.path.join(save_dir, "fb_agent_continual.pt"))

    # Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(perf, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(QUADRANTS)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"After P{i+1} ({q})"
                                                   for i, q in enumerate(QUADRANTS)])
    ax.set_xlabel("Test target quadrant")
    ax.set_ylabel("Training phase completed")
    for i in range(4):
        for j in range(4):
            color = "white" if perf[i, j] < 0.5 else "black"
            ax.text(j, i, f"{perf[i,j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="fraction in target")
    ax.set_title("Continual FB (no replay/distillation) — Catastrophic Forgetting",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "continual_fb_matrix.png"),
                dpi=140, bbox_inches="tight")
    plt.close()

    # Per-quadrant retention curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    phases = np.arange(1, 5)
    for j, q in enumerate(QUADRANTS):
        ax.plot(phases, perf[:, j], 'o-', linewidth=2, markersize=10,
                label=f"Test {q}")
    ax.set_xlabel("Training phase completed")
    ax.set_ylabel("Fraction in target quadrant")
    ax.set_xticks(phases)
    ax.set_xticklabels([f"P{p}\n({QUADRANTS[p-1]})" for p in phases])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Per-task Retention Across Continual Phases", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "continual_fb_retention.png"),
                dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\nAll results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
