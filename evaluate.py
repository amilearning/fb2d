"""
evaluate.py — Zero-shot evaluation of a trained FB agent.

Demonstrates the FB property: a single trained model solves multiple tasks
zero-shot by inferring z from goals or rewards (no fine-tuning).

Tasks tested:
  1. Goal-reaching: navigate to a target point (z = B(goal))
  2. Reward inference: optimize reward = -||s - target||^2 (z = E[r * B(s)])
  3. Quadrant preference: reward = +1 in target quadrant, 0 elsewhere
"""

import argparse
import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

from env import Nav2D
from fb_agent import FBAgent


def find_latest_ckpt(save_dir="checkpoints"):
    runs = sorted(glob.glob(os.path.join(save_dir, "*/fb_agent.pt")))
    if not runs:
        raise FileNotFoundError(f"No checkpoints found in {save_dir}")
    return runs[-1]


def load_agent(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    env = Nav2D()
    agent = FBAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        z_dim=args["z_dim"],
        hidden_dim=args["hidden_dim"],
        device=device,
    )
    agent.forward_net.load_state_dict(ckpt["forward_net"])
    agent.backward_net.load_state_dict(ckpt["backward_net"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.forward_net.eval()
    agent.backward_net.eval()
    agent.actor.eval()
    return agent, env


def rollout(agent, env, z, start_state, max_steps=200):
    obs = env.reset(state=start_state)
    trajectory = [obs.copy()]
    for _ in range(max_steps):
        action = agent.act(obs, z, noise=False)
        obs, _, done, _ = env.step(action)
        trajectory.append(obs.copy())
        if done:
            break
    return np.array(trajectory)


def task_goal_reaching(agent, env, output_dir):
    """z = B(goal)."""
    goals = [
        np.array([0.7, 0.7], dtype=np.float32),
        np.array([-0.7, 0.7], dtype=np.float32),
        np.array([-0.7, -0.7], dtype=np.float32),
        np.array([0.7, -0.7], dtype=np.float32),
        np.array([0.0, 0.8], dtype=np.float32),
    ]
    starts = [
        np.array([-0.8, -0.8], dtype=np.float32),
        np.array([0.8, -0.8], dtype=np.float32),
        np.array([0.8, 0.8], dtype=np.float32),
        np.array([-0.8, 0.8], dtype=np.float32),
        np.array([0.0, -0.8], dtype=np.float32),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(goals)))

    final_dists = []
    for i, (goal, start) in enumerate(zip(goals, starts)):
        z = agent.infer_z_from_goal(goal)
        traj = rollout(agent, env, z, start)
        dist = np.linalg.norm(traj[-1] - goal)
        final_dists.append(dist)

        ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], linewidth=2, alpha=0.7)
        ax.plot(start[0], start[1], 'o', color=colors[i], markersize=10,
                markeredgecolor='black')
        ax.plot(goal[0], goal[1], '*', color=colors[i], markersize=20,
                markeredgecolor='black', label=f"Goal {i+1} (d={dist:.2f})")

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Zero-shot Goal Reaching\n(z = B(goal))", fontweight="bold")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    path = os.path.join(output_dir, "task_goal_reaching.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Goal reaching: mean final dist = {np.mean(final_dists):.4f}")
    print(f"  Saved: {path}")
    return final_dists


def task_reward_inference(agent, env, output_dir):
    """z inferred from sampled (s, r) pairs. Reward = -||s - target||^2."""
    targets = [
        np.array([0.5, 0.0], dtype=np.float32),
        np.array([-0.5, 0.5], dtype=np.float32),
        np.array([0.0, -0.6], dtype=np.float32),
    ]

    n_samples = 4096
    rng = np.random.RandomState(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    final_dists = []

    for i, (target, ax) in enumerate(zip(targets, axes)):
        # Sample states uniformly and compute rewards
        sample_obs = rng.uniform(-1, 1, size=(n_samples, 2)).astype(np.float32)
        rewards = -np.sum((sample_obs - target) ** 2, axis=-1).astype(np.float32)

        # Infer z
        z = agent.infer_z_from_rewards(sample_obs, rewards)

        # Rollout from random starts
        starts = [
            np.array([-0.8, -0.8], dtype=np.float32),
            np.array([0.8, 0.8], dtype=np.float32),
            np.array([-0.8, 0.8], dtype=np.float32),
        ]
        for j, start in enumerate(starts):
            traj = rollout(agent, env, z, start)
            d = np.linalg.norm(traj[-1] - target)
            final_dists.append(d)
            ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=2, alpha=0.7)
            ax.plot(start[0], start[1], 'o', markersize=10, markeredgecolor='black')

        ax.plot(target[0], target[1], '*', color='red', markersize=22,
                markeredgecolor='black')
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_title(f"Target [{target[0]:.1f}, {target[1]:.1f}]\nreward = -||s-t||^2",
                     fontsize=11)

    fig.suptitle("Zero-shot Reward Inference  (z = E[r * B(s)])",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "task_reward_inference.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Reward inference: mean final dist = {np.mean(final_dists):.4f}")
    print(f"  Saved: {path}")
    return final_dists


def task_quadrant(agent, env, output_dir):
    """Reward = 1 in a target quadrant, 0 elsewhere."""
    quadrants = {
        "top-right":    lambda s: float((s[0] > 0) and (s[1] > 0)),
        "top-left":     lambda s: float((s[0] < 0) and (s[1] > 0)),
        "bottom-left":  lambda s: float((s[0] < 0) and (s[1] < 0)),
        "bottom-right": lambda s: float((s[0] > 0) and (s[1] < 0)),
    }

    n_samples = 4096
    rng = np.random.RandomState(0)
    sample_obs = rng.uniform(-1, 1, size=(n_samples, 2)).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, (name, reward_fn) in enumerate(quadrants.items()):
        rewards = np.array([reward_fn(s) for s in sample_obs], dtype=np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rewards)

        ax = axes[i]
        # Shade target quadrant
        if name == "top-right":
            ax.fill_between([0, 1], 0, 1, color='green', alpha=0.1)
        elif name == "top-left":
            ax.fill_between([-1, 0], 0, 1, color='green', alpha=0.1)
        elif name == "bottom-left":
            ax.fill_between([-1, 0], -1, 0, color='green', alpha=0.1)
        elif name == "bottom-right":
            ax.fill_between([0, 1], -1, 0, color='green', alpha=0.1)

        for start in [np.array([-0.8, -0.8]), np.array([0.8, 0.8])]:
            traj = rollout(agent, env, z, start.astype(np.float32))
            ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=2, alpha=0.7)
            ax.plot(start[0], start[1], 'o', markersize=10, markeredgecolor='black')
            ax.plot(traj[-1, 0], traj[-1, 1], 'X', markersize=12, markeredgecolor='black')

        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(f"Reward: {name}", fontsize=11)

    fig.suptitle("Zero-shot Quadrant Preference",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "task_quadrant.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = find_latest_ckpt()
    print(f"Loading checkpoint: {args.ckpt}")

    agent, env = load_agent(args.ckpt, device=args.device)
    output_dir = os.path.dirname(args.ckpt)

    print("\n=== Task 1: Goal Reaching ===")
    task_goal_reaching(agent, env, output_dir)

    print("\n=== Task 2: Reward Inference ===")
    task_reward_inference(agent, env, output_dir)

    print("\n=== Task 3: Quadrant Preference ===")
    task_quadrant(agent, env, output_dir)

    print(f"\nAll evaluation outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
