"""
train_quadrants.py — Train 4 separate FB models, one per quadrant.

Each model only sees data from its own quadrant. Saves all 4 checkpoints.
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch

from env import ReplayBuffer
from env_quadrant import Nav2DQuadrant, QUADRANT_BOUNDS
from fb_agent import FBAgent


def train_one(quadrant, args, device, save_dir):
    print(f"\n{'='*60}\n  TRAINING {quadrant}\n{'='*60}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = Nav2DQuadrant(quadrant=quadrant, max_steps=args.max_steps_ep, seed=args.seed)
    buffer = ReplayBuffer(args.buffer_capacity, env.obs_dim, env.action_dim)

    agent = FBAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device,
    )

    obs = env.reset()
    z = agent.sample_z(1).squeeze(0)
    start = time.time()

    for step in range(1, args.total_steps + 1):
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
                print(f"    {quadrant} step {step:6d}/{args.total_steps}  "
                      f"fb={metrics['fb_loss']:.2f}  "
                      f"ortho={metrics['ortho_loss']:.2f}  "
                      f"actor={metrics['actor_loss']:.2f}  "
                      f"({elapsed:.0f}s)")

    ckpt = {
        "forward_net": agent.forward_net.state_dict(),
        "backward_net": agent.backward_net.state_dict(),
        "actor": agent.actor.state_dict(),
        "args": vars(args),
        "quadrant": quadrant,
    }
    ckpt_path = os.path.join(save_dir, f"fb_agent_{quadrant}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"  Saved: {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=25_000)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--update_z_every", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--max_steps_ep", type=int, default=200)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints_quadrant")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        train_one(q, args, device, save_dir)

    print(f"\nAll 4 models trained. Save dir: {save_dir}")


if __name__ == "__main__":
    main()
