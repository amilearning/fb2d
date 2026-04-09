"""
train.py — Train an FB agent on the 2D navigation env (no rewards).

The agent collects random + on-policy data, fills a replay buffer,
and trains F, B, and the actor jointly. After training, the same model
solves arbitrary tasks zero-shot via z inference.
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch

from env import Nav2D, ReplayBuffer
from fb_agent import FBAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--update_z_every", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--max_steps_ep", type=int, default=200)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    env = Nav2D(max_steps=args.max_steps_ep, seed=args.seed)
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

    losses_log = {"step": [], "fb_loss": [], "ortho_loss": [], "actor_loss": []}
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

        # Resample z periodically (matches FB-DDPG protocol)
        if step % args.update_z_every == 0:
            z = agent.sample_z(1).squeeze(0)

        # Update
        if step >= args.warmup_steps and buffer.size >= args.batch_size:
            batch = buffer.sample(args.batch_size, device=device)
            metrics = agent.update(batch)

            if step % 500 == 0:
                losses_log["step"].append(step)
                losses_log["fb_loss"].append(metrics["fb_loss"])
                losses_log["ortho_loss"].append(metrics["ortho_loss"])
                losses_log["actor_loss"].append(metrics["actor_loss"])

            if step % 2000 == 0:
                elapsed = time.time() - start
                print(f"  step {step:6d}/{args.total_steps}  "
                      f"fb={metrics['fb_loss']:.3f}  "
                      f"ortho={metrics['ortho_loss']:.3f}  "
                      f"actor={metrics['actor_loss']:.3f}  "
                      f"buf={buffer.size}  "
                      f"({elapsed:.1f}s)")

    print("Training done.")

    # Save checkpoint
    ckpt = {
        "forward_net": agent.forward_net.state_dict(),
        "backward_net": agent.backward_net.state_dict(),
        "actor": agent.actor.state_dict(),
        "args": vars(args),
        "losses": losses_log,
    }
    ckpt_path = os.path.join(save_dir, "fb_agent.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Save loss plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(losses_log["step"], losses_log["fb_loss"]); axes[0].set_title("FB loss")
    axes[1].plot(losses_log["step"], losses_log["ortho_loss"]); axes[1].set_title("Ortho loss")
    axes[2].plot(losses_log["step"], losses_log["actor_loss"]); axes[2].set_title("Actor loss")
    for ax in axes:
        ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved curves: {save_dir}/training_curves.png")


if __name__ == "__main__":
    main()
