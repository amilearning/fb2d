"""
train_continual_fb_distill.py — Continual FB with simple distillation.

Same setup as train_continual_fb.py, but at the start of every phase k>=2 we
snapshot the agent (forward_net, backward_net, actor) as a frozen `teacher`
and add two distillation losses to the per-step update:

  L_distill_actor = MSE(actor(s_d, z_d),  teacher_actor(s_d, z_d))
  L_distill_F     = MSE(F(s_d, a_t, z_d), teacher_F(s_d, a_t, z_d))

where (s_d, z_d) are sampled fresh each step:
  s_d : uniform in [-1,1]^2  (we know the env's state space — data-free)
  z_d : agent.sample_z()      (uniform on z-sphere)
  a_t : teacher_actor(s_d, z_d)

Purpose: see if a vanilla distillation regularizer alone (no replay buffer of
old states) is enough to reduce forgetting on the z_dim=2 baseline. This is
the "naive FB + distillation" cell of the experiment matrix.
"""

import argparse
import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from env import Nav2D, ReplayBuffer
from env_quadrant import Nav2DQuadrant, QUADRANT_BOUNDS
from fb_agent import FBAgent
from train_continual_fb import (
    QUADRANTS,
    in_quadrant,
    evaluate_on_all_quadrants,
)


def snapshot_teacher(agent):
    """Frozen deepcopy of (forward_net, backward_net, actor)."""
    teacher = {
        "forward": copy.deepcopy(agent.forward_net).eval(),
        "backward": copy.deepcopy(agent.backward_net).eval(),
        "actor": copy.deepcopy(agent.actor).eval(),
    }
    for net in teacher.values():
        for p in net.parameters():
            p.requires_grad = False
    return teacher


def distillation_step(agent, teacher, batch, args, device):
    """
    One optimizer step that backprops only the distillation losses.
    States come from the *current task's* training batch (not uniform).
    """
    s_d = batch["obs"]
    n = s_d.size(0)
    z_d = agent.sample_z(n)

    with torch.no_grad():
        a_teacher = teacher["actor"](s_d, z_d)
        Ft1, Ft2 = teacher["forward"](s_d, a_teacher, z_d)
        Bt = teacher["backward"](s_d)

    # Actor distillation
    a_student = agent.actor(s_d, z_d)
    actor_distill_loss = F.mse_loss(a_student, a_teacher)

    agent.actor_opt.zero_grad()
    (args.distill_actor_coef * actor_distill_loss).backward()
    agent.actor_opt.step()

    # F + B distillation
    Fs1, Fs2 = agent.forward_net(s_d, a_teacher, z_d)
    Bs = agent.backward_net(s_d)
    f_distill_loss = 0.5 * (F.mse_loss(Fs1, Ft1) + F.mse_loss(Fs2, Ft2))
    b_distill_loss = F.mse_loss(Bs, Bt)
    fb_distill_loss = f_distill_loss + b_distill_loss

    agent.fb_opt.zero_grad()
    (args.distill_fb_coef * fb_distill_loss).backward()
    agent.fb_opt.step()

    return actor_distill_loss.item(), fb_distill_loss.item()


def train_one_phase(agent, teacher, quadrant, args, device):
    print(f"\n{'='*60}\n  PHASE: training on {quadrant}"
          f"  (distill={'on' if teacher is not None else 'off'})\n{'='*60}")

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

            if teacher is not None:
                ad, fd = distillation_step(agent, teacher, batch, args, device)
            else:
                ad = fd = 0.0

            if step % 5000 == 0:
                elapsed = time.time() - start
                print(f"    {quadrant} step {step:5d}/{args.steps_per_phase}  "
                      f"fb={metrics['fb_loss']:.2f}  "
                      f"actor={metrics['actor_loss']:.2f}  "
                      f"d_actor={ad:.4f}  d_fb={fd:.4f}  "
                      f"({elapsed:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_phase", type=int, default=15_000)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--update_z_every", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--max_steps_ep", type=int, default=200)
    parser.add_argument("--z_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill_batch_size", type=int, default=512)
    parser.add_argument("--distill_actor_coef", type=float, default=1.0)
    parser.add_argument("--distill_fb_coef", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints_continual_distill")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  z_dim={args.z_dim}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    agent = FBAgent(
        obs_dim=2,
        action_dim=2,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device,
    )

    perf = np.zeros((4, 4))

    teacher = None  # phase 1: no teacher
    for phase_idx, q in enumerate(QUADRANTS):
        train_one_phase(agent, teacher, q, args, device)
        print(f"\n[Eval] After phase {phase_idx+1} (just trained on {q}):")
        row = evaluate_on_all_quadrants(agent, args, device)
        perf[phase_idx] = row
        for j, target_q in enumerate(QUADRANTS):
            print(f"    target {target_q}: {row[j]:.3f}")
        # Snapshot teacher for the *next* phase
        teacher = snapshot_teacher(agent)

    print(f"\n{'='*60}")
    print(f"  CONTINUAL FB + DISTILL — PERFORMANCE MATRIX  (z_dim={args.z_dim})")
    print(f"{'='*60}")
    print(f"  {'After phase':<15}" + "".join(f"{q:>10}" for q in QUADRANTS))
    print(f"  {'-'*55}")
    for i, q in enumerate(QUADRANTS):
        vals = "".join(f"{perf[i,j]:>10.3f}" for j in range(4))
        print(f"  {f'P{i+1} ({q})':<15}{vals}")

    print(f"\n  FORGETTING:")
    forgets = []
    for i in range(4):
        forgetting = perf[i, i] - perf[3, i]
        forgets.append(forgetting)
        print(f"    {QUADRANTS[i]}: trained={perf[i,i]:.3f}  final={perf[3,i]:.3f}  "
              f"forgot={forgetting:.3f}")
    print(f"  Mean forgetting: {np.mean(forgets):.3f}")

    ckpt = {
        "forward_net": agent.forward_net.state_dict(),
        "backward_net": agent.backward_net.state_dict(),
        "actor": agent.actor.state_dict(),
        "args": vars(args),
        "perf_matrix": perf,
    }
    torch.save(ckpt, os.path.join(save_dir, "fb_agent_continual_distill.pt"))

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
    ax.set_title(f"Continual FB + Distillation (z_dim={args.z_dim})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "continual_fb_distill_matrix.png"),
                dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\nSaved to: {save_dir}/")


if __name__ == "__main__":
    main()
