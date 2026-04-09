"""
train_continual_fb_ssvmf.py — Continual FB with Sensitivity-Seeded vMF (SSVMF).

For each phase k:
  1. Sample z from a fixed vMF(mu_k, kappa) on the unit circle (scaled by
     sqrt(2) per FB convention).
  2. Train F, B, Actor on the current quadrant with the standard FB recipe,
     plus a distillation step against a frozen teacher snapshot.

After each phase:
  - Snapshot teacher (forward_net, backward_net, actor).
  - Compute the actor's sensitivity to z on a 1D angular grid (theta in
    [-pi, pi)) using finite differences over a probe set of states from
    the next quadrant.
  - Choose mu_{k+1} = argmin_theta (sensitivity - gamma * angular_dist_to_prev_mus).
    The repulsion term ensures the new mu is far from previously used ones
    (otherwise argmin can collapse onto an old mu when the sensitivity
    curve is locally flat).

Distillation z's are drawn from a categorical built from softmax(+beta*s),
i.e. concentrated where the teacher actor cares most about z. Distillation
states come from the current quadrant's batch.
"""

import argparse
import copy
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
from torch.distributions import VonMises

from env import Nav2D, ReplayBuffer
from env_quadrant import Nav2DQuadrant, QUADRANT_BOUNDS
from fb_agent import FBAgent
from train_continual_fb import (
    QUADRANTS,
    in_quadrant,
    evaluate_on_all_quadrants,
)


SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# vMF prior (non-reparam — used as a fixed sampling distribution per phase)
# ---------------------------------------------------------------------------

def make_grid(n_grid, device):
    return torch.linspace(-math.pi, math.pi - 2 * math.pi / n_grid, n_grid,
                          device=device)


def theta_to_z(theta):
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * SQRT2


def sample_vmf_z(mu, kappa, n, device):
    """n z's drawn from vMF(mu, kappa) on the unit circle, scaled by sqrt(2)."""
    with torch.no_grad():
        d = VonMises(loc=torch.tensor(float(mu), device=device),
                     concentration=torch.tensor(float(kappa), device=device))
        theta = d.sample((n,))
    return theta_to_z(theta)


def sample_from_grid_probs(probs, n, device):
    n_bins = probs.size(0)
    bin_width = 2 * math.pi / n_bins
    cdf = torch.cumsum(probs, dim=-1)
    cdf = cdf / cdf[-1].clamp(min=1e-12)
    u = torch.rand(n, device=device)
    idx = torch.searchsorted(cdf, u, right=False).clamp(max=n_bins - 1)
    cdf_left = torch.where(idx > 0, cdf[idx - 1], torch.zeros_like(u))
    p_i = probs[idx]
    theta_left = -math.pi + idx.float() * bin_width
    theta = theta_left + (u - cdf_left) / (p_i + 1e-12) * bin_width
    theta = ((theta + math.pi) % (2 * math.pi)) - math.pi
    return theta


# ---------------------------------------------------------------------------
# Sensitivity computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_actor_sensitivity(teacher_actor, states, theta_grid):
    """
    s(theta_g) = E_s ||a_T(s, z(theta_g + delta)) - a_T(s, z(theta_g))||^2
    averaged over states. Returns (n_grid,) tensor.
    """
    n_grid = theta_grid.size(0)
    dtheta = 2 * math.pi / n_grid
    n = states.size(0)
    s = torch.zeros(n_grid, device=states.device)
    for g in range(n_grid):
        z = theta_to_z(theta_grid[g]).unsqueeze(0).expand(n, -1)
        z_plus = theta_to_z(theta_grid[g] + dtheta).unsqueeze(0).expand(n, -1)
        a0 = teacher_actor(states, z)
        ap = teacher_actor(states, z_plus)
        s[g] = (ap - a0).pow(2).sum(dim=-1).mean()
    return s


def pick_next_mu(s_curve, theta_grid, prev_mus, gamma_repulse, sigma_repulse):
    """
    Choose mu_{k+1} = argmin over theta of:
        s_norm(theta) - gamma_repulse * exp(-(angular_dist(theta, mu_j))^2 / sigma^2)
    Wait — we want to MINIMIZE sensitivity but be FAR from prev mus.
    So minimize: s_norm(theta) + gamma_repulse * sum_j exp(-d_j^2 / sigma^2)
    (the second term is a soft "you can't sit on top of an existing mu" penalty)
    """
    s = s_curve / (s_curve.mean() + 1e-12)
    score = s.clone()
    for mu_j in prev_mus:
        d = theta_grid - mu_j
        d = ((d + math.pi) % (2 * math.pi)) - math.pi  # wrap to [-pi, pi]
        score = score + gamma_repulse * torch.exp(-d.pow(2) / (sigma_repulse ** 2))
    idx = int(torch.argmin(score).item())
    return float(theta_grid[idx].item()), score


def snapshot_teacher(agent):
    teacher = {
        "forward": copy.deepcopy(agent.forward_net).eval(),
        "backward": copy.deepcopy(agent.backward_net).eval(),
        "actor": copy.deepcopy(agent.actor).eval(),
    }
    for net in teacher.values():
        for p in net.parameters():
            p.requires_grad = False
    return teacher


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def patch_agent_sample_z(agent, mu_k_ref, kappa, device):
    """Override agent.sample_z to draw from the current phase's vMF prior."""
    def _sample(n):
        return sample_vmf_z(mu_k_ref[0], kappa, n, device)
    agent.sample_z = _sample


def distillation_step(agent, teacher, batch, p_crit, args, device):
    """One distillation step on (current batch states, z_old~p_crit)."""
    s_d = batch["obs"]
    n = s_d.size(0)
    n_d = min(n, args.n_distill)
    s_d = s_d[:n_d]

    with torch.no_grad():
        theta_old = sample_from_grid_probs(p_crit, n_d, device)
        z_old = theta_to_z(theta_old)
        a_t = teacher["actor"](s_d, z_old)
        Ft1, Ft2 = teacher["forward"](s_d, a_t, z_old)
        Bt = teacher["backward"](s_d)

    # Actor distillation
    a_s = agent.actor(s_d, z_old)
    actor_loss = Fn.mse_loss(a_s, a_t)
    agent.actor_opt.zero_grad()
    (args.distill_actor_coef * actor_loss).backward()
    agent.actor_opt.step()

    # F + B distillation
    Fs1, Fs2 = agent.forward_net(s_d, a_t, z_old)
    Bs = agent.backward_net(s_d)
    fb_loss = 0.5 * (Fn.mse_loss(Fs1, Ft1) + Fn.mse_loss(Fs2, Ft2)) + Fn.mse_loss(Bs, Bt)
    agent.fb_opt.zero_grad()
    (args.distill_fb_coef * fb_loss).backward()
    agent.fb_opt.step()
    return actor_loss.item(), fb_loss.item()


def train_one_phase(agent, mu_k_ref, teacher, p_crit, quadrant, args, device):
    print(f"\n{'='*60}\n  PHASE: training on {quadrant}  "
          f"(mu={mu_k_ref[0]:+.3f} rad, kappa={args.kappa})  "
          f"distill={'on' if teacher else 'off'}\n{'='*60}")

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
                ad, fd = distillation_step(agent, teacher, batch, p_crit, args, device)
            else:
                ad = fd = 0.0

            if step % 5000 == 0:
                el = time.time() - start
                print(f"    {quadrant} step {step:5d}/{args.steps_per_phase}  "
                      f"fb={metrics['fb_loss']:.2f}  actor={metrics['actor_loss']:.2f}  "
                      f"d_a={ad:.4f}  d_fb={fd:.4f}  ({el:.0f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--mu0", type=float, default=0.0)
    parser.add_argument("--n_grid", type=int, default=128)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma_repulse", type=float, default=2.0)
    parser.add_argument("--sigma_repulse", type=float, default=0.6)
    parser.add_argument("--distill_actor_coef", type=float, default=1.0)
    parser.add_argument("--distill_fb_coef", type=float, default=1.0)
    parser.add_argument("--n_distill", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints_continual_ssvmf")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    assert args.z_dim == 2

    device = torch.device(args.device)
    print(f"Device: {device}  z_dim={args.z_dim}  kappa={args.kappa}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    agent = FBAgent(
        obs_dim=2, action_dim=2, z_dim=args.z_dim,
        hidden_dim=args.hidden_dim, lr=args.lr, device=device,
    )

    theta_grid = make_grid(args.n_grid, device)
    chosen_mus = [args.mu0]
    mu_k_ref = [args.mu0]  # mutable single-element holder for sample_z closure
    patch_agent_sample_z(agent, mu_k_ref, args.kappa, device)

    perf = np.zeros((4, 4))
    sensitivity_log = []
    teacher = None
    p_crit = None

    for phase_idx, q in enumerate(QUADRANTS):
        # Set the prior mu for this phase
        mu_k_ref[0] = chosen_mus[phase_idx]

        train_one_phase(agent, mu_k_ref, teacher, p_crit, q, args, device)

        # Eval
        print(f"\n[Eval] After phase {phase_idx+1} ({q}):")
        row = evaluate_on_all_quadrants(agent, args, device)
        perf[phase_idx] = row
        for j, target_q in enumerate(QUADRANTS):
            print(f"    target {target_q}: {row[j]:.3f}")

        # Snapshot teacher
        teacher = snapshot_teacher(agent)

        # Compute sensitivity for the NEXT quadrant's data (probe states)
        if phase_idx < len(QUADRANTS) - 1:
            next_q = QUADRANTS[phase_idx + 1]
            probe_env = Nav2DQuadrant(quadrant=next_q, max_steps=args.max_steps_ep,
                                       seed=args.seed + 9999)
            probe_states = []
            o = probe_env.reset()
            for _ in range(512):
                probe_states.append(o.copy())
                a = probe_env.random_action()
                o, _, d, _ = probe_env.step(a)
                if d:
                    o = probe_env.reset()
            probe_states = torch.as_tensor(np.array(probe_states),
                                            dtype=torch.float32, device=device)

            s_curve = compute_actor_sensitivity(teacher["actor"], probe_states, theta_grid)
            new_mu, score = pick_next_mu(s_curve, theta_grid, chosen_mus,
                                          args.gamma_repulse, args.sigma_repulse)
            chosen_mus.append(new_mu)
            print(f"    [SSVMF] sensitivity argmin (with repulsion): "
                  f"new mu = {new_mu:+.3f} rad")

            # Build distillation prior over high-sensitivity bins
            s_norm = s_curve / (s_curve.mean() + 1e-12)
            p_crit = Fn.softmax(args.beta * s_norm, dim=-1)

            sensitivity_log.append({
                "after_phase": q,
                "next_phase": next_q,
                "s": s_curve.cpu().numpy(),
                "score": score.cpu().numpy(),
                "chosen_mu": new_mu,
                "all_mus": list(chosen_mus),
            })

    # Print results
    print(f"\n{'='*60}\n  CONTINUAL FB + SSVMF — PERFORMANCE MATRIX\n{'='*60}")
    print(f"  {'After phase':<15}" + "".join(f"{q:>10}" for q in QUADRANTS))
    print(f"  {'-'*55}")
    for i, q in enumerate(QUADRANTS):
        vals = "".join(f"{perf[i,j]:>10.3f}" for j in range(4))
        print(f"  {f'P{i+1} ({q})':<15}{vals}")

    print(f"\n  Forgetting:")
    forgets = []
    for i in range(4):
        f_i = perf[i, i] - perf[3, i]
        forgets.append(f_i)
        print(f"    {QUADRANTS[i]}: trained={perf[i,i]:.3f}  final={perf[3,i]:.3f}  "
              f"forgot={f_i:.3f}")
    print(f"  Mean forgetting: {np.mean(forgets):.3f}")
    print(f"  Chosen mus: {chosen_mus}")

    # Save checkpoint
    ckpt = {
        "forward_net": agent.forward_net.state_dict(),
        "backward_net": agent.backward_net.state_dict(),
        "actor": agent.actor.state_dict(),
        "args": vars(args),
        "perf_matrix": perf,
        "chosen_mus": chosen_mus,
    }
    torch.save(ckpt, os.path.join(save_dir, "fb_agent_ssvmf.pt"))
    np.save(os.path.join(save_dir, "perf_matrix.npy"), perf)

    # Plot perf matrix
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(perf, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(QUADRANTS)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"After P{i+1} ({q})" for i, q in enumerate(QUADRANTS)])
    for i in range(4):
        for j in range(4):
            color = "white" if perf[i, j] < 0.5 else "black"
            ax.text(j, i, f"{perf[i,j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="fraction in target")
    ax.set_title("Continual FB + SSVMF (z_dim=2)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ssvmf_matrix.png"),
                dpi=140, bbox_inches="tight")
    plt.close()

    # Plot sensitivity curves and chosen mus per transition
    if sensitivity_log:
        n = len(sensitivity_log)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
        if n == 1: axes = [axes]
        theta_np = theta_grid.cpu().numpy()
        for ax, entry in zip(axes, sensitivity_log):
            s = entry["s"] / (entry["s"].mean() + 1e-12)
            score = entry["score"]
            ax.plot(theta_np, s, 'k-', label='sensitivity (norm)')
            ax.plot(theta_np, score, 'b--', alpha=0.6, label='score (with repulse)')
            for mu in entry["all_mus"][:-1]:
                ax.axvline(mu, color='C7', linestyle=':', alpha=0.7)
            ax.axvline(entry["chosen_mu"], color='C2', linewidth=2,
                       label=f'next mu={entry["chosen_mu"]:+.3f}')
            ax.set_title(f"After {entry['after_phase']}, picking mu for {entry['next_phase']}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_ylabel("sens / score")
        axes[-1].set_xlabel("theta (rad)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ssvmf_sensitivity.png"),
                    dpi=140, bbox_inches="tight")
        plt.close()

    print(f"\nSaved to: {save_dir}/")


if __name__ == "__main__":
    main()
