"""
train_continual_fb_vmf.py — Continual FB with a vMF mixture over z, plus
distillation against the previous-phase snapshot.

Key idea (z_dim=2, z lives on unit circle scaled by sqrt(2)):

  • Maintain a list of `ReparamVMFComponent`s, one added at the START of each
    phase. So during phase k there are k learnable vMF components on the
    unit circle.
  • Replace `agent.sample_z(n)` with sampling from a uniform mixture over the
    currently-active components. Reparam gradients flow into mu_angle and
    log_kappa of every component, so each component is pushed toward useful
    z directions.
  • The mix-with-B trick is preserved (still 50% mixed with B(next_obs)).
  • Distillation (same as train_continual_fb_distill.py): from phase 2 onward
    snapshot the previous agent and add MSE distill on (actor, F, B) using
    the *current task batch states* and z's sampled from the vMF mixture.

This is the "vMF mixture z-routing" cell — the planned next experiment in
the FB2D project memory.
"""

import argparse
import copy
import os
import sys
import time
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import vMF reparam from the Repres project
sys.path.insert(0, "/home/frl/Repres")
from vmf_reparam import ReparamVMFComponent  # noqa: E402

from env import Nav2D, ReplayBuffer
from env_quadrant import Nav2DQuadrant, QUADRANT_BOUNDS
from fb_agent import FBAgent
from train_continual_fb import (
    QUADRANTS,
    in_quadrant,
    evaluate_on_all_quadrants,
)


# Initial mu (in radians) for the per-quadrant components — pointed at the
# centre of the corresponding quadrant. Components are still fully learnable;
# this just gives a sane starting direction.
QUADRANT_INIT_ANGLE = {
    "Q1": math.pi / 4,           # +x +y
    "Q2": 3 * math.pi / 4,       # -x +y
    "Q3": -3 * math.pi / 4,      # -x -y
    "Q4": -math.pi / 4,          # +x -y
}


class VMFMixture:
    """Uniform mixture over a growing list of ReparamVMFComponents."""

    def __init__(self, lr, device):
        self.components = []  # list of ReparamVMFComponent
        self.lr = lr
        self.device = device
        self.opt = None  # rebuilt each time we add a component

    def add_component(self, init_angle, init_kappa=10.0):
        comp = ReparamVMFComponent(init_angle=init_angle,
                                    init_kappa=init_kappa).to(self.device)
        self.components.append(comp)
        params = []
        for c in self.components:
            params += list(c.parameters())
        self.opt = torch.optim.Adam(params, lr=self.lr)

    def sample(self, n):
        """Sample n z's from the uniform mixture, with reparam gradients."""
        k = len(self.components)
        assert k >= 1
        # assignments
        assigns = torch.randint(0, k, (n,), device=self.device)
        out = torch.empty(n, 2, device=self.device)
        for i, comp in enumerate(self.components):
            mask = (assigns == i)
            n_i = int(mask.sum().item())
            if n_i == 0:
                continue
            z_i = comp.sample(n_i, device=self.device)  # (n_i, 2) on unit circle
            out[mask] = z_i
        return out * math.sqrt(2.0)  # FB convention: |z| = sqrt(z_dim)


def patch_agent_with_mixture(agent, mixture):
    """Override agent.sample_z to draw from the vMF mixture (with grads)."""
    agent._mixture = mixture
    agent.sample_z = lambda n: mixture.sample(n)


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


def distillation_step(agent, teacher, batch, args):
    s_d = batch["obs"]
    n = s_d.size(0)
    with torch.no_grad():
        z_d = agent._mixture.sample(n)  # no grad path needed for distill z

        a_teacher = teacher["actor"](s_d, z_d)
        Ft1, Ft2 = teacher["forward"](s_d, a_teacher, z_d)
        Bt = teacher["backward"](s_d)

    a_student = agent.actor(s_d, z_d)
    actor_distill_loss = F.mse_loss(a_student, a_teacher)
    agent.actor_opt.zero_grad()
    (args.distill_actor_coef * actor_distill_loss).backward()
    agent.actor_opt.step()

    Fs1, Fs2 = agent.forward_net(s_d, a_teacher, z_d)
    Bs = agent.backward_net(s_d)
    f_distill_loss = 0.5 * (F.mse_loss(Fs1, Ft1) + F.mse_loss(Fs2, Ft2))
    b_distill_loss = F.mse_loss(Bs, Bt)
    fb_distill_loss = f_distill_loss + b_distill_loss
    agent.fb_opt.zero_grad()
    (args.distill_fb_coef * fb_distill_loss).backward()
    agent.fb_opt.step()

    return actor_distill_loss.item(), fb_distill_loss.item()


def train_one_phase(agent, mixture, teacher, quadrant, args, device):
    print(f"\n{'='*60}\n  PHASE: training on {quadrant}  "
          f"(n_components={len(mixture.components)}, distill={'on' if teacher else 'off'})\n"
          f"{'='*60}")

    env = Nav2DQuadrant(quadrant=quadrant, max_steps=args.max_steps_ep,
                         seed=args.seed + hash(quadrant) % 10000)
    buffer = ReplayBuffer(args.buffer_capacity, env.obs_dim, env.action_dim)

    obs = env.reset()
    z = agent.sample_z(1).detach().squeeze(0)
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
            z = agent.sample_z(1).detach().squeeze(0)

        if step >= args.warmup_steps and buffer.size >= args.batch_size:
            batch = buffer.sample(args.batch_size, device=device)
            # FBAgent.update will call agent.sample_z (now mixture.sample) —
            # we need the vMF optimizer to step alongside the FB optimizer.
            mixture.opt.zero_grad()
            metrics = agent.update(batch)
            # Clip vMF grads to avoid kappa blow-up via implicit reparam.
            for c in mixture.components:
                if c.log_kappa.grad is not None:
                    c.log_kappa.grad.clamp_(-1.0, 1.0)
                if c.mu_angle.grad is not None:
                    c.mu_angle.grad.clamp_(-1.0, 1.0)
            mixture.opt.step()
            # Hard-clamp log_kappa so kappa stays in [1, 50] — implicit
            # reparam quadrature gets unstable for very large kappa.
            with torch.no_grad():
                for c in mixture.components:
                    c.log_kappa.clamp_(math.log(1.0), math.log(50.0))

            if teacher is not None:
                ad, fd = distillation_step(agent, teacher, batch, args)
            else:
                ad = fd = 0.0

            if step % 5000 == 0:
                elapsed = time.time() - start
                k_str = " ".join(
                    f"({c.mu_angle.item():.2f},{c.kappa.item():.1f})"
                    for c in mixture.components
                )
                print(f"    {quadrant} step {step:5d}/{args.steps_per_phase}  "
                      f"fb={metrics['fb_loss']:.2f}  "
                      f"actor={metrics['actor_loss']:.2f}  "
                      f"d_a={ad:.4f}  d_fb={fd:.4f}  "
                      f"({elapsed:.0f}s)  comps={k_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_phase", type=int, default=15_000)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--update_z_every", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--max_steps_ep", type=int, default=200)
    parser.add_argument("--z_dim", type=int, default=2)  # MUST be 2 for vMF on circle
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vmf_lr", type=float, default=1e-4)
    parser.add_argument("--init_kappa", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill_actor_coef", type=float, default=1.0)
    parser.add_argument("--distill_fb_coef", type=float, default=1.0)
    parser.add_argument("--use_distill", action="store_true", default=True)
    parser.add_argument("--no_distill", dest="use_distill", action="store_false")
    parser.add_argument("--save_dir", type=str, default="checkpoints_continual_vmf")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    assert args.z_dim == 2, "vMF mixture variant requires z_dim=2 (unit circle)."

    device = torch.device(args.device)
    print(f"Device: {device}  z_dim={args.z_dim}  use_distill={args.use_distill}")

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

    mixture = VMFMixture(lr=args.vmf_lr, device=device)
    patch_agent_with_mixture(agent, mixture)

    perf = np.zeros((4, 4))

    teacher = None
    for phase_idx, q in enumerate(QUADRANTS):
        # Add a fresh vMF component for this phase, initialized at the
        # quadrant centre angle.
        mixture.add_component(init_angle=QUADRANT_INIT_ANGLE[q],
                              init_kappa=args.init_kappa)

        train_one_phase(agent, mixture,
                        teacher if args.use_distill else None,
                        q, args, device)

        print(f"\n[Eval] After phase {phase_idx+1} (just trained on {q}):")
        row = evaluate_on_all_quadrants(agent, args, device)
        perf[phase_idx] = row
        for j, target_q in enumerate(QUADRANTS):
            print(f"    target {target_q}: {row[j]:.3f}")

        teacher = snapshot_teacher(agent)

    print(f"\n{'='*60}")
    print(f"  CONTINUAL FB + vMF MIXTURE + DISTILL — PERFORMANCE MATRIX")
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

    print(f"\n  Final vMF components:")
    for i, c in enumerate(mixture.components):
        print(f"    [{QUADRANTS[i]}] mu_angle={c.mu_angle.item():+.3f} rad  "
              f"kappa={c.kappa.item():.2f}")

    ckpt = {
        "forward_net": agent.forward_net.state_dict(),
        "backward_net": agent.backward_net.state_dict(),
        "actor": agent.actor.state_dict(),
        "vmf_components": [
            {"mu_angle": c.mu_angle.item(), "kappa": c.kappa.item()}
            for c in mixture.components
        ],
        "args": vars(args),
        "perf_matrix": perf,
    }
    torch.save(ckpt, os.path.join(save_dir, "fb_agent_continual_vmf.pt"))

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
    ax.set_title("Continual FB + vMF mixture + distillation",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "continual_fb_vmf_matrix.png"),
                dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\nSaved to: {save_dir}/")


if __name__ == "__main__":
    main()
