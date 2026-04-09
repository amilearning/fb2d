"""
fb_agent.py — FB-DDPG agent for continuous control.

Implements the three losses:
  1. FB TD loss   — twin successor measure regression on M = F @ B^T
  2. Ortho loss   — push E[B B^T] toward I (prevents B collapse)
  3. Actor loss   — DDPG-style: -min(F1·z, F2·z).mean()

Polyak target nets for both F and B.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import ForwardNet, BackwardNet, Actor


class FBAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        z_dim=32,
        hidden_dim=256,
        lr=1e-4,
        discount=0.98,
        tau=0.01,
        ortho_coef=1.0,
        mix_ratio=0.5,
        actor_noise=0.2,
        device="cpu",
    ):
        self.z_dim = z_dim
        self.discount = discount
        self.tau = tau
        self.ortho_coef = ortho_coef
        self.mix_ratio = mix_ratio
        self.actor_noise = actor_noise
        self.device = device

        # Networks
        self.forward_net = ForwardNet(obs_dim, action_dim, z_dim, hidden_dim).to(device)
        self.backward_net = BackwardNet(obs_dim, z_dim, hidden_dim).to(device)
        self.actor = Actor(obs_dim, action_dim, z_dim, hidden_dim).to(device)

        # Target networks
        self.forward_target = copy.deepcopy(self.forward_net)
        self.backward_target = copy.deepcopy(self.backward_net)
        for p in self.forward_target.parameters():
            p.requires_grad = False
        for p in self.backward_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.fb_opt = optim.Adam(
            list(self.forward_net.parameters()) + list(self.backward_net.parameters()),
            lr=lr,
        )
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

    # -----------------------------------------------------------------
    # z sampling
    # -----------------------------------------------------------------

    def sample_z(self, n):
        """Sample z uniformly on sphere of radius sqrt(z_dim)."""
        z = torch.randn(n, self.z_dim, device=self.device)
        z = F.normalize(z, dim=-1) * math.sqrt(self.z_dim)
        return z

    def mix_z_with_B(self, z, obs_for_B):
        """Replace `mix_ratio` fraction of z with B(random_obs)."""
        n = z.size(0)
        n_mix = int(n * self.mix_ratio)
        if n_mix == 0:
            return z
        with torch.no_grad():
            perm = torch.randperm(n, device=self.device)[:n_mix]
            z_mixed = self.backward_net(obs_for_B[perm]).detach()
        z = z.clone()
        z[:n_mix] = z_mixed
        return z

    # -----------------------------------------------------------------
    # Acting
    # -----------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs, z, noise=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        action = self.actor(obs_t, z)
        if noise:
            action = action + torch.randn_like(action) * self.actor_noise
            action = action.clamp(-1.0, 1.0)
        return action.squeeze(0).cpu().numpy()

    # -----------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------

    def update(self, batch):
        obs = batch["obs"]
        act = batch["act"]
        next_obs = batch["next_obs"]
        n = obs.size(0)

        # Sample z and mix
        z = self.sample_z(n)
        z = self.mix_z_with_B(z, next_obs)

        # ---- FB loss ----
        with torch.no_grad():
            next_action = self.actor(next_obs, z)
            next_action = next_action + torch.randn_like(next_action) * self.actor_noise
            next_action = next_action.clamp(-1.0, 1.0)

            target_F1, target_F2 = self.forward_target(next_obs, next_action, z)
            target_B = self.backward_target(next_obs)
            target_M1 = target_F1 @ target_B.T
            target_M2 = target_F2 @ target_B.T
            target_M = torch.minimum(target_M1, target_M2)

        F1, F2 = self.forward_net(obs, act, z)
        B = self.backward_net(next_obs)

        M1 = F1 @ B.T  # (n, n)
        M2 = F2 @ B.T

        I = torch.eye(n, device=self.device)
        off_diag = ~I.bool()

        # Off-diagonal: TD regression
        fb_offdiag = 0.0
        for M in [M1, M2]:
            diff = M - self.discount * target_M
            fb_offdiag = fb_offdiag + 0.5 * (diff[off_diag].pow(2)).mean()

        # Diagonal: push up (Dirac of true next state)
        fb_diag = -sum(M.diag().mean() for M in [M1, M2])

        fb_loss = fb_offdiag + fb_diag

        # ---- Ortho loss on B ----
        Cov = B @ B.T
        ortho_diag = -2.0 * Cov.diag().mean()
        ortho_offdiag = (Cov[off_diag]).pow(2).mean()
        ortho_loss = ortho_diag + ortho_offdiag

        total_fb_loss = fb_loss + self.ortho_coef * ortho_loss

        self.fb_opt.zero_grad()
        total_fb_loss.backward()
        self.fb_opt.step()

        # ---- Actor loss ----
        # Resample z (fresh, no gradient through B for actor)
        with torch.no_grad():
            z_actor = self.sample_z(n)
        action_pi = self.actor(obs, z_actor)
        F1_pi, F2_pi = self.forward_net(obs, action_pi, z_actor)
        Q1 = (F1_pi * z_actor).sum(dim=-1)
        Q2 = (F2_pi * z_actor).sum(dim=-1)
        Q = torch.minimum(Q1, Q2)
        actor_loss = -Q.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Polyak target update ----
        with torch.no_grad():
            for p, tp in zip(self.forward_net.parameters(),
                             self.forward_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.backward_net.parameters(),
                             self.backward_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "fb_loss": fb_loss.item(),
            "fb_offdiag": fb_offdiag.item(),
            "fb_diag": fb_diag.item(),
            "ortho_loss": ortho_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    # -----------------------------------------------------------------
    # Zero-shot inference
    # -----------------------------------------------------------------

    @torch.no_grad()
    def infer_z_from_goal(self, goal_obs):
        """z = B(goal), normalized to sphere."""
        goal_t = torch.as_tensor(goal_obs, dtype=torch.float32, device=self.device)
        if goal_t.dim() == 1:
            goal_t = goal_t.unsqueeze(0)
        z = self.backward_net(goal_t)  # already normalized to sqrt(z_dim)
        return z.squeeze(0)

    @torch.no_grad()
    def infer_z_from_rewards(self, obs_samples, rewards):
        """z = (1/N) sum_i r_i * B(s_i), normalized to sphere of radius sqrt(z_dim)."""
        obs_t = torch.as_tensor(obs_samples, dtype=torch.float32, device=self.device)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        B = self.backward_net(obs_t)  # (N, z_dim)
        z = (r_t.unsqueeze(-1) * B).mean(dim=0)
        z = F.normalize(z, dim=-1) * math.sqrt(self.z_dim)
        return z
