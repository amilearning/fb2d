"""
networks.py — F (forward), B (backward), and Actor networks for FB.

Following the FB-DDPG recipe (Touati & Ollivier 2021, controllable_agent):
  F(s, a, z): twin critic, returns z_dim-dimensional vectors
  B(s):       maps state to z_dim feature, L2-normalized to sqrt(z_dim)
  Actor(s,z): tanh-squashed deterministic policy (DDPG/TD3 style)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ForwardNet(nn.Module):
    """
    F(s, a, z) -> R^{z_dim}, twin critic.

    Two preprocessing towers:
      - obs_action -> hidden
      - obs_z      -> hidden
    Concatenated and passed through trunk + two heads.
    """

    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim=256, feature_dim=128):
        super().__init__()
        self.obs_action_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )
        self.obs_z_net = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head1 = mlp([hidden_dim, hidden_dim, z_dim])
        self.head2 = mlp([hidden_dim, hidden_dim, z_dim])

    def forward(self, obs, action, z):
        h_oa = self.obs_action_net(torch.cat([obs, action], dim=-1))
        h_oz = self.obs_z_net(torch.cat([obs, z], dim=-1))
        h = self.trunk(torch.cat([h_oa, h_oz], dim=-1))
        return self.head1(h), self.head2(h)


class BackwardNet(nn.Module):
    """B(s) -> R^{z_dim}, L2-normalized to sphere of radius sqrt(z_dim)."""

    def __init__(self, obs_dim, z_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.z_dim = z_dim

    def forward(self, obs):
        b = self.net(obs)
        b = F.normalize(b, dim=-1) * math.sqrt(self.z_dim)
        return b


class Actor(nn.Module):
    """Deterministic actor: pi(s, z) -> a in [-1, 1]^{action_dim}."""

    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim=256, feature_dim=128):
        super().__init__()
        self.obs_z_net = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, z):
        h = self.obs_z_net(torch.cat([obs, z], dim=-1))
        return torch.tanh(self.trunk(h))
