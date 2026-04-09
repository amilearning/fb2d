"""
env.py — Simple 2D continuous navigation environment.

State:  s in [-1, 1]^2  (position)
Action: a in [-1, 1]^2  (velocity command, scaled internally)
Dynamics: s_{t+1} = clip(s_t + max_speed * a_t, -1, 1)

No reward is provided during FB training — rewards are inferred at test time.
"""

import numpy as np
import torch


class Nav2D:
    """Minimal 2D continuous navigation env (no reward during training)."""

    def __init__(self, max_speed=0.05, max_steps=200, seed=None):
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.obs_dim = 2
        self.action_dim = 2
        self.rng = np.random.RandomState(seed)
        self.state = None
        self.t = 0

    def reset(self, state=None):
        if state is None:
            self.state = self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        else:
            self.state = np.asarray(state, dtype=np.float32)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.state = np.clip(self.state + self.max_speed * action, -1.0, 1.0)
        self.t += 1
        done = self.t >= self.max_steps
        # No reward — FB doesn't need it during training
        return self.state.copy(), 0.0, done, {}

    def random_action(self):
        return self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)


class ReplayBuffer:
    """Simple FIFO replay buffer for (s, a, s', discount) transitions."""

    def __init__(self, capacity, obs_dim, action_dim, discount=0.98):
        self.capacity = capacity
        self.discount = discount
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, act, next_obs):
        self.obs[self.idx] = obs
        self.act[self.idx] = act
        self.next_obs[self.idx] = next_obs
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(device),
            "act": torch.from_numpy(self.act[idx]).to(device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(device),
        }
