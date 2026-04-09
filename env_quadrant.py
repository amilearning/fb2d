"""
env_quadrant.py — 2D nav env constrained to a single quadrant.

Quadrants:
  Q1: x in [0, 1], y in [0, 1]
  Q2: x in [-1, 0], y in [0, 1]
  Q3: x in [-1, 0], y in [-1, 0]
  Q4: x in [0, 1], y in [-1, 0]
"""

import numpy as np


QUADRANT_BOUNDS = {
    "Q1": ((0.0, 1.0), (0.0, 1.0)),
    "Q2": ((-1.0, 0.0), (0.0, 1.0)),
    "Q3": ((-1.0, 0.0), (-1.0, 0.0)),
    "Q4": ((0.0, 1.0), (-1.0, 0.0)),
}


class Nav2DQuadrant:
    """2D nav env restricted to a single quadrant."""

    def __init__(self, quadrant="Q1", max_speed=0.05, max_steps=200, seed=None):
        assert quadrant in QUADRANT_BOUNDS
        self.quadrant = quadrant
        (self.xlo, self.xhi), (self.ylo, self.yhi) = QUADRANT_BOUNDS[quadrant]
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.obs_dim = 2
        self.action_dim = 2
        self.rng = np.random.RandomState(seed)
        self.state = None
        self.t = 0

    def reset(self, state=None):
        if state is None:
            x = self.rng.uniform(self.xlo, self.xhi)
            y = self.rng.uniform(self.ylo, self.yhi)
            self.state = np.array([x, y], dtype=np.float32)
        else:
            s = np.asarray(state, dtype=np.float32)
            s[0] = np.clip(s[0], self.xlo, self.xhi)
            s[1] = np.clip(s[1], self.ylo, self.yhi)
            self.state = s
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        new_state = self.state + self.max_speed * action
        new_state[0] = np.clip(new_state[0], self.xlo, self.xhi)
        new_state[1] = np.clip(new_state[1], self.ylo, self.yhi)
        self.state = new_state.astype(np.float32)
        self.t += 1
        done = self.t >= self.max_steps
        return self.state.copy(), 0.0, done, {}

    def random_action(self):
        return self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
