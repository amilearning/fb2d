"""
env_quaddyn.py — Nav2D where the action transformation depends on the
QUADRANT of the current state.

  Q1 (x>=0, y>=0): a' = (+ax, +ay)
  Q2 (x<0,  y>=0): a' = (-ax, +ay)   x flipped
  Q3 (x<0,  y<0):  a' = (-ax, -ay)   x and y flipped
  Q4 (x>=0, y<0):  a' = (+ax, -ay)   y flipped

Two env classes:
  Nav2DQuadDyn          — free movement in [-1,1]^2 with state-dependent dynamics
                           (used for evaluation rollouts on the full box)
  Nav2DQuadDynRestricted — same dynamics but training is restricted to one
                           quadrant (used for per-quadrant data collection)
"""

import numpy as np
from env_quadrant import QUADRANT_BOUNDS


def state_quadrant(s):
    x, y = float(s[0]), float(s[1])
    if x >= 0 and y >= 0: return "Q1"
    if x < 0  and y >= 0: return "Q2"
    if x < 0  and y < 0:  return "Q3"
    return "Q4"


def transform_action(a, q):
    sx = 1.0 if q in ("Q1", "Q4") else -1.0
    sy = 1.0 if q in ("Q1", "Q2") else -1.0
    return np.array([sx * a[0], sy * a[1]], dtype=np.float32)


class Nav2DQuadDyn:
    """Full [-1,1]^2 box with quadrant-dependent action signs."""

    def __init__(self, max_speed=0.05, max_steps=200, seed=None):
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.obs_dim = 2; self.action_dim = 2
        self.rng = np.random.RandomState(seed)
        self.state = None; self.t = 0

    def reset(self, state=None):
        if state is None:
            self.state = self.rng.uniform(-1, 1, size=(2,)).astype(np.float32)
        else:
            self.state = np.asarray(state, dtype=np.float32)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        q = state_quadrant(self.state)
        a_eff = transform_action(action, q)
        self.state = np.clip(self.state + self.max_speed * a_eff, -1.0, 1.0)
        self.t += 1
        return self.state.copy(), 0.0, self.t >= self.max_steps, {}

    def random_action(self):
        return self.rng.uniform(-1, 1, size=(2,)).astype(np.float32)


class Nav2DQuadDynRestricted:
    """Same per-state dynamics but states restricted to one quadrant."""

    def __init__(self, quadrant="Q1", max_speed=0.05, max_steps=200, seed=None):
        assert quadrant in QUADRANT_BOUNDS
        self.quadrant = quadrant
        (self.xlo, self.xhi), (self.ylo, self.yhi) = QUADRANT_BOUNDS[quadrant]
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.obs_dim = 2; self.action_dim = 2
        self.rng = np.random.RandomState(seed)
        self.state = None; self.t = 0

    def reset(self, state=None):
        if state is None:
            self.state = np.array([self.rng.uniform(self.xlo, self.xhi),
                                    self.rng.uniform(self.ylo, self.yhi)], dtype=np.float32)
        else:
            s = np.asarray(state, dtype=np.float32)
            self.state = np.array([np.clip(s[0], self.xlo, self.xhi),
                                    np.clip(s[1], self.ylo, self.yhi)], dtype=np.float32)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Always apply this restricted env's quadrant transform (the state
        # never leaves this quadrant by construction).
        a_eff = transform_action(action, self.quadrant)
        new = self.state + self.max_speed * a_eff
        new[0] = np.clip(new[0], self.xlo, self.xhi)
        new[1] = np.clip(new[1], self.ylo, self.yhi)
        self.state = new.astype(np.float32)
        self.t += 1
        return self.state.copy(), 0.0, self.t >= self.max_steps, {}

    def random_action(self):
        return self.rng.uniform(-1, 1, size=(2,)).astype(np.float32)
