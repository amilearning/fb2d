"""
env_quaddisk.py — 2D nav env, valid region = (one quadrant) ∪ (central disk).

Each task k corresponds to one quadrant Q1..Q4 plus a shared central disk of
radius `disk_r` around the origin. The disk pokes into all four quadrants, so
every task overlaps every other task on that central area.

Step semantics: clip the proposed next state to the nearest valid point.
"""

import numpy as np
from env_quadrant import QUADRANT_BOUNDS


def _clip_to_quadrant(s, q):
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
    return np.array([np.clip(s[0], xlo, xhi), np.clip(s[1], ylo, yhi)],
                    dtype=np.float32)


def _clip_to_disk(s, r):
    n = float(np.linalg.norm(s))
    if n <= r or n == 0.0:
        return s.astype(np.float32)
    return (s * (r / n)).astype(np.float32)


def _in_quadrant(s, q):
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
    return (s[0] >= xlo) and (s[0] <= xhi) and (s[1] >= ylo) and (s[1] <= yhi)


def _in_disk(s, r):
    return float(np.linalg.norm(s)) <= r


def in_quaddisk(s, q, r):
    return _in_quadrant(s, q) or _in_disk(s, r)


def project_to_quaddisk(s, q, r):
    """Snap a state to the nearest valid point in (quadrant_q ∪ disk_r)."""
    s = np.asarray(s, dtype=np.float32)
    if in_quaddisk(s, q, r):
        return s
    cand_q = _clip_to_quadrant(s, q)
    cand_d = _clip_to_disk(s, r)
    if np.linalg.norm(cand_q - s) <= np.linalg.norm(cand_d - s):
        return cand_q
    return cand_d


class Nav2DQuadDisk:
    """Nav2D restricted to one quadrant plus a shared central disk."""

    def __init__(self, quadrant="Q1", disk_r=0.4, max_speed=0.05,
                 max_steps=200, seed=None):
        assert quadrant in QUADRANT_BOUNDS
        self.quadrant = quadrant
        self.disk_r = float(disk_r)
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.obs_dim = 2
        self.action_dim = 2
        self.rng = np.random.RandomState(seed)
        self.state = None
        self.t = 0

    def _sample_reset(self):
        # Rejection-sample uniformly from (quadrant ∪ disk).
        (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[self.quadrant]
        for _ in range(200):
            # Pick uniformly from the bounding box of (quadrant ∪ disk).
            bx_lo = min(xlo, -self.disk_r); bx_hi = max(xhi, self.disk_r)
            by_lo = min(ylo, -self.disk_r); by_hi = max(yhi, self.disk_r)
            x = self.rng.uniform(bx_lo, bx_hi)
            y = self.rng.uniform(by_lo, by_hi)
            s = np.array([x, y], dtype=np.float32)
            if in_quaddisk(s, self.quadrant, self.disk_r):
                return s
        # Fallback: random point inside the quadrant.
        return np.array([self.rng.uniform(xlo, xhi),
                          self.rng.uniform(ylo, yhi)], dtype=np.float32)

    def reset(self, state=None):
        if state is None:
            self.state = self._sample_reset()
        else:
            self.state = project_to_quaddisk(state, self.quadrant, self.disk_r)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        proposed = self.state + self.max_speed * action
        # Also keep inside the global [-1,1]^2 box.
        proposed = np.clip(proposed, -1.0, 1.0)
        self.state = project_to_quaddisk(proposed, self.quadrant, self.disk_r)
        self.t += 1
        done = self.t >= self.max_steps
        return self.state.copy(), 0.0, done, {}

    def random_action(self):
        return self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
