"""Residual-RL wrapper: action = clip(baseline_action + k * policy_action).

The baseline guidance sees only the navigation estimate, as does the policy. The
baseline action is appended to the observation so the policy can learn *when* to
deviate from it (e.g. when sigma is large or terrain is featureless).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .guidance import PolyGuidance


class ResidualEnv(gym.Env):
    def __init__(self, inner, k: float = 0.5, guidance: PolyGuidance | None = None):
        super().__init__()
        self.inner, self.k = inner, k
        self.g = guidance or PolyGuidance()
        self.action_space = inner.action_space
        d = inner.observation_space.shape[0] + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (d,), np.float32)
        self.last_obs = None
        self._base = np.zeros(3, np.float32)

    @property
    def history(self):
        return self.inner.history

    def _aug(self, obs):
        self.last_obs = np.concatenate([obs, self._base]).astype(np.float32)
        return self.last_obs

    def reset(self, *, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        self._base = self.g.act(self.inner)
        return self._aug(obs), info

    def step(self, action):
        a = np.clip(self._base + self.k * np.asarray(action, np.float32), -1.0, 1.0)
        obs, r, te, tr, info = self.inner.step(a)
        if not (te or tr):
            self._base = self.g.act(self.inner)
        return self._aug(obs), r, te, tr, info
