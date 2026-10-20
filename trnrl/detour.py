"""Learned-detour layer: the baseline guidance flies everything; a policy only decides, at a few decision
points, whether to fly a lateral detour and in which direction (to get terrain fixes when matching is ambiguous).
Action 0 = no detour (= baseline). Actions 1..8 = direction k at 400 m, 9..16 = direction at 800 m, 17 = hover up to 25 s.
The same guidance object serves hand-written/learned choosers (eval) and the Gym wrapper (training)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .guidance import PolyGuidance
from .matchability import matchability_map
from .nav import G_VEC

DIRS = np.array([[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4])
RADII = (400.0, 800.0)
N_ACT = 1 + 8 * len(RADII) + 1          # last action = hold (hover) for up to HOLD_S seconds
HOLD_S = 25.0
N_FEAT = 7 + 8 * len(RADII) + 8


def site_match(site):
    if not hasattr(site, "match_map"):
        site.match_map = matchability_map(site.map.h)
    return site.match_map


class DetourGuidance2(PolyGuidance):
    def __init__(self, chooser=None, sigma_ok: float = 60.0, min_agl: float = 400.0, fix_age_trigger: float = 12.0,
                 gap_s: float = 15.0, max_decisions: int = 3, speed: float = 15.0, max_detour_s: float = 60.0, **kw):
        super().__init__(**kw)
        self.chooser = chooser
        self.sigma_ok, self.min_agl, self.fix_age_trigger = sigma_ok, min_agl, fix_age_trigger
        self.gap_s, self.max_decisions, self.speed, self.max_detour_s = gap_s, max_decisions, speed, max_detour_s
        self.reset_episode()

    def reset_episode(self):
        self.wp, self.n_dec, self.last_dec_t, self.wp_t0, self.pending = None, 0, -1e9, 0.0, False
        self.hold_until = -1.0
        self.log = []

    def _agl(self, env, r):
        return r[2] - env.site.map.height(r[0], r[1])

    def eligible(self, env) -> bool:
        if self.wp is not None or self.n_dec >= self.max_decisions or env.t < self.hold_until:
            return False
        r = env.nav.pos
        return (env.nav.sigma_h() > self.sigma_ok and env.fix_age >= self.fix_age_trigger
                and self._agl(env, r) > self.min_agl and env.t - self.last_dec_t >= self.gap_s)

    def waypoints(self, env):
        r = env.nav.pos[:2]
        ext_x, ext_y = env.site.truth.extent
        out = []
        for rad in RADII:
            for d in DIRS:
                w = r + rad * d
                out.append(np.clip(w, 200.0, [ext_x - 200.0, ext_y - 200.0]))
        return np.array(out)

    def features(self, env) -> np.ndarray:
        n = env.nav
        r = n.pos
        tgt = env.target
        mm = site_match(env.site)
        dx = env.site.map.dx

        def mval(p):
            i = int(np.clip(p[0] / dx, 0, mm.shape[1] - 1)); j = int(np.clip(p[1] / dx, 0, mm.shape[0] - 1))
            return float(mm[j, i])
        wps = self.waypoints(env)
        to_t = tgt[:2] - r[:2]
        to_t_u = to_t / max(np.linalg.norm(to_t), 1e-6)
        f = [n.sigma_h() / 600.0, min(env.fix_age, 60.0) / 60.0, self._agl(env, r) / 1500.0,
             to_t[0] / 1500.0, to_t[1] / 1500.0, env.t / 250.0, (self.max_decisions - self.n_dec) / self.max_decisions]
        f += [mval(w) for w in wps]
        f += [float(d @ to_t_u) for d in DIRS]
        return np.array(f, np.float32)

    def set_action(self, env, a: int):
        self.n_dec += 1
        self.last_dec_t = env.t
        self.log.append(int(a))
        if a == N_ACT - 1:
            self.hold_until = env.t + HOLD_S
        elif a > 0:
            self.wp = self.waypoints(env)[a - 1]
            self.wp_t0 = env.t

    def act(self, env) -> np.ndarray:
        if self.chooser is not None and self.eligible(env):
            self.set_action(env, int(self.chooser(self.features(env))))
        est = env.estimate()
        if est["t"] < self.last_dec_t:        # new episode (time went backwards)
            self.reset_episode()
        r, v, m = est["r"], est["v"], est["mass"]
        agl = self._agl(env, r)
        sig = env.nav.sigma_h()
        if self.wp is not None:
            if sig <= self.sigma_ok or np.hypot(*(self.wp - r[:2])) < 60.0 or agl <= self.min_agl * 0.7 \
                    or env.t - self.wp_t0 > self.max_detour_s:
                self.wp = None
            else:
                dh = self.wp - r[:2]
                vhd = dh / max(np.hypot(*dh), 1e-6) * min(self.speed, 0.1 * np.hypot(*dh))
                a = np.array([(vhd[0] - v[0]) / 4.0, (vhd[1] - v[1]) / 4.0, (-3.0 - v[2]) / 4.0])
                return self._to_action(env, m * (a - G_VEC))
        if env.t < self.hold_until and sig > self.sigma_ok and agl > self.min_agl * 0.7:
            return self._to_action(env, m * (-v / 3.0 - G_VEC))
        self.speed_floor = 15.0 if agl > 300.0 else 0.0
        return super().act(env)


class DetourEnv(gym.Env):
    """Semi-MDP: one step = one decision; the baseline flies in between. Reward only at the end of the episode."""

    def __init__(self, inner, detour_cost: float = 0.02, **gkw):
        super().__init__()
        self.inner = inner
        self.g = DetourGuidance2(chooser=None, **gkw)
        self.detour_cost = detour_cost
        self.action_space = spaces.Discrete(N_ACT)
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_FEAT,), np.float32)
        self.last_info = {}
        self._tries = 0

    @property
    def history(self):
        return self.inner.history

    def _advance(self):
        """Fly with the baseline until the next decision or the end of the episode."""
        done, info = False, {}
        while not done and not self.g.eligible(self.inner):
            _, _, te, tr, info = self.inner.step(self.g.act(self.inner))
            done = te or tr
        return done, info

    def reset(self, *, seed=None, options=None):
        """Start an episode and fly to the first decision point. Episodes where a terrain fix lands early
        (sigma low) never need a decision, so they are abandoned after a few seconds instead of simulated to the end."""
        for k in range(400):
            self.inner.reset(seed=seed, options=options)
            self.g.reset_episode()
            done = False
            while not done:
                if self.g.eligible(self.inner):
                    return self.g.features(self.inner), {}
                if self.inner.t >= 3.0 and self.inner.nav.sigma_h() <= self.g.sigma_ok:
                    break                                   # fixed early: baseline-equal, abandon
                _, _, te, tr, _ = self.inner.step(self.g.act(self.inner))
                done = te or tr
            seed = None if seed is None else seed + 100_000
        return np.zeros(N_FEAT, np.float32), {}

    def step(self, action):
        a = int(action)
        self.g.set_action(self.inner, a)
        done, info = self._advance()
        r = -self.detour_cost if a > 0 else 0.0
        if done:
            out = info.get("outcome")
            prec = out == "success" and info.get("landing_err", 1e9) <= 50.0
            r += 1.0 if prec else (0.3 if out == "success" else -0.5)
            self.last_info = info
            return np.zeros(N_FEAT, np.float32), float(r), True, False, info
        return self.g.features(self.inner), float(r), False, False, info
