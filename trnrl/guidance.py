"""Baseline guidance: two-phase approach to a fixed designated site.

Phase 1 (approach): polynomial / E-guidance style. Acceleration is linear in time
with terminal position and velocity constraints at a gate point above the target:
    a0 = 6*dr/T^2 - 2*dv/T          (net acceleration incl. gravity)
    thrust accel = a0 - g
Phase 2 (terminal descent): constant-rate vertical descent with PD horizontal
station-keeping over the target. (Pure polynomial guidance stalls near zero
speed because time-to-go collapses with altitude, so real landers use a
terminal vertical phase.)

Uses only the navigation estimate (never truth).
"""
from __future__ import annotations

import numpy as np

from .nav import G_VEC


class PolyGuidance:
    _tgt_override = None     # set by subclasses to fly toward a waypoint instead of the designated target

    def __init__(self, gate_alt: float = 30.0, gate_vz: float = -3.0,
                 term_alt: float = 40.0, term_dh: float = 25.0,
                 tgo_min: float = 4.0, tgo_max: float = 250.0):
        self.gate_alt, self.gate_vz = gate_alt, gate_vz
        self.term_alt, self.term_dh = term_alt, term_dh
        self.tgo_min, self.tgo_max = tgo_min, tgo_max
        self.speed_floor = 0.0   # lower bound on |vz| used for time-to-go (0 = classic behaviour)

    def act(self, env) -> np.ndarray:
        est = env.estimate()
        r, v, m, tgt = est["r"], est["v"], est["mass"], est["target"]
        if self._tgt_override is not None:
            tgt = self._tgt_override
        agl = r[2] - tgt[2]
        dh = tgt[:2] - r[:2]
        if agl <= self.term_alt and np.linalg.norm(dh) <= self.term_dh:
            vz_des = -float(np.clip(0.5 + 0.04 * agl, 0.5, 1.5))
            a = np.array([0.02 * dh[0] - 0.2 * v[0],
                          0.02 * dh[1] - 0.2 * v[1],
                          (vz_des - v[2]) / 2.0])
        else:
            rf = tgt + np.array([0.0, 0.0, self.gate_alt])
            vf = np.array([0.0, 0.0, self.gate_vz])
            dz = max(r[2] - rf[2], 1.0)
            tgo = np.clip(2.0 * dz / (max(abs(v[2]), self.speed_floor) + abs(vf[2])), self.tgo_min, self.tgo_max)
            dr = rf - r - v * tgo
            dv = vf - v
            a = 6.0 * dr / tgo ** 2 - 2.0 * dv / tgo
        return self._to_action(env, m * (a - G_VEC))

    @staticmethod
    def _to_action(env, F: np.ndarray) -> np.ndarray:
        L = env.L
        tmin = L.throttle_min * L.thrust_max
        F = F.copy()
        F[2] = max(F[2], 1e-3)
        T = float(np.clip(np.linalg.norm(F), tmin, L.thrust_max))
        d = F / np.linalg.norm(F)
        tmax = np.radians(L.tilt_max_deg)
        if np.arccos(np.clip(d[2], -1, 1)) > tmax:
            h = d[:2] / max(np.linalg.norm(d[:2]), 1e-9)
            d = np.array([*(np.sin(tmax) * h), np.cos(tmax)])
        a0 = 2.0 * (T - tmin) / (L.thrust_max - tmin) - 1.0
        uxy = d[:2] / np.sin(tmax)
        return np.array([a0, *uxy], dtype=np.float32)


class WaitGuidance(PolyGuidance):
    """Baseline plus a covariance-triggered hold.

    While the horizontal 1-sigma of the nav solution is above ``sigma_ok`` and the
    lander is still above ``wait_alt`` (so TRN is still in range), hover instead of
    descending, up to ``max_wait`` seconds of cumulative hold and while the
    propellant reserve allows. Resumes as soon as a fix brings sigma down.
    Uses only nav outputs (estimate and its covariance).
    """

    def __init__(self, sigma_ok: float = 40.0, wait_alt: float = 500.0, max_wait: float = 90.0,
                 reserve_kg: float = 220.0, **kw):
        super().__init__(**kw)
        self.sigma_ok, self.wait_alt, self.max_wait, self.reserve = sigma_ok, wait_alt, max_wait, reserve_kg
        self.waited = 0.0
        self._last_t = -1.0

    def act(self, env) -> np.ndarray:
        est = env.estimate()
        if est["t"] < self._last_t:          # new episode
            self.waited = 0.0
        self._last_t = est["t"]
        r, v, m = est["r"], est["v"], est["mass"]
        agl = r[2] - env.site.map.height(r[0], r[1])
        fuel_left = m - env.L.m_dry
        if (env.nav.sigma_h() > self.sigma_ok and agl > self.wait_alt and self.waited < self.max_wait
                and fuel_left > self.reserve):
            self.waited += env.cfg.dt
            a = -v / 3.0                      # damp to a hover
            return self._to_action(env, m * (a - G_VEC))
        # resuming from (near) rest: assume a nominal descent speed so time-to-go is sane
        self.speed_floor = 15.0 if agl > 300.0 else 0.0
        return super().act(env)


class LandingGuidance(PolyGuidance):
    """'Baseline v2': the same approach law, then a landing sequence like real landers fly it.

    1. Approach (polynomial law) down to ``funnel_top`` above the target.
    2. Final approach inside a glide-slope funnel: altitude above the target must stay above
       ``funnel_k * horizontal_offset``. Outside the funnel the lander holds or climbs while it
       nulls the horizontal offset (a late large position correction no longer ends in a dive).
    3. Terminal descent (vertical, slow) only once the offset is inside ``term_dh`` and the
       horizontal speed is low; it aborts back to the funnel if the offset grows again.
    Uses only the navigation estimate, like the other guidance classes.
    """

    def __init__(self, funnel_top: float = 150.0, funnel_k: float = 0.7, funnel_floor: float = 30.0,
                 vh_max_cmd: float = 8.0, tau_h: float = 4.0, term_dh: float = 15.0, term_alt: float = 40.0,
                 gate_alt: float = 60.0, **kw):
        super().__init__(gate_alt=gate_alt, term_alt=term_alt, term_dh=term_dh, **kw)
        self.funnel_top, self.funnel_k, self.funnel_floor = funnel_top, funnel_k, funnel_floor
        self.vh_max_cmd, self.tau_h = vh_max_cmd, tau_h

    def act(self, env) -> np.ndarray:
        est = env.estimate()
        r, v, m, tgt = est["r"], est["v"], est["mass"], est["target"]
        agl = r[2] - tgt[2]
        dh = tgt[:2] - r[:2]
        d = float(np.linalg.norm(dh))
        if agl > self.funnel_top:                  # high up: normal polynomial approach
            return super().act(env)
        need = max(self.funnel_floor, self.funnel_k * d)       # altitude the funnel requires at this offset
        vhd = dh / max(d, 1e-6) * min(self.vh_max_cmd, 0.15 * d)   # desired horizontal velocity toward target
        if d <= self.term_dh and agl <= self.term_alt:         # terminal descent
            vz_des = -float(np.clip(0.5 + 0.04 * agl, 0.5, 1.5))
            vhd = dh * 0.1
        elif agl < need:                                       # below the funnel: climb / hold, close the offset
            vz_des = float(np.clip(0.1 * (need - agl), 0.0, 3.0))
        else:                                                  # inside the funnel: descend, rate set by margin
            vz_des = -float(np.clip(0.05 * (agl - need) + 0.5, 0.5, 5.0))
        a = np.array([(vhd[0] - v[0]) / self.tau_h, (vhd[1] - v[1]) / self.tau_h, (vz_des - v[2]) / 2.0])
        return self._to_action(env, m * (a - G_VEC))


class DetourGuidance(PolyGuidance):
    """Hand-written active-perception test: when the nav solution is poor and no fix has come for a while,
    fly a lateral detour toward the best-textured ground (onboard map texture) near the estimated position,
    then resume the normal descent. Uses only nav outputs and the onboard map."""

    def __init__(self, sigma_ok: float = 60.0, min_agl: float = 400.0, fix_age_trigger: float = 12.0,
                 radius: float = 600.0, speed: float = 15.0, max_detours: int = 1, tex_weight: float = 1.0, **kw):
        super().__init__(**kw)
        self.sigma_ok, self.min_agl, self.fix_age_trigger = sigma_ok, min_agl, fix_age_trigger
        self.radius, self.speed, self.max_detours, self.tex_weight = radius, speed, max_detours, tex_weight
        self.wp, self.used, self._last_t = None, 0, -1.0

    def _pick(self, env, r, tgt):
        m = env.site.map
        n = int(self.radius / m.dx)
        ci, cj = int(r[0] / m.dx), int(r[1] / m.dx)
        j0, j1, i0, i1 = max(cj - n, 0), min(cj + n + 1, m.shape[0]), max(ci - n, 0), min(ci + n + 1, m.shape[1])
        tex = env.site.map_tex[j0:j1, i0:i1]
        jj, ii = np.mgrid[j0:j1, i0:i1]
        px, py = ii * m.dx, jj * m.dx
        dist = np.hypot(px - r[0], py - r[1])
        ok = (dist <= self.radius) & (dist >= 100.0)
        to_t = np.hypot(px - tgt[0], py - tgt[1])
        ok &= to_t <= np.hypot(*(r[:2] - tgt[:2])) + 300.0           # do not wander away from the target
        if not ok.any():
            return None
        score = np.where(ok, tex / max(float(env.site.map_tex.max()), 1e-6) * self.tex_weight - dist / self.radius * 0.3, -1e9)
        k = np.unravel_index(int(np.argmax(score)), score.shape)
        return np.array([px[k], py[k]])

    def act(self, env) -> np.ndarray:
        est = env.estimate()
        if est["t"] < self._last_t:
            self.wp, self.used = None, 0
        self._last_t = est["t"]
        r, v, m, tgt = est["r"], est["v"], est["mass"], est["target"]
        agl = r[2] - env.site.map.height(r[0], r[1])
        sig = env.nav.sigma_h()
        if self.wp is None and self.used < self.max_detours and sig > self.sigma_ok and agl > self.min_agl \
                and env.fix_age >= self.fix_age_trigger:
            self.wp = self._pick(env, r, tgt)
            if self.wp is not None:
                self.used += 1
        if self.wp is not None:
            if sig <= self.sigma_ok or np.hypot(*(self.wp - r[:2])) < 60.0 or agl <= self.min_agl * 0.7:
                self.wp = None
            else:
                dh = self.wp - r[:2]
                vhd = dh / max(np.hypot(*dh), 1e-6) * min(self.speed, 0.1 * np.hypot(*dh))
                a = np.array([(vhd[0] - v[0]) / 4.0, (vhd[1] - v[1]) / 4.0, (-3.0 - v[2]) / 4.0])
                return self._to_action(env, m * (a - G_VEC))
        self.speed_floor = 15.0 if agl > 300.0 else 0.0
        return super().act(env)
