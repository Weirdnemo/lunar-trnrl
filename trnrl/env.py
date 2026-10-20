"""Gymnasium environment: 3-DoF lunar powered descent with TRN-based navigation.

The agent only ever sees the *navigation estimate* (and its covariance) plus
crops of the *onboard map*. Truth is used for dynamics, sensors, reward and
success evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy import ndimage

from .nav import G_VEC, NavConfig, NavSystem
from .terrain import Terrain, synthetic_lunar

G0 = 9.80665


@dataclass
class LanderConfig:
    m0: float = 2000.0
    m_dry: float = 1500.0
    thrust_max: float = 6000.0       # T/W ~1.85 at lunar gravity
    throttle_min: float = 0.2
    isp: float = 311.0
    tilt_max_deg: float = 40.0


@dataclass
class EnvConfig:
    lander: LanderConfig = field(default_factory=LanderConfig)
    nav: NavConfig = field(default_factory=NavConfig)
    map_factor: int = 2              # onboard map is this much coarser than truth
    map_noise_m: float = 0.0         # std [m] of spatially correlated vertical error added to the onboard map (realism knob)
    map_noise_corr_m: float = 150.0  # correlation length of that error [m]
    dt: float = 1.0
    substeps: int = 4
    t_max: float = 250.0
    # initial conditions
    range_h: tuple = (1000.0, 1600.0)
    agl0: tuple = (1000.0, 1400.0)
    v_h0: tuple = (20.0, 30.0)
    v_z0: tuple = (-25.0, -15.0)
    edge_margin: float = 250.0
    # touchdown limits
    vz_max: float = 2.0
    vh_max: float = 1.0
    slope_max_deg: float = 10.0
    rough_max_m: float = 0.6
    min_target_clearance_cells: int = 3
    # reward
    w_pot_pos: float = 2.0
    w_pot_vel: float = 1.0
    w_fuel: float = 0.02
    w_time: float = 0.01
    lambda_sigma: float = 0.0        # penalty on horizontal nav sigma (ablation knob)
    r_success: float = 100.0
    r_fail: float = -100.0
    # observation ablation: hide covariance / fix-age features
    obs_covariance: bool = True
    # TRN blackout: with probability p_outage a window [start, start+dur] has no TRN
    p_outage: float = 0.0
    oracle_nav: bool = False         # guidance sees the TRUE state (ceiling experiment); nav filter still runs
    outage_start: tuple = (0.0, 80.0)
    outage_dur: tuple = (30.0, 70.0)


class Site:
    """One terrain with everything derived from it (map, hazard, texture)."""

    def __init__(self, truth: Terrain, cfg: EnvConfig):
        self.truth = truth
        self.map = truth.downsample(cfg.map_factor)
        if cfg.map_noise_m > 0:
            g = np.random.default_rng(int(abs(float(truth.h.sum())) * 1000) % (2 ** 32))
            z = ndimage.gaussian_filter(g.normal(size=self.map.h.shape), cfg.map_noise_corr_m / self.map.dx, mode="wrap")
            z *= cfg.map_noise_m / max(float(z.std()), 1e-9)
            self.map = Terrain(self.map.h + z, self.map.dx, self.map.name)
        self.map_safe = self.map.safe_map(cfg.slope_max_deg, cfg.rough_max_m, 15.0)
        self.map_tex = self.map.texture(300.0)
        slope = truth.slope_deg()
        self.truth_ok = (slope <= cfg.slope_max_deg) & (truth.roughness() <= cfg.rough_max_m)
        # candidate designated targets: safe, with clearance, away from the edges
        edt = ndimage.distance_transform_edt(self.map_safe)
        ny, nx = self.map.shape
        m = int(round((cfg.range_h[1] + cfg.edge_margin) / self.map.dx))
        region = np.zeros_like(self.map_safe)
        region[m:ny - m, m:nx - m] = True
        cand = (edt >= cfg.min_target_clearance_cells) & region
        self.targets = np.argwhere(cand)
        self.map_err_rms = float(np.sqrt(np.mean(
            (truth.height(*np.meshgrid(np.arange(nx) * self.map.dx, np.arange(ny) * self.map.dx)) - self.map.h) ** 2)))


def make_sites(n: int, cfg: EnvConfig, seed0: int = 0, size: int = 512, dx: float = 10.0, **terrain_kw) -> list[Site]:
    sites = []
    s = seed0
    while len(sites) < n:
        site = Site(synthetic_lunar(size, dx, seed=s, **terrain_kw), cfg)
        s += 1
        if len(site.targets) >= 20:
            sites.append(site)
    return sites


class LunarLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    CROP_N, CROP_STRIDE_FINE, CROP_STRIDE_COARSE, TEX_N, TEX_STRIDE = 9, 1, 4, 7, 8

    def __init__(self, sites: list[Site] | None = None, cfg: EnvConfig | None = None, seed: int = 0):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.sites = sites if sites is not None else make_sites(8, self.cfg, seed0=1000)
        self.L = self.cfg.lander
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        d = 11 + 2 * self.CROP_N ** 2 + self.TEX_N ** 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (d,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.site: Site | None = None
        self.history: dict = {}

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng, cfg = self._rng, self.cfg
        opts = options or {}
        self.site = self.sites[opts.get("site", int(rng.integers(len(self.sites))))]
        s = self.site
        tj, ti = s.targets[int(rng.integers(len(s.targets)))]
        self.target = np.array([ti * s.map.dx, tj * s.map.dx, s.map.h[tj, ti]])
        ext_x, ext_y = s.truth.extent
        for _ in range(50):
            th = rng.uniform(0, 2 * np.pi)
            rh = rng.uniform(*cfg.range_h)
            xy = self.target[:2] + rh * np.array([np.cos(th), np.sin(th)])
            if cfg.edge_margin < xy[0] < ext_x - cfg.edge_margin and cfg.edge_margin < xy[1] < ext_y - cfg.edge_margin:
                break
        agl0 = rng.uniform(*cfg.agl0)
        self.r = np.array([xy[0], xy[1], s.truth.height(*xy) + agl0])
        to_t = (self.target[:2] - xy) / np.linalg.norm(self.target[:2] - xy)
        vh = rng.uniform(*cfg.v_h0)
        ang = rng.normal(0, np.radians(8))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        self.v = np.array([*(vh * rot @ to_t), rng.uniform(*cfg.v_z0)])
        self.mass = self.L.m0
        self.t = 0.0

        ncfg = NavConfig(**{**self.cfg.nav.__dict__, "map_height_sigma": max(s.map_err_rms, 0.5)})
        if cfg.p_outage > 0 and rng.uniform() < cfg.p_outage:
            ncfg.outage_start = float(rng.uniform(*cfg.outage_start))
            ncfg.outage_end = ncfg.outage_start + float(rng.uniform(*cfg.outage_dur))
        self.nav = NavSystem(s.truth, s.map, ncfg, rng)
        self.nav.reset(self.r, self.v)
        self.fix_age = 0.0
        self.last_info = {"trn": "none"}
        self._phi = self._potential()
        self.history = {k: [] for k in ["t", "r", "r_hat", "sigma_h", "sigma_v", "trn", "trn_err", "mass", "thrust", "nees"]}
        self._log(0.0)
        return self._obs(), self._info()

    # ----------------------------------------------------------------- helpers
    def _potential(self) -> float:
        d = np.linalg.norm(self.r - self.target)
        return -(self.cfg.w_pot_pos * d / 1500.0 + self.cfg.w_pot_vel * np.linalg.norm(self.v) / 40.0)

    def _log(self, thrust: float) -> None:
        h = self.history
        h["t"].append(self.t); h["r"].append(self.r.copy()); h["r_hat"].append(self.nav.pos.copy())
        h["sigma_h"].append(self.nav.sigma_h()); h["sigma_v"].append(self.nav.sigma_v())
        h["trn"].append(self.last_info.get("trn", "none")); h["trn_err"].append(self.last_info.get("trn_err", np.nan))
        h["mass"].append(self.mass); h["thrust"].append(thrust); h["nees"].append(self.nav.nees(self.r, self.v))

    def _crop(self, arr: np.ndarray, n: int, stride: int, order: int) -> np.ndarray:
        m = self.site.map
        c = (n - 1) / 2.0
        o = (np.arange(n) - c) * stride
        gi, gj = np.meshgrid(o, o)
        ci = self.nav.pos[0] / m.dx + gi
        cj = self.nav.pos[1] / m.dx + gj
        return ndimage.map_coordinates(arr, [cj, ci], order=order, mode="nearest")

    def estimate(self) -> dict:
        """What guidance may use: nav estimate, designated target, mass, limits."""
        r, v = (self.r.copy(), self.v.copy()) if self.cfg.oracle_nav else (self.nav.pos.copy(), self.nav.vel.copy())
        return {"r": r, "v": v, "target": self.target.copy(), "mass": self.mass, "t": self.t}

    def _obs(self) -> np.ndarray:
        n = self.nav
        rel = (n.pos - self.target) / 1500.0
        vel = n.vel / 40.0
        agl_est = (n.pos[2] - self.site.map.height(n.pos[0], n.pos[1])) / 1500.0
        fuel = (self.mass - self.L.m_dry) / (self.L.m0 - self.L.m_dry)
        cov = [np.log10(max(n.sigma_h(), 1.0)) / 2.0, np.log10(max(n.sigma_v(), 0.5)) / 2.0,
               min(self.fix_age, 60.0) / 60.0] if self.cfg.obs_covariance else [0.0, 0.0, 0.0]
        scal = np.array([*rel, *vel, agl_est, fuel, *cov], dtype=np.float32)
        fine = self._crop(self.site.map_safe.astype(np.float32), self.CROP_N, self.CROP_STRIDE_FINE, 0)
        coarse = self._crop(self.site.map_safe.astype(np.float32), self.CROP_N, self.CROP_STRIDE_COARSE, 0)
        tex = self._crop(self.site.map_tex, self.TEX_N, self.TEX_STRIDE, 1) / 20.0
        return np.concatenate([scal, fine.ravel(), coarse.ravel(), tex.ravel()]).astype(np.float32)

    def _info(self) -> dict:
        return {"pos_err": float(np.linalg.norm(self.nav.pos - self.r)),
                "pos_err_h": float(np.linalg.norm(self.nav.pos[:2] - self.r[:2])),
                "sigma_h": self.nav.sigma_h(), "trn": self.last_info.get("trn", "none"),
                "dist_target": float(np.linalg.norm(self.r[:2] - self.target[:2])), "t": self.t,
                "fuel_used": self.L.m0 - self.mass}

    def action_to_thrust(self, a: np.ndarray) -> tuple[float, np.ndarray]:
        L = self.L
        a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        tmin = L.throttle_min * L.thrust_max
        T = tmin + (L.thrust_max - tmin) * 0.5 * (a[0] + 1.0)
        u = a[1:3]
        nrm = np.linalg.norm(u)
        if nrm > 1.0:
            u = u / nrm
        dxy = u * np.sin(np.radians(L.tilt_max_deg))
        d = np.array([dxy[0], dxy[1], np.sqrt(max(1.0 - dxy @ dxy, 0.0))])
        return float(T), d

    # -------------------------------------------------------------------- step
    def step(self, action):
        cfg, L, s = self.cfg, self.L, self.site
        T, d = self.action_to_thrust(action)
        ds = cfg.dt / cfg.substeps
        fuel0 = self.mass
        f_sum = np.zeros(3)
        touched = False
        elapsed = 0.0
        for _ in range(cfg.substeps):
            Teff = T if self.mass > L.m_dry + 1e-6 else 0.0
            acc_t = Teff * d / self.mass
            acc = acc_t + G_VEC
            r_new = self.r + self.v * ds + 0.5 * acc * ds ** 2
            v_new = self.v + acc * ds
            agl_new = r_new[2] - s.truth.height(r_new[0], r_new[1])
            if agl_new <= 0.0:
                agl_old = self.r[2] - s.truth.height(self.r[0], self.r[1])
                frac = np.clip(agl_old / max(agl_old - agl_new, 1e-9), 0.0, 1.0)
                self.r = self.r + (r_new - self.r) * frac
                self.v = self.v + (v_new - self.v) * frac
                elapsed += ds * frac
                self.t += ds * frac
                self.mass -= Teff / (L.isp * G0) * ds * frac
                touched = True
                break
            f_sum += acc_t * ds
            self.r, self.v = r_new, v_new
            self.mass = max(self.mass - Teff / (L.isp * G0) * ds, L.m_dry)
            elapsed += ds
            self.t += ds

        terminated = truncated = False
        outcome = "flying"
        if not touched:
            self.nav.t = self.t - elapsed
            self.nav.predict(f_sum / max(elapsed, 1e-9), elapsed)
            self.last_info = self.nav.measure(self.r, self.v)
            self.fix_age = 0.0 if self.last_info["trn"] == "fix" else self.fix_age + elapsed
            ext_x, ext_y = s.truth.extent
            if not (30 < self.r[0] < ext_x - 30 and 30 < self.r[1] < ext_y - 30):
                terminated, outcome = True, "out_of_bounds"
            elif self.t >= cfg.t_max:
                truncated, outcome = True, "timeout"
        else:
            terminated = True
            outcome = self._classify_touchdown()

        # reward: potential shaping + fuel/time/uncertainty costs + terminal
        phi = self._potential()
        fuel = (fuel0 - self.mass) / (L.m0 - L.m_dry)
        rew = (phi - self._phi) - cfg.w_fuel * 100 * fuel / 10.0 - cfg.w_time - cfg.lambda_sigma * self.nav.sigma_h() / 100.0
        self._phi = phi
        info = self._info()
        info["outcome"] = outcome
        if terminated or truncated:
            dist = float(np.linalg.norm(self.r[:2] - self.target[:2]))
            if outcome == "success":
                used = (L.m0 - self.mass) / (L.m0 - L.m_dry)
                rew += cfg.r_success + 30.0 * (1.0 - used) - 20.0 * min(dist / 50.0, 1.0)
            else:
                rew += cfg.r_fail - 2.0 * min(float(np.linalg.norm(self.v)), 20.0)
            info.update({"touch_vz": float(-self.v[2]), "touch_vh": float(np.hypot(*self.v[:2])),
                         "landing_err": dist, "final_sigma_h": self.nav.sigma_h(),
                         "success": outcome == "success"})
        self._log(T)
        return self._obs(), float(rew), terminated, truncated, info

    def _classify_touchdown(self) -> str:
        s, cfg = self.site, self.cfg
        vz, vh = -self.v[2], float(np.hypot(*self.v[:2]))
        i = int(np.clip(round(self.r[0] / s.truth.dx), 0, s.truth.shape[1] - 1))
        j = int(np.clip(round(self.r[1] / s.truth.dx), 0, s.truth.shape[0] - 1))
        if vz > cfg.vz_max or vh > cfg.vh_max:
            return "hard_landing"
        if not s.truth_ok[j, i]:
            return "unsafe_site"
        return "success"
