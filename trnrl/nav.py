"""Navigation: 9-state EKF (pos, vel, accel bias) with terrain-relative navigation.

Sensors simulated against the *truth* terrain, filtered against the coarser
*onboard map*:

* IMU           -> prediction (white noise + random-walk bias)
* Doppler       -> 3-axis velocity
* Radar altim.  -> height above ground, terrain-aided (innovation uses map height)
* TRN           -> lidar height patch matched (NCC) against the onboard map in a
                   search window around the current estimate. The fix is gated
                   on match quality/ambiguity and on a chi-square innovation test.
                   Its covariance comes from a Cramer-Rao-style bound on the map
                   gradients, so featureless terrain yields weak/rejected fixes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .terrain import Terrain

G_VEC = np.array([0.0, 0.0, -1.62])  # lunar gravity [m/s^2]

# chi-square 99.9% gates
CHI2_1DOF = 10.8
CHI2_2DOF = 13.8


@dataclass
class NavConfig:
    # initial knowledge error (1 sigma)
    init_pos_h: float = 120.0
    init_pos_v: float = 30.0
    init_vel: float = 0.3
    init_bias: float = 2e-3
    # IMU
    accel_noise: float = 2e-3        # [m/s^2] per 1 s sample
    bias_rw: float = 2e-5            # [m/s^2/sqrt(s)]
    # Doppler velocimeter
    dop_sigma0: float = 0.05
    dop_sigma_frac: float = 0.002
    # radar altimeter
    alt_sigma0: float = 0.5
    alt_sigma_frac: float = 0.005
    map_height_sigma: float = 2.0    # set from rms(truth - map) by the env
    # TRN
    trn_period: float = 2.0
    trn_min_agl: float = 150.0
    trn_max_agl: float = 2500.0
    patch_n: int = 15
    fov_deg: float | None = None     # None: fixed patch_n cells. Else patch side = 2*agl*tan(fov/2), in map cells, clipped to [patch_n_min, patch_n_max]
    patch_n_min: int = 7
    patch_n_max: int = 41
    reg_offset_m: float = 0.0        # 1-sigma/axis map-to-world registration offset (fixed per episode): TRN fixes are biased by it
    lidar_sigma0: float = 0.3
    lidar_sigma_frac: float = 0.002
    min_patch_relief: float = 0.3    # [m] std of measured patch, else "flat"
    ncc_min: float = 0.7
    lobe_drop: float = 0.10          # main lobe = connected region with ncc >= best - lobe_drop
    amb_margin: float = 0.03         # ambiguous if a peak outside the lobe is within this of best
    fix_floor_cells: float = 0.4
    fix_inflate: float = 2.0
    search_max_cells: int = 20       # must cover 3*init_pos_h / map dx (=18 cells at 20 m)
    alt_sync: bool = False           # after a TRN fix, inflate altitude variance by the map-height change (stops altimeter lock-out)
    enable_trn: bool = True
    alt_mode: str = "coupled"       # 'coupled' (EKF with map-gradient coupling) | 'decoupled' (Schmidt-style)
    outage_start: float = -1.0       # TRN blackout window [s]; -1 = none
    outage_end: float = -1.0


class NavSystem:
    def __init__(self, truth: Terrain, onboard: Terrain, cfg: NavConfig,
                 rng: np.random.Generator):
        self.truth, self.map, self.cfg, self.rng = truth, onboard, cfg, rng
        self.gy_map, self.gx_map = np.gradient(onboard.h, onboard.dx)
        self._offs_cache = {}
        self._ox, self._oy = self._offs(cfg.patch_n)
        self.reg_off = np.zeros(2)
        self.x = np.zeros(9)
        self.P = np.eye(9)
        self.b_true = np.zeros(3)
        self.last_fix_t = -1e9
        self.t = 0.0

    def _offs(self, n: int):
        if n not in self._offs_cache:
            c = (n - 1) / 2.0
            oi, oj = np.meshgrid((np.arange(n) - c) * self.map.dx, (np.arange(n) - c) * self.map.dx)
            self._offs_cache[n] = (oi.ravel(), oj.ravel())
        return self._offs_cache[n]

    def patch_n_for(self, agl: float) -> int:
        c = self.cfg
        if c.fov_deg is None:
            return c.patch_n
        side = 2.0 * max(agl, 1.0) * np.tan(np.radians(c.fov_deg) / 2.0)
        n = int(round(side / self.map.dx)) + 1
        return int(np.clip(n, c.patch_n_min, c.patch_n_max))

    # ----------------------------------------------------------------- set-up
    def reset(self, r: np.ndarray, v: np.ndarray) -> None:
        c = self.cfg
        sig = np.array([c.init_pos_h, c.init_pos_h, c.init_pos_v, c.init_vel, c.init_vel,
                        c.init_vel, c.init_bias, c.init_bias, c.init_bias])
        self.P = np.diag(sig ** 2)
        self.b_true = self.rng.normal(0, c.init_bias, 3)
        err = self.rng.normal(0, sig[:6])
        self.x = np.concatenate([r + err[:3], v + err[3:6], np.zeros(3)])
        self.reg_off = (self.rng.normal(0, c.reg_offset_m, 2) if c.reg_offset_m > 0 else np.zeros(2))
        self.t = 0.0
        self.last_fix_t = -1e9

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:6]

    def sigma_h(self) -> float:
        return float(np.sqrt(0.5 * (self.P[0, 0] + self.P[1, 1])))

    def sigma_v(self) -> float:
        return float(np.sqrt(self.P[2, 2]))

    def err_vec(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        e = self.x.copy()
        e[:3] -= r
        e[3:6] -= v
        e[6:9] = self.x[6:9] - self.b_true
        return e

    def nees(self, r: np.ndarray, v: np.ndarray) -> float:
        e = self.err_vec(r, v)
        return float(e @ np.linalg.solve(self.P, e))

    # ---------------------------------------------------------------- predict
    def predict(self, f_true: np.ndarray, dt: float) -> None:
        """Propagate with an IMU specific-force measurement of the true thrust accel."""
        c = self.cfg
        self.b_true = self.b_true + self.rng.normal(0, c.bias_rw * np.sqrt(dt), 3)
        f_meas = f_true + self.b_true + self.rng.normal(0, c.accel_noise, 3)
        a = f_meas - self.x[6:9] + G_VEC
        self.x[:3] += self.x[3:6] * dt + 0.5 * a * dt ** 2
        self.x[3:6] += a * dt
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * dt ** 2 * np.eye(3)
        F[3:6, 6:9] = -dt * np.eye(3)
        G = np.zeros((9, 3))
        G[0:3] = 0.5 * dt ** 2 * np.eye(3)
        G[3:6] = dt * np.eye(3)
        Q = (G * c.accel_noise ** 2) @ G.T
        Q[6:9, 6:9] += np.eye(3) * c.bias_rw ** 2 * dt
        self.P = F @ self.P @ F.T + Q
        self.t += dt

    # ----------------------------------------------------------------- update
    def _update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray,
                gate: float | None) -> tuple[bool, float]:
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        if gate is not None and nis > gate:
            return False, nis
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        IKH = np.eye(9) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True, nis

    def measure(self, r: np.ndarray, v: np.ndarray) -> dict:
        """Take Doppler, altimeter and (periodically) TRN measurements at the true state."""
        c = self.cfg
        info: dict = {"trn": "none", "trn_err": np.nan}

        # Doppler
        sd = c.dop_sigma0 + c.dop_sigma_frac * float(np.linalg.norm(v))
        zv = v + self.rng.normal(0, sd, 3)
        H = np.zeros((3, 9)); H[:, 3:6] = np.eye(3)
        self._update(zv - self.x[3:6], H, np.eye(3) * sd ** 2, CHI2_1DOF * 3)

        # radar altimeter (terrain aided)
        agl = float(r[2] - self.truth.height(r[0], r[1]))
        sa = c.alt_sigma0 + c.alt_sigma_frac * max(agl, 0.0)
        za = agl + self.rng.normal(0, sa)
        xe, ye = self.x[0], self.x[1]
        pred = self.x[2] - self.map.height(xe, ye)
        gx = float(ndimage.map_coordinates(self.gx_map, [[ye / self.map.dx], [xe / self.map.dx]], order=1, mode="nearest")[0])
        gy = float(ndimage.map_coordinates(self.gy_map, [[ye / self.map.dx], [xe / self.map.dx]], order=1, mode="nearest")[0])
        if c.alt_mode == "decoupled":
            # height error from horizontal position uncertainty is treated as noise,
            # not as information about horizontal position
            H = np.zeros((1, 9)); H[0, 2] = 1.0
            extra = gx ** 2 * self.P[0, 0] + gy ** 2 * self.P[1, 1]
            R = np.array([[sa ** 2 + c.map_height_sigma ** 2 + extra]])
        else:
            H = np.zeros((1, 9)); H[0, 0], H[0, 1], H[0, 2] = -gx, -gy, 1.0
            R = np.array([[sa ** 2 + c.map_height_sigma ** 2]])
        self._update(np.array([za - pred]), H, R, CHI2_1DOF)

        # TRN
        due = (self.t - self.last_fix_t) >= c.trn_period - 1e-9
        in_outage = c.outage_start <= self.t < c.outage_end
        if c.enable_trn and due and not in_outage and c.trn_min_agl <= za <= c.trn_max_agl:
            info.update(self._trn_fix(r, za))
        info["agl_meas"] = za
        return info

    # -------------------------------------------------------------------- TRN
    def _trn_fix(self, r: np.ndarray, agl: float) -> dict:
        c, m = self.cfg, self.map
        n = self.patch_n_for(agl)
        self._ox, self._oy = self._offs(n)
        n2 = n ** 2
        # lidar patch of the TRUE terrain under the true position
        sl = c.lidar_sigma0 + c.lidar_sigma_frac * agl
        meas = self.truth.height(r[0] + self._ox, r[1] + self._oy) + self.rng.normal(0, sl, n2)
        self.last_fix_t = self.t
        if meas.std() < c.min_patch_relief:
            return {"trn": "rejected_flat", "trn_err": np.nan}

        xe, ye = self.x[0], self.x[1]
        sh = np.sqrt(max(self.P[0, 0], self.P[1, 1]))
        S = int(np.clip(np.ceil(3.0 * sh / m.dx) + 1, 2, c.search_max_cells))
        ab = np.arange(-S, S + 1)
        A, B = np.meshgrid(ab, ab)                      # A: x offset, B: y offset (cells)
        cx = xe + A.ravel() * m.dx
        cy = ye + B.ravel() * m.dx
        mc = meas - meas.mean()
        mcn = np.linalg.norm(mc)
        ncc = np.empty(cx.size)
        chunk = max(1, 400_000 // n2)                    # bound memory: the wide-search / large-patch case is C x n2 floats
        for i0 in range(0, cx.size, chunk):
            sl_ = slice(i0, i0 + chunk)
            px = (cx[sl_, None] + self._ox[None, :]) / m.dx
            py = (cy[sl_, None] + self._oy[None, :]) / m.dx
            ref = ndimage.map_coordinates(m.h, [py, px], order=1, mode="nearest")
            refc = ref - ref.mean(1, keepdims=True)
            ncc[sl_] = (refc @ mc) / (np.linalg.norm(refc, axis=1) * mcn + 1e-9)
        grid = ncc.reshape(A.shape)
        k = int(np.argmax(ncc))
        bj, bi = divmod(k, A.shape[1])
        best = float(ncc[k])
        # ambiguity: strongest correlation outside the (dilated) main lobe of the best peak
        lab, _ = ndimage.label(grid >= best - c.lobe_drop, structure=np.ones((3, 3)))
        lobe = ndimage.binary_dilation(lab == lab[bj, bi], structure=np.ones((3, 3)))
        second = float(grid[~lobe].max()) if (~lobe).any() else -1.0
        if best < c.ncc_min:
            return {"trn": "rejected_ncc", "trn_err": float(np.hypot(xe + A[bj, bi] * m.dx - r[0], ye + B[bj, bi] * m.dx - r[1])), "trn_ncc": best, "trn_second": second}
        if second >= best - c.amb_margin:
            return {"trn": "rejected_ambiguous", "trn_err": float(np.hypot(xe + A[bj, bi] * m.dx - r[0], ye + B[bj, bi] * m.dx - r[1])), "trn_ncc": best, "trn_second": second}
        da = db = 0.0
        if 0 < bi < A.shape[1] - 1:
            l, c0, rr = grid[bj, bi - 1], grid[bj, bi], grid[bj, bi + 1]
            den = l - 2 * c0 + rr
            da = float(np.clip(0.5 * (l - rr) / den, -0.5, 0.5)) if den < 0 else 0.0
        if 0 < bj < A.shape[0] - 1:
            l, c0, rr = grid[bj - 1, bi], grid[bj, bi], grid[bj + 1, bi]
            den = l - 2 * c0 + rr
            db = float(np.clip(0.5 * (l - rr) / den, -0.5, 0.5)) if den < 0 else 0.0
        fx = xe + (A[bj, bi] + da) * m.dx
        fy = ye + (B[bj, bi] + db) * m.dx

        # covariance: CRB-style bound from map gradients over the matched patch
        cxb, cyb = fx + self._ox, fy + self._oy
        coords = [cyb / m.dx, cxb / m.dx]
        gxp = ndimage.map_coordinates(self.gx_map, coords, order=1, mode="nearest")
        gyp = ndimage.map_coordinates(self.gy_map, coords, order=1, mode="nearest")
        s_eff2 = sl ** 2 + c.map_height_sigma ** 2
        floor2 = (c.fix_floor_cells * m.dx) ** 2
        sx2 = c.fix_inflate ** 2 * s_eff2 / max(float(gxp @ gxp), 1e-6) + floor2
        sy2 = c.fix_inflate ** 2 * s_eff2 / max(float(gyp @ gyp), 1e-6) + floor2
        H = np.zeros((2, 9)); H[0, 0] = H[1, 1] = 1.0
        ok, nis = self._update(np.array([fx + self.reg_off[0] - xe, fy + self.reg_off[1] - ye]), H, np.diag([sx2, sy2]), CHI2_2DOF)
        if ok and c.alt_sync:
            # the horizontal jump changes the map height under the filter by dh; the old altitude state was
            # fitted to the old position, so its error can be as large as dh. Inflate the altitude variance
            # by dh^2 so the (gated) altimeter update can pull the altitude back instead of locking out.
            dh = float(self.map.height(self.x[0], self.x[1]) - self.map.height(xe, ye))
            self.P[2, 2] += dh ** 2
        fx, fy = fx + self.reg_off[0], fy + self.reg_off[1]
        err = float(np.hypot(fx - r[0], fy - r[1]))
        if not ok:
            return {"trn": "rejected_gate", "trn_err": err, "trn_ncc": best, "trn_second": second}
        return {"trn": "fix", "trn_err": err, "trn_ncc": best, "trn_second": second, "trn_sigma": float(np.sqrt(0.5 * (sx2 + sy2)))}
