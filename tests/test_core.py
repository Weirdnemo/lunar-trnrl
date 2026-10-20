import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from trnrl.env import EnvConfig, LunarLandingEnv, Site, make_sites, G0
from trnrl.guidance import PolyGuidance
from trnrl.nav import G_VEC, NavConfig, NavSystem
from trnrl.terrain import Terrain, synthetic_lunar


# ------------------------------------------------------------------ terrain
def test_slope_of_known_plane():
    dx = 10.0
    y, x = np.mgrid[0:64, 0:64]
    h = np.tan(np.radians(7.0)) * x * dx          # 7 deg ramp along x
    t = Terrain(h, dx)
    s = t.slope_deg()
    assert np.allclose(s[5:-5, 5:-5], 7.0, atol=0.05)


def test_height_interpolation_and_axes():
    h = np.zeros((10, 20))
    h[3, 7] = 5.0                                   # row=y index 3, col=x index 7
    t = Terrain(h, 2.0)
    assert t.height(14.0, 6.0) == pytest.approx(5.0)
    assert t.height(15.0, 6.0) == pytest.approx(2.5)  # halfway in x
    assert t.height(np.array([14.0]), np.array([6.0]))[0] == pytest.approx(5.0)


def test_downsample_keeps_registration():
    t = synthetic_lunar(128, 10.0, seed=3)
    m = t.downsample(2)
    assert m.dx == 20.0 and m.h.shape == (64, 64)
    xs = np.array([100.0, 400.0, 700.0])
    assert np.max(np.abs(t.height(xs, xs) - m.height(xs, xs))) < 3.0


def test_synthetic_terrain_is_deterministic():
    a = synthetic_lunar(64, 10.0, seed=5).h
    b = synthetic_lunar(64, 10.0, seed=5).h
    c = synthetic_lunar(64, 10.0, seed=6).h
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_safe_map_rejects_steep_cells():
    dx = 10.0
    y, x = np.mgrid[0:80, 0:80]
    h = np.where(x < 40, 0.0, 0.5 * (x - 40) * dx)  # flat half, steep half
    safe = Terrain(h.astype(float), dx).safe_map(10.0, 0.6, 15.0)
    assert safe[40, 10] and not safe[40, 70]


# ----------------------------------------------------------------- dynamics
def test_free_fall_matches_analytic():
    cfg = EnvConfig()
    sites = make_sites(1, cfg, seed0=1000)
    env = LunarLandingEnv(sites, cfg, seed=0)
    env.reset(seed=1)
    z0, vz0 = env.r[2], env.v[2]
    a_min = np.array([-1.0, 0.0, 0.0])               # min throttle, vertical
    T, d = env.action_to_thrust(a_min)
    env.step(a_min)
    acc = T / env.L.m0 * d[2] - 1.62                  # mass change over 1 s is small
    # mass drops by ~ T/(Isp*g0)*1 s ~ 0.4 kg of 2000, so allow 0.1% slack
    assert env.v[2] - vz0 == pytest.approx(acc * 1.0, rel=2e-3)
    assert env.r[2] - z0 == pytest.approx(vz0 + 0.5 * acc, rel=5e-3, abs=0.05)


def test_fuel_consumption_matches_rocket_equation():
    cfg = EnvConfig()
    env = LunarLandingEnv(make_sites(1, cfg, seed0=1000), cfg, seed=0)
    env.reset(seed=2)
    m0 = env.mass
    T, _ = env.action_to_thrust(np.array([1.0, 0.0, 0.0]))
    env.step(np.array([1.0, 0.0, 0.0]))
    assert m0 - env.mass == pytest.approx(T / (env.L.isp * G0) * 1.0, rel=1e-6)


def test_thrust_limits_and_tilt():
    cfg = EnvConfig()
    env = LunarLandingEnv(make_sites(1, cfg, seed0=1000), cfg, seed=0)
    T, d = env.action_to_thrust(np.array([5.0, 5.0, 5.0]))   # out-of-range action is clipped
    assert T == pytest.approx(env.L.thrust_max)
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert np.degrees(np.arccos(d[2])) <= env.L.tilt_max_deg + 1e-6


def test_observation_contains_no_truth():
    """Changing truth-only quantities must not change the observation."""
    cfg = EnvConfig()
    env = LunarLandingEnv(make_sites(2, cfg, seed0=1000), cfg, seed=0)
    obs, _ = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
    scal = obs[:11]
    n = env.nav
    rel = (n.pos - env.target) / 1500.0
    assert np.allclose(scal[:3], rel, atol=1e-5)       # built from the estimate, not env.r


def test_covariance_ablation_zeroes_features():
    cfg = EnvConfig()
    cfg.obs_covariance = False
    env = LunarLandingEnv(make_sites(1, cfg, seed0=1000), cfg, seed=0)
    obs, _ = env.reset(seed=3)
    assert np.all(obs[8:11] == 0.0)


# ---------------------------------------------------------------------- nav
def test_filter_covariance_stays_symmetric_psd():
    cfg = EnvConfig()
    env = LunarLandingEnv(make_sites(1, cfg, seed0=1000), cfg, seed=0)
    g = PolyGuidance()
    env.reset(seed=4)
    done = False
    while not done:
        _, _, te, tr, _ = env.step(g.act(env))
        done = te or tr
        P = env.nav.P
        assert np.allclose(P, P.T, atol=1e-6 * max(1.0, np.abs(P).max()))
        assert np.linalg.eigvalsh(P).min() > -1e-6


def test_nees_consistent_with_trn():
    """Averaged over episodes with TRN on, NEES should be near the state dimension (9)."""
    cfg = EnvConfig()
    env = LunarLandingEnv(make_sites(4, cfg, seed0=1000), cfg, seed=0)
    g = PolyGuidance()
    vals = []
    for s in range(20):
        env.reset(seed=500 + s)
        done = False
        while not done:
            _, _, te, tr, _ = env.step(g.act(env))
            done = te or tr
        vals.append(np.median(env.history["nees"]))
    m = float(np.mean(vals))
    assert 3.0 < m < 20.0, m


def test_trn_improves_over_dead_reckoning():
    errs = {}
    for trn in (True, False):
        cfg = EnvConfig()
        cfg.nav.enable_trn = trn
        env = LunarLandingEnv(make_sites(4, cfg, seed0=1000), cfg, seed=0)
        g = PolyGuidance()
        e = []
        for s in range(15):
            env.reset(seed=700 + s)
            done = False
            while not done:
                _, _, te, tr, info = env.step(g.act(env))
                done = te or tr
            e.append(info["pos_err_h"])
        errs[trn] = np.median(e)
    assert errs[True] < 0.25 * errs[False], errs


def test_featureless_terrain_never_yields_a_fix():
    """Near-flat terrain (0.1 m white-noise relief) must not produce accepted TRN fixes."""
    rng = np.random.default_rng(0)
    flat = Terrain(rng.normal(0, 0.1, (200, 200)), 10.0)
    onboard = flat.downsample(2)
    accepted = 0
    for s in range(60):
        nav = NavSystem(flat, onboard, NavConfig(), np.random.default_rng(s))
        r = np.array([1000.0, 1000.0, 600.0])
        nav.reset(r, np.zeros(3))
        accepted += nav._trn_fix(r, 600.0)["trn"] == "fix"
    assert accepted == 0, accepted


# ----------------------------------------------------------------- baseline
def test_baseline_lands_on_nominal_scenarios():
    cfg = EnvConfig()
    env = LunarLandingEnv(make_sites(4, cfg, seed0=1000), cfg, seed=0)
    g = PolyGuidance()
    ok = 0
    for s in range(10):
        env.reset(seed=900 + s)
        done = False
        while not done:
            _, _, te, tr, info = env.step(g.act(env))
            done = te or tr
        ok += info["outcome"] == "success" and info["landing_err"] < 50
    assert ok >= 8, ok


def test_reset_is_reproducible():
    cfg = EnvConfig()
    sites = make_sites(3, cfg, seed0=1000)
    a = LunarLandingEnv(sites, cfg, seed=0)
    b = LunarLandingEnv(sites, cfg, seed=99)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    assert np.array_equal(oa, ob)
    act = np.array([0.1, 0.2, -0.1], dtype=np.float32)
    ra = a.step(act)
    rb = b.step(act)
    assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1]


def test_residual_with_zero_action_equals_baseline():
    from trnrl.residual import ResidualEnv
    cfg = EnvConfig()
    sites = make_sites(2, cfg, seed0=1000)
    base_env = LunarLandingEnv(sites, cfg, seed=0)
    g = PolyGuidance()
    base_env.reset(seed=11)
    done = False
    while not done:
        _, _, te, tr, info_b = base_env.step(g.act(base_env))
        done = te or tr
    res = ResidualEnv(LunarLandingEnv(sites, cfg, seed=0), k=0.5)
    res.reset(seed=11)
    done = False
    while not done:
        _, _, te, tr, info_r = res.step(np.zeros(3, np.float32))
        done = te or tr
    assert info_b["outcome"] == info_r["outcome"]
    assert info_b["landing_err"] == pytest.approx(info_r["landing_err"], abs=1e-6)
