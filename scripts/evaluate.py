"""Monte-Carlo evaluation of a guidance policy. Usage:
    python scripts/evaluate.py baseline|wait|ppo:<path>:<cov|nocov> N tag scenario   (scenario: nominal|no_trn|low_relief|big_init|mixed_hard)
Writes results/<tag>.json (per-episode) and prints a summary.
"""
import sys, json, time, collections
sys.path.insert(0, ".")
import numpy as np
from trnrl.env import LunarLandingEnv, EnvConfig, make_sites
from trnrl.guidance import PolyGuidance, WaitGuidance
from trnrl.benchmarks import build


def run_episode(env, act, seed, site=None):
    env.reset(seed=seed, options={} if site is None else {"site": site})
    done = False
    while not done:
        _, _, te, tr, info = env.step(act(env))
        done = te or tr
    h = env.history
    trn = collections.Counter(h["trn"])
    attempts = sum(v for k, v in trn.items() if k != "none")
    fix_err = [e for t, e in zip(h["trn"], h["trn_err"]) if t == "fix"]
    return {
        "seed": seed, "outcome": info["outcome"], "landing_err": info.get("landing_err", np.nan),
        "touch_vz": info.get("touch_vz", np.nan), "touch_vh": info.get("touch_vh", np.nan),
        "fuel": info["fuel_used"], "t_end": info["t"], "final_sigma_h": info.get("final_sigma_h", np.nan),
        "final_nav_err_h": info["pos_err_h"], "trn_attempts": attempts, "trn_fixes": trn.get("fix", 0),
        "trn_rejected": {k: v for k, v in trn.items() if k not in ("none", "fix")},
        "median_fix_err": float(np.median(fix_err)) if fix_err else float("nan"),
        "median_nees": float(np.median(h["nees"])), "max_nees": float(np.max(h["nees"])),
    }


def summarize(rows, precise_m=50.0):
    n = len(rows)
    safe = [r for r in rows if r["outcome"] == "success"]
    prec = [r for r in safe if r["landing_err"] <= precise_m]
    out = collections.Counter(r["outcome"] for r in rows)
    f = lambda k, rs: float(np.nanmedian([r[k] for r in rs])) if rs else float("nan")
    return {
        "episodes": n, "safe_landing_rate": len(safe) / n, f"precise_landing_rate_<={precise_m:.0f}m": len(prec) / n,
        "outcomes": dict(out), "median_landing_err_m": f("landing_err", safe), "p90_landing_err_m": float(np.nanpercentile([r["landing_err"] for r in safe], 90)) if safe else float("nan"),
        "median_fuel_kg": f("fuel", safe), "median_touch_vz": f("touch_vz", safe),
        "trn_fix_rate": sum(r["trn_fixes"] for r in rows) / max(sum(r["trn_attempts"] for r in rows), 1),
        "episodes_with_zero_fixes": sum(r["trn_fixes"] == 0 for r in rows),
        "median_of_median_NEES": f("median_nees", rows), "frac_episodes_median_NEES_gt_2x_dof": float(np.mean([r["median_nees"] > 18 for r in rows])),
    }


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    tag = sys.argv[3] if len(sys.argv) > 3 else which
    scenario = sys.argv[4] if len(sys.argv) > 4 else "nominal"
    split = sys.argv[5] if len(sys.argv) > 5 else "test"
    cfg, sites = build(scenario, split=split)
    env = LunarLandingEnv(sites, cfg, seed=0)
    g = PolyGuidance()
    t0 = time.time()
    if which.startswith("ppo:"):
        from stable_baselines3 import PPO
        from trnrl.residual import ResidualEnv
        _, path, variant = which.split(":")
        kres = float(variant.split("@")[1]) if "@" in variant else 0.5      # residual scale, e.g. nocov@1.0
        variant = variant.split("@")[0]
        cfg.obs_covariance = (variant == "cov")
        env = ResidualEnv(LunarLandingEnv(sites, cfg, seed=0), k=kres)
        model = PPO.load(path, device="cpu")
        act = lambda e: model.predict(e.last_obs, deterministic=True)[0]
    elif which.startswith("wait"):
        mw = float(which.split(":")[1]) if ":" in which else 90.0
        act = WaitGuidance(max_wait=mw).act
    else:
        act = g.act
    rows = [run_episode(env, act, seed=10_000 + i) for i in range(N)]
    summ = summarize(rows)
    summ["wall_s"] = time.time() - t0
    json.dump({"config": tag, "summary": summ, "episodes": rows}, open(f"results/{tag}.json", "w"), indent=1, default=float)
    for k, v in summ.items():
        print(f"{k:42s} {v}")
