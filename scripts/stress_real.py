"""Does stressing navigation create headroom on the real tiles? Baseline vs hold rule, 100 episodes each.
    python scripts/stress_real.py
Settings: default (40% blackouts of 30-70 s), big_init (300 m initial position error), long blackout
(always, 60-100 s starting in the first 80 s), both. Tiles: data/site04.npy (train split), data/site23.npy (test split)."""
import sys, json, copy
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from trnrl.env import LunarLandingEnv
from trnrl.guidance import PolyGuidance, WaitGuidance
from trnrl.benchmarks import build
from evaluate import run_episode, summarize

SETTINGS = {
    "default": {},
    "big_init": {"nav.init_pos_h": 300.0},
    "long_blackout": {"p_outage": 1.0, "outage_dur": (60.0, 100.0)},
    "both": {"nav.init_pos_h": 300.0, "p_outage": 1.0, "outage_dur": (60.0, 100.0)},
}
out = {}
for split in ("train", "test"):
    for name, kw in SETTINGS.items():
        cfg, sites = build("real+dec", split)
        for k, v in kw.items():
            if k.startswith("nav."): setattr(cfg.nav, k[4:], v)
            else: setattr(cfg, k, v)
        for pol in ("baseline", "wait70"):
            act = PolyGuidance().act if pol == "baseline" else WaitGuidance(max_wait=70.0).act
            env = LunarLandingEnv(sites, cfg, seed=0)
            rows = [run_episode(env, act, seed=10_000 + i) for i in range(100)]
            s = summarize(rows)
            tile = sites[0].truth.name
            out[f"{tile}|{name}|{pol}"] = s
            o = s["outcomes"]
            print(f"{tile} {name:14s} {pol:9s} precise={s['precise_landing_rate_<=50m']:.2f} safe={s['safe_landing_rate']:.2f} "
                  f"p90={s['p90_landing_err_m']:.1f} fuel={s['median_fuel_kg']:.0f} zero_fix={s['episodes_with_zero_fixes']} {o}", flush=True)
json.dump(out, open("results/stress_real.json", "w"), indent=1, default=float)
