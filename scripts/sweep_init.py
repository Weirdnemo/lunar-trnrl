"""Baseline vs initial position error, with the TRN search window narrow (20 cells, the old default) or
widened to 3 sigma. Real tiles data/site04.npy and data/site23.npy, 100 episodes per cell.
    python scripts/sweep_init.py [sigma1 sigma2 ...]      default: 120 300 600 1000"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from trnrl.env import LunarLandingEnv
from trnrl.guidance import PolyGuidance
from trnrl.dem_source import BigDEM
from trnrl.windows import window_cfg, site_from_height
from evaluate import run_episode, summarize

sigmas = [float(a) for a in sys.argv[1:]] or [120, 300, 600, 1000]
out = {}
for tile in ("data/site04.npy", "data/site23.npy"):
    h = np.load(tile).astype("float32")
    for sig in sigmas:
        for label, cells in (("narrow20", 20), ("wide3sigma", None)):
            cfg = window_cfg(init_pos_h=sig, search_cells=cells)
            site = site_from_height(h, cfg, tile)
            env = LunarLandingEnv([site], cfg, seed=0)
            rows = [run_episode(env, PolyGuidance().act, seed=10_000 + i) for i in range(100)]
            s = summarize(rows)
            out[f"{tile}|{sig:.0f}|{label}"] = s
            print(f"{tile[5:11]} sigma={sig:5.0f} {label:10s} cells={cfg.nav.search_max_cells:3d} "
                  f"precise={s['precise_landing_rate_<=50m']:.2f} p90={s['p90_landing_err_m']:.1f} "
                  f"zero_fix={s['episodes_with_zero_fixes']} {s['outcomes']}", flush=True)
json.dump(out, open("results/sweep_init.json", "w"), indent=1, default=float)
