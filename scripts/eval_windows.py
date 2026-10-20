"""Compare policies on random windows from the held-out (right 30%) or train region of a big DEM.

    python scripts/eval_windows.py <dem> <baseline|wait:70|ppo:<model.zip>:<cov|nocov>> N tag [split=test] [n_windows=24]

Every policy sees the same windows and the same episode seeds for a given (tag-independent) N,
so results are paired. Writes results/<tag>.json.
"""
import sys, json, time
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
from trnrl.env import LunarLandingEnv
from trnrl.guidance import PolyGuidance, WaitGuidance
from trnrl.dem_source import BigDEM
from trnrl.windows import window_cfg, sample_sites, pop_nav_flags
from evaluate import run_episode, summarize

SPLIT_FRAC = 0.7
NAV_FLAGS = pop_nav_flags(sys.argv)   # --init_pos_h X  --search_cells N
if "--frac" in sys.argv:            # --frac 1.0 = no split inside the map (train and test on different files)
    i = sys.argv.index("--frac"); SPLIT_FRAC = float(sys.argv[i + 1]); del sys.argv[i:i + 2]

if __name__ == "__main__":
    path, which, N, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    split = sys.argv[5] if len(sys.argv) > 5 else "test"
    n_win = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    dem = BigDEM(path, dx=10.0 if path.endswith(".npy") else None)
    cfg = window_cfg(**NAV_FLAGS)
    sites, origins = sample_sites(dem, split, n_win, seed=777 if split == "test" else 778, cfg=cfg, frac=SPLIT_FRAC)
    t0 = time.time()
    if which.startswith("ppo:"):
        from stable_baselines3 import PPO
        from trnrl.residual import ResidualEnv
        _, mpath, variant = which.split(":")
        cfg.obs_covariance = (variant == "cov")
        env = ResidualEnv(LunarLandingEnv(sites, cfg, seed=0), k=0.5)
        model = PPO.load(mpath, device="cpu")
        act = lambda e: model.predict(e.last_obs, deterministic=True)[0]
    else:
        env = LunarLandingEnv(sites, cfg, seed=0)
        act = WaitGuidance(max_wait=float(which.split(":")[1])).act if which.startswith("wait") else PolyGuidance().act
    rows = [run_episode(env, act, seed=10_000 + i) for i in range(N)]
    summ = summarize(rows)
    summ["wall_s"] = time.time() - t0
    json.dump({"config": tag, "dem": str(path), "split": split, "n_windows": len(sites), "window_origins": origins,
               "summary": summ, "episodes": rows}, open(f"results/{tag}.json", "w"), indent=1, default=float)
    print(f"{tag}: windows={len(sites)} N={N}")
    for k, v in summ.items():
        print(f"{k:42s} {v}")
