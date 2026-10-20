"""Train the residual PPO policy on many random 8 km windows of a big DEM (train region only).

    python scripts/train_ppo_windows.py <dem.tif|dem.npy> <cov|nocov> <seed> <steps> [n_windows=48] [tag] [--frac 1.0] [--ckpt_every N]

The left 70% of the map (by columns) is the train region; the right 30% is never used here,
so eval_windows.py can test on terrain the policy has not seen. Needs a map at least 8 km tall
and wide enough for both regions. Writes results/<tag>.zip (final), results/<tag>_step<N>.zip (checkpoints), results/<tag>_best.zip and
results/<tag>_train.json. Evaluate several checkpoints, not only the final one.
"""
import sys, time, json
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from trnrl.env import LunarLandingEnv
from trnrl.residual import ResidualEnv
from trnrl.dem_source import BigDEM
from trnrl.windows import window_cfg, sample_sites, pop_nav_flags
from train_ppo import ScaleReward, OutcomeLog

torch.set_num_threads(1)
SPLIT_FRAC = 0.7
NAV_FLAGS = pop_nav_flags(sys.argv)   # --init_pos_h X  --search_cells N
CKPT_EVERY = None                   # --ckpt_every N: log and save a checkpoint every N steps (default steps/20)
for _flag in ("--frac", "--ckpt_every"):
    if _flag in sys.argv:
        _i = sys.argv.index(_flag); _v = float(sys.argv[_i + 1]); del sys.argv[_i:_i + 2]
        if _flag == "--frac": SPLIT_FRAC = _v
        else: CKPT_EVERY = int(_v)


class CkptLog(OutcomeLog):
    """OutcomeLog that also saves results/<tag>_step<N>.zip at every log line and results/<tag>_best.zip
    (highest rolling training precision; stochastic policy on training windows, so optimistic).
    Needed because PPO can degrade late (E019): evaluate several checkpoints, not only the last."""
    def __init__(self, tag, every):
        super().__init__(every=every); self.tag, self.best = tag, -1.0

    def _on_step(self):
        before = self.next
        r = super()._on_step()
        if self.next != before:
            steps, _, prec = self.curve[-1]
            self.model.save(f"results/{self.tag}_step{steps}")
            if prec > self.best:
                self.best = prec
                self.model.save(f"results/{self.tag}_best")
        return r
if "--frac" in sys.argv:            # --frac 1.0 = no split inside the map (train and test on different files)
    i = sys.argv.index("--frac"); SPLIT_FRAC = float(sys.argv[i + 1]); del sys.argv[i:i + 2]

if __name__ == "__main__":
    path, variant, seed, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    n_win = int(sys.argv[5]) if len(sys.argv) > 5 else 48
    tag = sys.argv[6] if len(sys.argv) > 6 else f"ppo_win_{variant}_s{seed}"
    dem = BigDEM(path, dx=10.0 if path.endswith(".npy") else None)
    cfg = window_cfg(**NAV_FLAGS)
    cfg.obs_covariance = (variant == "cov")
    t0 = time.time()
    sites, origins = sample_sites(dem, "train", n_win, seed=seed, cfg=cfg, frac=SPLIT_FRAC)
    print(f"{len(sites)} train windows ready in {time.time()-t0:.0f}s", flush=True)

    def mk(i):
        def f():
            return Monitor(ScaleReward(ResidualEnv(LunarLandingEnv(sites, cfg, seed=seed * 100 + i), k=0.5)))
        return f
    venv = DummyVecEnv([mk(i) for i in range(8)])
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=512, batch_size=512, n_epochs=8,
                learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, ent_coef=0.0, clip_range=0.2,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.5),
                verbose=0, device="cpu")
    cb = CkptLog(tag, CKPT_EVERY or max(steps // 20, 20_000))
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    model.save(f"results/{tag}")
    json.dump({"dem": str(path), "variant": variant, "seed": seed, "steps": steps, "n_windows": len(sites),
               "split_frac": SPLIT_FRAC, "window_origins": origins, "wall_s": time.time() - t0,
               "train_curve": cb.curve}, open(f"results/{tag}_train.json", "w"))
    print("done", tag, f"{time.time()-t0:.0f}s", flush=True)
