"""Train PPO on a benchmark scenario.
    python scripts/train_ppo.py <variant: cov|nocov> <seed> <total_steps> [scenario=mixed_hard] [tag]
"""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from trnrl.env import LunarLandingEnv
from trnrl.residual import ResidualEnv
from trnrl.benchmarks import build

torch.set_num_threads(1)


class ScaleReward(gym.RewardWrapper):
    def __init__(self, env, k=0.05):
        super().__init__(env); self.k = k
    def reward(self, r):
        return r * self.k


class OutcomeLog(BaseCallback):
    """Rolling success/precision stats from episode infos."""
    def __init__(self, every=50_000):
        super().__init__(); self.every, self.buf, self.next = every, [], every
        self.curve = []
    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "outcome" in info:
                self.buf.append((info["outcome"] == "success", info["outcome"] == "success" and info["landing_err"] <= 50.0))
        if self.num_timesteps >= self.next and self.buf:
            b = np.array(self.buf[-500:], dtype=float)
            self.curve.append((self.num_timesteps, float(b[:, 0].mean()), float(b[:, 1].mean())))
            print(f"[{self.num_timesteps:>9}] safe={b[:,0].mean():.2f} precise={b[:,1].mean():.2f} (last {len(b)} eps)", flush=True)
            self.next += self.every
        return True


KRES = 0.5
if "--k" in sys.argv:                      # residual scale: action = clip(baseline + k * policy)
    _i = sys.argv.index("--k"); KRES = float(sys.argv[_i + 1]); del sys.argv[_i:_i + 2]

if __name__ == "__main__":
    variant, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    scenario = sys.argv[4] if len(sys.argv) > 4 else "mixed_hard"
    tag = sys.argv[5] if len(sys.argv) > 5 else f"ppo_{variant}_{scenario}_s{seed}"
    cfg, sites = build(scenario, split="train")
    cfg.obs_covariance = (variant == "cov")

    def mk(i):
        def f():
            e = ResidualEnv(LunarLandingEnv(sites, cfg, seed=seed * 100 + i), k=KRES)
            return Monitor(ScaleReward(e))
        return f
    venv = DummyVecEnv([mk(i) for i in range(8)])
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=512, batch_size=512, n_epochs=8,
                learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, ent_coef=0.0, clip_range=0.2,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.5), verbose=0, device="cpu")
    cb = OutcomeLog(every=max(steps // 20, 20_000))
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    model.save(f"results/{tag}")
    json.dump({"variant": variant, "k": KRES, "seed": seed, "steps": steps, "scenario": scenario,
               "wall_s": time.time() - t0, "train_curve": cb.curve}, open(f"results/{tag}_train.json", "w"))
    print("done", tag, f"{time.time()-t0:.0f}s", flush=True)
