"""Train the learned-detour decision (baseline guidance flies, policy picks detours) with PPO.
    python scripts/train_detour.py <tile.npy | scenario> <steps> <seed> <tag> [--init_pos_h 600 --alt_sync --patch_n 31]
Each PPO step is one decision (about 1 to 3 per episode), so use steps in the thousands, not millions.
Initial policy strongly prefers 'no detour' (= baseline). Saves results/<tag>.zip and results/<tag>_train.json."""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from trnrl.env import LunarLandingEnv
from trnrl.windows import window_cfg, site_from_height, pop_nav_flags, apply_nav_flags
import trnrl.benchmarks as _bm
from trnrl.benchmarks import build
from trnrl.detour import DetourEnv, N_ACT

torch.set_num_threads(1)
NAV = pop_nav_flags(sys.argv)
if "--patch_n" in sys.argv:
    i = sys.argv.index("--patch_n"); NAV["patch_n"] = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
if "--init_hold" in sys.argv:
    sys.argv.remove("--init_hold"); _H = True
else:
    _H = False
COLD = "--cold" in sys.argv                      # no warm start: all action logits start equal
if COLD: sys.argv.remove("--cold")
LR = 1e-4
if "--lr" in sys.argv:
    _i = sys.argv.index("--lr"); LR = float(sys.argv[_i + 1]); del sys.argv[_i:_i + 2]
tile, steps, seed, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]


class Log(BaseCallback):
    def __init__(self):
        super().__init__(); self.rets, self.acts, self.t0 = [], [], time.time()

    def _on_step(self):
        a = int(self.locals["actions"][0]); self.acts.append(a)
        for inf, d in zip(self.locals["infos"], self.locals["dones"]):
            if d:
                o = inf.get("outcome"); self.rets.append(1.0 if (o == "success" and inf.get("landing_err", 1e9) <= 50) else 0.0)
        if self.num_timesteps % 500 == 0 and self.rets:
            r = self.rets[-100:]
            print(f"step {self.num_timesteps} episodes {len(self.rets)} rolling precise {np.mean(r):.3f} "
                  f"detour rate {np.mean(np.array(self.acts[-500:]) > 0):.2f} wall {time.time()-self.t0:.0f}s", flush=True)
        return True


if tile.endswith(".npy"):
    cfg = window_cfg(**NAV)
    sites = [site_from_height(np.load(tile).astype("float32"), cfg, tile)]
else:                                              # a benchmark scenario name, e.g. mixed_hard+dec (train sites)
    _bm.CFG_TWEAK = lambda c: apply_nav_flags(c, NAV)   # was missing: --realistic / nav flags were silently ignored for scenario names
    cfg, sites = build(tile, "train")
env = DetourEnv(LunarLandingEnv(sites, cfg, seed=seed))
model = PPO("MlpPolicy", env, seed=seed, n_steps=128, batch_size=64, learning_rate=LR, ent_coef=0.01,
            gamma=1.0, gae_lambda=1.0, clip_range=0.1, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
INIT = "hold" if _H else "none"       # --init_hold: start at 'always hover' (87.5% on mixed_hard)
with torch.no_grad():                      # start at a known-good policy: no detour (baseline) or hover
    model.policy.action_net.bias.zero_()
    if not COLD: model.policy.action_net.bias[N_ACT - 1 if INIT == "hold" else 0] = 4.0
cb = Log()
model.learn(total_timesteps=steps, callback=cb)
model.save(f"results/{tag}")
json.dump({"tile": tile, "steps": steps, "seed": seed, "nav": NAV, "rolling_precise": cb.rets,
           "detour_rate_last500": float(np.mean(np.array(cb.acts[-500:]) > 0))}, open(f"results/{tag}_train.json", "w"))
