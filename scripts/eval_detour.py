"""Paired evaluation: baseline vs learned detour (or never/always detour) on one tile.
    python scripts/eval_detour.py <tile.npy | scenario> <baseline|rule|hold|away|away400|ppo:<model.zip>> N tag [--init_pos_h 600 --alt_sync --patch_n 31]
rule = hand-written one detour toward the best matchability within 800 m. Same episode seeds 10000+i for all."""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from trnrl.env import LunarLandingEnv
from trnrl.guidance import PolyGuidance, WaitGuidance
from trnrl.windows import window_cfg, site_from_height, pop_nav_flags, apply_nav_flags
import trnrl.benchmarks as _bm
from trnrl.benchmarks import build
from trnrl.detour import DetourGuidance2, site_match, N_ACT
from evaluate import run_episode, summarize

NAV = pop_nav_flags(sys.argv)
ORACLE = bool(NAV.get("oracle"))
if "--patch_n" in sys.argv:
    i = sys.argv.index("--patch_n"); NAV["patch_n"] = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
tile, which, N, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
if tile.endswith(".npy"):
    cfg = window_cfg(**NAV)
    sites = [site_from_height(np.load(tile).astype("float32"), cfg, tile)]
else:                                              # scenario name: held-out TEST sites
    _bm.CFG_TWEAK = lambda c: apply_nav_flags(c, NAV)
    cfg, sites = build(tile, "test")
    if ORACLE:
        cfg.oracle_nav = True; cfg.nav.enable_trn = False
env = LunarLandingEnv(sites, cfg, seed=0)
if which == "baseline":
    act = PolyGuidance().act
elif which == "hold":
    act = WaitGuidance(max_wait=70.0).act
elif which == "holdall":                          # always take the hover action when a decision is due
    g = DetourGuidance2(chooser=lambda f: N_ACT - 1); act = g.act
elif which == "rule":
    def chooser(f):                                  # features 7..22 are matchability at the 16 waypoints
        m = f[7:7 + 16]; k = int(np.argmax(m)); return k + 1 if m[k] > 0.05 else 0
    g = DetourGuidance2(chooser=chooser); act = g.act
elif which in ("awayfirst", "away400first"):      # same rule, but ONLY at the first decision (no further detours): the policy of the headline 93.5% / 84.5%
    off = 9 if which == "awayfirst" else 1
    _n = [0]
    def _ch(f, off=off, _n=_n):
        _n[0] += 1
        return off + int(np.argmin(np.asarray(f)[23:31])) if _n[0] == 1 else 0
    class _G1(DetourGuidance2):
        def reset_episode(self):
            super().reset_episode(); _n[0] = 0
    g = _G1(chooser=_ch); act = g.act
elif which in ("away", "away400"):                # one-line rule: detour directly away from the target (features 23..30 = alignment of each direction with the target)
    off = 9 if which == "away" else 1
    g = DetourGuidance2(chooser=lambda f: off + int(np.argmin(np.asarray(f)[23:31]))); act = g.act
else:
    from stable_baselines3 import PPO
    model = PPO.load(which.split(":", 1)[1], device="cpu")
    _det = not which.startswith("ppos:")           # ppos:<zip> samples actions from the policy (stochastic) instead of the greedy action
    g = DetourGuidance2(chooser=lambda f: int(model.predict(f, deterministic=_det)[0])); act = g.act
rows = [run_episode(env, act, seed=10_000 + i) for i in range(N)]
s = summarize(rows)
json.dump({"tile": tile, "which": which, "nav": NAV, "summary": s, "episodes": rows}, open(f"results/{tag}.json", "w"), indent=1, default=float)
print(tag, {k: s[k] for k in ("safe_landing_rate", "precise_landing_rate_<=50m", "outcomes", "episodes_with_zero_fixes")})
