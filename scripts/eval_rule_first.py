"""Hand-written matchability rule, but only at the FIRST decision point (then no more detours).
Compare with eval_detour.py `rule` (up to 3 decisions). Same seeds 10000+i.
    python scripts/eval_rule_first.py <scenario|tile.npy> N tag [nav flags]"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from trnrl.env import LunarLandingEnv
from trnrl.windows import window_cfg, site_from_height, pop_nav_flags, apply_nav_flags
import trnrl.benchmarks as _bm
from trnrl.benchmarks import build
from trnrl.detour import DetourGuidance2
from evaluate import run_episode, summarize
NAV = pop_nav_flags(sys.argv)
tile, N, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if tile.endswith(".npy"):
    cfg = window_cfg(**NAV); sites = [site_from_height(np.load(tile).astype("float32"), cfg, tile)]
else:
    _bm.CFG_TWEAK = lambda c: apply_nav_flags(c, NAV); cfg, sites = build(tile, "test")
env = LunarLandingEnv(sites, cfg, seed=0)
rows = []
for i in range(N):
    calls = [0]
    def chooser(f):
        calls[0] += 1
        if calls[0] > 1: return 0
        m = f[7:23]; k = int(np.argmax(m)); return k + 1 if m[k] > 0.05 else 0
    g = DetourGuidance2(chooser=chooser)
    rows.append(run_episode(env, g.act, seed=10_000 + i))
s = summarize(rows)
json.dump({"tile": tile, "which": "rule_first", "nav": NAV, "summary": s, "episodes": rows}, open(f"results/{tag}.json", "w"), indent=1, default=float)
print(tag, {k: s[k] for k in ("safe_landing_rate", "precise_landing_rate_<=50m", "outcomes", "episodes_with_zero_fixes")})
