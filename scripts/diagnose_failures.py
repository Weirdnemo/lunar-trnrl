"""Phase 1 diagnosis: why does baseline guidance fail at large initial position error on a real tile?
    python scripts/diagnose_failures.py <tile.npy> <init_pos_h> [episodes=60] [tag] [policy=baseline|wait:70]
Uses the auto-widened TRN search window. Classifies every non-precise episode:
  zero_fix        - no accepted TRN fix all episode
  false_fix       - a fix was accepted that was >100 m wrong
  late_fix        - first good fix after altitude fell below 300 m
  nav_ok_hazard   - navigation error at touchdown <= 15 m but touchdown unsafe/hard (terminal-phase problem)
  nav_bad         - navigation error at touchdown > 15 m (fixed late/partially)
  other
Writes results/<tag>.json."""
import sys, json, collections
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from trnrl.env import LunarLandingEnv
from trnrl.guidance import PolyGuidance, WaitGuidance, LandingGuidance, DetourGuidance
from trnrl.windows import window_cfg, site_from_height

tile, sig = sys.argv[1], float(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
tag = sys.argv[4] if len(sys.argv) > 4 else f"diag_{tile.split('/')[-1][:-4]}_{int(sig)}"
from trnrl.windows import pop_nav_flags
NAVF = pop_nav_flags(sys.argv)          # --alt_sync --patch_n --fov --reg_offset --map_noise --realistic --oracle ...
NAVF["init_pos_h"] = sig
cfg = window_cfg(**NAVF)
if "--oracle" in sys.argv:            # guidance sees the true state; TRN off for speed (ceiling experiment)
    sys.argv.remove("--oracle"); cfg.oracle_nav = True; cfg.nav.enable_trn = False
env = LunarLandingEnv([site_from_height(np.load(tile).astype("float32"), cfg, tile)], cfg, seed=0)
POL = sys.argv[5] if len(sys.argv) > 5 else "baseline"
SEED0 = 10_000
if "--seed0" in sys.argv:
    _i = sys.argv.index("--seed0"); SEED0 = int(sys.argv[_i + 1]); del sys.argv[_i:_i + 2]
act = WaitGuidance(max_wait=float(POL.split(":")[1])).act if POL.startswith("wait") else (LandingGuidance().act if POL == "v2" else (DetourGuidance().act if POL == "detour" else PolyGuidance().act))
rows, cats = [], collections.Counter()
for i in range(N):
    env.reset(seed=SEED0 + i); done = False
    while not done:
        _, _, te, tr, info = env.step(act(env)); done = te or tr
    h = env.history
    r = np.array(h["r"]); alt = r[:, 2] if r.ndim == 2 and r.shape[1] >= 3 else np.full(len(r), np.nan)
    trn, ferr = h["trn"], h["trn_err"]
    fixes = [k for k, t in enumerate(trn) if t == "fix"]
    first = fixes[0] if fixes else None
    max_fix_err = max([ferr[k] for k in fixes], default=float("nan"))
    nav_end = float(info["pos_err_h"]); ok = info["outcome"] == "success" and info.get("landing_err", 1e9) <= 50
    row = {"seed": SEED0 + i, "outcome": info["outcome"], "landing_err": float(info.get("landing_err", np.nan)),
           "n_fix": len(fixes), "t_first_fix": float(h["t"][first]) if first is not None else None,
           "alt_first_fix": float(alt[first]) if first is not None else None, "max_fix_err": float(max_fix_err),
           "nav_err_end": nav_end, "nav_err_v_end": float(np.array(h["r_hat"])[-1][2] - r[-1][2]), "rejects": dict(collections.Counter(t for t in trn if t not in ("none", "fix")))}
    if ok: c = "precise"
    elif not fixes: c = "zero_fix"
    elif max_fix_err > 100: c = "false_fix"
    elif row["alt_first_fix"] is not None and row["alt_first_fix"] < 300: c = "late_fix"
    elif nav_end <= 15 and info["outcome"] in ("unsafe_site", "hard_landing"): c = "nav_ok_hazard"
    elif nav_end > 15: c = "nav_bad"
    else: c = "other"
    row["category"] = c; cats[c] += 1; rows.append(row)
    print(i, c, row["outcome"], f"land_err={row['landing_err']:.0f} nav_end={nav_end:.0f} n_fix={len(fixes)} t1={row['t_first_fix']}", flush=True)
print(dict(cats))
json.dump({"tile": tile, "sigma": sig, "categories": dict(cats), "episodes": rows}, open(f"results/{tag}.json", "w"), indent=1, default=float)
