"""First-fix TRN statistics at the initial state vs ambiguity thresholds."""
import sys, copy
sys.path.insert(0, ".")
import numpy as np
from trnrl.env import LunarLandingEnv, EnvConfig, make_sites

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
base = EnvConfig()
sites = make_sites(6, base, seed0=1000)
print(f"{N} first-fix attempts at episode start (sigma_h0={base.nav.init_pos_h} m, map dx={sites[0].map.dx} m)")
print("lobe_drop amb_margin | accepted  wrong(>60m | accepted)  median_err_accepted  reasons")
for lobe_drop, amb in [(0.10, 0.03), (0.10, 0.05), (0.05, 0.03), (0.15, 0.03), (0.10, 0.01), (0.10, 0.10)]:
    cfg = EnvConfig()
    cfg.nav.lobe_drop, cfg.nav.amb_margin = lobe_drop, amb
    env = LunarLandingEnv(sites, cfg, seed=0)
    acc = wrong = 0; errs = []; reasons = {}
    for i in range(N):
        env.reset(seed=i)
        agl = env.r[2] - env.site.truth.height(env.r[0], env.r[1])
        out = env.nav._trn_fix(env.r, agl)
        reasons[out["trn"]] = reasons.get(out["trn"], 0) + 1
        if out["trn"] == "fix":
            acc += 1; errs.append(out["trn_err"]); wrong += out["trn_err"] > 60
    med = np.median(errs) if errs else float("nan")
    print(f"  {lobe_drop:.2f}      {amb:.2f}     | {acc/N:6.2f}   {wrong/max(acc,1):6.2f}              {med:7.1f} m          {reasons}")
