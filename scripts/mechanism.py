"""Mechanism checks (three modes).
  geom  <scenario> N tag [--realistic ...]   Why does 'detour away from the target' help? At the first decision of each episode that has one,
        compare time-matched hovering (25 s and 60 s) with detours of equal length in different directions relative to the target
        (away / toward / perpendicular, 400 and 800 m). Records time to the next accepted terrain fix, accepted fixes after the decision,
        ambiguity rejections after the decision, outcome.
  noise <scenario> N tag [R=10] [--realistic] How much of the hindsight cap is luck? Replays each decision episode R times per action
        (none, hover 25 s, away 800 m, toward 800 m) with the SAME initial state and a different number of random draws burned at the
        decision (-> different measurement noise afterwards). Reports per-episode success probabilities and the hindsight cap obtained
        by picking the best of R replicates of the SAME action (pure luck, no action effect).
  fix   <tile.npy> N tag [sigma=600] [legacy,fov40,fov60 subset]   Altitude-fix mechanism. Baseline guidance on one tile for patch/fov in {legacy, fov40, fov60} x alt_sync in {off, on}
        (registration offset 6 m, map noise 1 m): time of the first accepted fix, fraction of episodes with a fix before 60 s AGL-time,
        altitude-state error at touchdown (lock-out proxy), precise rate.
Seeds 10000+i. Writes results/<tag>.json and prints a summary."""
import sys, json, collections
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from trnrl.env import LunarLandingEnv
from trnrl.guidance import PolyGuidance
from trnrl.windows import window_cfg, site_from_height, pop_nav_flags, apply_nav_flags
import trnrl.benchmarks as _bm
from trnrl.benchmarks import build
from trnrl.detour import DetourGuidance2, N_ACT

mode = sys.argv[1]
NAV = pop_nav_flags(sys.argv)
args = [a for a in sys.argv[2:] if not a.startswith("--")]
SEED0 = 10_000

class G(DetourGuidance2):
    hold_s = None
    def set_action(self, env, a):
        super().set_action(env, a)
        if a == N_ACT - 1 and self.hold_s:
            self.hold_until = env.t + self.hold_s

def play(env, seed, chooser, hold_s=None, burn=0):
    """Fly one episode. chooser(features)->action at the first decision, then none. Returns outcome row and decision stats."""
    st = {"n": 0, "t_dec": None, "next_fix": None, "fixes_after": 0, "amb_after": 0, "rej_after": 0, "feat": None}
    def ch(f):
        st["n"] += 1
        if st["n"] == 1:
            st["t_dec"] = float(env.t); st["feat"] = [float(x) for x in f]
            if burn: env._rng.random(burn)            # same generator that drives the sensors: shifts the noise afterwards
            return chooser(f)
        return 0
    g = G(chooser=ch); g.hold_s = hold_s
    env.reset(seed=seed); done = False
    while not done:
        _, _, te, tr, info = env.step(g.act(env)); done = te or tr
        if st["t_dec"] is not None:
            k = env.last_info.get("trn", "none")
            if k == "fix":
                st["fixes_after"] += 1
                if st["next_fix"] is None: st["next_fix"] = float(env.t - st["t_dec"])
            elif k == "rejected_ambiguous": st["amb_after"] += 1
            elif k != "none": st["rej_after"] += 1
    ok = info["outcome"] == "success"
    return {"seed": seed, "outcome": info["outcome"], "precise": bool(ok and info.get("landing_err", 1e9) <= 50.0),
            "landing_err": float(info.get("landing_err", float("nan"))), "t_end": float(info["t"]), **st}

def make_env(tile):
    if tile.endswith(".npy"):
        cfg = window_cfg(**NAV); sites = [site_from_height(np.load(tile).astype("float32"), cfg, tile)]
    else:
        _bm.CFG_TWEAK = lambda c: apply_nav_flags(c, NAV); cfg, sites = build(tile, "test")
    return LunarLandingEnv(sites, cfg, seed=0)

def idx_dir(f):
    al = np.array(f[23:31]); return int(np.argmin(al)), int(np.argmax(al)), int(np.argmin(np.abs(al)))   # away, toward, perpendicular

if mode == "geom":
    tile, N, tag = args[0], int(args[1]), args[2]
    env = make_env(tile)
    specs = {"none": (lambda f: 0, None), "hover25": (lambda f: 17, None), "hover60": (lambda f: 17, 60.0)}
    for rad, off in (("400", 1), ("800", 9)):
        specs[f"away{rad}"] = (lambda f, o=off: o + idx_dir(f)[0], None)
        specs[f"toward{rad}"] = (lambda f, o=off: o + idx_dir(f)[1], None)
        specs[f"perp{rad}"] = (lambda f, o=off: o + idx_dir(f)[2], None)
    res = {k: [] for k in specs}; has = []
    for i in range(N):
        seed = SEED0 + i
        first = play(env, seed, specs["none"][0])
        has.append(first["n"] > 0)
        for k, (c, hs) in specs.items():
            res[k].append(first if k == "none" else (play(env, seed, c, hs) if first["n"] > 0 else first))
        if (i + 1) % 10 == 0: print(i + 1, flush=True)
    has = np.array(has); out = {}
    for k, v in res.items():
        sel = [r for r, h in zip(v, has) if h]
        nf = [r["next_fix"] for r in sel]
        out[k] = {"precise_all": float(np.mean([r["precise"] for r in v])), "precise_on_decision_eps": float(np.mean([r["precise"] for r in sel])) if sel else None,
                  "got_fix_after": float(np.mean([x is not None for x in nf])) if sel else None,
                  "median_time_to_fix_s": float(np.median([x for x in nf if x is not None])) if any(x is not None for x in nf) else None,
                  "mean_fixes_after": float(np.mean([r["fixes_after"] for r in sel])) if sel else None,
                  "mean_amb_rejections_after": float(np.mean([r["amb_after"] for r in sel])) if sel else None,
                  "median_t_end_s": float(np.median([r["t_end"] for r in sel])) if sel else None}
    out["_n_decision_episodes"] = int(has.sum()); out["_N"] = N
    json.dump({"mode": mode, "nav": NAV, "summary": out, "episodes": res}, open(f"results/{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1))

elif mode == "noise":
    tile, N, tag = args[0], int(args[1]), args[2]; R = int(args[3]) if len(args) > 3 else 10
    env = make_env(tile)
    acts = {"none": lambda f: 0, "hover25": lambda f: 17, "away800": lambda f: 9 + idx_dir(f)[0], "toward800": lambda f: 9 + idx_dir(f)[1]}
    P = {k: [] for k in acts}; sel = []
    for i in range(N):
        seed = SEED0 + i
        first = play(env, seed, acts["none"])
        if first["n"] == 0: continue
        sel.append(seed)
        for k, c in acts.items():
            P[k].append([play(env, seed, c, burn=1 + 997 * r)["precise"] for r in range(R)])
        if len(sel) % 10 == 0: print(len(sel), "decision episodes", flush=True)
    P = {k: np.array(v, float) for k, v in P.items()}; out = {"R": R, "n_decision_eps": len(sel)}
    for k, v in P.items():
        out[k] = {"mean_success_prob": float(v.mean()), "share_of_episodes_always_ok": float((v.min(1) == 1).mean()),
                  "share_never_ok": float((v.max(1) == 0).mean()), "luck_cap_best_of_R_same_action": float(v.max(1).mean()),
                  "share_episodes_mixed_outcome": float(((v.max(1) == 1) & (v.min(1) == 0)).mean())}
    allv = np.stack(list(P.values())); out["best_of_all_actions_and_replicates"] = float(allv.max((0, 2)).mean())
    json.dump({"mode": mode, "nav": NAV, "seeds": sel, "summary": out, "P": {k: v.tolist() for k, v in P.items()}}, open(f"results/{tag}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

elif mode == "fix":
    tile, N, tag = args[0], int(args[1]), args[2]; sig = float(args[3]) if len(args) > 3 else 600.0
    only = args[4].split(",") if len(args) > 4 else ["legacy", "fov40", "fov60"]   # run a subset so the three can run in parallel
    out, eps = {}, {}
    for name, fov in (("legacy", None), ("fov40", 40.0), ("fov60", 60.0)):
        if name not in only: continue
        for sync in (False, True):
            key = f"{name}_{'sync' if sync else 'nosync'}"
            cfg = window_cfg(init_pos_h=sig, fov_deg=fov, reg_offset=6.0, map_noise=1.0, alt_sync=sync)
            env = LunarLandingEnv([site_from_height(np.load(tile).astype("float32"), cfg, tile)], cfg, seed=0)
            rows = []
            for i in range(N):
                env.reset(seed=SEED0 + i); done = False; t_fix = None; alt_err = float("nan"); n_rej = 0
                g = PolyGuidance()
                while not done:
                    _, _, te, tr, info = env.step(g.act(env)); done = te or tr
                    k = env.last_info.get("trn", "none")
                    if t_fix is None and k == "fix": t_fix = float(env.t)
                    elif t_fix is None and k != "none": n_rej += 1
                alt_err = float(abs(env.nav.x[2] - env.r[2]))
                ok = info["outcome"] == "success"
                if (i + 1) % 10 == 0: print(key, i + 1, "of", N, flush=True)
                rows.append({"seed": SEED0 + i, "outcome": info["outcome"], "precise": bool(ok and info.get("landing_err", 1e9) <= 50.0), "t_first_fix": t_fix, "rejected_before_first_fix": n_rej, "alt_state_err_end": alt_err, "t_end": float(info["t"])})
            eps[key] = rows
            tf = [r["t_first_fix"] for r in rows if r["t_first_fix"] is not None]
            out[key] = {"precise": float(np.mean([r["precise"] for r in rows])), "no_fix_episodes": int(sum(r["t_first_fix"] is None for r in rows)),
                        "median_t_first_fix_s": float(np.median(tf)) if tf else None, "p90_t_first_fix_s": float(np.percentile(tf, 90)) if tf else None,
                        "mean_rejected_before_first_fix": float(np.mean([r["rejected_before_first_fix"] for r in rows])), "share_alt_err_over_15m": float(np.mean([r["alt_state_err_end"] > 15 for r in rows])),
                        "median_alt_err_m": float(np.median([r["alt_state_err_end"] for r in rows]))}
            print(key, out[key], flush=True)
    json.dump({"mode": mode, "sigma": sig, "summary": out, "episodes": eps}, open(f"results/{tag}.json", "w"), indent=1, default=float)
else:
    sys.exit(__doc__)
