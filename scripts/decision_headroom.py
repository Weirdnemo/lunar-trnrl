"""How much could ANY decision layer gain? For every held-out episode, try each fixed first-decision action
(0 none, 1..8 detour 400 m, 9..16 detour 800 m, 17 hover) followed by no further detours, plus 'hover at every decision',
then take the best outcome in hindsight per episode (an upper bound for a policy that picks one action per episode).
    python scripts/decision_headroom.py <scenario|tile.npy> N tag [--realistic ...nav flags]
Writes results/<tag>.json with per-episode outcomes for every action (also usable for paired CIs)."""
import sys, json, collections
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from trnrl.env import LunarLandingEnv
from trnrl.windows import window_cfg, site_from_height, pop_nav_flags, apply_nav_flags
import trnrl.benchmarks as _bm
from trnrl.benchmarks import build
from trnrl.detour import DetourGuidance2, N_ACT

NAV = pop_nav_flags(sys.argv)
SPLIT = "train" if "--train" in sys.argv else "test"
if "--train" in sys.argv: sys.argv.remove("--train")
SEED0 = 50_000 if SPLIT == "train" else 10_000
tile, N, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if tile.endswith(".npy"):
    cfg = window_cfg(**NAV); sites = [site_from_height(np.load(tile).astype("float32"), cfg, tile)]
else:
    _bm.CFG_TWEAK = lambda c: apply_nav_flags(c, NAV); cfg, sites = build(tile, SPLIT)
env = LunarLandingEnv(sites, cfg, seed=0)

def run(seed, first, every=False):
    calls = [0]; feat = [None]
    def chooser(f):
        calls[0] += 1
        if calls[0] == 1: feat[0] = [float(x) for x in f]
        return first if (every or calls[0] == 1) else 0
    g = DetourGuidance2(chooser=chooser)
    env.reset(seed=seed); done = False
    while not done:
        _, _, te, tr, info = env.step(g.act(env)); done = te or tr
    ok = info["outcome"] == "success"
    return {"outcome": info["outcome"], "precise": bool(ok and info.get("landing_err", 1e9) <= 50.0), "decisions": calls[0], "features": feat[0], "landing_err": float(info.get("landing_err", float("nan")))}

names = [f"first_{a}" for a in range(N_ACT)] + ["hover_every"]
res = {n: [] for n in names}
for i in range(N):
    seed = SEED0 + i
    for a in range(N_ACT): res[f"first_{a}"].append(run(seed, a))
    res["hover_every"].append(run(seed, N_ACT - 1, every=True))
    if (i + 1) % 10 == 0: print(i + 1, flush=True)
P = {n: np.array([r["precise"] for r in v]) for n, v in res.items()}
decided = np.array([res["first_0"][i]["decisions"] > 0 for i in range(N)])        # episodes where a decision point occurred
best_any = np.any([P[n] for n in names], axis=0)
best_two = P["first_0"] | P["first_17"]
summary = {n: float(P[n].mean()) for n in names}
summary.update({"hindsight_best_any_action": float(best_any.mean()), "hindsight_best_of_none_or_hover": float(best_two.mean()),
                "episodes_with_a_decision_point": int(decided.sum()), "N": N})
json.dump({"tile": tile, "split": SPLIT, "nav": NAV, "summary": summary, "episodes": res}, open(f"results/{tag}.json", "w"), indent=1, default=float)
for k, v in summary.items(): print(k, round(v, 3) if isinstance(v, float) else v)
