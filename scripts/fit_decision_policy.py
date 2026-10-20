"""Offline learning of the first-decision action from full-information logs (decision_headroom.py output).
Train on --train json, evaluate EXACTLY on --test json (every action's outcome was observed on every test episode).
    python scripts/fit_decision_policy.py results/P_H_train_nominal.json results/P_H_test_nominal.json tag
Policies compared (precise rate over ALL test episodes; episodes without a decision point are identical for all actions):
  none, hover, best fixed action chosen on train, random detour, rule (argmax matchability),
  conditional-logit (shared weights over actions: matchability, alignment with target, radius, sigma, per-action intercepts),
  hindsight best (upper bound)."""
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

trj, tej, tag = sys.argv[1], sys.argv[2], sys.argv[3]
TR, TE = json.load(open(trj)), json.load(open(tej))
NA = 18

def load(d):
    ep = d["episodes"]; n = len(ep["first_0"])
    P = np.array([[ep[f"first_{a}"][i]["precise"] for a in range(NA)] for i in range(n)], float)
    F = [ep["first_0"][i]["features"] for i in range(n)]
    has = np.array([f is not None for f in F])
    return P, F, has

def act_feats(f):
    """rows = 18 actions; features shared across actions + one-hot intercepts."""
    f = np.asarray(f); sig, agl = f[0], f[2]
    rows = []
    for a in range(NA):
        det = 1 <= a <= 16
        m = f[7 + a - 1] if det else 0.0
        al = f[23 + ((a - 1) % 8)] if det else 0.0
        far = 1.0 if a >= 9 and det else 0.0
        hov = 1.0 if a == 17 else 0.0
        rows.append([m, al, m * sig, al * sig, far, far * sig, hov * sig, hov * agl, det * agl, det * sig])
    X = np.array(rows); onehot = np.eye(NA)
    return np.hstack([X, onehot])

Ptr, Ftr, htr = load(TR); Pte, Fte, hte = load(TE)
def design(P, F, has):
    X, y, g = [], [], []
    for i in np.where(has)[0]:
        A = act_feats(F[i]); X.append(A); y.append(P[i]); g.append(i)
    return np.vstack(X), np.concatenate(y), np.array(g)
Xtr, ytr, gtr = design(Ptr, Ftr, htr)
best = None
for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
    clf = LogisticRegression(C=C, max_iter=2000)
    groups = np.repeat(np.arange(len(gtr)), NA)
    folds = groups % 5; sc = []
    for k in range(5):
        tr, va = folds != k, folds == k
        clf.fit(Xtr[tr], ytr[tr])
        p = clf.predict_proba(Xtr[va])[:, 1].reshape(-1, NA)
        yv = ytr[va].reshape(-1, NA)
        sc.append(yv[np.arange(len(p)), p.argmax(1)].mean())
    m = float(np.mean(sc)); print(f"C={C}: cross-validated precise on decision episodes {m:.3f}")
    if best is None or m > best[0]: best = (m, C)
clf = LogisticRegression(C=best[1], max_iter=2000).fit(Xtr, ytr)

def rate(choose):
    out = []
    for i in range(len(Pte)):
        a = choose(i) if hte[i] else 0
        out.append(Pte[i, a])
    return float(np.mean(out))
fixed_train = int(np.argmax(Ptr[htr].mean(0)))
rng = np.random.default_rng(0)
res = {
  "N_test": len(Pte), "N_test_with_decision": int(hte.sum()), "N_train_with_decision": int(htr.sum()),
  "none": rate(lambda i: 0), "hover": rate(lambda i: 17),
  "best_fixed_on_train": rate(lambda i: fixed_train), "best_fixed_action": fixed_train,
  "random_detour": float(np.mean([rate(lambda i: int(rng.integers(1, 17))) for _ in range(50)])),
  "rule_argmax_matchability": rate(lambda i: (int(np.argmax(np.array(Fte[i])[7:23])) + 1) if np.max(np.array(Fte[i])[7:23]) > 0.05 else 0),
  "conditional_logit": rate(lambda i: int(clf.predict_proba(act_feats(Fte[i]))[:, 1].argmax())), "chosen_C": best[1],
  "away_800_rule": rate(lambda i: 9 + int(np.argmin(np.array(Fte[i])[23:31]))),
  "away_400_rule": rate(lambda i: 1 + int(np.argmin(np.array(Fte[i])[23:31]))),
  "hindsight_best": float(np.mean(np.where(hte, Pte.max(1), Pte[:, 0]))),
}
res["per_action_test_precise_on_decision_episodes"] = [float(x) for x in Pte[hte].mean(0)]
# how often does the learned policy pick each action
ch = [int(clf.predict_proba(act_feats(Fte[i]))[:, 1].argmax()) for i in np.where(hte)[0]]
res["conditional_logit_action_counts"] = {str(a): ch.count(a) for a in sorted(set(ch))}
json.dump(res, open(f"results/{tag}.json", "w"), indent=1)
for k, v in res.items():
    if not isinstance(v, (list, dict)): print(k, round(v, 3) if isinstance(v, float) else v)
print("action counts", res["conditional_logit_action_counts"])
