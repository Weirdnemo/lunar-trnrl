"""Paired comparisons between result jsons (same seeds => same episodes). Prints markdown, writes results/paired_ci.md.
precise = outcome success and landing error <= 50 m (matches the summaries). 95% CI: paired bootstrap over episodes (B=5000),
plus discordant counts (b = only A precise, c = only B precise) and an exact two-sided sign-test p-value on the discordant pairs.
Rows whose files are missing are skipped."""
import json, os, math
import numpy as np
R = "results/"
def prec(tag):
    f = f"{R}{tag}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); ep = d.get("episodes")
    if not isinstance(ep, list) or not ep or "landing_err" not in ep[0]: return None
    return {e["seed"]: float(e["outcome"] == "success" and e["landing_err"] <= 50) for e in ep}
def sign_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
rng = np.random.default_rng(0)
def cmp(label, a, b):
    A, B = prec(a), prec(b)
    if A is None or B is None: return None
    seeds = sorted(set(A) & set(B))
    if not seeds: return None
    x = np.array([A[s] for s in seeds]); y = np.array([B[s] for s in seeds]); d = x - y; n = len(d)
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
    bb, cc = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return (label, n, 100 * x.mean(), 100 * y.mean(), 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), bb, cc, sign_p(bb, cc))
C = []
# A: default regime (alt_sync = corrected window sizing, "_fixed")
for t in ("site04", "site23"):
    C += [(f"A {t}: baseline vs oracle", f"P_A_{t}_baseline_fixed", f"P_A_{t}_oracle"),
          (f"A {t}: alt_sync vs legacy windows (baseline)", f"P_A_{t}_baseline_fixed", f"P_A_{t}_baseline"),
          (f"A {t}: hold vs baseline (alt_sync)", f"P_A_{t}_hold_fixed", f"P_A_{t}_baseline_fixed")]
for sp in ("test", "train"):
    C += [(f"A haworth {sp}: baseline vs oracle", f"P_A_haw_baseline_{sp}_fixed", f"P_A_haw_oracle_{sp}"),
          (f"A haworth {sp}: hold(wait70) vs baseline (alt_sync)", f"P_A_haw_wait_{sp}_fixed", f"P_A_haw_baseline_{sp}_fixed"),
          (f"A haworth {sp}: alt_sync vs legacy (baseline)", f"P_A_haw_baseline_{sp}_fixed", f"P_A_haw_baseline_{sp}")]
# B: initial-error sweep
for t in ("site04", "site23"):
    for s in (300, 600, 1000):
        C += [(f"B {t} sigma{s}: wide vs narrow window", f"P_B_{t}_{s}_wide", f"P_B_{t}_{s}_narrow"),
              (f"B {t} sigma{s}: alt_sync vs wide", f"P_B_{t}_{s}_sync", f"P_B_{t}_{s}_wide"),
              (f"B {t} sigma{s}: sync+fov60 vs sync", f"P_B_{t}_{s}_sync_fov60", f"P_B_{t}_{s}_sync"),
              (f"B {t} sigma{s}: hold vs baseline (sync)", f"P_B_{t}_{s}_sync_hold", f"P_B_{t}_{s}_sync")]
# D: residual PPO vs baseline
for v in ("nocov", "cov"):
    for sd in (0, 1, 2):
        C.append((f"D synthetic: PPO {v} s{sd} vs baseline", f"P_D_dec_{v}_s{sd}", "P_D_dec_baseline"))
C.append(("D synthetic: hold vs baseline", "P_D_dec_hold", "P_D_dec_baseline"))
for sd in (1, 2):
    for k in ("best", "final"):
        C.append((f"D haworth test: PPO nocov s{sd} {k} vs baseline", f"P_D_haw_nocov_s{sd}_{k}", "P_A_haw_baseline_test"))
# E, R: learned hold/detour
for sd in (0, 1, 2):
    C += [(f"E: learned s{sd} vs baseline", f"P_E_learned_s{sd}", "P_E_baseline"),
          (f"E: learned s{sd} vs hold", f"P_E_learned_s{sd}", "P_E_hold"),
          (f"R decision: learned s{sd} vs baseline", f"P_R_dec_learned_s{sd}", "P_R_dec_baseline"),
          (f"R decision: learned s{sd} vs hold", f"P_R_dec_learned_s{sd}", "P_R_dec_hold")]
C += [("E: hold vs baseline", "P_E_hold", "P_E_baseline"), ("E: rule vs baseline", "P_E_rule", "P_E_baseline"),
      ("E: baseline vs oracle", "P_E_baseline", "P_E_oracle"),
      ("R decision: hold vs baseline", "P_R_dec_hold", "P_R_dec_baseline"), ("R decision: rule vs baseline", "P_R_dec_rule", "P_R_dec_baseline")]
# R: realism sensitivity, Site 23, vs the realism preset (fov 40, offset 6, noise 1) at the same sigma (same seeds)
REF = {120: "P_A_site23_baseline_fixed", 600: "P_B_site23_600_sync"}
for knob in ("fov20", "fov30", "fov60", "off0", "off3", "off12", "noise0", "noise2", "noise3"):
    for s in (120, 600):
        C.append((f"R site23 sigma{s}: {knob} vs preset", f"P_R_{knob}_s{s}", REF[s]))
for s in (120, 600):
    C.append((f"R site23 sigma{s}: fov60 vs fov20", f"P_R_fov60_s{s}", f"P_R_fov20_s{s}"))
# X: extended Site 23 runs (400 episodes), sigma 600 and 1000
for s in (600, 1000):
    C += [(f"X site23 sigma{s}: alt_sync vs wide", f"P_X_site23_{s}_sync", f"P_X_site23_{s}_wide"),
          (f"X site23 sigma{s}: sync+fov60 vs sync", f"P_X_site23_{s}_sync_fov60", f"P_X_site23_{s}_sync"),
          (f"X site23 sigma{s}: hold vs baseline (sync)", f"P_X_site23_{s}_sync_hold", f"P_X_site23_{s}_sync")]
C += [("X site23 sigma600: fov20 vs sync (fov40)", "P_X_site23_600_sync_fov20", "P_X_site23_600_sync"),
      ("X site23 sigma600: fov30 vs sync (fov40)", "P_X_site23_600_sync_fov30", "P_X_site23_600_sync"),
      ("X site23 sigma600: fov60 vs fov20", "P_X_site23_600_sync_fov60", "P_X_site23_600_sync_fov20"),
      ("X site23 sigma1000 minus sigma600 (sync, different seeds' difficulty)", "P_X_site23_1000_sync", "P_X_site23_600_sync")]
# S: extra seeds and ablations (synthetic benchmark, n=200)
for v in ("nocov", "cov"):
    for sd in (3, 4):
        C.append((f"S: PPO {v} s{sd} (k=0.5) vs baseline", f"P_S_dec_{v}_s{sd}", "P_D_dec_baseline"))
for sd in (0, 1, 2):
    C.append((f"S: PPO nocov k=1.0 s{sd} vs baseline", f"P_S_dec_k1_nocov_s{sd}", "P_D_dec_baseline"))
for sd in range(5):
    C += [(f"S: cold-start decision PPO s{sd} vs baseline (nominal)", f"P_S_dec_cold_s{sd}", "P_E_baseline"),
          (f"S: cold-start decision PPO s{sd} vs hold (nominal)", f"P_S_dec_cold_s{sd}", "P_E_hold"),
          (f"S: cold-start decision PPO s{sd} vs away-800 rule (nominal)", f"P_S_dec_cold_s{sd}", "P_S_dec_away"),
          (f"S: cold-start decision PPO s{sd} vs baseline (realism)", f"P_S_real_cold_s{sd}", "P_R_dec_baseline"),
          (f"S: cold-start decision PPO s{sd} vs away-800 rule (realism)", f"P_S_real_cold_s{sd}", "P_S_real_away")]
for r, nm in (("away", "800 m"), ("away400", "400 m")):
    C += [(f"S: away rule {nm} AT EVERY DECISION vs baseline (nominal)", f"P_S_dec_{r}", "P_E_baseline"), (f"S: away rule {nm} AT EVERY DECISION vs hold (nominal)", f"P_S_dec_{r}", "P_E_hold"),
          (f"S: away rule {nm} AT EVERY DECISION vs baseline (realism)", f"P_S_real_{r}", "P_R_dec_baseline"), (f"S: away rule {nm} AT EVERY DECISION vs hold (realism)", f"P_S_real_{r}", "P_R_dec_hold")]
# S2: first-decision-only away rules; cold-start policies sampled
for r, nm in (("awayfirst", "800 m"), ("away400first", "400 m")):
    C += [(f"S2: away {nm} first decision only vs baseline (nominal)", f"P_S2_dec_{r}", "P_E_baseline"), (f"S2: away {nm} first decision only vs hold (nominal)", f"P_S2_dec_{r}", "P_E_hold"),
          (f"S2: away {nm} first decision only vs baseline (realism)", f"P_S2_real_{r}", "P_R_dec_baseline"), (f"S2: away {nm} first decision only vs hold (realism)", f"P_S2_real_{r}", "P_R_dec_hold")]
for sd in range(5):
    C += [(f"S2: cold-start PPO s{sd} sampled vs baseline (nominal)", f"P_S2_dec_sto_s{sd}", "P_E_baseline"),
          (f"S2: cold-start PPO s{sd} sampled vs hold (nominal)", f"P_S2_dec_sto_s{sd}", "P_E_hold"),
          (f"S2: cold-start PPO s{sd} sampled vs first-only away 800 (nominal)", f"P_S2_dec_sto_s{sd}", "P_S2_dec_awayfirst"),
          (f"S2: cold-start PPO s{sd} sampled vs baseline (realism)", f"P_S2_real_sto_s{sd}", "P_R_dec_baseline"),
          (f"S2: cold-start PPO s{sd} sampled vs first-only away 800 (realism)", f"P_S2_real_sto_s{sd}", "P_S2_real_awayfirst")]
rows = [cmp(*c) for c in C if c[2] is not None]
rows = [r for r in rows if r]
out = ["| comparison (A minus B) | n | A | B | diff [pts] | 95% CI | only A / only B | sign-test p |", "|---|---|---|---|---|---|---|---|"]
for l, n, a, b, d, lo, hi, bb, cc, p in rows:
    out.append(f"| {l} | {n} | {a:.1f}% | {b:.1f}% | {d:+.1f} | {lo:+.1f} to {hi:+.1f} | {bb} / {cc} | {p:.3f} |")
txt = "\n".join(out); open(R + "paired_ci.md", "w").write(txt); print(txt)
