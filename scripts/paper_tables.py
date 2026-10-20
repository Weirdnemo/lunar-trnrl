"""Collect results/P_*.json into markdown tables (results/paper_tables.md). Prints too.
Columns: precise landing rate with its binomial standard error, safe landing rate, outcome counts, episodes with zero TRN fixes."""
import glob, json, os, math, re, collections
rows = []
heads = []
fits = []
diag = []
for f in sorted(glob.glob("results/P_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    s = d.get("summary")
    if "conditional_logit" in d:                               # part H: offline fit result
        fits.append((os.path.basename(f)[:-5], d)); continue
    if s and "hindsight_best_any_action" in s:                 # part H: decision headroom (per-action precise rates)
        heads.append((os.path.basename(f)[:-5], s)); continue
    if not s and "categories" in d:                       # part C: failure-diagnosis json has category counts, no summary
        diag.append((os.path.basename(f)[:-5], d["categories"], len(d.get("episodes", []))))
        continue
    if not s:
        continue
    n = s["episodes"]; p = s["precise_landing_rate_<=50m"]
    rows.append((os.path.basename(f)[:-5], n, p, math.sqrt(p * (1 - p) / n), s["safe_landing_rate"], s["outcomes"], s["episodes_with_zero_fixes"], s.get("median_landing_err_m", float("nan")), s.get("p90_landing_err_m", float("nan"))))
parts = collections.defaultdict(list)
for r in rows:
    parts[r[0].split("_")[1]].append(r)
titles = {"A": "A. Default regime and ceiling", "B": "B. Initial-error sweep, realistic sensor/map (narrow / wide / +alt_sync / +fov 60 / hold)",
          "C": "C. Failure diagnosis", "D": "D. Residual PPO", "E": "E. Learned hold/detour (nominal conditions)", "R": "R. Realism sensitivity and RL robustness", "L": "L. Legacy idealised-map conditions", "H": "H. Rule restricted to the first decision", "S": "S. Extra training seeds, residual scale k=1.0, cold-start decision policy, away rules", "S2": "S2. Away rules at the first decision only; cold-start policies sampled", "X": "X. Extended Site 23 runs (400 episodes, sigma 600 / 1000)"}
out = []
for k in sorted(parts):
    out.append(f"### {titles.get(k, k)}\n\n| run | n | precise | SE | safe | median miss [m] | p90 miss [m] | outcomes | zero-fix |\n|---|---|---|---|---|---|---|---|---|")
    for tag, n, p, se, safe, oc, zf, med, p90 in parts[k]:
        out.append(f"| {tag} | {n} | {100*p:.1f}% | {100*se:.1f} | {100*safe:.1f}% | {med:.1f} | {p90:.1f} | {dict(oc)} | {zf} |")
    out.append("")
if diag:
    out.append("### C. Failure diagnosis (category counts per run)\n\n| run | n | categories |\n|---|---|---|")
    for tag, cats, n in diag:
        out.append(f"| {tag} | {n} | {dict(cats)} |")
    out.append("")
if heads:
    out.append("### H. Decision-layer headroom (first-decision action tried on every episode; precise rate)\n\n| run | N | episodes with a decision point | no detour | hover | mean single detour | best single action | hindsight best of all |\n|---|---|---|---|---|---|---|---|")
    for tag, sm in heads:
        det = [sm[f"first_{a}"] for a in range(1, 17)]
        acts = [sm[f"first_{a}"] for a in range(18)]
        out.append(f"| {tag} | {sm['N']} | {sm['episodes_with_a_decision_point']} | {100*sm['first_0']:.1f}% | {100*sm['first_17']:.1f}% | {100*sum(det)/16:.1f}% | {100*max(acts):.1f}% | {100*sm['hindsight_best_any_action']:.1f}% |")
    out.append("")
if fits:
    out.append("### H2. Learned first-decision chooser (offline fit on train logs, exact evaluation on test logs)\n\n| run | test N | none | hover | best fixed | random detour | matchability rule | away-800 rule | away-400 rule | conditional logit | hindsight best |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, d in fits:
        out.append(f"| {tag} | {d['N_test']} | {100*d['none']:.1f}% | {100*d['hover']:.1f}% | {100*d['best_fixed_on_train']:.1f}% | {100*d['random_detour']:.1f}% | {100*d['rule_argmax_matchability']:.1f}% | {100*d.get('away_800_rule', float('nan')):.1f}% | {100*d.get('away_400_rule', float('nan')):.1f}% | {100*d['conditional_logit']:.1f}% | {100*d['hindsight_best']:.1f}% |")
    out.append("")
if os.path.exists("results/paired_ci.md"):
    out.append("### Paired comparisons (same seeds; 95% paired-bootstrap CI; run scripts/paired_ci.py first)\n\n" + open("results/paired_ci.md").read() + "\n")
txt = "\n".join(out) if out else "no results/P_*.json files found"
open("results/paper_tables.md", "w").write(txt)
print(txt)
