"""Figures for the paper, built only from results/*.json and results/paper_tables.md.

    python scripts/paper_figures.py [1 2 3 ...]        (default: all)

Writes figures/paper/fig<N>_<name>.pdf (vector) and .png (200 dpi). Nothing is simulated here:
every plotted number is read from a results file, so figures stay in sync with the tables.
Run `PART=T bash scripts/run_paper.sh` first so results/paper_tables.md is current.
"""
import sys, json, re, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RES, OUT = Path("results"), Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#898781"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------- data helpers
def episodes(tag):
    return json.load(open(RES / f"{tag}.json"))["episodes"]


def prec(tag):
    """per-episode precise-landing indicator (success and landing error <= 50 m), by seed"""
    return {e["seed"]: float(e["outcome"] == "success" and e["landing_err"] <= 50.0) for e in episodes(tag)}


def rate(tag):
    v = np.array(list(prec(tag).values()))
    p = float(v.mean()) * 100
    return p, float(np.sqrt(v.mean() * (1 - v.mean()) / len(v))) * 100, len(v)


def h2_rows():
    """parse the H2 table in paper_tables.md: {row name: {column: percent}}"""
    txt = (RES / "paper_tables.md").read_text()
    sec = txt.split("### H2.")[1].split("###")[0]
    lines = [l for l in sec.splitlines() if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    out = {}
    for l in lines[2:]:
        c = [x.strip() for x in l.strip("|").split("|")]
        out[c[0]] = {h: float(v.rstrip("%")) for h, v in zip(head[2:], c[2:])}
    return out


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def label_end(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=8.5, ha=ha, va="center")


# ---------------------------------------------------------------------------- figure 1: system
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_axis_off(); ax.set_xlim(-0.5, 101.5); ax.set_ylim(0, 50)

    def box(x, y, w, h, title, sub="", fc="#eef3fb", ec=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top", fontsize=8.4, weight="bold", color=INK)
        if sub:
            ax.text(x + w / 2, y + h - 7.2, sub, ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.3)

    def arrow(p, q, color=INK2, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=1.0, linestyle=ls))

    box(1, 27, 22, 16, "Truth terrain", "LOLA DEM, 10 m cells\ndynamics, sensors,\nlanding judged here", fc="#f4f4f1", ec=MUTED)
    box(1, 3, 22, 18, "Onboard map", "20 m downsample\ntargets chosen here\nrealism layer: offset,\ncorrelated map error", fc="#f4f4f1", ec=MUTED)
    box(30, 27, 24, 16, "Sensors", "IMU, Doppler velocimeter,\nradar altimeter,\nlidar patch (field of view)")
    box(30, 3, 24, 18, "Terrain matching", "normalised cross-correlation\nlobe / ambiguity test\nchi-square gate, every 2 s")
    box(59, 13, 19, 20, "9-state EKF", "position, velocity,\naccel. bias\nestimate + covariance", fc="#e8f6f0", ec=AQUA)
    box(82, 27, 18, 16, "Guidance", "polynomial law\n(baseline)", fc="#fdf0e9", ec=ORANGE)
    box(82, 3, 18, 18, "Decision layer", "none / detour (16) /\nhover; rule, chooser\nor PPO", fc="#fdf0e9", ec=ORANGE)

    arrow((23, 35), (30, 35)); arrow((23, 12), (30, 12))
    arrow((42, 27), (42, 21.4)); ax.text(43, 24.2, "patch", fontsize=7, color=MUTED)
    arrow((54, 12), (59, 18)); arrow((54, 35), (59, 28))
    arrow((78, 28), (82, 33)); arrow((91, 21.4), (91, 26.6))
    ax.text(92.0, 24.0, "waypoint / hold", fontsize=7, color=MUTED, va="center")
    arrow((78, 18), (82, 12))
    ax.text(68.5, 9.8, "estimate only:\nguidance never\nsees the truth", fontsize=7.2, color=INK2, ha="center", va="top")
    ax.plot([91, 91, 12], [43.4, 46.6, 46.6], color=ORANGE, lw=1.0, ls=(0, (4, 2)))
    arrow((12, 46.6), (12, 43.5), color=ORANGE)
    ax.text(51, 48.0, "thrust command acts on the true lander state", fontsize=7.4, color=ORANGE, ha="center")
    save(fig, "fig1_system")


# ---------------------------------------------------------------------------- figure 2: default regime
def fig2():
    rows = [("Site 04", "P_A_site04_baseline", "P_A_site04_hold", "P_A_site04_oracle"),
            ("Site 23", "P_A_site23_baseline", "P_A_site23_hold", "P_A_site23_oracle"),
            ("Haworth (test)", "P_A_haw_baseline_test", "P_A_haw_wait_test", "P_A_haw_oracle_test")]
    series = [("Baseline guidance", BLUE, "o"), ("Hold rule", AQUA, "s"), ("Perfect navigation", MAGENTA, "D")]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, (name, *tags) in enumerate(rows):
        for j, (tag, (lab, col, mk)) in enumerate(zip(tags, series)):
            p, se, n = rate(tag); y = i + (j - 1) * 0.22
            ax.errorbar(p, y, xerr=se, fmt=mk, color=col, ms=5.5, mec="white", mew=0.8, elinewidth=1.2, capsize=0, label=lab if i == 0 else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(93, 100.5); ax.set_xlabel("Precise landings (within 50 m of target), %"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", fontsize=8, ncol=3, bbox_to_anchor=(0.45, 1.0), columnspacing=1.2, handletextpad=0.3)
    fig.text(0.0, -0.04, "Default regime (120 m initial error), 200 episodes each; bars: 1 SE.", fontsize=7.4, color=MUTED)
    save(fig, "fig2_default_regime")


# ---------------------------------------------------------------------------- figure 3: initial-error sweep
def fig3():
    variants = [("wide", "window widened to 3 sigma", BLUE, "s", -14), ("sync", "+ altitude fix", AQUA, "D", 0),
                ("sync_fov60", "+ 60 degree field of view", ORANGE, "^", 14)]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, tile in zip(axs, ("site04", "site23")):
        for v, lab, col, mk, dx in variants:
            xs, ys, es = [], [], []
            for s_ in (300, 600, 1000):
                p, se, n = rate(f"P_B_{tile}_{s_}_{v}"); xs.append(s_ + dx * (s_ / 600) ** 0.5); ys.append(p); es.append(se)
            ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=5, mec="white", mew=0.7, lw=1.2, elinewidth=1.0, label=lab, capsize=0)
        nar = [rate(f"P_B_{tile}_{s_}_narrow")[0] for s_ in (300, 600, 1000)]
        ax.text(0.5, 0.06, "20-cell search window (off scale): " + ", ".join(f"{v:.0f}" for v in nar) + " %", transform=ax.transAxes,
                ha="center", fontsize=7.6, color=INK2)
        ax.set_title("Site 04" if tile == "site04" else "Site 23", fontsize=9, loc="left", color=INK)
        ax.set_xticks([300, 600, 1000]); ax.set_xlabel("Initial horizontal error (1 sigma), m"); ax.set_ylim(84, 100.8)
    axs[0].set_ylabel("Precise landings, %")
    axs[0].legend(loc="lower left", fontsize=7.6, bbox_to_anchor=(0.0, 0.14))
    fig.text(0.0, -0.04, "100 episodes per point, realism layer on; bars: 1 SE. Points are offset sideways for legibility. The n = 400 reruns of Site 23 are in the paper's table.", fontsize=7.2, color=MUTED)
    fig.tight_layout()
    save(fig, "fig3_initial_error_sweep")


# ---------------------------------------------------------------------------- figure 4: first fix mechanism
def fig4():
    s = json.load(open(RES / "M_fix.json"))["summary"]
    cfgs = [("legacy", "20 deg-class\nlegacy patch"), ("fov40", "40 degree"), ("fov60", "60 degree")]
    cfgs[0] = ("legacy", "legacy\nfixed patch")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    x = np.arange(3)
    for ax, key, ttl, scale in ((axs[0], "precise", "Precise landings, %", 100.0),
                                (axs[1], "no_fix_episodes", "Episodes with no accepted fix\n(of 200)", 1.0),
                                (axs[2], "mean_rejected_before_first_fix", "Rejected attempts before\nthe first accepted fix", 1.0)):
        vals = [s[f"{c}_sync"][key] * scale for c, _ in cfgs]
        ax.bar(x, vals, width=0.55, color=BLUE, edgecolor="white", linewidth=1.2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}" if key != "no_fix_episodes" else f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in cfgs], fontsize=8); ax.set_title(ttl, fontsize=8.5, color=INK2, loc="left")
        ax.grid(axis="x", visible=False); ax.set_ylim(0, max(vals) * 1.18)
    fig.text(0.0, -0.04, "Site 23, 600 m initial error, realism layer, altitude fix on, 200 episodes (M_fix).", fontsize=7.4, color=MUTED)
    fig.tight_layout()
    save(fig, "fig4_first_fix")


# ---------------------------------------------------------------------------- figure 5: decision layer
def fig5():
    h2 = h2_rows()
    cols = [("none", "No detour"), ("hover", "Hover"), ("random detour", "Random detour"), ("best fixed", "Best fixed action"),
            ("matchability rule", "Matchability rule"), ("away-400 rule", "Away rule, 400 m"),
            ("away-800 rule", "Away rule, 800 m"), ("conditional logit", "Learned chooser"), ("hindsight best", "Hindsight best (bound)")]
    sets = [("Nominal", h2["P_H_fit_nominal"], "dec"), ("Realism layer", h2["P_H_fit_realistic"], "real")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, (ttl, row, tag) in zip(axs, sets):
        for i, (k, lab) in enumerate(cols):
            v = row[k]; col = MUTED if k in ("none", "hindsight best") else (ORANGE if "away" in k else BLUE)
            ax.plot([row["none"], v], [i, i], color=GRID, lw=1.0, zorder=1) if k != "none" else None
            ax.scatter(v, i, s=32, color=col, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(v + 0.9, i, f"{v:.1f}", va="center", fontsize=7.6, color=INK2)
        sto = [rate(f"P_S2_{tag}_sto_s{s}")[0] for s in range(5)]
        i = len(cols)
        ax.scatter(sto, [i] * 5, s=22, color=AQUA, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(max(sto) + 0.9, i, f"{min(sto):.0f} to {max(sto):.0f}", va="center", fontsize=7.6, color=INK2)
        ax.set_title(ttl, fontsize=9, loc="left", color=INK); ax.set_xlabel("Precise landings, % (200 test episodes)")
        ax.set_xlim(min(row["none"], min(sto)) - 4, 104); ax.grid(axis="y", visible=False)
    ys = [lab for _, lab in cols] + ["Cold-start PPO, sampled\n(5 seeds)"]
    axs[0].set_yticks(range(len(ys))); axs[0].set_yticklabels(ys, fontsize=8); axs[0].invert_yaxis()
    fig.text(0.0, -0.03, "First decision only, evaluated exactly on full-information logs; the PPO row is a sampled policy evaluated by simulation. 1 SE is 2 to 3 points.", fontsize=7.4, color=MUTED)
    fig.tight_layout()
    save(fig, "fig5_decision_layer")


# ---------------------------------------------------------------------------- figure 6: learned policies across seeds
def fig6():
    base, hold = rate("P_D_dec_baseline")[0], rate("P_D_dec_hold")[0]
    groups = [
        ("Residual PPO\nno cov., k = 0.5", [f"P_D_dec_nocov_s{s}" for s in (0, 1, 2)] + [f"P_S_dec_nocov_s{s}" for s in (3, 4)]),
        ("Residual PPO\ncov., k = 0.5", [f"P_D_dec_cov_s{s}" for s in (0, 1, 2)] + [f"P_S_dec_cov_s{s}" for s in (3, 4)]),
        ("Residual PPO\nno cov., k = 1.0", [f"P_S_dec_k1_nocov_s{s}" for s in (0, 1, 2)]),
        ("Decision PPO\ncold, greedy", [f"P_S_dec_cold_s{s}" for s in range(5)]),
        ("Decision PPO\ncold, sampled", [f"P_S2_dec_sto_s{s}" for s in range(5)]),
    ]
    away = rate("P_S2_dec_awayfirst")[0]
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    for i, (lab, tags) in enumerate(groups):
        v = [rate(t)[0] for t in tags]
        jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.07
        ax.scatter(np.full(len(v), i) + jit, v, s=28, color=BLUE if i < 3 else AQUA, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(v)] * 2, color=INK, lw=1.4, zorder=4)
    for y, lab, col in ((base, "baseline guidance", MUTED), (hold, "hold rule", ORANGE), (away, "away rule, first decision", MAGENTA)):
        ax.axhline(y, color=col, lw=1.0, ls=(0, (4, 2)), zorder=2)
        ax.text(-0.45, y + 0.5, f"{lab} {y:.1f}", color=INK2, fontsize=7.8, ha="left", va="bottom")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=7.6)
    ax.set_ylabel("Precise landings, % (200 test episodes)"); ax.set_xlim(-0.5, len(groups) - 0.4); ax.grid(axis="x", visible=False)
    ax.set_ylim(55, 98)
    fig.text(0.0, -0.05, "Synthetic benchmark, nominal conditions. One dot per training seed; black line: seed mean.", fontsize=7.4, color=MUTED)
    save(fig, "fig6_learned_policies")


# ---------------------------------------------------------------------------- figure 7: failure categories (appendix)
def fig7():
    names = [("P_C_diag_600_legacy", "600 m\nlegacy"), ("P_C_diag_600_sync", "600 m\n+ altitude fix"), ("P_C_diag_600_sync_fov60", "600 m\n+ 60 deg"),
             ("P_C_diag_1000_legacy", "1000 m\nlegacy"), ("P_C_diag_1000_sync", "1000 m\n+ altitude fix"), ("P_C_diag_1000_sync_fov60", "1000 m\n+ 60 deg")]
    cats = [("zero_fix", "No accepted fix", ORANGE), ("nav_bad", "Navigation error > 15 m at touchdown", MAGENTA),
            ("nav_ok_hazard", "Navigation good, unsafe or hard landing", YELLOW), ("other", "Other", GRAY)]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, (tag, lab) in enumerate(names):
        d = json.load(open(RES / f"{tag}.json"))
        cnt = d.get("categories") or d["summary"].get("categories")
        if cnt is None:                                   # fall back to paper_tables.md section C
            txt = (RES / "paper_tables.md").read_text().split("### C. Failure diagnosis")[1].split("###")[0]
            m = re.search(rf"\| {tag} \| (\d+) \| (\{{.*?\}}) \|", txt); cnt = eval(m.group(2))
        bottom = 0
        for k, cl, col in cats:
            v = cnt.get(k, 0)
            if v:
                ax.bar(i, v, bottom=bottom, color=col, edgecolor="white", linewidth=1.2, width=0.62, label=cl if i == 0 or False else None); bottom += v
        ax.text(i, bottom + 0.6, f"{60 - bottom:.0f} of 60 precise" if False else f"{bottom}", ha="center", fontsize=7.6, color=INK2)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([l for _, l in names], fontsize=7.8)
    ax.set_ylabel("Non-precise episodes (of 60)"); ax.grid(axis="x", visible=False)
    h, l = [], []
    for k, cl, col in cats:
        h.append(plt.Rectangle((0, 0), 1, 1, color=col)); l.append(cl)
    ax.legend(h, l, fontsize=7.6, loc="upper right", ncol=1)
    ax.set_ylim(0, 12)
    fig.text(0.0, -0.05, "Site 23, realism layer. Numbers above bars: all non-precise episodes of the 60.", fontsize=7.4, color=MUTED)
    save(fig, "fig7_failure_categories")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7}
if __name__ == "__main__":
    sel = [int(a) for a in sys.argv[1:]] or list(FIGS)
    for k in sel:
        FIGS[k]()
