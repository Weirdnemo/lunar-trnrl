"""Render the shareable figure set from real simulation output.

    python scripts/make_figures.py [fig ...]      (default: all)

Figures go to figures/*.png. Every number comes from results/*.json or from re-running
the simulator with the same seeds. Terrain is SYNTHETIC (stated on every figure).
"""
import sys, json, collections
sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from scipy import ndimage

from trnrl.benchmarks import build
from trnrl.env import LunarLandingEnv
from trnrl.guidance import PolyGuidance, WaitGuidance

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
RES = Path("results")

# ------------------------------------------------------------------ style tokens
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREEN_TXT, GRAY = "#006300", "#b9b8b0"
BLUE_STEPS = {250: "#86b6ef", 400: "#3987e5", 600: "#184f95"}
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
NOTE = "Simulated lunar-like terrain (synthetic), not real NASA data"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False,
})


def title(fig, main, sub=None, y=0.985):
    fig.text(0.035, y, main, fontsize=17, fontweight="bold", ha="left", va="top", color=INK)
    if sub:
        fig.text(0.035, y - 0.052, sub, fontsize=11.5, ha="left", va="top", color=INK2, linespacing=1.35)


def footer(fig, extra=""):
    fig.text(0.035, 0.012, (extra + "   " if extra else "") + NOTE, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("wrote", p)


def J(tag):
    return json.load(open(RES / f"{tag}.json"))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def precise_k(rows):
    return sum(1 for r in rows if r["outcome"] == "success" and r["landing_err"] <= 50.0)


def hill(terr, exag=5.0, az=315, alt=35):
    return LightSource(azdeg=az, altdeg=alt).hillshade(terr.h, vert_exag=exag, dx=terr.dx, dy=terr.dx)


def run_episode(sites, cfg, seed, guidance):
    env = LunarLandingEnv(sites, cfg, seed=0)
    env.reset(seed=seed)
    outage = (env.nav.cfg.outage_start, env.nav.cfg.outage_end)
    hold = []
    done = False
    while not done:
        before = getattr(guidance, "waited", 0.0)
        a = guidance.act(env)
        hold.append(getattr(guidance, "waited", 0.0) > before)
        _, _, te, tr, info = env.step(a)
        done = te or tr
    return env, info, outage, np.array(hold)


# =============================================================== fig: terrain 3D
def fig_terrain_3d():
    cfg, sites = build("mixed_hard+dec", "test")
    site = sites[3]
    env, info, _, _ = run_episode(sites, cfg, 10001, PolyGuidance())
    T = site.truth
    k = 4
    h = T.h[::k, ::k] - T.h.min()
    ny, nx = h.shape
    xs = np.arange(nx) * T.dx * k / 1000.0
    ys = np.arange(ny) * T.dx * k / 1000.0
    X, Y = np.meshgrid(xs, ys)
    EX = 4.0
    tex = ndimage.zoom(site.map_tex, 2, order=1)[::k, ::k]
    tex = tex[:ny, :nx]
    norm = np.clip(tex / 12.0, 0, 1)
    rgb = SEQ(norm)[..., :3]
    ls = LightSource(azdeg=300, altdeg=40)
    shaded = ls.shade_rgb(rgb, h * EX, vert_exag=1.0, blend_mode="soft", dx=T.dx * k, dy=T.dx * k)
    fig = plt.figure(figsize=(12.5, 7.3))
    ax = fig.add_axes([-0.03, -0.06, 0.82, 0.96], projection="3d", computed_zorder=False)
    ax.plot_surface(X, Y, h * EX / 1000.0, facecolors=shaded, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False, zorder=1)
    H = env.history
    r = np.array(H["r"]); rh = np.array(H["r_hat"])
    gz = np.array([T.height(p[0], p[1]) - T.h.min() for p in r])
    z = (gz * EX + (r[:, 2] - np.array([T.height(p[0], p[1]) for p in r]))) / 1000.0
    ax.plot(r[:, 0] / 1000, r[:, 1] / 1000, z, color=INK, lw=2.6, zorder=10, label="Actual path")
    ax.scatter(*(r[0, 0] / 1000, r[0, 1] / 1000, z[0]), s=70, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=11, depthshade=False)
    tg = env.target
    ax.scatter(tg[0] / 1000, tg[1] / 1000, (tg[2] - T.h.min()) * EX / 1000.0, marker="*", s=420, color="#ffd23f",
               edgecolor=INK, linewidth=1.0, zorder=12, depthshade=False)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(0, ys[-1]); ax.set_zlim(0, 1.45)
    ax.set_box_aspect((1, 1, 0.42))
    ax.view_init(elev=27, azim=-58)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.set_facecolor(SURF); a_.pane.set_edgecolor(GRID); a_._axinfo["grid"]["color"] = GRID
    ax.set_xlabel("km east", labelpad=6, color=INK2); ax.set_ylabel("km north", labelpad=6, color=INK2)
    ax.set_zlabel("altitude (km)", labelpad=4, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=9)
    # legend / colorbar / callouts
    start_alt = float(r[0, 2] - T.height(r[0, 0], r[0, 1]))
    fig.text(0.80, 0.76, "Ground colour", fontsize=11, fontweight="bold", color=INK, ha="left")
    fig.text(0.80, 0.735, "Darker = more bumps and\ncraters, so the ground is\neasier to recognise.", fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.35)
    cax = fig.add_axes([0.80, 0.50, 0.016, 0.16])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 12), cmap=SEQ), cax=cax)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)
    cb.set_label("Height change (m)", color=INK2, fontsize=9.5, labelpad=10)
    fig.text(0.80, 0.40, "Marks", fontsize=11, fontweight="bold", color=INK, ha="left")
    fig.text(0.80, 0.375, "●  Where the descent starts\n━  Actual flight path\n★  Designated landing site", fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.6)
    fig.text(0.80, 0.24, f"This flight landed {info['landing_err']:.1f} m\nfrom the target after\nabout {info['t']:.0f} seconds.", fontsize=10.5, color=INK, ha="left", va="top", linespacing=1.4)
    title(fig, "A simulated Moon landing in 3D",
          f"The lander starts about {start_alt/1000:.1f} km up and does not know exactly where it is. It looks at the ground below,\nrecognises the terrain, and steers to the star. Terrain heights are exaggerated 4x so the shape is visible.")
    footer(fig, "Site 3 of 8 test sites, flight seed 10001.")
    save(fig, "01_terrain_3d")


# ===================================================== fig: where navigation works
def fig_navigability():
    cfg, sites = build("mixed_hard+dec", "test")
    rows = [(sites[0], "Rough, cratered terrain", "Easy to navigate"),
            (sites[4], "Flat plains", "Hard to navigate")]
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 8.6), gridspec_kw={"wspace": 0.08, "hspace": 0.16, "left": 0.10, "right": 0.915, "top": 0.78, "bottom": 0.10})
    VMAX = 12.0
    cols = ["Shape of the ground", "Where it is safe to land", "How recognisable the ground is"]
    meds = []
    for i, (s, name, verdict) in enumerate(rows):
        T = s.truth
        ext = [0, (T.shape[1] - 1) * T.dx / 1000, 0, (T.shape[0] - 1) * T.dx / 1000]
        hs = hill(T, exag=6.0)
        a0, a1, a2 = axs[i]
        a0.imshow(hs, cmap="gray", origin="lower", extent=ext, vmin=0.15, vmax=1.0)
        a1.imshow(hs, cmap="gray", origin="lower", extent=ext, vmin=0.15, vmax=1.0)
        safe = np.ma.masked_where(~s.truth_ok, np.ones_like(hs))
        a1.imshow(safe, cmap=matplotlib.colors.ListedColormap([AQUA]), origin="lower", extent=ext, alpha=0.55, vmin=0, vmax=1)
        disp = ndimage.gaussian_filter(s.map_tex, 2.0)   # smoothed for display only
        im = a2.imshow(disp, cmap=SEQ, origin="lower", extent=ext, vmin=0, vmax=VMAX)
        meds.append(float(np.median(s.map_tex)))
        for a in (a0, a1, a2):
            a.set_xticks([0, 1, 2, 3, 4, 5]); a.set_yticks([0, 1, 2, 3, 4, 5])
            a.tick_params(length=2, labelsize=9)
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color(AXIS)
            if a is not a0:
                a.set_yticklabels([])
            if i == 0:
                a.set_xticklabels([])
        if i == 1:
            for a in (a0, a1, a2): a.set_xlabel("km east", fontsize=10)
        axs[i][0].text(-0.20, 0.5, name, transform=axs[i][0].transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold", color=INK)
        axs[i][0].text(-0.13, 0.5, verdict, transform=axs[i][0].transAxes, rotation=90, va="center", ha="center", fontsize=10.5, color=INK2)
        a1.text(0.02, 0.02, f"{100*s.truth_ok.mean():.0f}% of ground is safe", transform=a1.transAxes, fontsize=10, color="white", fontweight="bold",
                bbox=dict(facecolor="#0b0b0b", alpha=0.65, pad=3, edgecolor="none"))
    for j, c in enumerate(cols):
        axs[0][j].set_title(c, fontsize=12, fontweight="bold", loc="left", color=INK, pad=8)
    cax = fig.add_axes([0.93, 0.18, 0.016, 0.50])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, VMAX), cmap=SEQ), cax=cax)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)
    cb.set_label("Ground distinctiveness (m of height change)", color=INK2, fontsize=10)
    title(fig, "Why flat plains are hard: the lander has nothing to recognise",
          f"Terrain-based navigation works by recognising bumps, craters and slopes. On rough ground ({meds[0]:.0f} m typical height change in a 300 m window)\nthere is a lot to match against. On plains ({meds[1]:.1f} m) there is little, so position fixes often fail.\nBoth sites have plenty of safe ground, so the difference is navigation, not safety.")
    footer(fig, "Each panel is the same 5 km x 5 km site; axes in km. Right column smoothed for display. Test sites 0 and 4.")
    save(fig, "02_where_navigation_works")
    print("median distinctiveness rough/plains:", meds, "safe fractions:", [round(float(s.truth_ok.mean()), 3) for s, _, _ in rows])



# =================================================================== fig: how it works
def trn_snapshot(env):
    nav = env.nav; c = nav.cfg; m = nav.map
    n = c.patch_n
    r = env.r
    agl = r[2] - env.site.truth.height(r[0], r[1])
    sl = c.lidar_sigma0 + c.lidar_sigma_frac * agl
    rng = np.random.default_rng(5)
    meas = env.site.truth.height(r[0] + nav._ox, r[1] + nav._oy) + rng.normal(0, sl, n * n)
    xe, ye = nav.x[0], nav.x[1]
    sh = float(np.sqrt(max(nav.P[0, 0], nav.P[1, 1])))
    S = int(np.clip(np.ceil(3 * sh / m.dx) + 1, 2, c.search_max_cells))
    ab = np.arange(-S, S + 1); A, B = np.meshgrid(ab, ab)
    cx = xe + A.ravel() * m.dx; cy = ye + B.ravel() * m.dx
    px = (cx[:, None] + nav._ox[None, :]) / m.dx; py = (cy[:, None] + nav._oy[None, :]) / m.dx
    ref = ndimage.map_coordinates(m.h, [py, px], order=1, mode="nearest")
    refc = ref - ref.mean(1, keepdims=True); mc = meas - meas.mean()
    ncc = (refc @ mc) / (np.linalg.norm(refc, axis=1) * np.linalg.norm(mc) + 1e-9)
    grid = ncc.reshape(A.shape)
    k = int(np.argmax(ncc)); bj, bi = divmod(k, A.shape[1])
    return dict(meas=meas.reshape(n, n), grid=grid, S=S, dx=m.dx, est=(xe, ye), sigma=sh,
                match=(xe + A[bj, bi] * m.dx, ye + B[bj, bi] * m.dx), true=(r[0], r[1]), n=n, best=float(ncc[k]))


def fig_how_it_works():
    cfg, sites = build("mixed_hard+dec", "test")
    env0, info0, _, _ = run_episode(sites, cfg, 10001, PolyGuidance())
    H = env0.history
    env = LunarLandingEnv(sites, cfg, seed=0); env.reset(seed=10001)
    sn = trn_snapshot(env)
    m = env.site.map
    W = sn["S"] + sn["n"] // 2 + 4
    ie, je = int(round(sn["est"][0] / m.dx)), int(round(sn["est"][1] / m.dx))
    hsm = hill(m, exag=6.0)
    crop = hsm[je - W:je + W + 1, ie - W:ie + W + 1]
    half = W * m.dx
    first_fix = next(i for i, t in enumerate(H["trn"]) if t == "fix")
    sig_after = H["sigma_h"][first_fix]
    fig = plt.figure(figsize=(14.0, 5.6))
    w, y0 = 0.145, 0.37
    hh = w * 14.0 / 5.6
    xs = [0.04 + i * 0.1900 for i in range(5)]
    axs = [fig.add_axes([x, y0, w, hh]) for x in xs]
    for ax in axs:
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(AXIS)
        ax.set_xticks([]); ax.set_yticks([])
    # 1 stored map
    a = axs[0]
    a.imshow(crop, cmap="gray", origin="lower", extent=[-half, half, -half, half], vmin=0.15, vmax=1.0)
    sq = (sn["S"] + 0.5) * m.dx
    a.add_patch(Rectangle((-sq, -sq), 2 * sq, 2 * sq, fill=False, ec=BLUE, lw=2.0, ls="--"))
    a.plot([0], [0], marker="+", color=BLUE, ms=12, mew=2.2)
    a.text(0, -half * 0.93, "dashed box = where the lander\nthinks it could be", ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold",
           bbox=dict(facecolor="#0b0b0b", alpha=0.6, pad=2.5, edgecolor="none"))
    # 2 laser scan
    a = axs[1]
    pad = sn["n"] * m.dx / 2
    a.imshow(sn["meas"], cmap="gray", origin="lower", extent=[-pad, pad, -pad, pad], interpolation="nearest")
    # 3 match score
    a = axs[2]
    S = sn["S"]
    a.imshow(sn["grid"], cmap=SEQ.reversed(), origin="lower", extent=[-(S + .5) * m.dx, (S + .5) * m.dx] * 2, vmin=min(0, sn["grid"].min()), vmax=1.0)
    bx, by = sn["match"][0] - sn["est"][0], sn["match"][1] - sn["est"][1]
    a.plot([bx], [by], marker="o", mfc="none", mec="#ffd23f", mew=2.4, ms=15)
    a.text(0, -(S + .5) * m.dx * 0.93, "bright spot = best fit", ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold",
           bbox=dict(facecolor="#0b0b0b", alpha=0.6, pad=2.5, edgecolor="none"))
    # 4 position fix (uncertainty circles)
    a = axs[3]
    R0, R1 = sn["sigma"], sig_after
    lim = R0 * 1.25
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
    a.add_patch(Circle((0, 0), R0, fc="#f9d2c3", ec=ORANGE, lw=2))
    a.add_patch(Circle((0, 0), R1, fc="#cde2fb", ec=BLUE, lw=2))
    a.text(0, R0 * 0.62, f"before: ±{R0:.0f} m", ha="center", fontsize=10, color=INK, fontweight="bold")
    a.text(0, -R0 * 0.50 - 4, f"after the first\nfix: ±{R1:.0f} m", ha="center", va="top", fontsize=10, color=INK, fontweight="bold")
    a.plot([0], [0], marker="+", color=INK, ms=8, mew=1.6)
    # 5 steer & land
    a = axs[4]
    r = np.array(H["r"]); T = env0.site.truth
    dist = np.hypot(r[:, 0] - env0.target[0], r[:, 1] - env0.target[1]) / 1000.0
    agl = np.array([p[2] - T.height(p[0], p[1]) for p in r])
    a.plot(dist, agl, color=INK, lw=2.4)
    a.plot([0], [0], marker="*", ms=16, mfc="#ffd23f", mec=INK, mew=1.0, clip_on=False, zorder=5)
    a.set_xlim(max(dist) * 1.05, -0.05 * max(dist)); a.set_ylim(-60, max(agl) * 1.08)
    a.yaxis.tick_left()
    a.spines["left"].set_visible(True)
    a.set_xlabel("distance to target (km)", fontsize=9, labelpad=3)
    a.text(0.97, 0.97, "height above\nground (m)", transform=a.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    a.set_xticks([0, 0.5, 1.0, 1.5]); a.set_yticks([0, 500, 1000]); a.tick_params(labelsize=8, length=2)
    steps = [("1  A map is stored on board", "Before the flight, the lander is\ngiven a map of the landing area."),
             ("2  A laser scans the ground", "During descent it measures the\nheight of the ground just below."),
             ("3  Find the best match", "It slides the scan around the map\nand scores how well each spot fits."),
             ("4  That gives a position fix", "The match tells the lander where it\nreally is, so its uncertainty shrinks."),
             ("5  Steer to the target", "With a better idea of where it is,\nguidance steers to the landing site.")]
    for x, (t, d) in zip(xs, steps):
        fig.text(x, 0.275, t, fontsize=11.5, fontweight="bold", color=INK, ha="left", va="top")
        fig.text(x, 0.215, d, fontsize=10, color=INK2, ha="left", va="top", linespacing=1.4)
    for i in range(4):
        fig.patches.append(matplotlib.patches.FancyArrowPatch((xs[i] + w + 0.008, y0 + hh / 2), (xs[i + 1] - 0.010, y0 + hh / 2), transform=fig.transFigure,
                           arrowstyle="-|>", mutation_scale=18, lw=2, color=MUTED))
    fig.text(0.5, 0.085, "Steps 2 to 4 repeat every 2 seconds during the descent. The lander also has motion sensors that track its movement between fixes.",
             ha="center", fontsize=10, color=INK2)
    title(fig, "How the lander finds itself",
          "There is no GPS on the Moon. The lander works out where it is by recognising the ground below it.\nThe pictures below are from one real simulated flight.", y=0.975)
    footer(fig, "Panels 1 to 4 show the first position fix of flight seed 10001 (map crop, noisy laser scan, match scores, uncertainty).")
    save(fig, "03_how_it_works")


# ================================================================ fig: descent story
def crop_bounds(T, pts, margin=300.0):
    x0 = pts[:, 0].min() - margin; x1 = pts[:, 0].max() + margin
    y0 = pts[:, 1].min() - margin; y1 = pts[:, 1].max() + margin
    w = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = float(np.clip(cx - w / 2, 0, T.extent[0] - w)); y0 = float(np.clip(cy - w / 2, 0, T.extent[1] - w))
    return x0, y0, w


def fig_descent_story():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cfg, sites = build("mixed_hard+dec", "test")
    cases = [("Rough terrain", 10001, PolyGuidance(), "Position fixes keep arriving"),
             ("Flat plains + signal dropout", 10051, PolyGuidance(), "No position fix ever arrives"),
             ("Same flight, but it waits", 10051, WaitGuidance(max_wait=70), "Hovers up to 70 s, then descends")]
    fig = plt.figure(figsize=(14.0, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], left=0.065, right=0.985, top=0.80, bottom=0.17, wspace=0.16, hspace=0.30)
    for c, (name, seed, g, blurb) in enumerate(cases):
        env, info, outage, hold = run_episode(sites, cfg, seed, g)
        H = env.history
        t = np.array(H["t"]); r = np.array(H["r"]); rh = np.array(H["r_hat"]); sg = np.array(H["sigma_h"])
        err = np.hypot(rh[:, 0] - r[:, 0], rh[:, 1] - r[:, 1])
        fix_t = [tt for tt, k in zip(t, H["trn"]) if k == "fix"]
        T = env.site.truth
        pts = np.vstack([r[:, :2], rh[:, :2], env.target[:2][None, :]])
        x0, y0, w = crop_bounds(T, pts)
        i0, j0 = int(round(x0 / T.dx)), int(round(y0 / T.dx)); n = int(round(w / T.dx))
        hs = hill(T, 6.0)[j0:j0 + n, i0:i0 + n]
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(hs, cmap="gray", origin="lower", extent=[x0 / 1000, (x0 + w) / 1000, y0 / 1000, (y0 + w) / 1000], vmin=0.15, vmax=1.0)
        idx = [0, len(t) // 4, len(t) // 2, 3 * len(t) // 4]
        for k in idx:
            ax.add_patch(Circle((rh[k, 0] / 1000, rh[k, 1] / 1000), sg[k] / 1000, fill=False, ec=ORANGE, lw=1.6, alpha=0.95))
        ax.plot(rh[:, 0] / 1000, rh[:, 1] / 1000, color=ORANGE, lw=1.8, ls=(0, (4, 2)))
        ax.plot(r[:, 0] / 1000, r[:, 1] / 1000, color="#ffffff", lw=4.4)
        ax.plot(r[:, 0] / 1000, r[:, 1] / 1000, color=INK, lw=2.4)
        ax.plot(r[0, 0] / 1000, r[0, 1] / 1000, "o", color=INK, mec="white", ms=7, mew=1.2, zorder=6)
        ax.plot(env.target[0] / 1000, env.target[1] / 1000, "*", ms=19, mfc="#ffd23f", mec=INK, mew=1.0, zorder=7)
        ax.plot(r[-1, 0] / 1000, r[-1, 1] / 1000, "X", ms=9, mfc=INK, mec="white", mew=1.0, zorder=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(AXIS)
        ax.text(0.5, 1.13, name, transform=ax.transAxes, ha="center", fontsize=13.5, fontweight="bold", color=INK)
        nfix = len(fix_t)
        if info["outcome"] == "success" and info["landing_err"] <= 50:
            res = f"Landed {info['landing_err']:.1f} m from the target"
        else:
            res = f"Landed {info['landing_err']:.0f} m from the target (miss)"
        ax.text(0.5, 1.045, f"{res}  |  {nfix} position fixes", transform=ax.transAxes, ha="center", fontsize=10.5, color=INK2)
        ax.text(0.02, 0.02, f"{w/1000:.1f} km across", transform=ax.transAxes, fontsize=8.5, color="white", fontweight="bold",
                bbox=dict(facecolor="#0b0b0b", alpha=0.6, pad=2, edgecolor="none"))
        b = fig.add_subplot(gs[1, c])
        if outage[1] > 0:
            b.axvspan(outage[0], min(outage[1], t[-1] + 2), color=GRAY, alpha=0.35, lw=0)
        # hover spans
        if hold.any():
            tt = t[:-1] if len(t) - 1 == len(hold) else t[:len(hold)]
            on = False
            for ti, hv in zip(t[:len(hold)], hold):
                if hv and not on: s0 = ti; on = True
                if not hv and on: b.axvspan(s0, ti, color=BLUE, alpha=0.18, lw=0); on = False
            if on: b.axvspan(s0, t[len(hold) - 1], color=BLUE, alpha=0.18, lw=0)
        b.axhline(50, color=MUTED, lw=1, ls=":")
        b.plot(t, np.maximum(sg, 0.3), color=INK2, lw=1.8, ls=(0, (4, 2)))
        b.plot(t, np.maximum(err, 0.3), color=ORANGE, lw=2.4)
        b.plot(fix_t, np.full(len(fix_t), 0.95), "|", color=AQUA, ms=9, mew=1.8)
        b.set_yscale("log"); b.set_ylim(0.8, 600); b.set_xlim(0, 255)
        b.set_yticks([1, 10, 50, 100, 500]); b.set_yticklabels(["1", "10", "50", "100", "500"])
        b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        b.grid(axis="y", color=GRID, lw=0.7); b.set_axisbelow(True)
        b.set_xlabel("seconds since start", fontsize=10)
        if c == 0:
            b.set_ylabel("metres", fontsize=10)
        b.text(252, 56, "50 m = precise landing", fontsize=8, color=MUTED, ha="right", va="bottom")
        b.text(0.0, 1.07, blurb, transform=b.transAxes, fontsize=10.5, color=INK, fontweight="bold")
    handles = [Line2D([0], [0], color=INK, lw=2.4, label="Actual flight path"),
               Line2D([0], [0], color=ORANGE, lw=1.8, ls=(0, (4, 2)), label="Where the lander believed it was"),
               Line2D([0], [0], marker="o", mfc="none", mec=ORANGE, lw=0, ms=10, label="Circle: how far off it thinks it could be"),
               Line2D([0], [0], marker="*", mfc="#ffd23f", mec=INK, lw=0, ms=13, label="Target"),
               Line2D([0], [0], marker="X", mfc=INK, mec="white", lw=0, ms=9, label="Touchdown"),
               Line2D([0], [0], color=ORANGE, lw=2.4, label="Chart: actual position error"),
               Line2D([0], [0], color=INK2, lw=1.8, ls=(0, (4, 2)), label="Chart: error the lander thinks is possible"),
               Line2D([0], [0], marker="|", color=AQUA, lw=0, ms=10, mew=2, label="Position fix from terrain matching"),
               Patch(fc=GRAY, alpha=0.35, label="Signal dropout (no fixes)"),
               Patch(fc=BLUE, alpha=0.18, label="Hovering to wait for a fix")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 0.025), columnspacing=1.6, handletextpad=0.6)
    title(fig, "One landing, three stories",
          "Top: the flight seen from above. Bottom: how far off the lander really was (orange) against how far off it thought it could be (dashed).\nA good fix collapses both. With no fix, the lander drifts and lands where it wrongly believes the target is.")
    footer(fig, "Left: typical rough-terrain flight (seed 10001). Middle/right: one plains flight (seed 10051), picked from 7 of 50 where waiting helped.")
    save(fig, "04_descent_story")


# ==================================================================== fig: scorecard
def fig_scorecard():
    def rate(tag):
        rows = J(tag)["episodes"]
        return precise_k(rows), len(rows)
    sec1 = [("Rough terrain, navigation working", "baseline_synthetic_trn"),
            ("Flat plains", "baseline_low_relief"),
            ("No terrain matching at all", "baseline_no_trn")]
    sec2 = [("Original guidance", "baseline_mixed_hard", BLUE_STEPS[250]),
            ("+ better navigation filter", "baseline_mixed_hard_dec200", BLUE_STEPS[400]),
            ("+ waits for a fix when unsure", "wait70_test_mixed_hard+dec", BLUE_STEPS[600]),
            ("Learned policy (RL), no uncertainty input", "ppo_nocov_s0_test", ORANGE),
            ("Learned policy (RL), sees uncertainty", "ppo_cov_s0_test", ORANGE)]
    rows = []
    y = 0.0
    ypos, labels, colors, vals, los, his, ns = [], [], [], [], [], [], []
    headers = []
    headers.append((y, "Original guidance, under different conditions", "100 flights each")); y += 0.85
    for name, tag in sec1:
        k, n = rate(tag); lo, hi = wilson(k, n)
        ypos.append(y); labels.append(name); colors.append(GRAY); vals.append(100 * k / n); los.append(100 * lo); his.append(100 * hi); ns.append(n); y += 1
    y += 0.9
    headers.append((y, "On the hard mix: flat plains plus signal dropouts", "200 flights each, terrain the system never trained on")); y += 0.85
    for name, tag, col in sec2:
        k, n = rate(tag); lo, hi = wilson(k, n)
        ypos.append(y); labels.append(name); colors.append(col); vals.append(100 * k / n); los.append(100 * lo); his.append(100 * hi); ns.append(n); y += 1
    fig, ax = plt.subplots(figsize=(12.4, 7.6))
    fig.subplots_adjust(left=0.335, right=0.96, top=0.79, bottom=0.19)
    ax.barh(ypos, vals, height=0.62, color=colors, zorder=3)
    for yy, v, lo, hi, n in zip(ypos, vals, los, his, ns):
        ax.plot([lo, hi], [yy, yy], color=INK2, lw=1.3, zorder=4, solid_capstyle="butt")
        ax.plot([lo, lo], [yy - 0.13, yy + 0.13], color=INK2, lw=1.3, zorder=4)
        ax.plot([hi, hi], [yy - 0.13, yy + 0.13], color=INK2, lw=1.3, zorder=4)
        ax.text(min(hi, 100) + 1.6, yy, f"{v:.0f}%" if abs(v - round(v)) < 0.05 else f"{v:.1f}%", va="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=11.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    import matplotlib.transforms as mtr
    tr = mtr.blended_transform_factory(fig.transFigure, ax.transData)
    for yy, h1, h2 in headers:
        ax.text(0.035, yy - 0.12, h1, transform=tr, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="center")
        ax.text(0.035, yy + 0.32, h2, transform=tr, fontsize=9.5, color=MUTED, ha="left", va="center")
    ax.set_xlabel("Flights that touched down safely within 50 m of the target", fontsize=11, labelpad=8)
    from matplotlib.patches import Patch
    leg = [Patch(fc=BLUE_STEPS[400], label="Classical guidance (hand-designed rules)"), Patch(fc=ORANGE, label="Learned policy (first attempt, one training run)"), Patch(fc=GRAY, label="Context")]
    fig.legend(handles=leg, loc="lower left", ncol=3, fontsize=10, bbox_to_anchor=(0.035, 0.035), handlelength=1.2, columnspacing=1.8)
    title(fig, "How often does the lander touch down within 50 m of its target?",
          "Thin lines show the statistical margin of error (95%). Differences smaller than about 5 points are not reliable.\nThe learned policies were tested with the original navigation filter, so compare them with the light-blue\n\"Original guidance\" bar (75.5%).")
    footer(fig, "Safe touchdown = gentle enough and on safe ground. Learned policies: one training run each, 600,000 steps.")
    save(fig, "05_scorecard")
    return list(zip(labels, vals))


# ============================================================ fig: miss-distance curves
def fig_miss_distance():
    series = [("Original guidance", "baseline_mixed_hard", ORANGE, 2.2),
              ("+ better navigation filter", "baseline_mixed_hard_dec200", AQUA, 2.2),
              ("+ waits for a fix when unsure", "wait70_test_mixed_hard+dec", BLUE, 3.4),
              ("Learned policy, no uncertainty input", "ppo_nocov_s0_test", YELLOW, 2.2),
              ("Learned policy, sees uncertainty", "ppo_cov_s0_test", MAGENTA, 2.2)]
    xs = np.logspace(-0.3, 3.1, 600)
    fig, ax = plt.subplots(figsize=(12.0, 7.4))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.80, bottom=0.12)
    out = {}
    for name, tag, col, lw in series:
        rows = J(tag)["episodes"]
        err = np.array([r["landing_err"] if r["outcome"] == "success" else np.inf for r in rows])
        y = 100 * np.mean(err[:, None] <= xs[None, :], axis=0)
        ax.plot(xs, y, color=col, lw=lw, drawstyle="steps-post", label=name, zorder=4 if col == BLUE else 3)
        out[name] = float(100 * np.mean(err <= 50))
    ax.axvline(50, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.text(54, 3, "50 m: counts as a\nprecise landing", fontsize=9.5, color=MUTED, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.5, 1300); ax.set_ylim(0, 102)
    ax.set_xticks([1, 10, 50, 100, 1000]); ax.set_xticklabels(["1 m", "10 m", "50 m", "100 m", "1 km"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_xlabel("Distance from the target at touchdown", fontsize=11, labelpad=8)
    ax.set_ylabel("Share of the 200 test flights", fontsize=11, labelpad=8)
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: -out[l[i]])
    ax.legend([h[i] for i in order], [f"{l[i]}  ({out[l[i]]:.1f}% within 50 m)" for i in order], loc="lower right", bbox_to_anchor=(0.99, 0.10), fontsize=10.5)
    title(fig, "How close do landings get to the target?",
          "Each line is the share of 200 test flights that touched down safely within a given distance of the target.\nHigher and further left is better. Unsafe or too-hard landings count as misses. All flights are on terrain the system never saw in training.")
    footer(fig, "Hard mix: flat plains and rough terrain, with signal dropouts in 40% of flights. Learned policies used the original navigation filter.")
    save(fig, "06_miss_distance")
    return out


# =================================================================== fig: why misses
def fig_why_misses():
    cfg, sites = build("mixed_hard+dec", "test")
    rows = J("baseline_mixed_hard_dec200")["episodes"]
    env = LunarLandingEnv(sites, cfg, seed=0)
    prec = lambda r: r["outcome"] == "success" and r["landing_err"] <= 50
    cell = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        env.reset(seed=r["seed"])
        kind = "Rough terrain" if sites.index(env.site) < 4 else "Flat plains"
        blk = "Signal dropout" if env.nav.cfg.outage_end > 0 else "No dropout"
        cell[(kind, blk)][0] += 1; cell[(kind, blk)][1] += prec(r)
    fx = [r for r in rows if r["trn_fixes"] > 0]; nf = [r for r in rows if r["trn_fixes"] == 0]
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 6.6), gridspec_kw={"width_ratios": [0.8, 1.2], "wspace": 0.22})
    fig.subplots_adjust(left=0.07, right=0.975, top=0.76, bottom=0.14)
    a = axs[0]
    vals = [(len(fx), 100 * np.mean([prec(r) for r in fx]), BLUE, "Got at least\none position fix"), (len(nf), 100 * np.mean([prec(r) for r in nf]), ORANGE, "Never got\na position fix")]
    for i, (n, v, col, lab) in enumerate(vals):
        a.bar(i, v, color=col, width=0.6, zorder=3)
        a.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=15, fontweight="bold", color=INK)
        a.text(i, -3, f"{n} flights", ha="center", va="top", fontsize=9.5, color=MUTED)
    a.set_xticks([0, 1]); a.set_xticklabels([v[3] for v in vals], fontsize=11)
    a.set_ylim(0, 110); a.set_yticks([0, 25, 50, 75, 100]); a.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    a.yaxis.grid(True, color=GRID, lw=0.7); a.set_axisbelow(True); a.tick_params(axis="x", length=0, pad=18)
    a.set_title("A position fix is what matters", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
    a.set_ylabel("Touched down safely within 50 m", fontsize=10.5)
    b = axs[1]
    groups = ["Rough terrain", "Flat plains"]; xpos = [0, 1]; bw = 0.36
    for gi, gname in enumerate(groups):
        for k, (blk, col, off) in enumerate([("No dropout", BLUE, -bw / 2 - 0.02), ("Signal dropout", ORANGE, bw / 2 + 0.02)]):
            n, kk = cell[(gname, blk)]
            v = 100 * kk / n
            b.bar(gi + off, v, width=bw, color=col, zorder=3, label=blk if gi == 0 else None)
            b.text(gi + off, v + 2, f"{v:.0f}%", ha="center", fontsize=13.5, fontweight="bold", color=INK)
            b.text(gi + off, -3, f"{n} flights", ha="center", va="top", fontsize=9, color=MUTED)
    b.set_xticks(xpos); b.set_xticklabels(groups, fontsize=12); b.tick_params(axis="x", length=0, pad=18)
    b.set_ylim(0, 110); b.set_yticks([0, 25, 50, 75, 100]); b.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    b.yaxis.grid(True, color=GRID, lw=0.7); b.set_axisbelow(True)
    b.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=10.5, ncol=2)
    b.set_title("Almost every miss is on flat plains", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
    title(fig, "Why do some landings miss?",
          "The steering itself is not the problem: when the lander gets a position fix it lands within 50 m almost every time.\nMisses happen when it never gets a fix, and that is mostly on flat plains, especially during a signal dropout.")
    footer(fig, "Navigation-filter-improved guidance, 200 test flights. Dropout = terrain-matching signal unavailable for 30 to 70 seconds.")
    save(fig, "07_why_landings_miss")
    return {k: (v[0], 100 * v[1] / v[0]) for k, v in cell.items()}, vals


# ================================================================== fig: waiting trade-off
def fig_waiting():
    ws = [0, 30, 50, 70, 80, 90]
    data = [J(f"sweep_wait{w}_train") for w in ws]
    prec = [100 * d["summary"]["precise_landing_rate_<=50m"] for d in data]
    fuel = [d["summary"]["median_fuel_kg"] for d in data]
    tmo = [d["summary"]["outcomes"].get("timeout", 0) for d in data]
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 6.4), gridspec_kw={"wspace": 0.22})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.76, bottom=0.14)
    a, b = axs
    a.plot(ws, prec, color=BLUE, lw=2.6, marker="o", ms=8, zorder=3)
    i70 = ws.index(70)
    a.plot([70], [prec[i70]], marker="o", ms=14, mfc="none", mec=INK, mew=2, zorder=4)
    for w, p, t in zip(ws, prec, tmo):
        lab = f"{p:.1f}%\nchosen" if w == 70 else f"{p:.1f}%"
        a.text(w, p + 1.1, lab, ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK, linespacing=1.15)
        if t:
            a.text(w, 71.0, f"{t}\ntimed\nout", ha="center", va="bottom", fontsize=8.5, color=ORANGE, fontweight="bold", linespacing=1.15)
    a.set_ylim(70, 98); a.set_xlim(-6, 98)
    a.set_yticks([70, 75, 80, 85, 90, 95]); a.set_yticklabels(["70%", "75%", "80%", "85%", "90%", "95%"])
    a.set_xticks(ws); a.yaxis.grid(True, color=GRID, lw=0.7); a.set_axisbelow(True)
    a.set_xlabel("Longest the lander may hover and wait for a fix (seconds)", fontsize=10.5, labelpad=8)
    a.set_title("Landings within 50 m", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
    b.plot(ws, fuel, color=INK2, lw=2.6, marker="o", ms=8, zorder=3)
    for w, f in zip(ws, fuel):
        below = (w == 90)
        b.text(w, f - 12 if below else f + 10, f"{f:.0f}", ha="center", va="top" if below else "bottom", fontsize=11, fontweight="bold", color=INK)
    b.axhline(500, color=MUTED, lw=1.2, ls=":")
    b.text(96, 488, "Fuel available: about 500 kg", ha="right", va="top", fontsize=9.5, color=MUTED)
    b.set_ylim(0, 540); b.set_xlim(-6, 98); b.set_xticks(ws)
    b.yaxis.grid(True, color=GRID, lw=0.7); b.set_axisbelow(True)
    b.set_xlabel("Longest the lander may hover and wait for a fix (seconds)", fontsize=10.5, labelpad=8)
    b.set_ylabel("Median fuel used (kg)", fontsize=10.5)
    b.set_title("Fuel it costs", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
    title(fig, "Waiting for a fix helps, until the clock runs out",
          "When the lander is unsure where it is, it can hover and wait for terrain matching to work instead of descending.\nMore waiting means more accurate landings and more fuel, until the 250-second mission limit cuts landings short.")
    footer(fig, "Chosen on 200 training-terrain flights. On held-out test terrain the 70 s setting scored 89.0% (83.0% without waiting).\nOrange numbers: flights (of 200) that timed out, meaning still in the air at the 250-second mission limit.")
    save(fig, "08_waiting_tradeoff")
    return list(zip(ws, prec, fuel, tmo))


FIGS = {"terrain_3d": fig_terrain_3d, "navigability": fig_navigability, "how_it_works": fig_how_it_works, "descent_story": fig_descent_story,
        "scorecard": fig_scorecard, "miss_distance": fig_miss_distance, "why_misses": fig_why_misses, "waiting": fig_waiting}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for w in which:
        res = FIGS[w]()
        if res is not None:
            print("  data:", res)
