"""TRN figures on the real LOLA tiles (data/site04.npy, data/site23.npy), run through the project's own
navigation code (trnrl.nav) and matchability map. Scenes are fixed below so the figures are reproducible.

    python scripts/trn_figures.py        -> figures/paper/fig_trn_match, fig_trn_cases, fig_trn_tiles (.pdf, .png)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from scipy import ndimage
from trnrl.terrain import Terrain
from trnrl.env import Site, EnvConfig
from trnrl.nav import NavSystem, NavConfig
from trnrl.matchability import matchability_map

INK, GRID, AXIS = "#000000", "#e4e4e4", "#222222"
BLUE, RED, GREEN, GREY = "#1f4e79", "#b5452a", "#3b7a57", "#6e6e6e"
plt.rcParams.update({
    "figure.facecolor": "w", "axes.facecolor": "w", "savefig.facecolor": "w",
    "font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

cfg = EnvConfig()
sites = {}
for name in ("site04", "site23"):
    sites[name] = Site(Terrain(np.load(ROOT / "data" / f"{name}.npy").astype(float), 10.0, name), cfg)


def make_nav(site, seed, sigma0=120.0, fov=40.0):
    nc = NavConfig(init_pos_h=sigma0, fov_deg=fov, map_height_sigma=max(site.map_err_rms, 0.5), search_max_cells=40)
    nav = NavSystem(site.truth, site.map, nc, np.random.default_rng(seed))
    return nav, nc


def surface(nav, r, agl, xe, ye, sh):
    """Replicates NavSystem._trn_fix up to the NCC surface (same rng stream, so the same measurement)."""
    c, m = nav.cfg, nav.map
    n = nav.patch_n_for(agl)
    ox, oy = nav._offs(n)
    n2 = n * n
    sl = c.lidar_sigma0 + c.lidar_sigma_frac * agl
    meas = nav.truth.height(r[0] + ox, r[1] + oy) + nav.rng.normal(0, sl, n2)
    S = int(np.clip(np.ceil(3.0 * sh / m.dx) + 1, 2, c.search_max_cells))
    ab = np.arange(-S, S + 1)
    A, B = np.meshgrid(ab, ab)
    cx, cy = xe + A.ravel() * m.dx, ye + B.ravel() * m.dx
    mc = meas - meas.mean(); mcn = np.linalg.norm(mc)
    px = (cx[:, None] + ox[None, :]) / m.dx
    py = (cy[:, None] + oy[None, :]) / m.dx
    ref = ndimage.map_coordinates(m.h, [py, px], order=1, mode="nearest")
    refc = ref - ref.mean(1, keepdims=True)
    ncc = (refc @ mc) / (np.linalg.norm(refc, axis=1) * mcn + 1e-9)
    grid = ncc.reshape(A.shape)
    k = int(np.argmax(ncc)); bj, bi = divmod(k, A.shape[1])
    best = float(ncc[k])
    lab, _ = ndimage.label(grid >= best - c.lobe_drop, structure=np.ones((3, 3)))
    lobe = ndimage.binary_dilation(lab == lab[bj, bi], structure=np.ones((3, 3)))
    out = np.where(lobe, -9, grid)
    j2, i2 = np.unravel_index(int(np.argmax(out)), out.shape)
    second = float(out[j2, i2])
    return dict(n=n, S=S, ox=ox, oy=oy, meas=meas.reshape(n, n), grid=grid, A=A, B=B, best=best, second=second,
                bj=bj, bi=bi, j2=j2, i2=i2, xe=xe, ye=ye, dx=m.dx, std=float(meas.std()))


def quick_status(surf, c):
    if surf["std"] < c.min_patch_relief: return "rejected_flat"
    if surf["best"] < c.ncc_min: return "rejected_ncc"
    if surf["second"] >= surf["best"] - c.amb_margin: return "rejected_ambiguous"
    return "fix"

def quick_status(surf, c):
    """accept/reject decision of NavSystem._trn_fix (flat, NCC threshold, ambiguity) without the chi-square gate"""
    if surf["std"] < c.min_patch_relief: return "rejected_flat"
    if surf["best"] < c.ncc_min: return "rejected_ncc"
    if surf["second"] >= surf["best"] - c.amb_margin: return "rejected_ambiguous"
    return "fix"


def hill(ax, h, dx, extent=None):
    ls = LightSource(azdeg=315, altdeg=35)
    ax.imshow(ls.hillshade(h, vert_exag=1.0, dx=dx, dy=dx), cmap="gray", origin="lower", extent=extent, vmin=0.2, vmax=1.0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# ------------------------------------------------------------------------------------------ plotting
def ncc_panel(ax, s, title, show_cb=False):
    A, B, dx = s["A"], s["B"], s["dx"]
    S = s["S"]
    ext = [(-S - .5) * dx, (S + .5) * dx, (-S - .5) * dx, (S + .5) * dx]
    im = ax.imshow(s["grid"], origin="lower", extent=ext, cmap="viridis", vmin=-0.2, vmax=1.0)
    tx, ty = s["r"][0] - s["xe"], s["r"][1] - s["ye"]
    ax.plot(0, 0, "+", color="w", ms=8, mew=1.2)
    ax.plot(tx, ty, "o", mfc="none", mec="w", ms=9, mew=1.3)
    bx, by = A[s["bj"], s["bi"]] * dx, B[s["bj"], s["bi"]] * dx
    ax.plot(bx, by, "x", color=RED, ms=7, mew=1.5)
    sx, sy = A[s["j2"], s["i2"]] * dx, B[s["j2"], s["i2"]] * dx
    ax.plot(sx, sy, "s", mfc="none", mec="#f2c14e", ms=7, mew=1.3)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("east offset from estimate (m)")
    return im


def fig_match(s, name, tile="site04"):
    site = sites[tile]
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.9, 1.1], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    r, xe, ye, dx = s["r"], s["xe"], s["ye"], s["dx"]
    half = 1900.0
    ex, ey = site.truth.extent
    cx_, cy_ = float(np.clip(r[0], half, ex - half)), float(np.clip(r[1], half, ey - half))
    x0, x1, y0, y1 = cx_ - half, cx_ + half, cy_ - half, cy_ + half
    i0, i1, j0, j1 = int(x0 / 10), int(x1 / 10), int(y0 / 10), int(y1 / 10)
    hill(ax, site.truth.h[j0:j1, i0:i1], 10.0, extent=[x0, x1, y0, y1])
    side = s["n"] * dx
    Sx = s["S"] * dx
    ax.add_patch(Rectangle((xe - Sx - side / 2, ye - Sx - side / 2), 2 * Sx + side, 2 * Sx + side, fill=False, ec=BLUE, lw=1.0, ls="--"))
    ax.add_patch(Rectangle((r[0] - side / 2, r[1] - side / 2), side, side, fill=False, ec=RED, lw=1.2))
    ax.plot(xe, ye, "+", color=BLUE, ms=9, mew=1.4)
    ax.plot(r[0], r[1], "o", mfc="none", mec=RED, ms=6, mew=1.2)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(a) Onboard map and search window", fontsize=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, ls="--", lw=1, label="search window"),
                       Line2D([], [], color=RED, lw=1.2, label="lidar footprint"),
                       Line2D([], [], color=BLUE, marker="+", ls="", label="estimate"),
                       Line2D([], [], mfc="none", mec=RED, marker="o", ls="", label="true position")],
              loc="upper left", fontsize=6, handlelength=1.4, framealpha=0.85, frameon=True, facecolor="w", edgecolor="none")
    ax2 = fig.add_subplot(gs[1])
    h = s["meas"]
    im2 = ax2.imshow(h - h.mean(), origin="lower", cmap="cividis", extent=[-side / 2, side / 2, -side / 2, side / 2])
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.05, pad=0.03); cb2.set_label("height $-$ mean (m)", fontsize=7, labelpad=2); cb2.ax.tick_params(labelsize=6)
    ax2.set_xlabel("east (m)")
    ax2.set_title("(b) Lidar height patch", fontsize=8)
    ax3 = fig.add_subplot(gs[2])
    im = ncc_panel(ax3, s, "(c) Correlation surface")
    cb = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03); cb.set_label("NCC", fontsize=7); cb.ax.tick_params(labelsize=6)
    for a in (ax, ax2, ax3):
        a.tick_params(labelsize=6.5)
        a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
    save(fig, name)


def fig_cases(trio, name):
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.8), gridspec_kw=dict(hspace=0.5, wspace=0.7))
    letters = "abc"
    for k, (s, ttl) in enumerate(trio):
        im = ncc_panel(axs[0, k], s, f"({letters[k]}) {ttl}")
        txt = f"best {s['best']:.2f}, second {s['second']:.2f}\n{s['status'].replace('rejected_', 'rejected: ').replace('fix', 'accepted')}"
        axs[0, k].text(0.03, 0.04, txt, transform=axs[0, k].transAxes, color="w", fontsize=6.5, va="bottom",
                       bbox=dict(fc="k", alpha=0.45, ec="none", pad=1.5))
        side = s["n"] * s["dx"]
        h = s["meas"]
        if k == 0: axs[0, k].set_ylabel("north offset (m)"); axs[1, k].set_ylabel("north (m)")
        imh = axs[1, k].imshow(h - h.mean(), origin="lower", cmap="cividis", extent=[-side / 2, side / 2, -side / 2, side / 2])
        axs[1, k].set_title(f"lidar patch, {s['n']}$\\times${s['n']} cells", fontsize=7.5)
        axs[1, k].set_xlabel("east (m)")
        cbh = fig.colorbar(imh, ax=axs[1, k], fraction=0.046, pad=0.03); cbh.ax.tick_params(labelsize=6)
        for a in axs[:, k]:
            a.tick_params(labelsize=6.5); a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
    cax = fig.add_axes([0.93, 0.56, 0.012, 0.31]); cb = fig.colorbar(im, cax=cax); cb.set_label("NCC", fontsize=7); cb.ax.tick_params(labelsize=6)
    save(fig, name)


def fig_tiles(name):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.2), gridspec_kw=dict(hspace=0.28, wspace=0.28))
    for row, (nm, lab) in enumerate((("site04", "Site 04"), ("site23", "Site 23"))):
        st = sites[nm]
        ax = axs[row, 0]
        hill(ax, st.truth.h, 10.0, extent=[0, 8000, 0, 8000])
        ax.set_title(f"({'ab'[0] if row == 0 else 'c'}) {lab}: relief" if False else f"{lab}: shaded relief", fontsize=8)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        G = matchability_map(st.map.h)
        ax = axs[row, 1]
        from matplotlib.colors import PowerNorm
        im = ax.imshow(G, origin="lower", cmap="magma", norm=PowerNorm(0.5, vmin=0, vmax=float(G.max())), extent=[0, 8000, 0, 8000])
        ax.set_title(f"{lab}: matchability map", fontsize=8)
        ax.set_xlabel("x (m)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("matchability", fontsize=7); cb.ax.tick_params(labelsize=6)
        print(nm, "mean matchability", float(G.mean()), "frac>0.5", float((G > 0.5).mean()))
        for a in axs[row]:
            a.tick_params(labelsize=6.5); a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
    for a, t in zip(axs.ravel(), "abcd"):
        a.set_title(f"({t}) " + a.get_title(), fontsize=8)
    save(fig, name)

# ------------------------------------------------------------------------------------------- scenes
# (tile, true position r = (x, y, z) [m], estimate (xe, ye) [m], sigma0 [m]); lander 500 m above ground.
SCENE_ACCEPT = ("site04", np.array([1257.54638123, 3602.17047097, 1995.50575125]), 757.5463812327475, 3723.072991253211)
SCENE_AMBIG = ("site23", np.array([6813.28426152, 3159.85921834, 4713.46394162]), 6622.078417629879, 2882.829958823471)
SIGMA0, NOISE_SEED = 250.0, 3


def scene(tile, r, xe, ye, fov):
    site = sites[tile]
    agl = float(r[2] - site.truth.height(r[0], r[1]))
    nav, nc = make_nav(site, 1, SIGMA0, fov)
    nav.reset(r, np.zeros(3))
    nav.rng = np.random.default_rng(NOISE_SEED)
    t = surface(nav, r, agl, xe, ye, SIGMA0)
    t["status"] = quick_status(t, nc); t["r"], t["agl"] = r, agl
    return t


if __name__ == "__main__":
    acc = scene(*SCENE_ACCEPT, 40.0)
    amb40 = scene(*SCENE_AMBIG, 40.0)
    amb60 = scene(*SCENE_AMBIG, 60.0)
    print("statuses:", acc["status"], amb40["status"], amb60["status"])
    assert (acc["status"], amb40["status"], amb60["status"]) == ("fix", "rejected_ambiguous", "fix")
    fig_match(acc, "fig_trn_match", SCENE_ACCEPT[0])
    fig_cases([(acc, "Site 04: accepted fix"), (amb40, "Site 23: ambiguous, rejected"),
               (amb60, "Same scene, 60$^\\circ$ field of view")], "fig_trn_cases")
    fig_tiles("fig_trn_tiles")
