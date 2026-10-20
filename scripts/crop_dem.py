"""Shrink a big DEM GeoTIFF into a small .npy tile (run this on YOUR computer).

    pip install rasterio numpy scipy
    python crop_dem.py Site04_final_adj_5mpp_surf.tif --size_km 8 --out site04.npy

Picks the size_km x size_km window with the most relief that still has gentle ground
(slope < 10 deg) in its central area to land on (use --centre for the middle instead), fills no-data holes, flips rows so +y is north (same as the project's
loader), and saves float32. 8 km at 10 m/pixel is 800x800, about 2.5 MB.
Output cells are 10 m (block-averaged from 5 m), which is what the simulator uses.
"""
import argparse
import numpy as np


def read(path):
    try:
        import rasterio
        with rasterio.open(path) as s:
            a = s.read(1).astype("float64")
            return a, abs(s.transform.a), s.nodata
    except ImportError:
        import tifffile
        return tifffile.imread(path).astype("float64"), None, None


def best_window(a, n, step, dx=5.0, margin_m=1850.0, min_cells=30):
    """Pick the n x n window with the most relief among those that have enough *usable
    landing ground in their central area*. The lander starts up to 1.6 km away and the
    simulator keeps 250 m from the edge, so targets must lie in the middle of the tile.
    Usable = slope < 10 deg, shrunk by ~80 m so there is clearance around it."""
    from scipy import ndimage
    k = 8
    sub = a[::k, ::k]
    fill = float(np.nanmean(a))
    gy, gx = np.gradient(np.nan_to_num(sub, nan=fill), dx * k)
    gentle = np.degrees(np.arctan(np.hypot(gx, gy))) < 10.0
    gentle = ndimage.binary_erosion(gentle, iterations=2)
    m, g = n // k, int(round(margin_m / (dx * k)))
    cands = []
    for r in range(0, sub.shape[0] - m + 1, max(step // k, 1)):
        for c in range(0, sub.shape[1] - m + 1, max(step // k, 1)):
            w = sub[r:r + m, c:c + m]
            if np.isnan(w).mean() > 0.05:
                continue
            cells = int(gentle[r + g:r + m - g, c + g:c + m - g].sum())
            cands.append((cells, float(np.nanstd(w)), r * k, c * k))
    ok = [t for t in cands if t[0] >= min_cells]
    best = max(ok, key=lambda t: t[1]) if ok else max(cands, key=lambda t: t[0])
    return (best[2], best[3]), best[0], best[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tif"); p.add_argument("--size_km", type=float, default=8.0)
    p.add_argument("--out", default="tile.npy"); p.add_argument("--centre", action="store_true")
    p.add_argument("--out_dx", type=float, default=10.0, help="output cell size in metres (block-averaged)")
    p.add_argument("--dx", type=float, default=None, help="metres per pixel if not read from file")
    o = p.parse_args()
    a, px, nodata = read(o.tif)
    dx = o.dx or px or 5.0
    if nodata is not None:
        a[a == nodata] = np.nan
    a[np.abs(a) > 1e6] = np.nan
    n = min(int(o.size_km * 1000 / dx), min(a.shape))
    if o.centre:
        rc = ((a.shape[0] - n) // 2, (a.shape[1] - n) // 2)
    else:
        rc, safe, _ = best_window(a, n, max(n // 8, 1), dx)
        print(f"usable landing cells (slope<10 deg, with clearance) in the window centre: {safe}")
        if safe < 30:
            print("WARNING: very little gentle ground in this site; results on it may not be usable.")
    w = a[rc[0]:rc[0] + n, rc[1]:rc[1] + n]
    w = np.where(np.isnan(w), np.nanmean(w), w)
    f = int(round(o.out_dx / dx))
    if f > 1:
        n2 = (w.shape[0] // f) * f
        w = w[:n2, :n2].reshape(n2 // f, f, n2 // f, f).mean(axis=(1, 3))
        dx = dx * f
    w = w[::-1].astype("float32")
    np.save(o.out, w)
    gy, gx = np.gradient(w.astype(float), dx)
    print(f"slope<10deg over whole tile: {(np.degrees(np.arctan(np.hypot(gx, gy))) < 10).mean():.0%}")
    print(f"saved {o.out}: {w.shape}, dx={dx} m, relief std={w.std():.1f} m, "
          f"range {w.min():.0f}..{w.max():.0f} m, window at row {rc[0]}, col {rc[1]}")


if __name__ == "__main__":
    main()
