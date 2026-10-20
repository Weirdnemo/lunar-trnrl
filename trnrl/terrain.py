"""Terrain: DEM loading, synthetic lunar terrain, hazard and texture maps.

Conventions
-----------
* Local ENU-like frame, metres. x east (column index i), y north (row index j),
  z up. ``h[j, i]`` is the height at ``(x, y) = (i*dx, j*dx)``.
* ``truth`` terrain is what the lander physically touches and what the lidar
  measures. The ``onboard map`` is a coarser, smoothed copy of it
  (``Terrain.downsample``), so sub-resolution detail acts as map error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass
class Terrain:
    h: np.ndarray          # heights [m], shape (ny, nx)
    dx: float              # cell size [m]
    name: str = "terrain"

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, int]:
        return self.h.shape

    @property
    def extent(self) -> tuple[float, float]:
        """(x_max, y_max) in metres."""
        return (self.h.shape[1] - 1) * self.dx, (self.h.shape[0] - 1) * self.dx

    def height(self, x, y):
        """Bilinear height at (x, y) [m]. Scalars in -> float out."""
        scalar = np.ndim(x) == 0 and np.ndim(y) == 0
        coords = np.array([np.atleast_1d(y) / self.dx, np.atleast_1d(x) / self.dx])
        out = ndimage.map_coordinates(self.h, coords, order=1, mode="nearest")
        return float(out[0]) if scalar else out

    def downsample(self, k: int) -> "Terrain":
        """Gaussian-filter then decimate by integer k (grid stays registered)."""
        if k == 1:
            return Terrain(self.h.copy(), self.dx, self.name + "-map")
        hs = ndimage.gaussian_filter(self.h, sigma=k / 2.0, mode="nearest")
        return Terrain(hs[::k, ::k].copy(), self.dx * k, self.name + "-map")

    def crop(self, x0: float, y0: float, width: float, height: float) -> "Terrain":
        i0, j0 = int(round(x0 / self.dx)), int(round(y0 / self.dx))
        i1, j1 = i0 + int(round(width / self.dx)), j0 + int(round(height / self.dx))
        return Terrain(self.h[j0:j1, i0:i1].copy(), self.dx, self.name + "-crop")

    # ------------------------------------------------------------- hazard maps
    def slope_deg(self) -> np.ndarray:
        gy, gx = np.gradient(self.h, self.dx)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def roughness(self, window_m: float = 30.0) -> np.ndarray:
        """Local RMS of height after removing the window-mean [m]."""
        w = max(3, int(round(window_m / self.dx)))
        hp = self.h - ndimage.uniform_filter(self.h, w, mode="nearest")
        return np.sqrt(ndimage.uniform_filter(hp ** 2, w, mode="nearest"))

    def texture(self, window_m: float = 300.0) -> np.ndarray:
        """Local relief (std of height) over a TRN-patch-sized window [m].

        Low values = featureless plains where terrain matching is ambiguous.
        """
        w = max(3, int(round(window_m / self.dx)))
        m = ndimage.uniform_filter(self.h, w, mode="nearest")
        m2 = ndimage.uniform_filter(self.h ** 2, w, mode="nearest")
        return np.sqrt(np.maximum(m2 - m ** 2, 0.0))

    def safe_map(self, slope_max_deg: float = 10.0, rough_max_m: float = 0.6,
                 footprint_radius_m: float = 15.0) -> np.ndarray:
        """Boolean map of cells where a lander footprint is safe."""
        ok = (self.slope_deg() <= slope_max_deg) & (self.roughness() <= rough_max_m)
        r = max(1, int(round(footprint_radius_m / self.dx)))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r ** 2
        return ndimage.binary_erosion(ok, structure=disk, border_value=0)


# ====================================================================== loaders
def load_dem(path: str | Path, dx: float | None = None, name: str | None = None) -> Terrain:
    """Load a DEM as a Terrain.

    Supports ``.npy`` (2-D float array, give ``dx``) and anything GDAL/rasterio
    can read (GeoTIFF, PDS3 ``.IMG`` with ``.LBL``...). For rasters in a
    projected CRS the pixel size is taken from the file unless ``dx`` is given.
    Rows are flipped so that +y is north.
    """
    path = Path(path)
    if path.suffix == ".npy":
        if dx is None:
            raise ValueError(".npy DEMs need an explicit dx (metres per cell)")
        return Terrain(np.load(path).astype(float), float(dx), name or path.stem)
    try:
        import rasterio  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading GeoTIFF/IMG needs `pip install rasterio`") from e
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        px = abs(src.transform.a)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    if np.isnan(arr).any():
        arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    return Terrain(arr[::-1].copy(), float(dx or px), name or path.stem)


def _craters(n: int, size_m: float, rng: np.random.Generator):
    """Diameters from a cumulative power law N(>D) ~ D^-2 between 40 m and 700 m."""
    dmin, dmax = 40.0, 700.0
    u = rng.uniform(size=n)
    d = (dmin ** -2 - u * (dmin ** -2 - dmax ** -2)) ** -0.5
    cx = rng.uniform(0, size_m, n)
    cy = rng.uniform(0, size_m, n)
    return d, cx, cy


def synthetic_lunar(n: int = 512, dx: float = 10.0, seed: int = 0,
                    relief_rms_m: float = 18.0, hurst: float = 0.85,
                    n_craters: int = 70) -> Terrain:
    """Fractal highlands + power-law crater population (fallback when no real DEM).

    This is a *stand-in* so the pipeline runs without downloads. Results on it
    must be re-validated on real LOLA/SLDEM tiles before they are reported.
    """
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(n, d=dx)
    k = np.hypot(*np.meshgrid(kx, kx))
    k[0, 0] = 1.0
    amp = k ** (-(hurst + 1.0))
    amp[0, 0] = 0.0
    h = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * amp))
    h *= relief_rms_m / h.std()

    size = n * dx
    xs = np.arange(n) * dx
    d_all, cx_all, cy_all = _craters(n_craters, size, rng)
    for d, cx, cy in zip(d_all, cx_all, cy_all):
        R = d / 2.0
        depth = 0.18 * d if d < 300 else 0.12 * d     # simple/complex depth-diameter
        i0, i1 = np.searchsorted(xs, [cx - 3 * R, cx + 3 * R])
        j0, j1 = np.searchsorted(xs, [cy - 3 * R, cy + 3 * R])
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1], xs[j0:j1])
        r = np.hypot(X - cx, Y - cy)
        bowl = -depth * (1 - (r / R) ** 2)
        rim = 0.04 * d * (R / np.maximum(r, R)) ** 3
        h[j0:j1, i0:i1] += np.where(r < R, bowl + rim * 0, 0.0) + np.where(r >= R, rim, 0.0)
    return Terrain(h - h.mean(), dx, f"synthetic-lunar-s{seed}")
