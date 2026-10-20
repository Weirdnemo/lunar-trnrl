"""Big-DEM access: overview, 8 km windows at the simulator's 10 m cell size, and a
geographic train/test split (left part of the map = train, right part = test).

Works with GeoTIFFs (via rasterio, read lazily so a 1 GB file is fine) and with
.npy height grids (give dx; row 0 = south, as written by scripts/crop_dem.py).
Window arrays use the project's convention: row 0 = south, +x east.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SIZE_M = 8000.0
OUT_DX = 10.0


class BigDEM:
    def __init__(self, path, dx: float | None = None):
        self.path = Path(path)
        self._ds = None
        self._arr = None
        if self.path.suffix == ".npy":
            if dx is None:
                raise ValueError(".npy sources need dx (metres per cell)")
            a = np.load(self.path, mmap_mode="r")
            self._arr = a[::-1]                       # north-up internally
            self.dx = float(dx)
            self.ny, self.nx = a.shape
        else:
            import rasterio
            self._ds = rasterio.open(self.path)
            self.dx = float(dx or abs(self._ds.transform.a))
            if self.dx < 0.5:
                raise ValueError("raster looks geographic (degrees); use a projected, metre-based DEM")
            self.ny, self.nx = self._ds.height, self._ds.width
        self.n_nat = int(round(SIZE_M / self.dx))
        if self.n_nat > min(self.nx, self.ny):
            raise ValueError(f"map is {self.nx*self.dx/1000:.1f} x {self.ny*self.dx/1000:.1f} km; "
                             f"need at least {SIZE_M/1000:.0f} km each way")

    # -------------------------------------------------------------- reading
    def _read(self, row0, col0, nrows, ncols, out_shape):
        """North-up float32 block with NaN for no-data, resampled to out_shape (area average)."""
        if self._ds is not None:
            from rasterio.enums import Resampling
            from rasterio.windows import Window
            a = self._ds.read(1, window=Window(col0, row0, ncols, nrows), out_shape=out_shape,
                              resampling=Resampling.average).astype("float32")
            nd = self._ds.nodata
            if nd is not None:
                a[a == nd] = np.nan
            a[np.abs(a) > 1e7] = np.nan
            return a
        blk = np.asarray(self._arr[row0:row0 + nrows, col0:col0 + ncols], dtype="float32")
        fy, fx = max(nrows // out_shape[0], 1), max(ncols // out_shape[1], 1)
        ny, nx = (blk.shape[0] // fy) * fy, (blk.shape[1] // fx) * fx
        return blk[:ny, :nx].reshape(ny // fy, fy, nx // fx, fx).mean(axis=(1, 3))

    def overview(self, max_px: int = 900):
        """(north-up array, metres per overview pixel)."""
        f = max(self.nx, self.ny) / max_px
        shape = (max(int(self.ny / f), 1), max(int(self.nx / f), 1))
        a = self._read(0, 0, self.ny, self.nx, shape)
        return a, self.dx * f

    def window(self, u: float, v: float):
        """8 km x 8 km window centred near fractional position (u east, v from north).
        Returns (h [south-up, 10 m cells, NaN filled], info)."""
        n = self.n_nat
        col0 = int(np.clip(round(u * self.nx - n / 2), 0, self.nx - n))
        row0 = int(np.clip(round(v * self.ny - n / 2), 0, self.ny - n))
        out_n = int(round(SIZE_M / OUT_DX))
        a = self._read(row0, col0, n, n, (out_n, out_n))
        nan_frac = float(np.isnan(a).mean())
        if nan_frac > 0:
            a = np.where(np.isnan(a), np.nanmean(a) if nan_frac < 1 else 0.0, a)
        info = {"col0": col0, "row0": row0, "nan_frac": nan_frac,
                "u": (col0 + n / 2) / self.nx, "v": (row0 + n / 2) / self.ny}
        return a[::-1].astype("float32"), info

    # ---------------------------------------------------------------- split
    def region_cols(self, split: str, frac: float = 0.7):
        if frac >= 1.0:                      # no geographic split: use the whole map
            return (0, self.nx)
        cut = int(frac * self.nx)
        return (0, cut) if split == "train" else (cut, self.nx)

    def region_ok(self, split: str, frac: float = 0.7) -> bool:
        lo, hi = self.region_cols(split, frac)
        return hi - lo >= self.n_nat

    def random_centre(self, split: str, rng, frac: float = 0.7):
        lo, hi = self.region_cols(split, frac)
        col0 = int(rng.integers(lo, hi - self.n_nat + 1))
        row0 = int(rng.integers(0, self.ny - self.n_nat + 1))
        return (col0 + self.n_nat / 2) / self.nx, (row0 + self.n_nat / 2) / self.ny
