"""Onboard 'where does terrain matching work' map.

For a coarse grid of locations, take the 15x15-cell patch the lander would match there (from the ONBOARD map),
correlate it (zero-mean normalised cross-correlation) against its +-`search`-cell neighbourhood and record how
strongly the best peak OUTSIDE the main lobe (the connected region within `lobe_drop` of the true peak) competes with it, the same logic as the lander's ambiguity test:
    matchability = 1 - second_peak_ncc   (0 = ambiguous, flat or only constrains one direction; larger = a sharp, unique match)
It uses only the onboard map, so a real lander could precompute it before descent."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve


def matchability_map(h: np.ndarray, patch_n: int = 15, search: int = 40, step: int = 6, lobe_drop: float = 0.10,
                     min_relief: float = 0.3) -> np.ndarray:
    ny, nx = h.shape
    half = patch_n // 2
    lo = half + search
    js = np.arange(lo, ny - lo, step)
    is_ = np.arange(lo, nx - lo, step)
    if len(js) < 2 or len(is_) < 2:
        return np.ones_like(h, dtype=np.float32)
    ones = np.ones((patch_n, patch_n))
    n = patch_n * patch_n
    G = np.zeros((len(js), len(is_)), np.float32)
    for a, j in enumerate(js):
        for b, i in enumerate(is_):
            reg = h[j - lo:j + lo + 1, i - lo:i + lo + 1].astype(np.float64)
            p = reg[search:search + patch_n, search:search + patch_n]
            if p.std() < min_relief:
                continue
            pz = p - p.mean()
            num = fftconvolve(reg, pz[::-1, ::-1], mode="valid")
            s1 = fftconvolve(reg, ones, mode="valid")
            s2 = fftconvolve(reg * reg, ones, mode="valid")
            den = np.sqrt(np.maximum(s2 - s1 ** 2 / n, 1e-9)) * np.sqrt((pz ** 2).sum())
            ncc = num / den
            c = ncc.shape[0] // 2
            lab, _ = ndimage.label(ncc >= 1.0 - lobe_drop)          # main lobe = connected region around the true peak
            lobe = lab == lab[c, c]
            outside = ncc[~lobe]
            sep = 1.0 - (float(outside.max()) if outside.size else 0.0)     # separation from the best competing peak
            r_lobe = float(np.sqrt(lobe.sum() / np.pi))                      # lobe radius [cells]: localisation sharpness
            G[a, b] = float(np.clip(sep / 0.3, 0.0, 1.0) / (1.0 + (r_lobe / 4.0) ** 2))
    f = RegularGridInterpolator((js, is_), G, bounds_error=False, fill_value=None, method="linear")
    jj, ii = np.mgrid[0:ny, 0:nx]
    pts = np.stack([np.clip(jj, js[0], js[-1]).ravel(), np.clip(ii, is_[0], is_[-1]).ravel()], axis=1)
    return np.clip(f(pts).reshape(ny, nx), 0.0, 1.0).astype(np.float32)
