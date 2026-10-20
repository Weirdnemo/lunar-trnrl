"""Named benchmark scenarios. 'train' and 'test' splits use disjoint terrain seeds."""
from __future__ import annotations

from .env import EnvConfig, Site, make_sites

SPLIT_SEED0 = {"train": 2000, "test": 1000}
LOW_RELIEF = dict(relief_rms_m=4.0, n_craters=8)


CFG_TWEAK = None      # optional callable(cfg) applied to every freshly made EnvConfig (e.g. realism flags)


def build(name: str, split: str = "test", n_sites: int | None = None):
    """Return (EnvConfig, [Site]) for a named scenario. Suffix '+dec' selects the decoupled altimeter."""
    cfg = EnvConfig()
    if CFG_TWEAK is not None:
        CFG_TWEAK(cfg)
    if name.endswith("+dec"):
        name = name[:-4]
        cfg.nav.alt_mode = "decoupled"
    if name == "real":
        # real LOLA/PGDA tiles (data/site04.npy, data/site23.npy; 10 m cells, see scripts/crop_dem.py).
        # train = Site 04 (Shackleton rim), test = Site 23 (Malapert massif). Same blackout rate as mixed_hard.
        import numpy as np
        from pathlib import Path
        from .terrain import Terrain
        cfg.p_outage = 0.4
        f = {"train": "site04", "test": "site23"}[split]
        p = Path(__file__).resolve().parents[1] / "data" / f"{f}.npy"
        return cfg, [Site(Terrain(np.load(p).astype(float), 10.0, f), cfg)]
    s0 = SPLIT_SEED0[split]
    n = n_sites or (16 if split == "train" else 8)
    if name == "nominal":
        return cfg, make_sites(n, cfg, seed0=s0)
    if name == "no_trn":
        cfg.nav.enable_trn = False
        return cfg, make_sites(n, cfg, seed0=s0)
    if name == "low_relief":
        return cfg, make_sites(n, cfg, seed0=s0, **LOW_RELIEF)
    if name == "big_init":
        cfg.nav.init_pos_h = 300.0
        return cfg, make_sites(n, cfg, seed0=s0)
    if name == "mixed_hard":
        cfg.p_outage = 0.4
        half = n // 2
        sites = make_sites(half, cfg, seed0=s0) + make_sites(n - half, cfg, seed0=s0 + 500, **LOW_RELIEF)
        return cfg, sites
    raise ValueError(name)
