"""Sites cut from a big DEM: the real-terrain version of make_sites()."""
from __future__ import annotations

import numpy as np

from .dem_source import BigDEM, OUT_DX
from .env import EnvConfig, Site
from .terrain import Terrain


def window_cfg(outage: float = 0.4, decoupled: bool = True,
               init_pos_h: float | None = None, search_cells: int | None = None,
               alt_sync: bool = False, patch_n: int | None = None, oracle: bool = False,
               fov_deg: float | None = None, reg_offset: float = 0.0, map_noise: float = 0.0,
               map_corr: float | None = None) -> EnvConfig:
    """init_pos_h = 1-sigma initial horizontal position error [m]. If it is set and search_cells is not,
    the TRN search window is widened to cover 3 sigma (+2 cells) of the 20 m onboard map."""
    cfg = EnvConfig()
    cfg.p_outage = outage
    if decoupled:
        cfg.nav.alt_mode = "decoupled"
    if init_pos_h is not None:
        cfg.nav.init_pos_h = float(init_pos_h)
        if search_cells is None:
            search_cells = max(cfg.nav.search_max_cells, int(np.ceil(3 * init_pos_h / 20.0)) + 2)
    if search_cells is not None:
        cfg.nav.search_max_cells = int(search_cells)
    cfg.nav.alt_sync = bool(alt_sync)
    if patch_n is not None:
        cfg.nav.patch_n = int(patch_n)
    if fov_deg is not None:
        cfg.nav.fov_deg = float(fov_deg)
    cfg.nav.reg_offset_m = float(reg_offset)
    cfg.map_noise_m = float(map_noise)
    if map_corr is not None:
        cfg.map_noise_corr_m = float(map_corr)
    if oracle:                                    # ceiling experiment: guidance sees the true state, TRN off for speed
        cfg.oracle_nav = True
        cfg.nav.enable_trn = False
    return cfg


def pop_nav_flags(argv: list) -> dict:
    """Remove --init_pos_h X and --search_cells N from argv (so positional args keep working)."""
    out = {}
    for flag, typ, key in (("--init_pos_h", float, "init_pos_h"), ("--search_cells", int, "search_cells"), ("--patch_n", int, "patch_n"),
                           ("--fov", float, "fov_deg"), ("--reg_offset", float, "reg_offset"),
                           ("--map_noise", float, "map_noise"), ("--map_corr", float, "map_corr")):
        if flag in argv:
            i = argv.index(flag); out[key] = typ(argv[i + 1]); del argv[i:i + 2]
    if "--realistic" in argv:                     # assumed (not published) realism preset; each knob is swept separately
        argv.remove("--realistic")
        out.setdefault("fov_deg", 40.0); out.setdefault("reg_offset", 6.0); out.setdefault("map_noise", 1.0)
    if "--alt_sync" in argv:
        argv.remove("--alt_sync"); out["alt_sync"] = True
    if "--oracle" in argv:
        argv.remove("--oracle"); out["oracle"] = True
    return out


def apply_nav_flags(cfg: EnvConfig, f: dict) -> EnvConfig:
    """Apply the pop_nav_flags dict to an existing config (scenario path); only the keys present are touched."""
    n = cfg.nav
    if "init_pos_h" in f:
        n.init_pos_h = float(f["init_pos_h"])
        n.search_max_cells = max(n.search_max_cells, int(np.ceil(3 * n.init_pos_h / 20.0)) + 2)
    if "search_cells" in f: n.search_max_cells = int(f["search_cells"])
    if "patch_n" in f: n.patch_n = int(f["patch_n"])
    if "fov_deg" in f: n.fov_deg = float(f["fov_deg"])
    if "reg_offset" in f: n.reg_offset_m = float(f["reg_offset"])
    if f.get("alt_sync"): n.alt_sync = True
    if "map_noise" in f: cfg.map_noise_m = float(f["map_noise"])
    if "map_corr" in f: cfg.map_noise_corr_m = float(f["map_corr"])
    if f.get("oracle"): cfg.oracle_nav = True; n.enable_trn = False
    return cfg


def site_from_height(h: np.ndarray, cfg: EnvConfig, name: str = "window") -> Site:
    return Site(Terrain(np.asarray(h, dtype=float), OUT_DX, name), cfg)


def sample_sites(dem: BigDEM, split: str, n: int, seed: int, cfg: EnvConfig,
                 frac: float = 0.7, min_targets: int = 20, max_tries: int = 400):
    """n random windows from the train (left) or test (right) part of the map.
    Windows with too many no-data cells or too few safe landing spots are skipped."""
    if not dem.region_ok(split, frac):
        raise ValueError(f"{split} region is narrower than 8 km; change the split fraction")
    rng = np.random.default_rng(seed)
    sites, origins = [], []
    for _ in range(max_tries):
        if len(sites) >= n:
            break
        u, v = dem.random_centre(split, rng, frac)
        h, info = dem.window(u, v)
        if info["nan_frac"] > 0.05:
            continue
        s = site_from_height(h, cfg, f"{split}-{len(sites)}")
        if len(s.targets) >= min_targets:
            sites.append(s)
            origins.append({k: info[k] for k in ("col0", "row0")})
    if len(sites) < n:
        print(f"warning: only {len(sites)} usable {split} windows found in {max_tries} tries")
    return sites, origins
