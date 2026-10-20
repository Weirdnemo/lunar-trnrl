import sys
sys.path.insert(0, ".")
import numpy as np
from trnrl.dem_source import BigDEM
from trnrl.windows import window_cfg, sample_sites


def _big(tmp_path):
    from trnrl.terrain import synthetic_lunar
    h = synthetic_lunar(3200, 10.0, seed=3).h.astype("float32")   # 32 km (test region 9.6 km >= 8 km), row 0 = south
    p = tmp_path / "big.npy"
    np.save(p, h)
    return BigDEM(p, dx=10.0)


def test_window_shape_and_orientation(tmp_path):
    d = _big(tmp_path)
    h, info = d.window(0.5, 0.5)
    assert h.shape == (800, 800) and info["nan_frac"] == 0.0
    full = np.load(d.path)
    r0 = full.shape[0] - info["row0"] - 800          # north-up window row -> south-up array row
    assert np.allclose(h, full[r0:r0 + 800, info["col0"]:info["col0"] + 800], atol=1e-4)


def test_train_test_windows_do_not_overlap(tmp_path):
    d = _big(tmp_path)
    cut = int(0.7 * d.nx)
    rng = np.random.default_rng(0)
    for _ in range(50):
        u, v = d.random_centre("train", rng)
        assert (u * d.nx + d.n_nat / 2) <= cut + 1
        u, v = d.random_centre("test", rng)
        assert (u * d.nx - d.n_nat / 2) >= cut - 1


def test_sample_sites_have_targets(tmp_path):
    d = _big(tmp_path)
    sites, _ = sample_sites(d, "train", 2, 0, window_cfg())
    assert len(sites) == 2 and all(len(s.targets) >= 20 for s in sites)
