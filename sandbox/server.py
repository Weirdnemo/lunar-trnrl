"""Landing sandbox: pick any 8 x 8 km window of a big DEM and watch landers fly it in 3D.

    python sandbox/server.py --dem Site04_final_adj_5mpp_surf.tif [--model results/ppo_win_cov_s0.zip --variant cov]
    then open http://localhost:8000

Runs the project's simulator on your machine (the DEM never leaves it). Policies shown on the same
window, same start and same random seed: baseline guidance, hover-and-wait rule, and (if --model is
given) the PPO residual policy. Uses only the Python standard library plus numpy/rasterio/Pillow/SB3.
"""
import argparse, base64, io, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from trnrl.dem_source import BigDEM, SIZE_M, OUT_DX
from trnrl.env import LunarLandingEnv
from trnrl.guidance import PolyGuidance, WaitGuidance
from trnrl.windows import window_cfg, site_from_height

SPLIT_FRAC = 0.7
MESH = 4                      # 10 m cells -> 40 m mesh cells for drawing
LOCK = threading.Lock()
STATE = {}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def block(a, k, how="mean"):
    n = (a.shape[0] // k) * k
    r = a[:n, :n].reshape(n // k, k, n // k, k)
    return r.mean(axis=(1, 3)) if how == "mean" else r


def hillshade_png(a, mpp):
    from PIL import Image
    z = np.nan_to_num(a, nan=float(np.nanmean(a)))
    gy, gx = np.gradient(z, mpp)
    shade = np.clip(0.55 + 0.9 * (-gx * 0.7 + gy * 0.7) / (1 + np.hypot(gx, gy)), 0, 1)
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    base = 0.35 + 0.5 * t
    g = np.clip(base * (0.45 + 0.8 * shade), 0, 1)
    rgb = np.stack([g * 0.97, g * 0.98, g], axis=-1)
    im = Image.fromarray((rgb * 255).astype("uint8"))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return buf.getvalue(), im.size


def get_site(u, v):
    dem = STATE["dem"]
    h, info = dem.window(u, v)
    key = (info["col0"], info["row0"])
    cache = STATE["sites"]
    if key not in cache:
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        cache[key] = site_from_height(h, STATE["cfg"], f"win{key}")
    return cache[key], info


def region_of(info):
    dem = STATE["dem"]
    if SPLIT_FRAC >= 1.0:
        return "all"
    cut = int(SPLIT_FRAC * dem.nx)
    if info["col0"] + dem.n_nat <= cut:
        return "train"
    if info["col0"] >= cut:
        return "held-out"
    return "straddles"


def api_terrain(q):
    u, v = float(q["u"][0]), float(q["v"][0])
    site, info = get_site(u, v)
    h = site.truth.h
    hm = block(h, MESH).astype("float32")
    safe = block(site.truth_ok.astype("float32"), MESH).astype("float32")
    tex = site.map_tex                                  # 20 m grid
    tm = block(tex, 2).astype("float32")
    tm = np.clip(tm / max(float(np.percentile(tm, 95)), 1e-6), 0, 1)
    n = hm.shape[0]
    return {"n": n, "cell": OUT_DX * MESH, "h": b64(hm), "safe": b64((safe * 255).astype("uint8")),
            "match": b64((tm[:n, :n] * 255).astype("uint8")), "zmin": float(h.min()), "zmax": float(h.max()),
            "targets": int(len(site.targets)), "region": region_of(info), "u": info["u"], "v": info["v"],
            "extent": float(site.truth.extent[0]), "nan_frac": info["nan_frac"]}


def make_policy(name):
    cfg = window_cfg()
    if name == "baseline":
        return cfg, (lambda site: LunarLandingEnv([site], cfg, seed=0)), PolyGuidance().act
    if name == "wait":
        w = WaitGuidance(max_wait=70.0)
        return cfg, (lambda site: LunarLandingEnv([site], cfg, seed=0)), w.act
    if name == "rl":
        from trnrl.residual import ResidualEnv
        cfg.obs_covariance = STATE["variant"] == "cov"
        model = STATE["model"]
        return cfg, (lambda site: ResidualEnv(LunarLandingEnv([site], cfg, seed=0), k=0.5)), \
            (lambda e: model.predict(e.last_obs, deterministic=True)[0])
    raise ValueError(name)


def fly(site, name, seed):
    cfg, mk, act = make_policy(name)
    env = mk(site)
    env.reset(seed=seed, options={"site": 0})
    done = False
    while not done:
        _, _, te, tr, info = env.step(act(env))
        done = te or tr
    inner = getattr(env, "inner", env)
    h = inner.history
    r, rh = np.array(h["r"]), np.array(h["r_hat"])
    trn = h["trn"]
    return {"name": name, "outcome": info["outcome"], "landing_err": float(info.get("landing_err", np.nan)),
            "fuel": float(info["fuel_used"]), "t_end": float(info["t"]),
            "t": [round(float(x), 2) for x in h["t"]], "r": np.round(r, 1).tolist(), "r_hat": np.round(rh, 1).tolist(),
            "sigma_h": np.round(h["sigma_h"], 1).tolist(), "mass": np.round(h["mass"], 1).tolist(),
            "fix": [1 if x == "fix" else 0 for x in trn], "fixes": int(sum(x == "fix" for x in trn)),
            "target": np.round(inner.target, 1).tolist(), "vz": float(info.get("touch_vz", np.nan)),
            "vh": float(info.get("touch_vh", np.nan))}


def api_run(q):
    u, v = float(q["u"][0]), float(q["v"][0])
    seed = int(q.get("seed", ["1"])[0])
    names = [n for n in q.get("policies", ["baseline,wait"])[0].split(",") if n]
    site, info = get_site(u, v)
    out = []
    for n in names:
        if n == "rl" and STATE.get("model") is None:
            continue
        out.append(fly(site, n, seed))
    return {"seed": seed, "runs": out, "m_dry": 1500.0}


def api_info(_q):
    d = STATE["dem"]
    a, mpp = STATE["overview"]
    return {"name": d.path.name, "nx": d.nx, "ny": d.ny, "dx": d.dx, "km_x": d.nx * d.dx / 1000, "km_y": d.ny * d.dx / 1000,
            "overview_w": a.shape[1], "overview_h": a.shape[0], "mpp": mpp, "window_px": SIZE_M / mpp,
            "split": SPLIT_FRAC, "rl": STATE.get("model") is not None, "rl_variant": STATE.get("variant"),
            "rl_name": STATE.get("model_name")}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._send(200, (ROOT / "sandbox" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/vendor/three.min.js":
                return self._send(200, (ROOT / "sandbox" / "vendor" / "three.min.js").read_bytes(), "application/javascript")
            if u.path == "/api/overview.png":
                return self._send(200, STATE["png"], "image/png")
            fn = {"/api/info": api_info, "/api/terrain": api_terrain, "/api/run": api_run}.get(u.path)
            if fn is None:
                return self._send(404, b"not found", "text/plain")
            with LOCK:
                body = json.dumps(fn(q)).encode()
            return self._send(200, body, "application/json")
        except Exception as e:  # report to the page instead of dropping the connection
            import traceback; traceback.print_exc()
            return self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dem", required=True)
    p.add_argument("--dx", type=float, default=None, help="metres per cell (needed for .npy)")
    p.add_argument("--model", default=None, help="trained PPO .zip (from train_ppo_windows.py)")
    p.add_argument("--variant", default="cov", choices=["cov", "nocov"])
    p.add_argument("--frac", type=float, default=0.7, help="train/held-out column split; 1.0 = no split")
    p.add_argument("--port", type=int, default=8000)
    o = p.parse_args()
    global SPLIT_FRAC
    SPLIT_FRAC = o.frac
    dem = BigDEM(o.dem, dx=o.dx or (10.0 if o.dem.endswith(".npy") else None))
    STATE.update(dem=dem, cfg=window_cfg(), sites={}, variant=o.variant)
    a, mpp = dem.overview(900)
    STATE["overview"] = (a, mpp)
    STATE["png"], _ = hillshade_png(a, mpp)
    if o.model:
        from stable_baselines3 import PPO
        STATE["model"] = PPO.load(o.model, device="cpu")
        STATE["model_name"] = Path(o.model).name
    print(f"map {dem.nx*dem.dx/1000:.1f} x {dem.ny*dem.dx/1000:.1f} km; RL model: {STATE.get('model_name', 'none')}")
    print(f"open http://localhost:{o.port}")
    ThreadingHTTPServer(("127.0.0.1", o.port), H).serve_forever()


if __name__ == "__main__":
    main()
