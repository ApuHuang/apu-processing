"""去光參數掃描（格子層級，幾秒一輪）：原始與參考去光的 32 px 格子各算一次存起來，改參數只重擬曲面。"""

import itertools
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

from apu_processing import background, stretch
from apu_processing.imageio import load_image
from tools.scorecard.data import Sample, splits

CACHE = Path(__file__).resolve().parents[2] / "output/analysis/gradient/cells.npz"


def build():
    data = {}
    for k in splits()["tune"] + splits()["holdout"]:
        s = Sample(k)
        raw = s.load_raw()
        ref_gradient = load_image(s.stage_path("gradient"))[0]
        data[f"{k}_raw"] = np.stack([background._cells(p) for p in raw])
        data[f"{k}_gc"] = np.stack([background._cells(p) for p in ref_gradient])
        data[f"{k}_sigma"] = np.array(stretch.measure(ref_gradient).sigma)
        print(k, flush=True)
    np.savez_compressed(CACHE, **data)


def surface_cells(cells, settings):
    """background.model 的格子版（同一套步驟，只是不展開到全尺寸）。"""
    C, rows, cols = cells.shape
    L = cells.mean(0)
    valid = np.ones((rows, cols), bool)
    my, mx = max(1, rows // 50), max(1, cols // 50)
    valid[:my] = valid[-my:] = False
    valid[:, :mx] = valid[:, -mx:] = False
    mask = valid & (L <= np.quantile(L[valid], 0.5))
    yy, xx = np.meshgrid(background._coords(rows), background._coords(cols), indexing="ij")
    for _ in range(settings.iterations):
        coef = background._fit(L, mask, settings.degree)
        resid = L - legendre.legval2d(yy, xx, coef)
        s = background._robust_sigma(resid[mask])
        new = valid & (resid < settings.clip_high * s) & (resid > -settings.clip_low * s)
        if new.sum() < 12 or np.array_equal(new, mask):
            break
        mask = new
    return np.stack([legendre.legval2d(yy, xx, background._fit(cells[c], mask, settings.degree)) for c in range(C)]), valid


def score(z, key, settings):
    raw, ref_gradient, sig = z[f"{key}_raw"], z[f"{key}_gc"], float(z[f"{key}_sigma"])
    surf, valid = surface_cells(raw, settings)
    d = (raw - surf) - ref_gradient
    d -= np.median(d[:, valid], axis=1)[:, None, None]
    d /= sig
    lum = d.mean(0)[valid]
    chroma = np.abs(d - d.mean(0, keepdims=True))[:, valid]
    return float(np.quantile(lum, 0.99)), float(-np.quantile(lum, 0.01)), float(np.quantile(chroma, 0.99))


def main():
    if not CACHE.exists():
        build()
    z = np.load(CACHE)
    g = splits()
    rows = []
    for degree, hi, lo in itertools.product((1, 2, 3, 4, 5), (1.0, 1.5, 2.0, 3.0), (3.0,)):
        st = background.BackgroundSettings(degree=degree, clip_high=hi, clip_low=lo)
        res = {k: score(z, k, st) for k in g["tune"] + g["holdout"]}
        def agg(keys):
            a = np.array([res[k] for k in keys])
            # 綜合：殘留、過扣、色偏三者的中位數和，外加最差一張的 1/3
            return float(np.median(a.sum(1)) + a.sum(1).max() / 3), np.median(a, 0)
        t, tm = agg(g["tune"])
        h, hm = agg(g["holdout"])
        rows.append((t, degree, hi, lo, tm, h, hm))
    rows.sort()
    for t, degree, hi, lo, tm, h, hm in rows[:12]:
        print(f"degree {degree} clip_high {hi:.1f}  調參 {t:.2f}（殘留 {tm[0]:.2f} 過扣 {tm[1]:.2f} 色偏 {tm[2]:.2f}）"
              f"  驗收 {h:.2f}（殘留 {hm[0]:.2f} 過扣 {hm[1]:.2f} 色偏 {hm[2]:.2f}）")


if __name__ == "__main__":
    main()
