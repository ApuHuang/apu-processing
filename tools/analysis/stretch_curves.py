"""從參考降噪線性檔 → 參考成品成品，反推每張的色調曲線（亮度，全域部分）。

x 軸試兩種歸一化：
  u = (L − 天空) / σ       σ = 像素噪聲（相鄰像素差的 MAD / √2），分別量參考降噪與原始堆疊
  v = (L − 天空) / (P99.9 − 天空)   訊號範圍
看哪一種讓 17 條曲線最一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from tools.scorecard.__main__ import load_regions
from tools.scorecard.data import Sample, all_samples

OUT = Path(__file__).resolve().parents[2] / "output/analysis"


def pixel_noise(L: np.ndarray, mask: np.ndarray) -> float:
    d = (L[:, 1:] - L[:, :-1])[mask[:, 1:] & mask[:, :-1]]
    d = d[:: max(1, d.size // 400_000)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def curve(sample: Sample) -> dict:
    reference = sample.reference
    R = load_regions(sample, reference)
    lin = sample.load_linear_final()
    raw = sample.load_raw()
    L, Lr, M = lin.mean(0), raw.mean(0), reference.mean(0)
    sky = float(np.median(L[R.sky]))
    sigma_ref = pixel_noise(L, R.sky)
    sigma_raw = pixel_noise(Lr, R.sky)
    top = float(np.quantile(L[R.valid][::5], 0.999))
    sel = R.valid
    x = (L[sel] - sky)
    y = M[sel]
    # 依 u 分箱，取參考成品中位數
    u_edges = np.geomspace(0.25, 5000, 60)
    u = x / sigma_ref
    idx = np.digitize(u, u_edges)
    pts = []
    for b in range(1, len(u_edges)):
        m = idx == b
        if m.sum() >= 200:
            pts.append((float(np.sqrt(u_edges[b - 1] * u_edges[b])), float(np.median(y[m])), int(m.sum())))
    return {
        "key": sample.key, "sky_linear": sky, "sigma_ref": sigma_ref, "sigma_raw": sigma_raw,
        "p999": top, "range_over_sigma": (top - sky) / sigma_ref,
        "ref_sky": R.sky_level, "points_u": pts,
    }


def main() -> None:
    keys = sys.argv[1:] or [s.key for s in all_samples()]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in keys:
        r = curve(Sample(k))
        rows.append(r)
        at = {round(p[0], 1): p[1] for p in r["points_u"]}
        def near(u):
            p = min(r["points_u"], key=lambda q: abs(np.log(q[0] / u)))
            return p[1]
        print(f"{k:12} sky {r['sky_linear']:.4f} σref {r['sigma_ref']:.2e} σraw {r['sigma_raw']:.2e} "
              f"(raw/ref {r['sigma_raw'] / r['sigma_ref']:.2f})  range/σ {r['range_over_sigma']:7.0f}  "
              f"參考成品 sky {r['ref_sky']:.3f} | y@u=1 {near(1):.3f} 3 {near(3):.3f} 10 {near(10):.3f} "
              f"30 {near(30):.3f} 100 {near(100):.3f} 300 {near(300):.3f} 1000 {near(1000):.3f}", flush=True)
    (OUT / "stretch_curves.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
