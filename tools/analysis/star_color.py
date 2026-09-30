"""星點平均顏色：參考校色校正後的星，平均起來是不是接近某個固定顏色（可以拿來取代參考校色）。

對原始與參考校色各量一次：每顆星的各色版光通量（小尺度 top-hat 在星點範圍內加總），取 R/G、B/G 的中位數。
若參考校色後各張都接近同一個值，就用「原始檔的星點平均顏色 → 那個值」求三色倍率，再跟參考校色的倍率比。
"""

import sys

import numpy as np
from scipy import ndimage

from apu_processing.imageio import load_image
from tools.scorecard.data import Sample, all_samples


def star_ratios(img: np.ndarray, max_stars: int = 3000) -> tuple[float, float, int]:
    L = img.mean(0)
    top = [c - ndimage.grey_opening(c, size=(9, 9)) for c in img]
    Lt = L - ndimage.grey_opening(L, size=(9, 9))
    sig = 1.4826 * np.median(np.abs(Lt - np.median(Lt)))
    mask = Lt > 8 * sig
    sat = np.quantile(L, 0.9999)
    lab, n = ndimage.label(mask)
    if n == 0:
        return float("nan"), float("nan"), 0
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(L), lab, idx)
    peak = ndimage.maximum(L, lab, idx)
    flux = [ndimage.sum(t, lab, idx) for t in top]
    ok = (area >= 4) & (area <= 400) & (peak < sat) & (flux[1] > 0)
    order = np.argsort(-flux[1][ok])[:max_stars]
    r = (flux[0][ok] / flux[1][ok])[order]
    b = (flux[2][ok] / flux[1][ok])[order]
    return float(np.median(r)), float(np.median(b)), int(ok.sum())


def main():
    keys = sys.argv[1:] or [s.key for s in all_samples()]
    for k in keys:
        s = Sample(k)
        raw = s.load_raw()
        rr, rb, n = star_ratios(raw)
        line = f"{k:12} 原始星色 R/G {rr:.3f} B/G {rb:.3f} ({n} 顆)"
        color_path = s.stage_path("color")
        if color_path:
            ref_color = load_image(color_path)[0]
            sr, sb, m = star_ratios(ref_color)
            # 參考校色倍率（相對 G），用原始 vs 參考校色的亮像素比
            ref_gradient = load_image(s.stage_path("gradient"))[0]
            Lg = ref_gradient.mean(0)
            sel = (Lg > np.quantile(Lg, 0.9)) & (Lg < np.quantile(Lg, 0.999))
            k_ = [float(np.median(ref_color[c][sel] / np.maximum(ref_gradient[c][sel], 1e-9))) for c in range(3)]
            pred_r, pred_b = sr / rr, sb / rb  # 若星色對上，需要的倍率
            line += (f" | 參考校色後星色 R/G {sr:.3f} B/G {sb:.3f} | 參考校色倍率 R/G {k_[0] / k_[1]:.3f} B/G {k_[2] / k_[1]:.3f}"
                     f"  (星色推得 {pred_r:.3f} / {pred_b:.3f})")
        print(line, flush=True)


if __name__ == "__main__":
    main()
