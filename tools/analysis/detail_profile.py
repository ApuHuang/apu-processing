"""參考細節做了什麼：參考校色後 → 參考細節後。

- 星點：在參考細節前的影像找沒飽和、孤立的星，兩張各量二階矩半高寬（FWHM），比例＝縮星程度；光通量比＝有沒有保住星的亮度
- 星雲：同一組拉伸參數轉顯示值、補掉星點，明亮星雲各尺度的振幅比（後／前），> 1 表示銳化
"""

import sys

import numpy as np
from scipy import ndimage

from apu_processing import stretch
from apu_processing.imageio import load_image
from tools.analysis.denoise_profile import band_vectors
from tools.scorecard.__main__ import load_regions
from tools.scorecard.data import Sample
from tools.scorecard.metrics import BAND_LABELS, sample_indices

KEYS = ["bubble", "ic1805", "ic2177", "ic405", "ic434", "lbn576", "m33", "m45", "m8", "ngc1893", "ngc2244",
        "ngc6960", "sh2_308", "vdb152", "wr134"]


def find_stars(L, n_max=400):
    top = L - ndimage.grey_opening(L, size=(9, 9))
    sig = 1.4826 * np.median(np.abs(top - np.median(top)))
    lab, n = ndimage.label(top > 15 * sig)
    idx = np.arange(1, n + 1)
    peak = ndimage.maximum(L, lab, idx)
    area = ndimage.sum_labels(np.ones_like(L), lab, idx)
    com = ndimage.center_of_mass(top, lab, idx)
    sat = np.quantile(L, 0.9995)
    stars = [(int(round(y)), int(round(x))) for (y, x), p, a in zip(com, peak, area) if p < sat and 5 <= a <= 150]
    # 孤立：12 px 內沒有別的星
    pts = np.array(stars)
    keep = []
    for i, (y, x) in enumerate(stars):
        d = np.hypot(pts[:, 0] - y, pts[:, 1] - x)
        if (d < 12).sum() == 1:
            keep.append((y, x))
    return keep[:n_max]


def moments(L, y, x, r=7):
    h, w = L.shape
    if y < r + 3 or x < r + 3 or y > h - r - 4 or x > w - r - 4:
        return None
    box = L[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
    ring = L[y - r - 3:y + r + 4, x - r - 3:x + r + 4]
    bg = np.median(np.r_[ring[:3].ravel(), ring[-3:].ravel(), ring[:, :3].ravel(), ring[:, -3:].ravel()])
    f = np.clip(box - bg, 0, None)
    tot = f.sum()
    if tot <= 0:
        return None
    yy, xx = np.indices(f.shape)
    cy, cx = (f * yy).sum() / tot, (f * xx).sum() / tot
    var = (f * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / tot / 2
    return 2.3548 * np.sqrt(var), tot


def main():
    keys = sys.argv[1:] or KEYS
    rows = []
    for k in keys:
        s = Sample(k)
        before = load_image(s.stage_path("color"))[0]
        after = load_image(s.stage_path("detail"))[0]
        Lb, La = before.mean(0), after.mean(0)
        fw, fl = [], []
        for y, x in find_stars(Lb):
            mb, ma = moments(Lb, y, x), moments(La, y, x)
            if mb and ma and mb[0] > 1:
                fw.append(ma[0] / mb[0])
                fl.append(ma[1] / mb[1])
        fwhm_b = np.median([moments(Lb, y, x)[0] for y, x in find_stars(Lb)[:200] if moments(Lb, y, x)])
        R = load_regions(s, s.reference)
        n = stretch.measure(before)
        idx = sample_indices(R)
        vb = band_vectors(stretch.apply(before, noise=n), R, idx)
        va = band_vectors(stretch.apply(after, noise=n), R, idx)
        amp = [float(np.sqrt((va[0][i]["bright"] ** 2).mean() / (vb[0][i]["bright"] ** 2).mean())) for i in range(6)]
        rows.append((np.median(fw), np.median(fl), amp))
        print(f"{k:10} 星點 {len(fw):3} 顆  FWHM {fwhm_b:4.2f}px → ×{np.median(fw):.2f}  光通量 ×{np.median(fl):.2f}  "
              f"| 亮雲振幅（後/前） " + " ".join(f"{a:.2f}" for a in amp), flush=True)
    a = np.array([r[2] for r in rows])
    print(f"中位數  FWHM ×{np.median([r[0] for r in rows]):.2f}  光通量 ×{np.median([r[1] for r in rows]):.2f}  "
          "亮雲振幅 " + " ".join(f"{b}:{x:.2f}" for b, x in zip(BAND_LABELS, np.median(a, 0))))


if __name__ == "__main__":
    main()
