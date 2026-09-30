"""校色對參考校色的比對：我們（去光＋星點白平衡）後的星色與星雲顏色，跟參考去光→參考校色比。"""
import numpy as np

from apu_processing import background, color
from apu_processing.imageio import load_image
from tools.analysis.star_color import star_ratios
from tools.scorecard.data import Sample, splits

errs = []
for k in splits()["tune"] + splits()["holdout"]:
    s = Sample(k)
    raw = s.load_raw()
    ours = background.correct(raw)
    bal = color.measure_balance(ours)
    ours = color.apply(ours, bal)
    color_path = s.stage_path("color")
    if not color_path:
        print(f"{k:12} 沒有參考校色（倍率 {np.round(bal.gains, 3)}）")
        continue
    ref_color = load_image(color_path)[0]
    # 兩者都扣天空後，比「亮結構」的顏色（非星像素的色比），代表星雲顏色
    def neb_color(img):
        sky = np.array([np.quantile(c[::9, ::9], 0.2) for c in img])
        x = img - sky[:, None, None]
        L = x.mean(0)
        sel = (L > np.quantile(L, 0.95)) & (L < np.quantile(L, 0.999))
        tot = x[:, sel].sum(1)
        return tot / tot[1]
    oc, sc = neb_color(ours), neb_color(ref_color)
    e = np.abs(np.log(oc / sc))
    errs.append(e)
    print(f"{k:12} 倍率 {np.round(bal.gains, 2)}  星雲色 R/G 我們 {oc[0]:.2f} 參考校色 {sc[0]:.2f}  B/G 我們 {oc[2]:.2f} 參考校色 {sc[2]:.2f}"
          f"  誤差 {100 * e[0]:.0f}% / {100 * e[2]:.0f}%", flush=True)
a = np.array(errs)
print(f"中位數誤差 R/G {100 * np.median(a[:, 0]):.0f}%  B/G {100 * np.median(a[:, 2]):.0f}%   最差 {100 * a[:, 0].max():.0f}% / {100 * a[:, 2].max():.0f}%")
