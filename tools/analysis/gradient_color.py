"""參考去光（去光梯度）與參考校色（色彩校正）各做了什麼。

- 梯度：原始 − 參考去光的大尺度（σ=64 px 模糊）範圍，以參考去光圖的 σ₈ 噪聲為單位（拉伸後看不看得到）
- 參考校色：參考校色 ≈ k·參考去光 + b（每色版），k 的比例就是白平衡
"""

import sys

import numpy as np

from apu_processing import stretch
from apu_processing.imageio import load_image
from tools.scorecard.data import Sample, all_samples
from tools.scorecard.metrics import blur


def main():
    keys = sys.argv[1:] or [s.key for s in all_samples()]
    for k in keys:
        s = Sample(k)
        raw = s.load_raw()
        gradient_path, color_path = s.stage_path("gradient"), s.stage_path("color")
        ref_gradient = load_image(gradient_path)[0]
        n = stretch.measure(ref_gradient)
        diff = raw - ref_gradient
        big = np.stack([blur(d, 64) for d in diff])
        h, w = big.shape[1:]
        inner = big[:, h // 20:-h // 20, w // 20:-w // 20]
        span = [(float(np.quantile(c, 0.99) - np.quantile(c, 0.01))) / n.sigma for c in inner]
        off = [float(np.median(c)) for c in diff]
        line = f"{k:12} 梯度幅度（σ₈ 倍）R {span[0]:6.1f} G {span[1]:6.1f} B {span[2]:6.1f}  參考去光減掉的平均 {np.mean(off):.4f}"
        if color_path:
            ref_color = load_image(color_path)[0]
            # 用亮的非飽和像素做 k（過原點）；b 用天空
            L = ref_gradient.mean(0)
            sel = (L > np.quantile(L, 0.9)) & (L < np.quantile(L, 0.999))
            ks = [float(np.median(ref_color[c][sel] / np.maximum(ref_gradient[c][sel], 1e-9))) for c in range(3)]
            line += f"  參考校色 k R/G {ks[0] / ks[1]:.3f} B/G {ks[2] / ks[1]:.3f}"
            ns = stretch.measure(ref_color)
            sky = np.array(ns.sky)
            line += f"  參考校色天空 R:G:B {sky[0] / sky[1]:.2f}:1:{sky[2] / sky[1]:.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
