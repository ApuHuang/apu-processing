"""校色：不靠星表，用「很多顆星的平均顏色」白平衡。

17 組參考成品用星表校色後，星點光通量比的中位數 R/G 1.07、B/G 1.00（每張差約 ±0.1，tools/analysis/star_color.py）。
所以量出影像裡星點的平均顏色，求三色倍率把它對到這個值，就近似用星表校色的結果。倍率只乘在天空以上的訊號，天空不動。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .stretch import StretchNoise, measure

TARGET_RG = 1.07
TARGET_BG = 1.00


@dataclass(frozen=True)
class ColorBalance:
    gains: tuple[float, float, float]
    star_count: int
    star_ratios: tuple[float, float]   # 校正前量到的 R/G、B/G


def star_colors(image: np.ndarray, max_stars: int = 3000) -> tuple[float, float, int]:
    """星點的 R/G、B/G 光通量比（中位數）。星點＝8 px 以下的結構超過 8σ、面積 4–400 px、沒飽和。"""
    from .denoise import starlet

    L = image.mean(0)
    Lt = L - starlet(L, 3)[1]           # 8 px 以下的結構（比形態學 opening 快很多）
    sig = 1.4826 * float(np.median(np.abs(Lt - np.median(Lt))))
    lab, n = ndimage.label(Lt > 8 * max(sig, 1e-12))
    if n == 0:
        return float("nan"), float("nan"), 0
    idx = np.arange(1, n + 1)
    area = ndimage.sum_labels(np.ones_like(L), lab, idx)
    peak = np.zeros(n + 1, np.float32)
    on = lab > 0
    np.maximum.at(peak, lab[on], L[on])      # 只看星點範圍內的像素（ndimage.maximum 會排序整張圖）
    peak = peak[1:]
    sat = float(np.quantile(L[:: max(1, L.shape[0] // 1024), :: max(1, L.shape[1] // 1024)], 0.9995))
    flux = [ndimage.sum_labels(c - starlet(c, 3)[1], lab, idx) for c in image]
    ok = (area >= 4) & (area <= 400) & (peak < sat) & (flux[1] > 0) & (flux[0] > 0) & (flux[2] > 0)
    if ok.sum() < 20:
        return float("nan"), float("nan"), int(ok.sum())
    order = np.argsort(-flux[1][ok])[:max_stars]
    r = (flux[0][ok] / flux[1][ok])[order]
    b = (flux[2][ok] / flux[1][ok])[order]
    return float(np.median(r)), float(np.median(b)), int(ok.sum())


def measure_balance(image: np.ndarray) -> ColorBalance:
    if image.ndim == 2:
        return ColorBalance((1.0, 1.0, 1.0), 0, (1.0, 1.0))
    r, b, n = star_colors(image)
    if not (np.isfinite(r) and np.isfinite(b)) or r <= 0 or b <= 0:
        return ColorBalance((1.0, 1.0, 1.0), n, (r, b))
    g = np.array([TARGET_RG / r, 1.0, TARGET_BG / b])
    g /= g.mean()
    return ColorBalance(tuple(float(x) for x in g), n, (r, b))


def apply(image: np.ndarray, balance: ColorBalance, noise: StretchNoise | None = None) -> np.ndarray:
    """(x − 天空)·倍率 + 天空。

    原始檔裡已經飽和的星核（三色都頂到上限）乘上不同倍率會變色（例如紅 ×0.67、藍 ×1.56 → 青色星核），
    所以飽和的地方改回中性：三色都用倍率後的最大值，邊緣平滑過渡。
    """
    if image.ndim == 2:
        return image
    noise = noise or measure(image)
    sky = np.asarray(noise.sky, np.float32)[:, None, None]
    gains = np.asarray(balance.gains, np.float32)[:, None, None]
    out = ((image - sky) * gains + sky).astype(np.float32)
    top = float(image.max())
    near = image.max(axis=0) >= 0.97 * top
    if top > 0 and near.mean() < 0.01 and near.any():
        soft = ndimage.gaussian_filter(ndimage.binary_dilation(near, iterations=2).astype(np.float32), 1.5)
        neutral = out.max(axis=0, keepdims=True)
        out = out + soft[None] * (neutral - out)
    return out
