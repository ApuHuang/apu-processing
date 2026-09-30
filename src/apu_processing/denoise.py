"""降噪：多尺度（starlet）收縮，再跟原圖按比例混合。

依據（tools/analysis/denoise_profile.py，17 組參考成品降噪前後）：
- 像素尺度的噪聲留下約一半、16–32 px 幾乎不動 —— 很像「完全降噪」與原圖各半混合
- 明亮星雲的結構 4 px 以上保留 84–100%；色彩噪聲在 8 px 以下同樣壓一半
所以這裡先做「盡量乾淨但不傷結構」的降噪 D(x)，輸出 = x + amount·(D(x) − x)。
混回一部分原圖會留下自然的細顆粒，D 本身的小瑕疵也減半（v0.9.8 刻意壓大尺度，結果變成斑駁與色塊）。

D(x)：亮度與兩個色差分開做 starlet（B3 樣條 à trous）分解；每個尺度的噪聲 σ_j 在天空量，
亮處噪聲較大（光子噪聲），每個尺度用「區塊亮度 → 係數離散度」擬合 σ² = a + b·亮度 來放大門檻；
係數用非負 garrote 收縮：w·max(0, 1 − (kσ)²/w²)，強的結構幾乎不受影響。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .progress import CancelCheck, check, never_cancel

_POOL = ThreadPoolExecutor(max_workers=3)

B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16
SCALES = 5
TILE = 64


@dataclass(frozen=True)
class DenoiseSettings:
    amount: float = 0.5                                  # 0＝不降噪，1＝完全用降噪結果
    # 每個尺度（1、2、4、8、16、32 px）的門檻，以該尺度的噪聲 σ 為單位
    k_luminance: tuple[float, ...] = (3.0, 2.5, 2.0, 1.5, 1.0)
    k_chroma: tuple[float, ...] = (3.5, 3.0, 2.5, 2.0, 1.5)
    # True：噪聲隨亮度增加的比例只在最小尺度量（那裡幾乎全是噪聲），各尺度只在天空量基準；
    # False：每個尺度各自擬合（大尺度在亮處會把結構當噪聲，門檻變高、結構被壓掉）
    shared_shape: bool = True


def _atrous_axis(x: np.ndarray, step: int, axis: int) -> np.ndarray:
    """B3 樣條帶洞平滑的一個方向：只加 5 個位移後的影像（長卷積核大部分是 0，直接卷積很浪費）。"""
    n = x.shape[axis]
    pad = [(0, 0), (0, 0)]
    pad[axis] = (2 * step, 2 * step)
    xp = np.pad(x, pad, mode="symmetric") if 2 * step < n else np.pad(x, pad, mode="edge")
    out = None
    for k, wt in enumerate(B3):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(k * step, k * step + n)
        term = xp[tuple(sl)] * wt
        out = term if out is None else out + term
    return out.astype(np.float32, copy=False)


def _atrous(plane: np.ndarray, j: int) -> np.ndarray:
    step = 2 ** j
    return _atrous_axis(_atrous_axis(plane, step, 0), step, 1)


def starlet(plane: np.ndarray, scales: int = SCALES) -> tuple[list[np.ndarray], np.ndarray]:
    coeffs = []
    c = plane
    for j in range(scales):
        smooth = _atrous(c, j)
        coeffs.append(c - smooth)
        c = smooth
    return coeffs, c


def _opponent(image: np.ndarray) -> np.ndarray:
    r, g, b = image
    return np.stack([(r + g + b) / 3, (r - b) / 2, (2 * g - r - b) / 4]).astype(np.float32)


def _from_opponent(opp: np.ndarray) -> np.ndarray:
    L, c1, c2 = opp
    r = L + c1 - 2 * c2 / 3
    b = L - c1 - 2 * c2 / 3
    g = L + 4 * c2 / 3
    return np.stack([r, g, b]).astype(np.float32)


def _tile_stats(values: np.ndarray, level: np.ndarray, tile: int = TILE) -> tuple[np.ndarray, np.ndarray]:
    h, w = values.shape
    th, tw = h // tile, w // tile
    # 每格抽 1/4 的像素（隔點取樣）算中位數，統計上夠用、快 4 倍以上
    v = values[:th * tile:2, :tw * tile:2].reshape(th, tile // 2, tw, tile // 2)
    lv = level[:th * tile:2, :tw * tile:2].reshape(th, tile // 2, tw, tile // 2)
    med = np.median(np.abs(v).transpose(0, 2, 1, 3).reshape(th, tw, -1), axis=2) / 0.6745
    return med.ravel(), np.median(lv.transpose(0, 2, 1, 3).reshape(th, tw, -1), axis=2).ravel()


def noise_model(w1: np.ndarray, level: np.ndarray) -> tuple[float, float, float]:
    """最小尺度係數的離散度 vs 區塊亮度：σ² = a + b·(亮度 − 天空)。回傳 (a, b, 天空)。
    亮的區塊可能有結構，所以每個亮度區間取離散度較低的 25% 分位（純噪聲的區塊）。"""
    sig, lev = _tile_stats(w1, level)
    sky = float(np.quantile(lev, 0.1))
    x = np.maximum(lev - sky, 0)
    bins = np.quantile(x, np.linspace(0, 0.98, 12))
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (x >= lo) & (x <= hi)
        if sel.sum() >= 8:
            xs.append(float(np.median(x[sel])))
            ys.append(float(np.quantile(sig[sel], 0.25)) ** 2)
    xs, ys = np.array(xs), np.array(ys)
    if xs.size >= 3 and np.ptp(xs) > 0:
        b, a = np.polyfit(xs, ys, 1)
    else:
        a, b = float(np.median(sig) ** 2), 0.0
    a = max(float(a), float(ys.min()) if ys.size else 1e-12, 1e-20)
    return a, max(float(b), 0.0), sky


def sky_noise(w: np.ndarray, level: np.ndarray, sky: float) -> float:
    """係數在「天空」區塊（亮度最暗的 1/4）的離散度²：大尺度的噪聲基準只在天空量，亮處的係數混著結構。"""
    sig, lev = _tile_stats(w, level)
    dark = lev <= np.quantile(lev, 0.25)
    return max(float(np.median(sig[dark])) ** 2, 1e-20)


def _shrink(w: np.ndarray, thr: np.ndarray | float) -> np.ndarray:
    w2 = w * w
    return w * np.maximum(0.0, 1.0 - (thr * thr) / np.maximum(w2, 1e-30))


def clean(image: np.ndarray, settings: DenoiseSettings = DenoiseSettings(),
          cancel: CancelCheck = never_cancel) -> np.ndarray:
    """D(x)：盡量乾淨、不傷結構的降噪結果（還沒混回原圖）。"""
    mono = image.ndim == 2
    planes = image[None] if mono else _opponent(image)
    L = planes[0]
    level = _atrous(_atrous(L, 0), 1)  # 局部亮度（約 3 px 平滑）
    def one(p: int) -> np.ndarray:
        ks = settings.k_luminance if p == 0 else settings.k_chroma
        coeffs, residual = starlet(planes[p], len(ks))
        acc = residual
        a0, b0, sky = noise_model(coeffs[0], level)
        shape = np.sqrt(1 + (b0 / a0) * np.maximum(level - sky, 0)).astype(np.float32)
        for j, w in enumerate(coeffs):
            # 每個尺度直接量噪聲（疊圖的噪聲有空間相關，不能用白噪聲的比例換算）
            if settings.shared_shape:
                local = (np.sqrt(a0 if j == 0 else sky_noise(w, level, sky)) * shape).astype(np.float32)
            else:
                a, b, sky_j = noise_model(w, level)
                local = np.sqrt(a + b * np.maximum(level - sky_j, 0)).astype(np.float32)
            acc = acc + _shrink(w, ks[j] * local)
            check(cancel)
        return acc

    out = np.stack(list(_POOL.map(one, range(planes.shape[0]))))
    return out[0] if mono else _from_opponent(out)


def reduce(image: np.ndarray, settings: DenoiseSettings = DenoiseSettings(),
           cancel: CancelCheck = never_cancel) -> np.ndarray:
    if settings.amount <= 0:
        return image
    d = clean(image, settings, cancel)
    return (image + np.float32(settings.amount) * (d - image)).astype(np.float32)
