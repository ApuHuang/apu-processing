"""細節與縮星。

依據（tools/analysis/detail_profile.py，15 組參考成品細節處理前後）：星點 FWHM ×0.80、光通量 ×1.06（保住）；
明亮星雲 1–2 px 的振幅 ×1.08–1.15。

v0.9.8 的教訓：每顆星只在半徑 30 px 的方框裡「扣掉再畫小」，大亮星的光暈與繞射芒被方框切斷，
扣一半留一半 → 黑斑、繞射芒斷成虛線。這裡的做法：
1. 星點層 S ＝ 影像 − 星點底下的背景 B（B 用周圍像素平滑補，不是方框），只在星點遮罩內非零
2. 縮星只對 S 做 Richardson-Lucy 反卷積（保住光通量、重疊的星一起處理、沒有接縫）；B 完全不動，所以不會挖出洞
3. 大星（半徑 > 6 px）逐漸減少縮小，> 15 px 完全不動；飽和星也不動
4. 星雲細節：只加強超過噪聲門檻的 1–2 px 係數（garrote），噪聲不會被放大
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import denoise
from .progress import CancelCheck, check, never_cancel

_POOL = ThreadPoolExecutor(max_workers=3)


@dataclass(frozen=True)
class DetailSettings:
    star_reduction: float = 0.5   # 0＝不縮；0.5＝FWHM ×0.8（同參考成品）；1＝×0.6
    sharpen: float = 0.5          # 0＝不加強；0.5＝1–2 px 結構約 +15%
    big_star: float = 6.0         # 半徑超過這個（px）開始減少縮小
    huge_star: float = 15.0       # 半徑超過這個完全不縮

    def shrink_factor(self) -> float:
        return 1.0 - 0.4 * min(1.0, max(0.0, self.star_reduction))


@dataclass
class Stars:
    y: np.ndarray
    x: np.ndarray
    radius: np.ndarray
    saturated: np.ndarray
    mask: np.ndarray        # 星點範圍（weight > 0）
    weight: np.ndarray      # 星點層的權重：星心 1，往外用餘弦平滑降到 0（沒有硬邊，縮星後不會留下暗框）


# ---------------------------------------------------------------------- 工具


def _block_mean(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape
    h2, w2 = h // f * f, w // f * f
    return img[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def _upsample(small: np.ndarray, shape: tuple[int, int], f: int) -> np.ndarray:
    """_block_mean 的反向：可分離的雙線性放大（像素中心對齊）。"""
    def axis_weights(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.clip((np.arange(n_out) + 0.5) / f - 0.5, 0, n_in - 1)
        i0 = np.floor(z).astype(np.intp)
        return i0, np.minimum(i0 + 1, n_in - 1), (z - i0).astype(np.float32)

    h, w = shape
    y0, y1, wy = axis_weights(h, small.shape[0])
    x0, x1, wx = axis_weights(w, small.shape[1])
    rows = small[y0] * (1 - wy)[:, None] + small[y1] * wy[:, None]
    return (rows[:, x0] * (1 - wx) + rows[:, x1] * wx).astype(np.float32)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 6:
        return ndimage.gaussian_filter(img, sigma, truncate=3.0).astype(np.float32)
    f = 2 ** int(math.log2(sigma / 3))
    small = ndimage.gaussian_filter(_block_mean(img, f), math.sqrt(max(0.25, (sigma / f) ** 2 - 1 / 12)), truncate=3.0)
    return _upsample(small.astype(np.float32), img.shape, f)


def inpaint(planes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """遮罩內用周圍的值補（正規化卷積，由小範圍到大範圍）。"""
    out = planes.copy()
    known = (~mask).astype(np.float32)
    remaining = mask.copy()
    for sigma in (3, 10, 30, 90):
        wgt = blur(known, sigma)
        fill_now = remaining & (wgt > 0.05)
        if fill_now.any():
            def one(c: int, sigma: float = sigma, wgt: np.ndarray = wgt, fill_now: np.ndarray = fill_now) -> None:
                val = blur(out[c] * known, sigma) / np.maximum(wgt, 1e-6)
                out[c][fill_now] = val[fill_now]
            list(_POOL.map(one, range(out.shape[0])))
            known[fill_now] = 1.0
            remaining &= ~fill_now
        if not remaining.any():
            break
    for c in range(out.shape[0]):
        if remaining.any():
            out[c][remaining] = np.median(out[c][~mask])
    return out


# ---------------------------------------------------------------------- 找星


def find_stars(L: np.ndarray, threshold: float = 5.0) -> Stars:
    coeffs, smooth = denoise.starlet(L, 3)
    top = L - smooth                      # 8 px 以下的結構
    h, w = L.shape
    sample = top[:: max(1, h // 512), :: max(1, w // 512)]
    sig = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
    cand = top > threshold * max(sig, 1e-12)
    lab, n = ndimage.label(cand)
    ys, xs, rs, sat = [], [], [], []
    keep = np.zeros(n + 1, bool)
    if n:
        global_max = float(L.max())
        slices = ndimage.find_objects(lab)
        for i, sl in enumerate(slices, start=1):
            comp = lab[sl] == i
            area = int(comp.sum())
            if area < 3:
                continue
            py, px = np.nonzero(comp)
            if area >= 12:
                ev = np.linalg.eigvalsh(np.cov(np.vstack([py, px])))
                if ev[0] <= 0 or ev[1] / ev[0] > 4.0:   # 細長：星雲纖維，不是星
                    continue
            wts = top[sl][comp]
            ys.append(sl[0].start + float((py * wts).sum() / wts.sum()))
            xs.append(sl[1].start + float((px * wts).sum() / wts.sum()))
            rs.append(math.sqrt(area / math.pi))
            vals = L[sl][comp]
            peak = float(vals.max())
            # 飽和＝平頂：峰值接近整張的最大值，而且至少 5 個像素被截在同一個值附近
            sat.append(peak >= 0.97 * global_max and int((vals >= 0.995 * peak).sum()) >= 5)
            keep[i] = True
    core = keep[lab]
    radius = np.zeros(n + 1, np.float32)
    radius[np.flatnonzero(keep)] = rs
    # 星暈也算進星點層：從星心邊緣往外 3 + 2×半徑 px 用餘弦降到 0
    dist, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
    extent = 3.0 + 2.0 * radius[lab[iy, ix]]
    t = np.clip(dist / extent, 0.0, 1.0)
    weight = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
    weight[core] = 1.0
    del dist, iy, ix, t
    return Stars(np.array(ys), np.array(xs), np.array(rs), np.array(sat, bool), weight > 1e-3, weight)


# ---------------------------------------------------------------------- 縮星


def measure_fwhm(L: np.ndarray, stars: Stars, limit: int = 300) -> float:
    """沒飽和、半徑 1.5–5 px 的星，二階矩半高寬的中位數。"""
    h, w = L.shape
    vals = []
    order = np.argsort(-stars.radius)
    for i in order:
        if len(vals) >= limit:
            break
        if stars.saturated[i] or not 1.5 <= stars.radius[i] <= 5:
            continue
        y, x = int(round(stars.y[i])), int(round(stars.x[i]))
        r = 6
        if y < r + 3 or x < r + 3 or y >= h - r - 3 or x >= w - r - 3:
            continue
        box = L[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
        outer = L[y - r - 3:y + r + 4, x - r - 3:x + r + 4]
        bg = np.median(np.r_[outer[:3].ravel(), outer[-3:].ravel(), outer[:, :3].ravel(), outer[:, -3:].ravel()])
        f = np.clip(box - bg, 0, None)
        tot = f.sum()
        if tot <= 0:
            continue
        yy, xx = np.indices(f.shape)
        cy, cx = (f * yy).sum() / tot, (f * xx).sum() / tot
        vals.append(2.3548 * math.sqrt((f * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / tot / 2))
    return float(np.median(vals)) if vals else 3.0


def reduce_stars(image: np.ndarray, stars: Stars, settings: DetailSettings,
                 iterations: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """星點層做 Richardson-Lucy 反卷積（高斯 PSF），縮小星點、保住光通量；重疊的星一起處理，沒有接縫。

    星點層 S ＝ max(影像 − 背景, 0) × 平滑權重，背景 B 用周圍像素補、完全不動。
    整張一起做（切小框做的話，框邊切到旁邊星點的光，振鈴會順著迭代擴散成短線）。
    只在亮度上做，三色按原本的比例分配（星色不變）。回傳 (縮星後影像, 星點層 S)。"""
    planes = image[None] if image.ndim == 2 else image
    if len(stars.y) == 0:
        return image, np.zeros_like(planes)
    background = inpaint(planes, stars.mask)
    S = (np.maximum(planes - background, 0) * stars.weight[None]).astype(np.float32)
    del background
    f = settings.shrink_factor()
    if f >= 0.999:
        return image, S
    L = planes.mean(axis=0)
    fwhm = measure_fwhm(L, stars)
    # 高斯：FWHM_後² = FWHM_前² − FWHM_psf² → 要縮成 f 倍，PSF 的 FWHM = √(1 − f²)·FWHM
    psf_sigma = math.sqrt(max(0.0, 1 - f * f)) * fwhm / 2.3548
    SL = S.mean(axis=0)
    # 大星、飽和星不縮：用權重在原本與反卷積結果之間混合（同一塊裡有大星就整塊保守）
    lab, _ = ndimage.label(stars.mask)
    weight_by_star = np.clip((settings.huge_star - stars.radius) / (settings.huge_star - settings.big_star), 0, 1)
    weight_by_star[stars.saturated] = 0
    cy = np.clip(np.round(stars.y).astype(int), 0, L.shape[0] - 1)
    cx = np.clip(np.round(stars.x).astype(int), 0, L.shape[1] - 1)
    comp_weight = np.ones(lab.max() + 1, np.float32)
    np.minimum.at(comp_weight, lab[cy, cx], weight_by_star.astype(np.float32))
    comp_weight[0] = 0
    blend = comp_weight[lab]
    del lab
    eps = np.float32(max(1e-12, float(SL.max()) * 1e-7))
    radius = int(math.ceil(4 * psf_sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / max(psf_sigma, 1e-3)) ** 2).astype(np.float32)
    kernel /= kernel.sum()

    def conv(a: np.ndarray) -> np.ndarray:
        return ndimage.correlate1d(ndimage.correlate1d(a, kernel, axis=0, mode="constant"), kernel, axis=1,
                                   mode="constant")

    u = SL.copy()
    for _ in range(iterations):
        u *= conv(SL / np.maximum(conv(u), eps))
    new_L = SL + blend * (u - SL)
    ratio = np.where(SL > eps, new_L / np.maximum(SL, eps), 1.0).astype(np.float32)
    np.minimum(ratio, 8.0, out=ratio)
    out = planes + S * (ratio[None] - 1.0)
    return (out[0] if image.ndim == 2 else out.astype(np.float32)), S


# ---------------------------------------------------------------------- 星雲細節


def sharpen(image: np.ndarray, star_mask: np.ndarray, settings: DetailSettings) -> np.ndarray:
    """亮度的 1–2 px 係數中，超過噪聲門檻的部分加強；加到三個色版（不改顏色）。星點範圍不動（縮星已處理）。"""
    gain = 0.3 * settings.sharpen
    if gain <= 0:
        return image
    L = image if image.ndim == 2 else image.mean(axis=0)
    coeffs, _ = denoise.starlet(L, 2)
    level = denoise._atrous(denoise._atrous(L, 0), 1)
    boost = np.zeros_like(L)
    for j, w in enumerate(coeffs):
        a, b, sky = denoise.noise_model(w, level)
        local = np.sqrt(a + b * np.maximum(level - sky, 0)).astype(np.float32)
        boost += denoise._shrink(w, 3.0 * local)
    soft = 1.0 - (star_mask if star_mask.dtype != bool else blur(star_mask.astype(np.float32), 2.0))
    boost *= gain * soft
    return (image + boost) if image.ndim == 2 else (image + boost[None]).astype(np.float32)


def process(image: np.ndarray, settings: DetailSettings = DetailSettings(),
            cancel: CancelCheck = never_cancel) -> np.ndarray:
    L = image if image.ndim == 2 else image.mean(axis=0)
    stars = find_stars(L)
    check(cancel)
    out, _ = reduce_stars(image, stars, settings)
    check(cancel)
    return sharpen(out, stars.weight, settings)
