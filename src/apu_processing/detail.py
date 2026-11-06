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
    # 讓拉長的星點變圓（0–1）：量全畫面星點的橢圓，反卷積的模糊多一段沿長軸的部分，長軸縮到跟短軸一樣。
    # 追蹤、導星讓星點整張往同一個方向拉長時，圓形的模糊只會讓星更扁（NGC7635：橢圓率 0.10–0.13 → 縮星後 0.13–0.17）
    round_stars: float = 1.0
    # 星點與背景分層（pipeline 在降噪前拆開）：背景層降噪、加強星雲；星點層不降噪，只清掉邊緣的噪聲、縮星、收緊。
    # 降噪作用在有星的整張上會把星核壓暗、外圍墊高（星點變軟），邊緣的噪聲也被當成星的一部分留下（毛邊）
    layered: bool = True
    # 星點層邊緣：亮度低於 STAR_CLEAN_K 倍噪聲的部分用 garrote 清掉（負的全清）
    clean_stars: float = 1.0
    # 收緊：星點層低於局部峰值 TIGHTEN_LEVEL 的部分 × (相對亮度/TIGHTEN_LEVEL)^(TIGHTEN_GAMMA × 縮星 × 這個值)，
    # 峰值不變、星暈變小 → 拉伸後星點小而亮（剖面仍單調，不會有暗環）。預設關：權重往外平滑降，有一部分星暈留在背景層，
    # 星點層收緊後大星周圍露出淡淡的圓盤（NGC7635，2026-11-02 使用者看過選「不收緊」）
    tighten: float = 0.0

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
    noise: float = 0.0      # 找星時量的噪聲（8 px 以下結構的 MAD，亮度）


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
    return Stars(np.array(ys), np.array(xs), np.array(rs), np.array(sat, bool), weight > 1e-3, weight, sig)


# ---------------------------------------------------------------------- 縮星


def core_fwhm(L: np.ndarray, y: int, x: int, r: int = 8, up: int = 4) -> float | None:
    """一顆星核心的半高寬：亮度在峰值一半以上、連到星心的面積換成等面積圓的直徑（放大 up 倍量，減少像素格的誤差）。
    跟二階矩不同，星暈不算進去；拉長的星點也量得合理（不假設圓形高斯）。"""
    h, w = L.shape
    if y < r + 3 or x < r + 3 or y >= h - r - 3 or x >= w - r - 3:
        return None
    box = L[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
    outer = L[y - r - 3:y + r + 4, x - r - 3:x + r + 4]
    bg = np.median(np.r_[outer[:3].ravel(), outer[-3:].ravel(), outer[:, :3].ravel(), outer[:, -3:].ravel()])
    z = ndimage.zoom(box - bg, up, order=1, mode="nearest", grid_mode=True)
    c0, c1 = (r - 1) * up, (r + 2) * up
    k = int(np.argmax(z[c0:c1, c0:c1]))
    py, px = c0 + k // (c1 - c0), c0 + k % (c1 - c0)
    if z[py, px] <= 0:
        return None
    lab, _ = ndimage.label(z >= z[py, px] / 2)
    area = float((lab == lab[py, px]).sum()) / (up * up)
    return 2.0 * math.sqrt(area / math.pi)


def _typical_stars(stars: Stars):
    """量星點大小用的星：沒飽和、半徑 1.5–5 px，由大到小。"""
    for i in np.argsort(-stars.radius):
        if not stars.saturated[i] and 1.5 <= stars.radius[i] <= 5:
            yield int(round(stars.y[i])), int(round(stars.x[i]))


def measure_fwhm(L: np.ndarray, stars: Stars, limit: int = 300) -> float:
    """典型星點二階矩半高寬的中位數（含星暈，決定縮星力道）。"""
    h, w = L.shape
    vals = []
    for y, x in _typical_stars(stars):
        if len(vals) >= limit:
            break
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


def measure_shape(L: np.ndarray, stars: Stars, limit: int = 300) -> np.ndarray | None:
    """典型星點二階矩的中位數 [[syy, syx], [sxy, sxx]]（px²）。星暈、噪聲讓兩軸都變大，但長短軸的差不受影響。"""
    h, w = L.shape
    r = 6
    yy, xx = np.indices((2 * r + 1, 2 * r + 1))
    vals = []
    for y, x in _typical_stars(stars):
        if len(vals) >= limit:
            break
        if y < r + 3 or x < r + 3 or y >= h - r - 3 or x >= w - r - 3:
            continue
        outer = L[y - r - 3:y + r + 4, x - r - 3:x + r + 4]
        bg = np.median(np.r_[outer[:3].ravel(), outer[-3:].ravel(), outer[:, :3].ravel(), outer[:, -3:].ravel()])
        f = np.clip(L[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64) - bg, 0, None)
        tot = f.sum()
        if tot <= 0:
            continue
        cy, cx = (f * yy).sum() / tot, (f * xx).sum() / tot
        vals.append(((f * (yy - cy) ** 2).sum() / tot, (f * (yy - cy) * (xx - cx)).sum() / tot,
                     (f * (xx - cx) ** 2).sum() / tot))
    if len(vals) < 20:
        return None
    syy, syx, sxx = np.median(np.array(vals), axis=0)
    return np.array([[syy, syx], [syx, sxx]])


# 橢圓率（1 − 短軸／長軸）低於這個就當成圓的（量測本身的誤差），照舊用圓形的模糊
ROUND_MIN_ELLIPTICITY = 0.04


def elongation(shape: np.ndarray | None) -> tuple[float, np.ndarray]:
    """回傳 (長短軸變異數的差 px², 長軸方向 (dy, dx))；沒有量到或接近圓形時差是 0。"""
    if shape is None:
        return 0.0, np.array([1.0, 0.0])
    ev, vec = np.linalg.eigh(shape)
    if ev[0] <= 0 or 1 - math.sqrt(ev[0] / ev[1]) < ROUND_MIN_ELLIPTICITY:
        return 0.0, vec[:, 1]
    return float(ev[1] - ev[0]), vec[:, 1]


def _gauss1d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(4 * sigma)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / max(sigma, 1e-3)) ** 2).astype(np.float32)
    return k / k.sum()


def _diagonal(a: np.ndarray, kernel: np.ndarray, sign: int) -> np.ndarray:
    """沿對角線（往下一列、往右 sign 欄）做一維卷積，超出邊界當成 0。"""
    r = len(kernel) // 2
    h, w = a.shape
    p = np.pad(a, r)
    out = np.zeros_like(a)
    for k, wt in enumerate(kernel):
        d = k - r
        out += wt * p[r + d:r + d + h, r + sign * d:r + sign * d + w]
    return out


def gaussian_kernels(cov: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """任意方向的二維高斯（共變異數 cov，[[yy, yx], [xy, xx]]）拆成水平、垂直、對角三次一維卷積：
    cov ＝ a·x̂x̂ᵀ ＋ b·ŷŷᵀ ＋ c·d̂d̂ᵀ，d̂ ＝ (1, ±1)/√2、c ＝ 2|yx|；對角一步是 √2 px，所以步數的變異數是 c/2。"""
    syy, syx, sxx = float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])
    c = 2 * abs(syx)
    out = [("y", _gauss1d(math.sqrt(max(syy - c / 2, 1e-6)))), ("x", _gauss1d(math.sqrt(max(sxx - c / 2, 1e-6))))]
    if c > 1e-4:
        out.append(("d+" if syx > 0 else "d-", _gauss1d(math.sqrt(c / 2))))
    return out


def measure_core_fwhm(L: np.ndarray, stars: Stars, limit: int = 300) -> float:
    """典型星點核心半高寬（core_fwhm）的中位數。"""
    vals = []
    for y, x in _typical_stars(stars):
        if len(vals) >= limit:
            break
        v = core_fwhm(L, y, x)
        if v is not None:
            vals.append(v)
    return float(np.median(vals)) if vals else 3.0


# 反卷積的 PSF 半高寬最多是星點核心的這個比例。二階矩會把星暈和暗星周圍的噪聲算進去：小星點、噪聲多的資料
# （例如 2.4"/px 的小望遠鏡）量到核心的 1.7 倍，PSF 比核心還寬 → 縮過頭、核心外挖出一圈暗環（荷包蛋）。
# 實拍 M63（核心 2.6 px）：PSF 是核心的 0.8 倍以下剖面都單調，1.07 倍就有暗環。一般素材的 PSF 本來就在這以下
PSF_CORE_LIMIT = 0.8


def deconvolution_fwhm(L: np.ndarray, stars: Stars, shrink: float) -> float:
    """縮星反卷積用的高斯 PSF 半高寬。高斯星點 FWHM_後² = FWHM_前² − FWHM_psf²，要縮成 shrink 倍，
    PSF = √(1 − shrink²)·FWHM（二階矩，決定力道）；但不超過星點核心的 PSF_CORE_LIMIT 倍。"""
    return min(math.sqrt(max(0.0, 1 - shrink * shrink)) * measure_fwhm(L, stars),
               PSF_CORE_LIMIT * measure_core_fwhm(L, stars))


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
    ratio = _shrink_ratio(planes.mean(axis=0), S.mean(axis=0), stars, settings, iterations)
    if ratio is None:
        return image, S
    out = planes + S * (ratio[None] - 1.0)
    return (out[0] if image.ndim == 2 else out.astype(np.float32)), S


def _shrink_ratio(L: np.ndarray, SL: np.ndarray, stars: Stars, settings: DetailSettings,
                  iterations: int = 15) -> np.ndarray | None:
    """星點層亮度 SL 反卷積後 ÷ 前的比例（星點層乘上它＝縮星）；不縮時回傳 None。L 是量星點大小用的整張亮度。"""
    f = settings.shrink_factor()
    if f >= 0.999 or len(stars.y) == 0:
        return None
    psf_sigma = deconvolution_fwhm(L, stars, f) / 2.3548
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
    # 模糊＝圓形的部分（縮小）＋沿長軸多出來的部分（變圓）。拉長的程度分區量（傾斜、像差讓各處不同），每區用自己的
    zones = _zone_covariances(L, stars, psf_sigma, settings) if settings.round_stars > 0 else None
    if zones is None:
        u = _richardson_lucy(SL, gaussian_kernels(_psf_cov(L, stars, psf_sigma, settings)), iterations, eps)
    else:
        num = np.zeros_like(SL)
        den = np.zeros_like(SL)
        H, W = SL.shape
        for (y0, y1, x0, x1), cov in zones:
            ty0, ty1, tx0, tx1 = max(0, y0 - ZONE_OVERLAP), min(H, y1 + ZONE_OVERLAP), max(0, x0 - ZONE_OVERLAP), min(W, x1 + ZONE_OVERLAP)
            u_t = _richardson_lucy(SL[ty0:ty1, tx0:tx1], gaussian_kernels(cov), iterations, eps)
            # 權重：區內 1，往外到重疊帶盡頭線性降到 0（畫面邊緣不降）
            wy = _ramp(ty0, ty1, y0, y1, H)
            wx = _ramp(tx0, tx1, x0, x1, W)
            wgt = np.outer(wy, wx).astype(np.float32)
            num[ty0:ty1, tx0:tx1] += wgt * u_t
            den[ty0:ty1, tx0:tx1] += wgt
        u = num / np.maximum(den, 1e-6)
    new_L = SL + blend * (u - SL)
    ratio = np.where(SL > eps, new_L / np.maximum(SL, eps), 1.0).astype(np.float32)
    np.minimum(ratio, 8.0, out=ratio)
    return ratio


ZONES = 3            # 分區量星點形狀：ZONES × ZONES 區
ZONE_MIN_STARS = 60  # 一區少於這麼多顆典型星就用整張的形狀
ZONE_OVERLAP = 48    # 相鄰區反卷積重疊、平滑接起來的寬度（px）


def _psf_cov(L: np.ndarray, stars: Stars, psf_sigma: float, settings: DetailSettings) -> np.ndarray:
    extra, axis = elongation(measure_shape(L, stars)) if settings.round_stars > 0 else (0.0, None)
    cov = np.eye(2) * psf_sigma ** 2
    if extra > 0:
        cov = cov + min(1.0, settings.round_stars) * extra * np.outer(axis, axis)
    return cov


def _zone_covariances(L: np.ndarray, stars: Stars, psf_sigma: float, settings: DetailSettings):
    """[(區的範圍 (y0, y1, x0, x1), 該區的 PSF 共變異數)]；畫面太小、或各區都跟整張差不多時回傳 None（用整張一個）。"""
    H, W = L.shape
    if min(H, W) < ZONES * 4 * ZONE_OVERLAP:
        return None
    whole = _psf_cov(L, stars, psf_sigma, settings)
    ys = np.linspace(0, H, ZONES + 1).astype(int)
    xs = np.linspace(0, W, ZONES + 1).astype(int)
    out, differs = [], False
    for i in range(ZONES):
        for j in range(ZONES):
            sel = (stars.y >= ys[i]) & (stars.y < ys[i + 1]) & (stars.x >= xs[j]) & (stars.x < xs[j + 1])
            cov = whole
            if int(sel.sum()) >= ZONE_MIN_STARS:
                sub = Stars(stars.y[sel], stars.x[sel], stars.radius[sel], stars.saturated[sel], stars.mask, stars.weight,
                            stars.noise)
                cov = _psf_cov(L, sub, psf_sigma, settings)
                differs |= not np.allclose(cov, whole, atol=0.02 * psf_sigma ** 2 + 1e-6)
            out.append(((ys[i], ys[i + 1], xs[j], xs[j + 1]), cov))
    return out if differs else None


def _ramp(t0: int, t1: int, z0: int, z1: int, n: int) -> np.ndarray:
    """一維權重：區內 [z0, z1) 為 1，往重疊帶外緣線性降到 0；碰到畫面邊緣的那一側不降。"""
    idx = np.arange(t0, t1, dtype=np.float32)
    w = np.ones(t1 - t0, np.float32)
    if z0 > 0:
        w = np.minimum(w, np.clip((idx - t0 + 1) / max(1, z0 - t0 + 1), 0, 1))
    if z1 < n:
        w = np.minimum(w, np.clip((t1 - idx) / max(1, t1 - z1 + 1), 0, 1))
    return w


def _richardson_lucy(SL: np.ndarray, passes, iterations: int, eps: np.float32) -> np.ndarray:
    def conv(a: np.ndarray) -> np.ndarray:
        for kind, k in passes:
            if kind == "y":
                a = ndimage.correlate1d(a, k, axis=0, mode="constant")
            elif kind == "x":
                a = ndimage.correlate1d(a, k, axis=1, mode="constant")
            else:
                a = _diagonal(a, k, 1 if kind == "d+" else -1)
        return a

    u = SL.copy()
    for _ in range(iterations):
        u *= conv(SL / np.maximum(conv(u), eps))
    return u


# ---------------------------------------------------------------------- 星點與背景分層


STAR_CLEAN_K = 2.0      # 星點層：亮度低於幾倍噪聲完全換成 3×3 中位數的版本
STAR_CLEAN_HIGH = 6.0   # 高於幾倍噪聲完全照原樣
TIGHTEN_LEVEL = 0.5     # 收緊：局部峰值的這個比例以下才壓
TIGHTEN_GAMMA = 1.0     # 收緊的指數（× 縮星強度 × tighten）；縮星 50% 時 0.5
TIGHTEN_BOOST = 0.0     # 星核加亮：局部峰值 TIGHTEN_LEVEL 以上 × (1 ＋ 這個 × 縮星強度 × tighten × 高出的比例)
TIGHTEN_WINDOW = 11     # 局部峰值的範圍（px）：比這遠的星暈不算同一顆（大星本來就不縮）


@dataclass
class Layers:
    background: np.ndarray   # 背景層（星點底下用周圍補，邊緣跟原圖平滑接上）；形狀跟原圖一樣
    stars_layer: np.ndarray  # 星點層 (C, H, W)；background + stars_layer ＝ 原圖
    stars: Stars
    L: np.ndarray            # 拆開前的亮度（量星點大小用）


def split(image: np.ndarray) -> Layers:
    """拆成背景層與星點層（相加等於原圖）。星點層＝(影像 − 星點下的背景) × 平滑權重，有正有負（含噪聲）。"""
    planes = image[None] if image.ndim == 2 else image
    L = planes.mean(axis=0)
    stars = find_stars(L)
    if len(stars.y) == 0:
        return Layers(image.copy(), np.zeros_like(planes), stars, L)
    under = inpaint(planes, stars.mask)
    # 試過範圍再往外擴 8 px 補一次、兩個取暗的（想讓星暈留在星點層、收緊後不留圓盤）：補的值比真的背景暗，
    # 星點層清邊緣時把混在裡面的背景一起清掉 → 亮星周圍暗斑（17 組亮星黑塊 0.018→0.025、WR134 0.005→0.016），不用
    S = ((planes - under) * stars.weight[None]).astype(np.float32)
    del under
    bg = (planes - S).astype(np.float32)
    return Layers(bg[0] if image.ndim == 2 else bg, S, stars, L)


def clean_layer(S: np.ndarray, noise: float, strength: float = 1.0) -> np.ndarray:
    """星點層的邊緣噪聲：星點本體（亮度 > STAR_CLEAN_HIGH 倍噪聲）照原樣，接近噪聲的地方換成 3×3 中位數，
    中間平滑過渡。星暈的光留著，只是噪聲的毛邊不見。
    （試過把接近噪聲的部分用 garrote 扣掉：連亮星外圍真的星暈也扣掉，周圍一圈暗斑——17 組亮星黑塊 WR134 0.005→0.016）"""
    if strength <= 0 or noise <= 0:
        return S
    # 中位數：去掉孤立的噪聲點，但不會把星核的光往外推（試過高斯模糊：核心的光漏到外圍，每顆星一圈亮環）
    soft = np.stack(list(_POOL.map(lambda p: ndimage.median_filter(p, size=3), S)))
    level = blur(S.mean(axis=0), 1.0) / np.float32(noise * strength)
    a = np.clip((level - STAR_CLEAN_K) / (STAR_CLEAN_HIGH - STAR_CLEAN_K), 0.0, 1.0)
    a = (a * a * (3 - 2 * a)).astype(np.float32)
    return (soft + a[None] * (S - soft)).astype(np.float32)


def tighten_layer(S: np.ndarray, settings: DetailSettings) -> np.ndarray:
    """局部峰值 TIGHTEN_LEVEL 以下的部分壓低，峰值不變；r·h(r) 隨 r 遞增，剖面仍單調。"""
    strength = min(1.0, max(0.0, settings.star_reduction)) * settings.tighten
    gamma, boost = TIGHTEN_GAMMA * strength, TIGHTEN_BOOST * strength
    if gamma <= 0 and boost <= 0:
        return S
    SL = S.mean(axis=0)
    peak = ndimage.maximum_filter(SL, size=TIGHTEN_WINDOW, mode="constant")
    rel = np.clip(SL / np.maximum(peak, 1e-30), 0.0, 1.0)
    low = np.power(np.clip(rel / TIGHTEN_LEVEL, 0.0, 1.0), gamma, dtype=np.float32)
    high = 1.0 + boost * np.clip((rel - TIGHTEN_LEVEL) / (1.0 - TIGHTEN_LEVEL), 0.0, 1.0)
    h = np.where(SL > 0, np.where(rel < TIGHTEN_LEVEL, low, high), 1.0).astype(np.float32)
    return S * h[None]


def finish(background: np.ndarray, layers: Layers, settings: DetailSettings,
           cancel: CancelCheck = never_cancel) -> np.ndarray:
    """分層處理的後半：背景層（已降噪）加強星雲；星點層清邊緣、縮星（變圓）、收緊；再合起來。"""
    stars = layers.stars
    S = clean_layer(layers.stars_layer, stars.noise, settings.clean_stars)
    check(cancel)
    ratio = _shrink_ratio(layers.L, np.maximum(S.mean(axis=0), 0), stars, settings) if len(stars.y) else None
    if ratio is not None:
        S = S * ratio[None]
    S = tighten_layer(S, settings)
    check(cancel)
    bg = sharpen(background, stars.weight, settings)
    return (bg + S[0]) if bg.ndim == 2 else (bg + S).astype(np.float32)


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
