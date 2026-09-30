"""自動拉伸：依畫面剩下的噪聲決定拉多深。

擬合自 17 組參考成品（2026-09-30，tools/analysis/stretch_curves.py）：
把線性亮度換成「比天空亮幾倍噪聲」u，參考成品的色調曲線幾乎是同一條

    y = t + (1 − t)·(1 − (1 + u/a)^(−p))        a = 9.47、p = 0.563、t = 背景亮度

關鍵是噪聲的尺度：用 8 px 尺度的噪聲（8×8 平均後的像素噪聲 × 8）當單位，17 張（包括降噪尺度特別大的富士天鵝座）
都落在同一條曲線上，調參組誤差 0.034、驗收組 0.025（顯示亮度）；只用像素噪聲的話降噪過的圖會被拉太深、斑駁跑出來。
所以拉伸要在降噪之後、對「要拉的這張」量噪聲。

天空與噪聲只在最暗的四分之一區塊量（64 px 區塊的中位數），星點與星雲不影響。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CURVE_A = 9.47
CURVE_P = 0.563
NOISE_SCALE = 8
TILE = 64
# 預設比擬合值深 2^0.6 ≈ 1.5 倍（2026-10-01，使用者：「大膽一點」）。擬合用的是參考成品降噪後的資料，我們的降噪留下較多噪聲、
# σ₈ 較大，照擬合值拉會偏暗（17 張星雲亮度中位只有參考成品的 0.62 倍）。加深後星雲亮度 ×0.91、天空看得到的噪聲 ×1.03（都對參考成品），
# 最吵的 M45 ×1.49。tools/scorecard 的 strength-study
DEFAULT_BOOST = 0.6


@dataclass(frozen=True)
class StretchSettings:
    background: float = 0.13   # 乾淨天空落在成品上的亮度（參考成品的中位數）
    strength: float = 0.5      # 0.5＝預設（比擬合值深 DEFAULT_BOOST）；每 +0.25 約等於把噪聲單位縮小 1.4 倍（拉更深）
    linked: bool = True        # True：三色共用同一個噪聲尺度、各扣自己的天空（色彩比例較忠實）
    # 背景保留色彩：0＝最暗的天空校成中性；1＝背景帶上暗處微弱星雲的顏色（滿版 Hα 是暗紅、反射星雲是藍）。
    # 影像本身分不出「光害的底色」和「微弱的星雲」，做成滑桿（2026-10-01）；預設 50%（使用者決定）。見 faint_color
    background_color: float = 0.5

    def gain(self) -> float:
        return float(2.0 ** ((self.strength - 0.5) * 2.0 + DEFAULT_BOOST))


@dataclass(frozen=True)
class StretchNoise:
    """拉伸需要的量測：各色版天空、亮度的 8 px 尺度噪聲、各色版的 8 px 噪聲（非連動時用）。"""
    sky: tuple[float, ...]
    sigma: float
    channel_sigma: tuple[float, ...]
    # 暗處微弱訊號的顏色（三色、平均 1）與量（σ₈ 倍）：最暗 0.5–3% 區塊與 15–35% 區塊的差。
    # 光害底色整張大致均勻，相減時抵消；剩下的是微弱星雲（WR134、IC1805 偏紅，M45 偏藍，M33 幾乎沒有）
    faint_color: tuple[float, ...] = (1.0, 1.0, 1.0)
    faint_amount: float = 0.0


def _luminance(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else image.mean(axis=0)


def _block_mean(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape
    h2, w2 = h // f * f, w // f * f
    return img[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def sky_tiles(luminance: np.ndarray, tile: int = TILE) -> np.ndarray:
    """最暗四分之一的區塊（布林陣列，區塊座標）。外圍 1% 不算（疊圖邊緣）。"""
    h, w = luminance.shape
    th, tw = h // tile, w // tile
    if th < 2 or tw < 2:
        return np.ones((max(th, 1), max(tw, 1)), bool)
    med = np.median(luminance[:th * tile, :tw * tile].reshape(th, tile, tw, tile), axis=(1, 3))
    inner = np.zeros_like(med, bool)
    my, mx = max(1, th // 50), max(1, tw // 50)
    inner[my:th - my, mx:tw - mx] = True
    if not inner.any():
        inner[:] = True
    return inner & (med <= np.quantile(med[inner], 0.25))


def _pixel_noise(img: np.ndarray, mask: np.ndarray) -> float:
    """相鄰像素差的 MAD / √2（只用遮罩內的相鄰對）。"""
    both = mask[:, 1:] & mask[:, :-1]
    d = (img[:, 1:] - img[:, :-1])[both]
    if d.size < 50:
        return 0.0
    d = d[:: max(1, d.size // 400_000)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def measure(image: np.ndarray, pixel_scale: float = 1.0) -> StretchNoise:
    """量天空與 8 px 尺度噪聲（完整尺寸的 8 px）。

    pixel_scale：這張圖一個像素等於完整尺寸的幾個像素（面積平均縮圖的倍率）。快速預覽在縮圖上量時，
    改用 8/pixel_scale 的區塊，量到的仍是完整尺寸 8 px 的噪聲，拉伸深淺才跟完整結果一致。"""
    planes = image[None] if image.ndim == 2 else image
    L = _luminance(image)
    tile = max(8, int(round(TILE / pixel_scale)))
    tiles = sky_tiles(L, tile)
    th, tw = tiles.shape
    mask = np.zeros(L.shape, bool)
    mask[:th * tile, :tw * tile] = np.repeat(np.repeat(tiles, tile, 0), tile, 1)
    sky = tuple(float(np.median(p[mask])) for p in planes)

    f = max(1, int(round(NOISE_SCALE / pixel_scale)))
    coarse_mask = _block_mean(mask.astype(np.float32), f) > 0.999 if f > 1 else mask

    def scale_noise(plane: np.ndarray) -> float:
        coarse = _block_mean(plane, f) if f > 1 else plane
        # 縮圖的 f 區塊＝完整尺寸的 f·pixel_scale 區塊；k·σ_k 形式（f·pixel_scale＝8 時就是 σ₈）
        s = f * pixel_scale * _pixel_noise(coarse, coarse_mask)
        if s <= 0:  # 圖太小或沒有噪聲：退回像素噪聲，再不行用亮度範圍的 1/1000
            s = pixel_scale * _pixel_noise(plane, mask)
        if s <= 0:
            s = max(1e-6, float(np.quantile(plane, 0.999) - np.median(plane)) / 1000)
        return s

    sigma = scale_noise(L)
    color, amount = _faint_signal(planes, L, tile, sigma)
    return StretchNoise(sky=sky, sigma=sigma, channel_sigma=tuple(scale_noise(p) for p in planes),
                        faint_color=color, faint_amount=amount)


def _faint_signal(planes: np.ndarray, L: np.ndarray, tile: int, sigma: float) -> tuple[tuple[float, ...], float]:
    """暗處微弱訊號的顏色與量（見 StretchNoise.faint_color）。區塊中位數用隔點抽樣，夠用也快。"""
    if planes.shape[0] < 3:
        return (1.0,) * planes.shape[0], 0.0
    th, tw = L.shape[0] // tile, L.shape[1] // tile
    if th * tw < 40:
        return (1.0, 1.0, 1.0), 0.0
    step = 2 if tile >= 16 else 1

    def tile_median(p: np.ndarray) -> np.ndarray:
        v = p[:th * tile:step, :tw * tile:step].reshape(th, tile // step, tw, tile // step)
        return np.median(v.transpose(0, 2, 1, 3).reshape(th, tw, -1), axis=2).ravel()

    med = np.stack([tile_median(p) for p in planes])
    order = np.argsort(med.mean(axis=0))
    n = order.size
    lo = order[int(0.005 * n):max(int(0.005 * n) + 1, int(0.03 * n))]
    hi = order[int(0.15 * n):max(int(0.15 * n) + 1, int(0.35 * n))]
    d = med[:, hi].mean(axis=1) - med[:, lo].mean(axis=1)
    if d.mean() <= 0:
        return (1.0, 1.0, 1.0), 0.0
    color = np.clip(d / d.mean(), 0.0, 3.0)
    color = color / color.mean()
    return tuple(float(c) for c in color), float(d.mean() / sigma)


def curve(u: np.ndarray, background: float) -> np.ndarray:
    """u（天空以上幾倍噪聲）→ 顯示亮度。天空以下接同斜率的直線，最低 0。"""
    t = background
    out = np.empty_like(u, dtype=np.float32)
    pos = u >= 0
    out[pos] = t + (1 - t) * (1 - np.power(1 + u[pos] / CURVE_A, -CURVE_P))
    out[~pos] = t + (1 - t) * (CURVE_P / CURVE_A) * u[~pos]
    np.clip(out, 0.0, 1.0, out=out)
    return out


def apply(image: np.ndarray, settings: StretchSettings = StretchSettings(),
          noise: StretchNoise | None = None) -> np.ndarray:
    """線性影像 → 0–1 顯示影像（形狀不變）。"""
    noise = noise or measure(image)
    planes = image[None] if image.ndim == 2 else image
    gain = settings.gain()
    out = np.empty(planes.shape, np.float32)
    black = black_points(noise, settings)
    for c, plane in enumerate(planes):
        sigma = noise.sigma if settings.linked else noise.channel_sigma[c]
        u = (plane - np.float32(black[c])) * np.float32(gain / sigma)
        out[c] = curve(u, settings.background)
    return out[0] if image.ndim == 2 else out


BACKGROUND_COLOR_SCALE = 2.5


def black_points(noise: StretchNoise, settings: StretchSettings) -> list[float]:
    """各色版的黑點。背景中性時是各自的天空；保留背景色時，黑點沿著「暗處微弱星雲的顏色」偏移，
    天空帶上那個顏色、亮度不變（偏移量三色平均為 0），偏移大小跟暗處星雲的量成正比，乾淨的天空幾乎不受影響。
    100% 時 WR134、IC1805 的天空顏色大致跟參考成品一樣（2026-10-01 以 17 組參考成品校）。"""
    k = min(1.0, max(0.0, settings.background_color))
    shift = k * BACKGROUND_COLOR_SCALE * noise.faint_amount * noise.sigma
    return [s - shift * (c - 1.0) for s, c in zip(noise.sky, noise.faint_color)]


def apply_luminance(image: np.ndarray, settings: StretchSettings = StretchSettings(),
                    noise: StretchNoise | None = None) -> np.ndarray:
    """只拉亮度、三色比例不變的版本（比較用）。"""
    noise = noise or measure(image)
    if image.ndim == 2:
        return apply(image, settings, noise)
    sky = np.asarray(noise.sky, np.float32)[:, None, None]
    lin = image - sky
    L = lin.mean(axis=0)
    gain = settings.gain()
    Ly = curve(L * np.float32(gain / noise.sigma), settings.background)
    t = settings.background
    ratio = np.where(np.abs(L) > 1e-12, (Ly - t) / np.where(np.abs(L) > 1e-12, L, 1), 0).astype(np.float32)
    out = t + lin * ratio[None]
    return np.clip(out, 0, 1).astype(np.float32)
