"""拆疊評分的量測。

素材：同一組單張依時間輪流分成 B（3/4，答案）與 A4（另外 1/4）、A8（A4 隔張取），三份對齊到同一張參考 frame、不裁切。
B 的噪聲跟 A 無關，所以：
- 誤差 E[(P − Q)²] = E[(P − 真值)²] + B 的噪聲²，B 的噪聲從 A − B 的差量（A、B 張數比）算出來扣掉；
- 保留率 Σ p·q / (Σ q² − N·B 噪聲²)：P 的噪聲跟 Q 無關，分子不偏；不處理時應該 ≈ 1（自我檢查）。
都在線性資料、starlet 各尺度（1–32 px）上量，大尺度的天光差異（A、B 的天空本來就不完全一樣）不算。

去光與校色：曲面與倍率都在 B 上量，A、B 套用同一份，這裡只比降噪與細節；兩者在 A 上量到的差異另外報告。
細節：Q＝B 做同樣的細節處理（不降噪），所以量的是「噪聲讓細節處理差了多少」，不是細節處理本身對不對（B 的星點一樣模糊）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

from apu_processing import background, color, coverage, denoise, detail, pipeline, stretch

SCALES = 6                       # starlet 1、2、4、8、16、32 px
GROUPS = {"fine": (0, 1), "mid": (2, 3), "large": (4, 5)}   # 1–2、4–8、16–32 px
FAINT_U = 2.0                    # 比天空亮幾倍 σ₈（A 未處理）以上算星雲；星雲再依亮度中位數分暗／亮
MIN_SIGNAL = 1.0                 # 訊號功率 / B 噪聲功率 低於這個就不報保留率（分母太小，估計不穩）


@dataclass
class Split:
    name: str
    B: np.ndarray
    subs: dict[str, np.ndarray]
    frames: dict[str, int]
    mask: np.ndarray             # 三份都覆蓋 ≥ 90% 的像素（裁切後）


def _master(folder: Path):
    info = json.loads((folder / "done.json").read_text(encoding="utf-8"))
    o = info["outputs"][0]
    # 原始 ADU 值、三份用同一個比例（不能用 imageio 的各檔峰值正規化）
    img = fits.getdata(folder / o["file"]).astype(np.float32) / np.float32(65535.0)
    cov = fits.getdata(folder / o["coverage"]).astype(np.float32)
    return img, cov >= 0.9 * float(cov.max()), int(o["frames"])


def load_split(root: Path, name: str, subs=("A4", "A8")) -> Split:
    B, mb, nb = _master(root / name / "B")
    imgs, frames, mask = {}, {"B": nb}, mb
    for s in subs:
        a, ma, na = _master(root / name / s)
        imgs[s], frames[s] = a, na
        mask = mask & ma
    # 三份都覆蓋的最大矩形（多晚的視野會轉，邊緣斜切）
    box = coverage.suggest_crop(mask.astype(np.float32), 0.5)
    x0, y0, x1, y1 = box if box is not None else (0, 0, mask.shape[1], mask.shape[0])
    crop = np.s_[y0:y1, x0:x1]
    return Split(name, np.ascontiguousarray(B[(slice(None),) + crop]),
                 {s: np.ascontiguousarray(a[(slice(None),) + crop]) for s, a in imgs.items()},
                 frames, mask[crop])


@dataclass
class Front:
    """去光、校色在 B 上量，A、B 共用。"""
    surfaces: np.ndarray
    balance: color.ColorBalance | None

    def apply(self, img: np.ndarray) -> np.ndarray:
        x = img - self.surfaces
        return color.apply(x, self.balance) if self.balance else x


def front(split: Split, settings: pipeline.ProcessingSettings) -> tuple[Front, dict]:
    surf = background.model(split.B, settings.background, sample_mask=split.mask).surfaces \
        if settings.background_enabled else np.zeros_like(split.B)
    bal = color.measure_balance(split.B - surf) if settings.color_enabled else None
    report = {}
    # 在 A 上自己量會差多少（σ₈ 倍數、倍率比）：去光與校色對噪聲的穩定度
    for s, a in split.subs.items():
        r = {}
        if settings.background_enabled:
            sa = background.model(a, settings.background, sample_mask=split.mask).surfaces
            sig = stretch.measure(a - sa).sigma
            d = (sa - surf)[:, split.mask]
            r["background_diff"] = float(np.sqrt(np.mean((d - d.mean(axis=1, keepdims=True)) ** 2)) / sig)
        if bal is not None:
            ba = color.measure_balance(a - surf)
            r["color_diff"] = float(max(abs(ga / gb - 1) for ga, gb in zip(ba.gains, bal.gains)))
        report[s] = r
    return Front(surf, bal), report


def linear_stages(x: np.ndarray, settings: pipeline.ProcessingSettings) -> np.ndarray:
    """管線的降噪與細節兩段（跟 pipeline.Processor 相同的順序與條件）。"""
    if settings.detail_enabled and settings.detail.layered:
        layers = detail.split(x)
        bg = denoise.reduce(layers.background, settings.denoise) if settings.denoise_enabled else layers.background
        return detail.finish(bg, layers, settings.detail)
    if settings.denoise_enabled:
        x = denoise.reduce(x, settings.denoise)
    if settings.detail_enabled:
        x = detail.process(x, settings.detail)
    return x


@dataclass
class Regions:
    sky: np.ndarray
    faint: np.ndarray
    bright: np.ndarray
    halo: np.ndarray             # 星點往外平滑降到 0 的那一圈（縮星的暗環會在這裡）
    sky_noise: float             # A 未處理、天空的像素噪聲（亮度），暗環偏差的單位
    faint_stars: tuple[np.ndarray, np.ndarray]   # 暗星（B 上星心 5–50 倍像素噪聲、不飽和）的 (y, x)


def regions(xb: np.ndarray, xa_raw: np.ndarray, mask: np.ndarray) -> Regions:
    Lb = xb.mean(axis=0)
    stars = detail.find_stars(Lb)
    # 星點層的權重往外拉得很寬（3+2R px），整個排除的話星場裡幾乎不剩星雲；只排除星心附近
    star_any = ndimage.binary_dilation(stars.weight >= 0.3, iterations=2)
    halo = (stars.weight > 0) & (stars.weight < 0.999)
    na = stretch.measure(xa_raw)
    tiles = stretch.sky_tiles(xa_raw.mean(axis=0))
    t = stretch.TILE
    sky_px = np.zeros(Lb.shape, bool)
    th, tw = tiles.shape
    sky_px[:th * t, :tw * t] = np.repeat(np.repeat(tiles, t, 0), t, 1)
    sky_level = float(np.mean(na.sky))
    u = (detail.blur(Lb, 4.0) - sky_level) / na.sigma
    free = mask & ~star_any
    nebula = free & (u >= FAINT_U) & ~sky_px
    split = float(np.median(u[nebula])) if nebula.sum() > 1000 else np.inf   # 依亮度分半（同 tools/scorecard）
    faint = nebula & (u < split)
    bright = nebula & (u >= split)
    La = xa_raw.mean(axis=0)
    both = sky_px[:, 1:] & sky_px[:, :-1] & mask[:, 1:]
    d = (La[:, 1:] - La[:, :-1])[both]
    pix = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    db = (Lb[:, 1:] - Lb[:, :-1])[both]
    pix_b = float(1.4826 * np.median(np.abs(db - np.median(db))) / np.sqrt(2))
    H, W = Lb.shape
    cy = np.round(stars.y).astype(int)
    cx = np.round(stars.x).astype(int)
    inside = (cy >= 12) & (cx >= 12) & (cy < H - 12) & (cx < W - 12)
    cy, cx, sat = cy[inside], cx[inside], stars.saturated[inside]
    snr = (Lb[cy, cx] - sky_level) / max(pix_b, 1e-12)
    pick = (snr >= 5) & (snr < 50) & ~sat & mask[cy, cx]
    return Regions(free & sky_px, faint, bright, halo & mask, pix, (cy[pick], cx[pick]))


def star_flux(L: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """每顆星 r ≤ 2 px 的光通量，扣掉 6–9 px 圈的中位數。"""
    yy, xx = np.mgrid[-9:10, -9:10]
    r = np.hypot(yy, xx)
    core_ = r <= 2.0
    ring = (r >= 6) & (r <= 9)
    out = np.empty(len(ys), np.float64)
    for i, (y, x) in enumerate(zip(ys, xs)):
        box = L[y - 9:y + 10, x - 9:x + 10]
        out[i] = box[core_].sum() - core_.sum() * np.median(box[ring])
    return out


def _planes(img: np.ndarray, kind: str) -> list[np.ndarray]:
    if kind == "L":
        return [img.mean(axis=0)]
    return [img[0] - img[1], img[2] - img[1]]


def coefficients(img: np.ndarray, kind: str) -> list[list[np.ndarray]]:
    """[色版][尺度] 的 starlet 係數。"""
    return [denoise.starlet(p, SCALES)[0] for p in _planes(img, kind)]


def sums(cp, cq, cd, region: np.ndarray) -> dict:
    """每個尺度：Σ(p−q)²、Σp·q、Σq²、Σd²（d＝A−B 未處理）、N。色版加總。"""
    out = {k: np.zeros(SCALES) for k in ("err", "pq", "qq", "dd")}
    n = int(region.sum())
    for p_c, q_c, d_c in zip(cp, cq, cd):
        for j in range(SCALES):
            p, q, d = p_c[j][region], q_c[j][region], d_c[j][region]
            out["err"][j] += float(np.dot(p - q, p - q))
            out["pq"][j] += float(np.dot(p, q))
            out["qq"][j] += float(np.dot(q, q))
            out["dd"][j] += float(np.dot(d, d))
    out["n"] = n * len(cp)
    return out


def summarize(s: dict, s0: dict, na: int, nb: int) -> dict:
    """s：方法、s0：不降噪（同樣的細節處理）。B 噪聲² ＝ Σd² × na/(na+nb)。"""
    noise_b = s["dd"] * na / (na + nb)
    res = {}
    for g, js in GROUPS.items():
        js = list(js)
        e = max(0.0, float(s["err"][js].sum() - noise_b[js].sum()))
        e0 = max(1e-30, float(s0["err"][js].sum() - noise_b[js].sum()))
        res[f"remain_{g}"] = float(np.sqrt(e / e0))
        signal = float(s["qq"][js].sum() - noise_b[js].sum())
        if signal > MIN_SIGNAL * float(noise_b[js].sum()):
            res[f"keep_{g}"] = float(s["pq"][js].sum() / signal)
            res[f"keep0_{g}"] = float(s0["pq"][js].sum() / signal)
    return res
