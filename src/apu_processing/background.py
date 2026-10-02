"""去光梯度：在天空取樣、擬合平滑曲面、扣掉。

取樣：32 px 一格，每格先 4×4 平均再取 8×8 的中位數（星點不影響）。
擬合：每個色版一個 Legendre 多項式曲面；共用一份「哪些格子是天空」——用亮度反覆排除比曲面亮的格子（星雲），
三色用同一批格子，背景才不會出現紅綠色塊。扣掉曲面後加回曲面的中位數，天空亮度維持原樣。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from .progress import CancelCheck, check, never_cancel

CELL = 32
SUB = 4


@dataclass(frozen=True)
class BackgroundSettings:
    degree: int = 4          # 曲面的多項式次數（2026-10-01 掃描：4 次在調參與驗收組都最好）
    clip_high: float = 2.0   # 比曲面亮幾倍殘差離散度就當星雲排除
    clip_low: float = 4.0
    iterations: int = 8


@dataclass
class BackgroundModel:
    surfaces: np.ndarray       # (C, H, W) 扣掉的曲面（已減去中位數）
    sky_cells: np.ndarray      # (rows, cols) 最後當作天空的格子
    cell_values: np.ndarray    # (C, rows, cols)


def _cells(plane: np.ndarray) -> np.ndarray:
    h, w = plane.shape
    rows, cols = h // CELL, w // CELL
    x = plane[:rows * CELL, :cols * CELL]
    sub = x.reshape(rows * CELL // SUB, SUB, cols * CELL // SUB, SUB).mean(axis=(1, 3))
    k = CELL // SUB
    blocks = sub.reshape(rows, k, cols, k).transpose(0, 2, 1, 3).reshape(rows, cols, k * k)
    return np.median(blocks, axis=2).astype(np.float64)


def _coords(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n * 2 - 1


def _fit(values: np.ndarray, mask: np.ndarray, degree: int) -> np.ndarray:
    rows, cols = values.shape
    yy, xx = np.meshgrid(_coords(rows), _coords(cols), indexing="ij")
    V = legendre.legvander2d(yy[mask], xx[mask], [degree, degree])
    # 只留總次數 ≤ degree 的項，避免角落亂翹
    keep = np.array([i + j <= degree for i in range(degree + 1) for j in range(degree + 1)])
    coef = np.zeros((degree + 1) * (degree + 1))
    sol, *_ = np.linalg.lstsq(V[:, keep], values[mask], rcond=None)
    coef[keep] = sol
    return coef.reshape(degree + 1, degree + 1)


def _evaluate(coef: np.ndarray, h: int, w: int) -> np.ndarray:
    """在每個像素中心算曲面值（可分離：先對 y 再對 x）。"""
    ys = (np.arange(h) + 0.5) / (h // CELL * CELL) * 2 - 1
    xs = (np.arange(w) + 0.5) / (w // CELL * CELL) * 2 - 1
    Vy = legendre.legvander(ys, coef.shape[0] - 1)
    Vx = legendre.legvander(xs, coef.shape[1] - 1)
    return (Vy @ coef @ Vx.T).astype(np.float32)


def _robust_sigma(v: np.ndarray) -> float:
    m = np.median(v)
    return float(1.4826 * np.median(np.abs(v - m))) or 1e-12


def model(image: np.ndarray, settings: BackgroundSettings = BackgroundSettings(),
          cancel: CancelCheck = never_cancel, sample_mask: np.ndarray | None = None) -> BackgroundModel:
    """sample_mask：(H, W) 布林，可以取樣的像素（例如疊圖覆蓋足夠的範圍）；None＝全部都可以。"""
    planes = image[None] if image.ndim == 2 else image
    C, h, w = planes.shape
    cells = np.stack([_cells(p) for p in planes])
    check(cancel)
    L = cells.mean(axis=0)
    rows, cols = L.shape
    valid = np.ones((rows, cols), bool)
    my, mx = max(1, rows // 50), max(1, cols // 50)
    valid[:my] = valid[-my:] = False
    valid[:, :mx] = valid[:, -mx:] = False
    valid &= np.all(cells != 0, axis=0)
    if sample_mask is not None and sample_mask.shape == (h, w):
        # 整格都在可取樣範圍裡才用；剩下的格子太少就不管遮罩（照原本的方式）
        m = sample_mask[:rows * CELL, :cols * CELL].reshape(rows, CELL, cols, CELL).all(axis=(1, 3))
        if (valid & m).sum() >= 24:
            valid &= m
    if not valid.any():
        # 沒有任何可取樣的格子（例如整張同一個值）：不扣
        return BackgroundModel(np.zeros((C, h, w), np.float32), valid, cells)
    # 起點：亮度最暗的一半格子
    mask = valid & (L <= np.quantile(L[valid], 0.5))
    for _ in range(settings.iterations):
        coef = _fit(L, mask, settings.degree)
        rows_, cols_ = np.meshgrid(_coords(rows), _coords(cols), indexing="ij")
        surf = legendre.legval2d(rows_, cols_, coef)
        resid = L - surf
        s = _robust_sigma(resid[mask])
        new = valid & (resid < settings.clip_high * s) & (resid > -settings.clip_low * s)
        if new.sum() < 12 or np.array_equal(new, mask):
            break
        mask = new
    surfaces = np.empty((C, h, w), np.float32)
    for c in range(C):
        coef = _fit(cells[c], mask, settings.degree)
        s = _evaluate(coef, h, w)
        surfaces[c] = s - np.float32(np.median(s))
        check(cancel)
    return BackgroundModel(surfaces, mask, cells)


def correct(image: np.ndarray, settings: BackgroundSettings = BackgroundSettings(),
            cancel: CancelCheck = never_cancel, sample_mask: np.ndarray | None = None) -> np.ndarray:
    m = model(image, settings, cancel, sample_mask)
    out = (image[None] if image.ndim == 2 else image) - m.surfaces
    return out[0] if image.ndim == 2 else out
