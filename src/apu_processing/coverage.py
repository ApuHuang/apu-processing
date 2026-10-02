"""疊圖覆蓋率圖：每個像素被幾張 frame 蓋到（跟 master 同尺寸、同方向，檔名加 _coverage）。

疊圖邊緣（沒裁切、拼接、dither 範圍大）覆蓋的張數少、噪聲高，可能還有對齊的邊界痕跡：
- 建議裁切：覆蓋 ≥ 最多張數 90% 的範圍裡最大的矩形（非破壞，使用者按「套用」才裁）
- 去光只在覆蓋足夠的地方取樣，免得背景模型被邊緣帶歪

覆蓋率一律換成「最多張數的幾成」（0–1），跟原本的整數單位無關；好幾張合成時取每個像素最小的那個。
引擎不 import tkinter。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import imageio

SUFFIX = "_coverage"
GOOD_FRACTION = 0.9      # 覆蓋 ≥ 最多張數的 90% 才算足夠
_POOL = 8                # 找矩形時先 8×8 縮小（取最小值，縮小後的格子整格都夠才算夠）


def find_for(path: Path | str) -> Path | None:
    """master 旁邊同名加 _coverage 的檔案（同副檔名優先，再試其他 FITS 副檔名）。"""
    path = Path(path)
    for suffix in dict.fromkeys([path.suffix, ".fits", ".fit", ".fts"]):
        candidate = path.with_name(f"{path.stem}{SUFFIX}{suffix}")
        if candidate.is_file():
            return candidate
    return None


def load_fraction(path: Path | str) -> np.ndarray | None:
    """覆蓋率圖 → 最多張數的幾成（float32，0–1）。讀檔走 imageio，方向跟 master 一致。全 0 就回傳 None。"""
    data, _ = imageio.load_image(path)
    if data.ndim == 3:
        data = data.min(axis=0)
    top = float(data.max())
    if top <= 0:
        return None
    return (data / np.float32(top)).astype(np.float32)


def combine(fractions: list[np.ndarray | None], shape: tuple[int, int]) -> np.ndarray | None:
    """好幾張的覆蓋率取每個像素最小的；尺寸對不上或都沒有就回傳 None。"""
    usable = [f for f in fractions if f is not None and f.shape == shape]
    if not usable or len(usable) != len(fractions):
        return None
    return np.minimum.reduce(usable)


def good_mask(fraction: np.ndarray, threshold: float = GOOD_FRACTION) -> np.ndarray:
    return fraction >= np.float32(threshold - 1e-6)


def suggest_crop(fraction: np.ndarray, threshold: float = GOOD_FRACTION) -> tuple[int, int, int, int] | None:
    """覆蓋足夠的範圍裡最大的矩形 (x0, y0, x1, y1)。整張都夠（不用裁）或夠的範圍太小時回傳 None。"""
    good = good_mask(fraction, threshold)
    h, w = good.shape
    if good.all():
        return None
    f = _POOL
    hs, ws = h // f, w // f
    if hs < 4 or ws < 4:
        return None
    small = good[:hs * f, :ws * f].reshape(hs, f, ws, f).all(axis=(1, 3))
    best, box = 0, None
    heights = np.zeros(ws, np.int32)
    for row in range(hs):
        heights = np.where(small[row], heights + 1, 0)
        # 直方圖裡最大的矩形（單調堆疊）
        stack: list[int] = []
        for col in range(ws + 1):
            current = heights[col] if col < ws else 0
            while stack and heights[stack[-1]] >= current:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = int(heights[top]) * (col - left)
                if area > best:
                    best = area
                    box = (left, row - int(heights[top]) + 1, col, row + 1)
            stack.append(col)
    if box is None:
        return None
    x0, y0, x1, y1 = (v * f for v in box)
    # 縮小時丟掉的右、下零頭：那一邊碰到原本的邊界就延伸回去
    if x1 == ws * f and good[y0:y1, ws * f:].all():
        x1 = w
    if y1 == hs * f and good[hs * f:, x0:x1].all():
        y1 = h
    area = (x1 - x0) * (y1 - y0)
    if area >= 0.995 * h * w or area < 0.25 * h * w:
        return None
    return x0, y0, x1, y1
