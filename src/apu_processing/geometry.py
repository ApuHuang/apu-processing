"""幾何：翻轉、旋轉 90°、矩形裁切。都在記憶體裡做，不改原始檔；逐像素複製，不重新取樣。

影像形狀 (H, W) 或 (C, H, W)；最後兩軸是列、欄。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """原始檔 → 顯示中影像的幾何操作（依序：裁切 → 旋轉 → 翻轉）。裁切框以原始檔的像素座標表示。"""
    crop: tuple[int, int, int, int] | None = None   # (x0, y0, x1, y1)，右下不含
    quarter_turns: int = 0                           # 逆時針 90° 的次數
    flip_h: bool = False
    flip_v: bool = False

    def is_identity(self) -> bool:
        return self.crop is None and self.quarter_turns % 4 == 0 and not self.flip_h and not self.flip_v


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[..., :, ::-1])


def flip_vertical(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[..., ::-1, :])


def rotate(image: np.ndarray, quarter_turns: int) -> np.ndarray:
    """逆時針 90° × quarter_turns。"""
    return np.ascontiguousarray(np.rot90(image, quarter_turns % 4, axes=(-2, -1)))


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = clamp_box(box, image.shape[-1], image.shape[-2])
    return np.ascontiguousarray(image[..., y0:y1, x0:x1])


def clamp_box(box: tuple[float, float, float, float], width: int, height: int,
              minimum: int = 16) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(width, int(round(x1))), min(height, int(round(y1)))
    if x1 - x0 < minimum:
        x1 = min(width, x0 + minimum)
        x0 = max(0, x1 - minimum)
    if y1 - y0 < minimum:
        y1 = min(height, y0 + minimum)
        y0 = max(0, y1 - minimum)
    return x0, y0, x1, y1


def apply(image: np.ndarray, g: Geometry) -> np.ndarray:
    out = image
    if g.crop is not None:
        out = crop(out, g.crop)
    if g.quarter_turns % 4:
        out = rotate(out, g.quarter_turns)
    if g.flip_h:
        out = flip_horizontal(out)
    if g.flip_v:
        out = flip_vertical(out)
    return out if out is not image else image


def displayed_box_to_source(box: tuple[float, float, float, float], g: Geometry,
                            source_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """顯示中影像上畫的框 → 原始檔座標（把翻轉、旋轉倒回去，再加上原本的裁切起點）。"""
    w, h = source_size
    if g.crop is not None:
        cx0, cy0, cx1, cy1 = g.crop
        w, h = cx1 - cx0, cy1 - cy0
    else:
        cx0 = cy0 = 0
    # 顯示中影像（旋轉、翻轉後）的尺寸
    turns = g.quarter_turns % 4
    dw, dh = (h, w) if turns % 2 else (w, h)
    pts = [(box[0], box[1]), (box[2], box[3]), (box[0], box[3]), (box[2], box[1])]
    back = []
    for x, y in pts:
        if g.flip_v:
            y = dh - y
        if g.flip_h:
            x = dw - x
        # 倒轉逆時針 90°×turns：顯示中影像＝rot90(轉之前)，轉之前的寬＝顯示中的高；
        # np.rot90 逆時針一次：新 (x', y') 對應轉之前的 (W − y', x')
        cw, ch = dw, dh
        for _ in range(turns):
            x, y = ch - y, x
            cw, ch = ch, cw
        back.append((x, y))
    xs, ys = [p[0] for p in back], [p[1] for p in back]
    return clamp_box((cx0 + min(xs), cy0 + min(ys), cx0 + max(xs), cy0 + max(ys)), source_size[0], source_size[1])


def source_box_to_displayed(box: tuple[float, float, float, float], g: Geometry,
                            source_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """原始檔座標的框 → 顯示中影像上的框（displayed_box_to_source 的反方向）。框跟目前的裁切範圍沒有交集就回傳 None。"""
    x0, y0, x1, y1 = box
    w, h = source_size
    if g.crop is not None:
        cx0, cy0, cx1, cy1 = g.crop
        x0, y0, x1, y1 = max(x0, cx0) - cx0, max(y0, cy0) - cy0, min(x1, cx1) - cx0, min(y1, cy1) - cy0
        w, h = cx1 - cx0, cy1 - cy0
    if x1 <= x0 or y1 <= y0:
        return None
    turns = g.quarter_turns % 4
    pts = [(x0, y0), (x1, y1), (x0, y1), (x1, y0)]
    out = []
    for x, y in pts:
        cw, ch = w, h
        for _ in range(turns):
            # np.rot90 逆時針一次：轉之前的 (x, y) → (y, W − x)，寬高互換
            x, y = y, cw - x
            cw, ch = ch, cw
        if g.flip_h:
            x = cw - x
        if g.flip_v:
            y = ch - y
        out.append((x, y))
    dw, dh = (h, w) if turns % 2 else (w, h)
    xs, ys = [p[0] for p in out], [p[1] for p in out]
    return clamp_box((min(xs), min(ys), max(xs), max(ys)), dw, dh)
