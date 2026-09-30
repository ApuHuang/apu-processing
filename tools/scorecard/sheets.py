"""並排比較圖：每張挑四個 100% 裁切（明亮星雲、暗淡星雲、最大的亮星、天空），各方法與參考成品並排。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .metrics import Regions

CROP = 400


def _densest(mask: np.ndarray, avoid: list[tuple[int, int]], tile: int = CROP) -> tuple[int, int] | None:
    h, w = mask.shape
    best, where = -1.0, None
    for y in range(0, h - tile, tile // 2):
        for x in range(0, w - tile, tile // 2):
            if any(abs(y - ay) < tile and abs(x - ax) < tile for ay, ax in avoid):
                continue
            v = float(mask[y:y + tile, x:x + tile].mean())
            if v > best:
                best, where = v, (y, x)
    return where


def crop_spots(regions: Regions) -> list[tuple[str, int, int]]:
    h, w = regions.valid.shape
    spots: list[tuple[str, int, int]] = []
    for label, mask in (("明亮星雲", regions.bright), ("暗淡星雲", regions.faint)):
        pos = _densest(mask, [(y, x) for _, y, x in spots])
        if pos:
            spots.append((label, *pos))
    big = [s for s in sorted(regions.stars, key=lambda s: -s[2])
           if CROP / 2 < s[0] < h - CROP / 2 and CROP / 2 < s[1] < w - CROP / 2]
    if big:
        y, x = int(big[0][0]) - CROP // 2, int(big[0][1]) - CROP // 2
        spots.append(("最大的亮星", y, x))
    pos = _densest(regions.sky, [(y, x) for _, y, x in spots])
    if pos:
        spots.append(("天空", *pos))
    return spots


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
                 "C:/Windows/Fonts/msjh.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_sheet(path: Path, title: str, images: dict[str, np.ndarray], regions: Regions) -> None:
    spots = crop_spots(regions)
    names = list(images)
    pad, head = 6, 28
    sheet = Image.new("RGB", (len(names) * (CROP + pad) + pad + 90, len(spots) * (CROP + pad) + pad + head + 24),
                      (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    draw.text((pad, 4), title, fill=(235, 235, 235), font=font)
    for j, name in enumerate(names):
        draw.text((90 + pad + j * (CROP + pad), head), name, fill=(200, 200, 200), font=font)
    for i, (label, y, x) in enumerate(spots):
        top = head + 24 + i * (CROP + pad)
        draw.text((pad, top + CROP // 2 - 8), label, fill=(200, 200, 200), font=font)
        for j, name in enumerate(names):
            crop = images[name][:, y:y + CROP, x:x + CROP]
            rgb = (np.clip(np.moveaxis(crop, 0, -1), 0, 1) * 255 + 0.5).astype(np.uint8)
            sheet.paste(Image.fromarray(rgb), (90 + pad + j * (CROP + pad), top))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
