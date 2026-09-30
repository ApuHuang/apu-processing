"""成品微調：只作用在顯示／成品輸出（PNG、JPEG、TIFF），FITS 仍是線性資料。

順序與語意照 Swift 版 DisplayAdjustment：
  明暗（×2^exposure）→ 對比（正值＝柔和 S 曲線，負值＝往中間灰壓）→ 各色版曲線 → 主曲線
  → 去綠（SCNR 平均中性保護：綠色比紅藍平均多的部分往回拉）→ 飽和度（Rec.709 亮度為軸）
前四步都是逐值的一維函數，合成每個色版一張 4096 階的查表，拖滑桿時每幀只查表。
曲線是保單調的三次 Hermite（Fritsch–Carlson 切線），控制點拉起來的手感與 Swift 版相同。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

LUT_SIZE = 4096
IDENTITY: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0))


def _identity() -> list[list[float]]:
    return [list(p) for p in IDENTITY]


@dataclass
class ToneCurves:
    master: list[list[float]] = field(default_factory=_identity)
    red: list[list[float]] = field(default_factory=_identity)
    green: list[list[float]] = field(default_factory=_identity)
    blue: list[list[float]] = field(default_factory=_identity)

    def is_identity(self) -> bool:
        return all(_is_identity(getattr(self, k)) for k in ("master", "red", "green", "blue"))


def _is_identity(points: list[list[float]]) -> bool:
    return all(abs(x - y) < 1e-9 for x, y in points)


@dataclass
class DisplayAdjustments:
    exposure: float = 0.0        # −1～1：×2^exposure
    contrast: float = 0.0        # −1～1
    saturation: float = 1.0      # 0～2
    green_removal: float = 0.0   # 0～1
    curves: ToneCurves = field(default_factory=ToneCurves)

    def is_neutral(self) -> bool:
        return (abs(self.exposure) < 1e-9 and abs(self.contrast) < 1e-9 and abs(self.saturation - 1) < 1e-9
                and self.green_removal < 1e-9 and self.curves.is_identity())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DisplayAdjustments:
        if not isinstance(data, dict):
            return cls()
        c = data.get("curves") or {}
        curves = ToneCurves(**{k: [list(map(float, p)) for p in c[k]] for k in ("master", "red", "green", "blue")
                               if isinstance(c.get(k), list) and len(c[k]) >= 2})
        kw = {k: float(data[k]) for k in ("exposure", "contrast", "saturation", "green_removal") if k in data}
        return cls(curves=curves, **kw)


def curve_values(points: list[list[float]], x: np.ndarray) -> np.ndarray:
    """保單調的三次 Hermite 曲線（與 Swift ToneCurve.Evaluator 同一套切線）。"""
    pts = sorted((min(1.0, max(0.0, px)), min(1.0, max(0.0, py))) for px, py in points)
    if len(pts) < 2:
        return np.clip(x, 0, 1)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    slopes = (ys[1:] - ys[:-1]) / np.maximum(1e-8, xs[1:] - xs[:-1])
    tang = np.empty(len(pts))
    tang[0], tang[-1] = slopes[0], slopes[-1]
    for i in range(1, len(pts) - 1):
        a, b = slopes[i - 1], slopes[i]
        tang[i] = 0.0 if a * b <= 0 else 2 * a * b / (a + b)
    x = np.clip(x, 0, 1)
    seg = np.clip(np.searchsorted(xs, x, side="left") - 1, 0, len(pts) - 2)
    width = np.maximum(1e-8, xs[seg + 1] - xs[seg])
    t = (x - xs[seg]) / width
    t2, t3 = t * t, t * t * t
    y = ((2 * t3 - 3 * t2 + 1) * ys[seg] + (t3 - 2 * t2 + t) * width * tang[seg]
         + (-2 * t3 + 3 * t2) * ys[seg + 1] + (t3 - t2) * width * tang[seg + 1])
    y = np.where(x <= xs[0], ys[0], np.where(x >= xs[-1], ys[-1], y))
    return np.clip(y, 0, 1)


def _contrast(v: np.ndarray, c: float) -> np.ndarray:
    if c >= 0:
        smooth = v * v * (3 - 2 * v)
        return np.clip(v + (smooth - v) * c, 0, 1)
    soft = 0.5 + (v - 0.5) * (1 + c)
    return np.clip(v + (soft - v) * -c, 0, 1)


def lookup_tables(adj: DisplayAdjustments, channels: int = 3) -> np.ndarray:
    """(色版數, LUT_SIZE+1) 的查表：明暗、對比、色版曲線、主曲線合成一張。"""
    x = np.linspace(0, 1, LUT_SIZE + 1)
    base = _contrast(np.clip(x * 2.0 ** min(1, max(-1, adj.exposure)), 0, 1), min(1, max(-1, adj.contrast)))
    per = [adj.curves.red, adj.curves.green, adj.curves.blue] if channels >= 3 else [IDENTITY]
    return np.stack([curve_values(adj.curves.master, curve_values(p, base)) for p in per]).astype(np.float32)


def apply(display: np.ndarray, adj: DisplayAdjustments) -> np.ndarray:
    """0–1 顯示影像 → 微調後的 0–1 顯示影像（形狀不變）。"""
    if adj.is_neutral():
        return display
    mono = display.ndim == 2
    planes = display[None] if mono else display
    tables = lookup_tables(adj, planes.shape[0])
    out = np.empty(planes.shape, np.float32)
    for c in range(planes.shape[0]):
        idx = np.clip(planes[c] * LUT_SIZE + 0.5, 0, LUT_SIZE).astype(np.uint16)
        out[c] = tables[min(c, len(tables) - 1)][idx]
    if not mono and planes.shape[0] >= 3:
        g = min(1.0, max(0.0, adj.green_removal))
        if g > 1e-6:
            neutral = (out[0] + out[2]) * 0.5
            excess = np.maximum(out[1] - neutral, 0)
            out[1] -= excess * np.float32(g)
        s = min(2.0, max(0.0, adj.saturation))
        if abs(s - 1) > 1e-6:
            lum = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]
            out = np.clip(lum[None] + (out - lum[None]) * np.float32(s), 0, 1)
    return out[0] if mono else out
