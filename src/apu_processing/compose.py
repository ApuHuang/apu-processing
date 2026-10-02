"""多濾鏡合成：幾張單色的線性 master 依色版組成一張 RGB，再進原本的處理流程。

預設組合（角色 → 色版）：
- HOO：Ha → R，OIII → G、B
- SHO：SII → R，Ha → G，OIII → B
- RGB：R、G、B 各一張

不同濾鏡之間的亮度不能直接比（Ha 通常比 OIII 亮很多），所以每個通道先扣自己的天空、除以自己的 8 px 尺度噪聲
（σ₈，跟拉伸用的是同一個量法）再組合——噪聲對齊：合成後各色版的天空顆粒一樣，弱的通道不會被放大成一片噪聲。
想讓某個通道更明顯，調它的強度（預設 1）。2026-10-02 用 NGC1499 的 Ha／OIII 比較「噪聲對齊」與
「亮處對齊」後選定前者。

讀檔時浮點資料超過 1 會各自除以自己的峰值（imageio 的 APUSCALE），每張的單位可能不同；
這裡只用扣完天空後的噪聲比，跟原本單位無關。

只合成同尺寸、已經對齊的影像（同一套器材一起疊出來的 master）；尺寸不同就拒絕，不做對齊。
引擎不 import tkinter。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from . import stretch
from .i18n import Msg

# 角色 → 這個通道放進 R、G、B 的比例
PRESETS: dict[str, tuple[tuple[str, tuple[float, float, float]], ...]] = {
    "HOO": (("Ha", (1.0, 0.0, 0.0)), ("OIII", (0.0, 1.0, 1.0))),
    "SHO": (("SII", (1.0, 0.0, 0.0)), ("Ha", (0.0, 1.0, 0.0)), ("OIII", (0.0, 0.0, 1.0))),
    "RGB": (("R", (1.0, 0.0, 0.0)), ("G", (0.0, 1.0, 0.0)), ("B", (0.0, 0.0, 1.0))),
}
# 窄帶合成的星點顏色不是真實顏色，用星點校色沒有意義
NARROWBAND = frozenset({"HOO", "SHO"})
# 自動建議時先比對通道多的組合
_SUGGEST_ORDER = ("SHO", "HOO", "RGB")
STRENGTH_RANGE = (0.0, 4.0)


def filter_key(name: object) -> str:
    """濾鏡名稱的比對鍵：只忽略大小寫與前後空白（Ha 與 H-alpha 這種拼法不同的不當成同一個）。"""
    return str(name).strip().casefold() if name is not None else ""


@dataclass(frozen=True)
class Channel:
    role: str               # 預設裡的角色，例如 "Ha"
    source: int = -1        # 用第幾張輸入；-1＝沒有指定（這個角色的色版只剩天空）
    strength: float = 1.0   # 噪聲對齊後再乘的倍率


@dataclass(frozen=True)
class ComposeSettings:
    preset: str = "HOO"
    channels: tuple[Channel, ...] = ()

    @classmethod
    def for_preset(cls, preset: str, filters: Sequence[object] = ()) -> ComposeSettings:
        """這個組合的設定；每個角色找濾鏡名稱相同、還沒被用掉的輸入，找不到就照順序補。"""
        roles = [role for role, _ in PRESETS[preset]]
        keys = [filter_key(f) for f in filters]
        used: set[int] = set()
        sources: list[int] = []
        for role in roles:
            match = next((i for i, k in enumerate(keys) if k == role.casefold() and i not in used), -1)
            if match >= 0:
                used.add(match)
            sources.append(match)
        free = [i for i in range(len(keys)) if i not in used]
        sources = [s if s >= 0 else (free.pop(0) if free else -1) for s in sources]
        return cls(preset, tuple(Channel(role, s) for role, s in zip(roles, sources)))

    @classmethod
    def recommended(cls, filters: Sequence[object]) -> ComposeSettings:
        """依 FILTER 建議組合：每個角色都找得到同名濾鏡的組合優先；都對不上就用 HOO、照順序指定。"""
        keys = {filter_key(f) for f in filters}
        for preset in _SUGGEST_ORDER:
            if all(role.casefold() in keys for role, _ in PRESETS[preset]):
                return cls.for_preset(preset, filters)
        return cls.for_preset("HOO", filters)

    def with_strength(self, index: int, strength: float) -> ComposeSettings:
        lo, hi = STRENGTH_RANGE
        channels = list(self.channels)
        channels[index] = replace(channels[index], strength=min(hi, max(lo, float(strength))))
        return replace(self, channels=tuple(channels))

    def with_source(self, index: int, source: int) -> ComposeSettings:
        channels = list(self.channels)
        channels[index] = replace(channels[index], source=int(source))
        return replace(self, channels=tuple(channels))

    @property
    def narrowband(self) -> bool:
        return self.preset in NARROWBAND


@dataclass(frozen=True)
class PlaneStats:
    sky: float
    sigma: float    # 8 px 尺度噪聲（stretch.measure 的 σ₈）


class Composer:
    """一組要合成的單色影像：建立時檢查尺寸、量好每張的天空與噪聲；compose() 可以換設定重算。"""

    def __init__(self, planes: Sequence[np.ndarray], names: Sequence[str]):
        if not planes:
            raise ValueError(Msg("msg.compose_empty"))
        for plane, name in zip(planes, names):
            if plane.ndim != 2:
                raise ValueError(Msg("msg.compose_not_mono", name=name))
            # 整張同一個值（例如疊圖失敗輸出全 0）：沒有天空也沒有噪聲可以對齊
            if float(np.ptp(plane[::4, ::4])) == 0.0:
                raise ValueError(Msg("msg.compose_blank", name=name))
        shape = planes[0].shape
        for plane, name in zip(planes[1:], names[1:]):
            if plane.shape != shape:
                raise ValueError(Msg("msg.compose_size", name=name, size=f"{plane.shape[1]}×{plane.shape[0]}",
                                     first=names[0], first_size=f"{shape[1]}×{shape[0]}"))
        self.planes = [np.asarray(p, dtype=np.float32) for p in planes]
        self.names = list(names)
        self.stats = [self._measure(p) for p in self.planes]

    @staticmethod
    def _measure(plane: np.ndarray) -> PlaneStats:
        noise = stretch.measure(plane)
        return PlaneStats(sky=float(noise.sky[0]), sigma=max(float(noise.sigma), 1e-12))

    @property
    def shape(self) -> tuple[int, int]:
        return self.planes[0].shape

    def compose(self, settings: ComposeSettings) -> np.ndarray:
        """(3, H, W) float32。每個通道 (x − 天空) / σ₈ × 強度，依比例放進色版（同一色版有幾個來源就取加權平均），
        再換回最小 σ₈ 的單位、加上共同的天空——量級跟原本的 master 相近，後面的處理不用改。"""
        weights = dict(PRESETS[settings.preset])
        used = [c for c in settings.channels if 0 <= c.source < len(self.planes)]
        h, w = self.shape
        if not used:
            return np.zeros((3, h, w), np.float32)
        unit = min(self.stats[c.source].sigma for c in used)
        pedestal = float(np.median([self.stats[c.source].sky for c in used]))
        out = np.zeros((3, h, w), np.float32)
        total = np.zeros(3, np.float64)
        for c in used:
            s = self.stats[c.source]
            gain = np.float32(c.strength * unit / s.sigma)
            normalized = (self.planes[c.source] - np.float32(s.sky)) * gain
            for k, share in enumerate(weights[c.role]):
                if share:
                    out[k] += np.float32(share) * normalized
                    total[k] += share
        for k in range(3):
            if total[k] > 0:
                out[k] /= np.float32(total[k])
        out += np.float32(pedestal)
        return out
