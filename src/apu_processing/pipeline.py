"""處理管線：線性階段（去光 → 校色 → 降噪 → 細節／縮星）＋ 顯示階段（拉伸 → 成品微調）。

引擎不 import tkinter；命令列、測試、評分工具與介面都呼叫這裡。
各階段的設定分開，之後做分階段快取時，只改後段設定不必重算前段。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from . import background, color, compose, denoise, detail, stretch
from .i18n import Msg
from .progress import CancelCheck, Progress, check, never_cancel, no_progress


@dataclass(frozen=True)
class ProcessingSettings:
    background_enabled: bool = True
    background: background.BackgroundSettings = field(default_factory=background.BackgroundSettings)
    color_enabled: bool = True
    denoise_enabled: bool = True
    denoise: denoise.DenoiseSettings = field(default_factory=denoise.DenoiseSettings)
    detail_enabled: bool = True
    detail: detail.DetailSettings = field(default_factory=detail.DetailSettings)
    stretch: stretch.StretchSettings = field(default_factory=stretch.StretchSettings)
    # 多濾鏡合成的哈伯色調（拉伸後去綠，0–1）。只有合成模式會設；單張影像一律 0
    palette: float = 0.0

    @classmethod
    def recommended(cls) -> ProcessingSettings:
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ProcessingSettings:
        """讀舊設定檔：缺的欄位用預設值補，不認得的欄位略過。"""
        def build(kind, values):
            if not isinstance(values, dict):
                return kind()
            kwargs = {}
            for f in fields(kind):
                if f.name not in values:
                    continue
                sub = getattr(kind(), f.name)
                kwargs[f.name] = build(type(sub), values[f.name]) if hasattr(sub, "__dataclass_fields__") else values[f.name]
            return kind(**kwargs)
        return build(cls, data)


@dataclass
class Result:
    linear: np.ndarray
    display: np.ndarray
    balance: color.ColorBalance | None
    noise: stretch.StretchNoise
    computed: tuple[str, ...] = ()   # 這次實際重算的階段（其他沿用快取）


STAGES = ("background", "color", "denoise", "detail", "stretch")   # 另有選用的 "palette"（合成的哈伯色調）


class Processor:
    """分階段快取：同一張來源，設定改了只重算那一段之後的部分。

    每段的快取鍵＝前一段的鍵＋這段的設定。校色只存倍率（套用很快），省一份完整尺寸影像。
    pixel_scale：來源是縮圖時（快速預覽），一個像素等於完整尺寸的幾個像素，拉伸量噪聲時要換算。
    """

    def __init__(self, pixel_scale: float = 1.0):
        self.pixel_scale = pixel_scale
        self.source: np.ndarray | None = None
        self.sample_mask: np.ndarray | None = None
        self.revision = 0
        self._cache: dict[str, tuple[tuple, object]] = {}

    def set_source(self, image: np.ndarray | None, sample_mask: np.ndarray | None = None) -> None:
        """sample_mask：(H, W) 布林，去光可以取樣的像素（疊圖覆蓋足夠的範圍）；None＝全部。"""
        self.source = image
        self.sample_mask = sample_mask
        self.revision += 1
        self._cache.clear()

    def _get(self, stage: str, key: tuple):
        hit = self._cache.get(stage)
        return hit[1] if hit is not None and hit[0] == key else None

    def run(self, settings: ProcessingSettings, progress: Progress = no_progress,
            cancel: CancelCheck = never_cancel) -> Result:
        if self.source is None:
            raise ValueError("no source")
        computed = []
        key = (self.revision,)

        key = key + (settings.background_enabled, settings.background if settings.background_enabled else None)
        x = self._get("background", key)
        if x is None:
            progress(0.1, Msg("stage.background"))
            x = background.correct(self.source, settings.background, cancel, self.sample_mask) \
                if settings.background_enabled else self.source
            self._cache["background"] = (key, x)
            computed.append("background")
        check(cancel)

        key = key + (settings.color_enabled,)
        balance = self._get("color", key)
        if balance is None:
            if settings.color_enabled and x.ndim == 3:
                progress(0.3, Msg("stage.color"))
                balance = color.measure_balance(x)
            else:
                balance = False
            self._cache["color"] = (key, balance)
            computed.append("color")
        check(cancel)

        key = key + (settings.denoise_enabled, settings.denoise if settings.denoise_enabled else None)
        y = self._get("denoise", key)
        if y is None:
            y = color.apply(x, balance) if balance else x
            if settings.denoise_enabled:
                progress(0.5, Msg("stage.denoise"))
                y = denoise.reduce(y, settings.denoise, cancel)
            self._cache["denoise"] = (key, y)
            computed.append("denoise")
        check(cancel)

        key = key + (settings.detail_enabled, settings.detail if settings.detail_enabled else None)
        hit = self._get("detail", key)
        if hit is None:
            z = y
            if settings.detail_enabled:
                progress(0.65, Msg("stage.detail"))
                z = detail.process(y, settings.detail, cancel)
            noise = stretch.measure(z, self.pixel_scale)
            hit = (z, noise)
            self._cache["detail"] = (key, hit)
            computed.append("detail")
        z, noise = hit
        check(cancel)

        key = key + (settings.stretch,)
        shown = self._get("stretch", key)
        if shown is None:
            progress(0.85, Msg("stage.stretch"))
            shown = stretch.apply(z, settings.stretch, noise)
            self._cache["stretch"] = (key, shown)
            computed.append("stretch")

        if settings.palette > 0:
            key = key + (settings.palette,)
            hit = self._get("palette", key)
            if hit is None:
                progress(0.95, Msg("stage.palette"))
                hit = compose.hubble_palette(shown, settings.palette, self.pixel_scale)
                self._cache["palette"] = (key, hit)
                computed.append("palette")
            shown = hit
        progress(1.0, Msg("stage.done"))
        return Result(z, shown, balance or None, noise, tuple(computed))


def process(image: np.ndarray, settings: ProcessingSettings = ProcessingSettings(),
            progress: Progress = no_progress, cancel: CancelCheck = never_cancel) -> Result:
    """不留快取的一次處理（命令列、評分工具用）。"""
    p = Processor()
    p.set_source(image)
    return p.run(settings, progress, cancel)


def preview_factor(shape: tuple[int, ...], long_side: int = 1600) -> int:
    """快速預覽的縮小倍率：2、4、8 裡最小、能讓長邊 ≤ long_side 的一個（8 的因數，拉伸量噪聲才換算得準）。"""
    longest = max(shape[-2:])
    for f in (1, 2, 4, 8):
        if longest / f <= long_side:
            return f
    return 8
