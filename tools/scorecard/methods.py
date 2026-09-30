"""被評分的「方法」：輸入一組素材，輸出 (3, H, W) 0–1 顯示影像。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from apu_processing import pipeline, stretch
from apu_processing.imageio import load_image

from .data import Sample

BASELINES = Path(__file__).resolve().parents[2] / "output/baselines"
_FIT = stretch.StretchSettings(strength=0.5 - stretch.DEFAULT_BOOST / 2)


def simple_stretch(linear: np.ndarray, target: float = 0.15) -> np.ndarray:
    """最簡單的自動拉伸（三色各自）：各色版黑點＝中位數 − 2.8 MAD，中間調把中位數拉到 target。
    只當「沒有任何處理」的對照組。"""
    out = np.empty_like(linear)
    for c in range(linear.shape[0]):
        x = linear[c]
        sample = x[::7, ::7]
        med = float(np.median(sample))
        mad = float(np.median(np.abs(sample - med))) * 1.4826
        c0 = med - 2.8 * mad
        top = float(np.quantile(sample, 0.99995))
        y = np.clip((x - c0) / max(1e-9, top - c0), 0, 1)
        x0 = (med - c0) / max(1e-9, top - c0)
        m = x0 * (1 - target) / (x0 - 2 * target * x0 + target)
        out[c] = ((m - 1) * y / ((2 * m - 1) * y - m)).astype(np.float32)
    return out


_PROCESSOR: dict[str, pipeline.Processor] = {}


def _cached_processor(sample: Sample) -> pipeline.Processor:
    """同一張素材的幾個方法共用分階段快取：只改拉伸或後段設定時不重跑前段。只留最近一張，記憶體才夠。"""
    p = _PROCESSOR.get(sample.key)
    if p is None:
        _PROCESSOR.clear()
        p = pipeline.Processor()
        p.set_source(sample.load_raw())
        _PROCESSOR[sample.key] = p
    return p


def _from_folder(name: str) -> Callable[[Sample], np.ndarray]:
    def load(sample: Sample) -> np.ndarray:
        return load_image(BASELINES / name / f"{sample.key}.tif")[0]
    return load


METHODS: dict[str, Callable[[Sample], np.ndarray]] = {
    "ref": lambda s: s.reference,
    "v098": _from_folder("v098"),
    "ref-simple": lambda s: simple_stretch(s.load_linear_final()),
    "raw-simple": lambda s: simple_stretch(s.load_raw()),
    # 新的依噪聲拉伸，套在參考流程最後的線性檔上：只驗拉伸本身
    # 擬合值本身（不含 DEFAULT_BOOST 的加深）：strength 0.5 − BOOST/2
    "ref-apu": lambda s: stretch.apply(s.load_linear_final(), _FIT),
    "ref-apu-lum": lambda s: stretch.apply_luminance(s.load_linear_final(), _FIT),
    # APU Processing 的完整管線（從原始堆疊）
    "apu": lambda s: _cached_processor(s).run(pipeline.ProcessingSettings.recommended()).display,
}


def _parse_value(text: str):
    if "," in text:
        return tuple(float(v) for v in text.split(","))
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return float(text)
    except ValueError:
        return text


def settings_with(overrides: str) -> pipeline.ProcessingSettings:
    """「denoise.amount=0.6;detail.sharpen=0.8」→ 建議設定加上這些改動。"""
    from dataclasses import replace

    s = pipeline.ProcessingSettings.recommended()
    for item in filter(None, overrides.split(";")):
        path, value = item.split("=", 1)
        parts = path.strip().split(".")
        v = _parse_value(value.strip())
        if len(parts) == 1:
            s = replace(s, **{parts[0]: v})
        else:
            sub = getattr(s, parts[0])
            s = replace(s, **{parts[0]: replace(sub, **{parts[1]: v})})
    return s


def resolve(name: str) -> Callable[[Sample], np.ndarray]:
    """方法名稱；「apu:設定改動」是帶參數的完整管線，例如 apu:denoise.k_chroma=3.5,3,2.5,2.5,2.5"""
    if name in METHODS:
        return METHODS[name]
    if name.startswith("apu:"):
        settings = settings_with(name[4:])
        return lambda s: _cached_processor(s).run(settings).display
    raise KeyError(name)
