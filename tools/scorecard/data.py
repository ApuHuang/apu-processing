"""測試素材：每組有原始堆疊，以及參考流程每一步的結果（去光 → 校色 → 細節 → 降噪 → 最終成品）。

素材放哪裡、資料夾與檔名規則寫在本機設定檔 `samples.local.json`（不進 git，跟私人交接說明一起帶），
路徑可用環境變數 APU_SAMPLES 指定；素材根目錄可用 APU_TESTDATA 蓋掉。格式：

    {"root": "~/…/TestData",
     "stages": {"gradient": "*_A.fit", "color": "*_B.fit", "detail": "*_C.fit", "denoise": "*_D.fit", "final": "*_E.png"},
     "raw_patterns": {"代號": "原始堆疊檔名規則（預設＝檔名開頭.fit）"},
     "exclude_right": {"代號": 右側排除比例},
     "samples": {"代號": ["資料夾", "檔名開頭"]}}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from apu_processing.imageio import load_image

HERE = Path(__file__).resolve().parent
LINEAR_STAGES = ("gradient", "color", "detail", "denoise")


@cache
def config() -> dict:
    path = Path(os.environ.get("APU_SAMPLES", HERE / "samples.local.json"))
    if not path.is_file():
        raise SystemExit(f"找不到測試素材設定檔 {path}（見 tools/scorecard/data.py 的說明）")
    return json.loads(path.read_text(encoding="utf-8"))


def root() -> Path:
    return Path(os.environ.get("APU_TESTDATA") or os.path.expanduser(config()["root"]))


def splits() -> dict[str, list[str]]:
    return json.loads((HERE / "splits.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Sample:
    key: str

    @property
    def folder(self) -> Path:
        return root() / config()["samples"][self.key][0]

    @property
    def stem(self) -> str:
        return config()["samples"][self.key][1]

    def _find(self, pattern: str) -> Path | None:
        hits = sorted(self.folder.glob(pattern))
        return hits[0] if hits else None

    @property
    def raw_path(self) -> Path:
        pattern = config().get("raw_patterns", {}).get(self.key)
        return self._find(pattern) if pattern else self.folder / f"{self.stem}.fit"

    def stage_path(self, stage: str) -> Path | None:
        """stage：gradient（去光）、color（校色）、detail（細節）、denoise（降噪）、final（最終成品）。沒有這一步就回傳 None。"""
        return self._find(config()["stages"][stage])

    @property
    def linear_final_path(self) -> Path:
        """最終成品之前的最後一個線性檔（通常是降噪後）。"""
        for stage in reversed(LINEAR_STAGES):
            path = self.stage_path(stage)
            if path is not None:
                return path
        return self.raw_path

    def before(self, stage: str) -> Path:
        """某一步之前的最後一個線性檔（例如 before("denoise")：降噪前、細節後；沒有細節就用校色後……）。"""
        for earlier in reversed(LINEAR_STAGES[:LINEAR_STAGES.index(stage)]):
            path = self.stage_path(earlier)
            if path is not None:
                return path
        return self.raw_path

    def load_raw(self) -> np.ndarray:
        return load_image(self.raw_path)[0]

    def load_linear_final(self) -> np.ndarray:
        return load_image(self.linear_final_path)[0]

    @cached_property
    def reference(self) -> np.ndarray:
        """參考流程的最終成品（使用者自己的後製），(3, H, W) float32 0–1。"""
        with Image.open(self.stage_path("final")) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return np.ascontiguousarray(np.moveaxis(rgb, -1, 0))

    def valid_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """參與評分的範圍：去掉最外圈 1%（疊圖邊緣）與設定檔裡要排除的區域。"""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        my, mx = max(4, h // 100), max(4, w // 100)
        right = w - mx - int(round(w * config().get("exclude_right", {}).get(self.key, 0.0)))
        mask[my:h - my, mx:right] = True
        return mask


def all_samples() -> list[Sample]:
    return [Sample(k) for k in config()["samples"]]
