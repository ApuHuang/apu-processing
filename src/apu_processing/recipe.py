"""疊圖專案紀錄（*.recipe.json）：一次疊圖輸出了哪些 master、各是什麼濾鏡、哪些屬於同一個對齊組。

只讀需要的欄位（outputs[] 的 file、filter、align_group、frames、coverage），不認格式名稱：
有 outputs 列表、每一項有 file 就能用。檔名是相對於紀錄檔所在資料夾。
同一個對齊組（同一套器材）的 master 尺寸相同、像素對齊，可以直接合成；不同對齊組之間沒有對齊。
引擎不 import tkinter。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .i18n import Msg

SUFFIX = ".recipe.json"


@dataclass(frozen=True)
class StackOutput:
    path: Path
    filter: str
    group: str
    frames: int
    coverage: Path | None


@dataclass(frozen=True)
class StackProject:
    path: Path
    target: str
    outputs: tuple[StackOutput, ...]

    def groups(self) -> dict[str, list[StackOutput]]:
        grouped: dict[str, list[StackOutput]] = {}
        for o in self.outputs:
            grouped.setdefault(o.group, []).append(o)
        return grouped

    def main_group(self) -> tuple[str, list[StackOutput]]:
        """master 最多的對齊組（一樣多取先出現的）。"""
        return max(self.groups().items(), key=lambda item: len(item[1]))


def is_recipe(path: Path | str) -> bool:
    return Path(path).name.lower().endswith(SUFFIX)


def load(path: Path | str) -> StackProject:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(Msg("msg.recipe_unreadable", name=path.name)) from e
    entries = data.get("outputs") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError(Msg("msg.recipe_unreadable", name=path.name))
    outputs = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        file = path.parent / str(entry["file"])
        if not file.is_file():
            continue
        coverage = path.parent / str(entry["coverage"]) if entry.get("coverage") else None
        outputs.append(StackOutput(
            path=file,
            filter=str(entry.get("filter") or "").strip(),
            group=str(entry.get("align_group") or ""),
            frames=int(entry.get("frames") or 0),
            coverage=coverage if coverage is not None and coverage.is_file() else None,
        ))
    if not outputs:
        raise ValueError(Msg("msg.recipe_empty", name=path.name))
    target = str(data.get("target") or path.name[: -len(SUFFIX)])
    return StackProject(path, target, tuple(outputs))
