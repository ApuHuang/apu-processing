"""長時間運算的進度回報與取消（APU Astro 系列共用的慣例）。

引擎函式接受 progress(fraction, message) 與 cancel() 兩個 callback；cancel() 為真時丟 Cancelled。
"""

from __future__ import annotations

from collections.abc import Callable

from .i18n import Msg

Progress = Callable[[float, Msg | str], None]
CancelCheck = Callable[[], bool]


class Cancelled(Exception):
    """使用者按了停止。"""


def no_progress(_fraction: float, _message: Msg | str) -> None:
    pass


def never_cancel() -> bool:
    return False


def check(cancel: CancelCheck) -> None:
    if cancel():
        raise Cancelled
