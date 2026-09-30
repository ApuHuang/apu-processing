"""曲線編輯器：主控／紅／綠／藍四條曲線，背後畫直方圖。

操作與 Swift 版相同：拖曳控制點；雙擊空白處新增、雙擊控制點刪除（頭尾兩點不能刪）。
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

import numpy as np

from .darkroom import Darkroom, Segmented, _trace
from .display import IDENTITY, ToneCurves, curve_values
from .i18n import tr

CHANNELS = ("master", "red", "green", "blue")
CHANNEL_COLOR = {"master": "#ebebeb", "red": "#ff6961", "green": "#7ee081", "blue": "#6ab0ff"}
HIST_COLOR = {"master": "#4a4a4a", "red": "#5a2e2e", "green": "#2e5a33", "blue": "#2e405a"}


class CurveEditor(tk.Frame):
    def __init__(self, master: tk.Misc, app, on_change: Callable[[ToneCurves], None]):
        super().__init__(master, bg=Darkroom.panel)
        self.app = app
        self.on_change = on_change
        self.curves = ToneCurves()
        self.channel = tk.StringVar(value="master")
        self.hist: dict[str, np.ndarray] = {}
        Segmented(self, app, [(k, tr(f"gui.curve.{k}")) for k in CHANNELS], self.channel, stretch=True).pack(
            fill="x", pady=(0, app.px(6)))
        size = app.px(260)
        self.size = size
        self.pad = app.px(6)
        self.canvas = tk.Canvas(self, width=size, height=size, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack()
        self._drag: int | None = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))
        self.canvas.bind("<Double-Button-1>", self._double)
        _trace(self, self.channel, self.redraw)
        self.redraw()

    # ------------------------------------------------------------------ 資料

    def set_curves(self, curves: ToneCurves) -> None:
        self.curves = curves
        self.redraw()

    def set_histogram(self, display: np.ndarray | None) -> None:
        """display：(3, H, W) 的顯示影像（通常是預覽縮圖）。"""
        self.hist = {}
        if display is not None:
            planes = display if display.ndim == 3 else display[None]
            sub = planes[:, ::2, ::2]
            bins = np.linspace(0, 1, 129)
            self.hist["master"] = np.histogram(sub.mean(axis=0), bins)[0].astype(float)
            for i, k in enumerate(("red", "green", "blue")):
                self.hist[k] = np.histogram(sub[min(i, len(sub) - 1)], bins)[0].astype(float)
        self.redraw()

    def _points(self) -> list[list[float]]:
        return getattr(self.curves, self.channel.get())

    def _set_points(self, pts: list[list[float]]) -> None:
        setattr(self.curves, self.channel.get(), pts)
        self.on_change(self.curves)
        self.redraw()

    # ------------------------------------------------------------------ 座標

    def _to_screen(self, x: float, y: float) -> tuple[float, float]:
        span = self.size - 2 * self.pad
        return self.pad + x * span, self.pad + (1 - y) * span

    def _to_value(self, sx: float, sy: float) -> tuple[float, float]:
        span = self.size - 2 * self.pad
        return min(1, max(0, (sx - self.pad) / span)), min(1, max(0, 1 - (sy - self.pad) / span))

    def _hit(self, sx: float, sy: float) -> int | None:
        best, where = None, self.app.px(9)
        for i, (x, y) in enumerate(self._points()):
            px, py = self._to_screen(x, y)
            d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
            if d < where:
                best, where = i, d
        return best

    # ------------------------------------------------------------------ 操作

    def _press(self, e: tk.Event) -> None:
        self._drag = self._hit(e.x, e.y)

    def _move(self, e: tk.Event) -> None:
        if self._drag is None:
            return
        pts = [list(p) for p in self._points()]
        order = sorted(range(len(pts)), key=lambda i: pts[i][0])
        rank = order.index(self._drag)
        x, y = self._to_value(e.x, e.y)
        lo = pts[order[rank - 1]][0] + 0.01 if rank > 0 else 0.0
        hi = pts[order[rank + 1]][0] - 0.01 if rank < len(order) - 1 else 1.0
        if rank == 0:
            x = 0.0 if pts[self._drag][0] == 0.0 else min(x, hi)
        elif rank == len(order) - 1:
            x = 1.0 if pts[self._drag][0] == 1.0 else max(x, lo)
        else:
            x = min(max(x, lo), hi)
        pts[self._drag] = [x, y]
        self._set_points(pts)

    def _double(self, e: tk.Event) -> None:
        pts = [list(p) for p in self._points()]
        hit = self._hit(e.x, e.y)
        if hit is not None:
            order = sorted(range(len(pts)), key=lambda i: pts[i][0])
            if hit not in (order[0], order[-1]):
                del pts[hit]
                self._set_points(pts)
            return
        x, _ = self._to_value(e.x, e.y)
        y = float(curve_values(pts, np.array([x]))[0])
        pts.append([x, y])
        self._set_points(sorted(pts))

    def reset_channel(self) -> None:
        self._set_points([list(p) for p in IDENTITY])

    # ------------------------------------------------------------------ 繪製

    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        ch = self.channel.get()
        s, p = self.size, self.pad
        span = s - 2 * p
        hist = self.hist.get(ch)
        if hist is not None and hist.max() > 0:
            h = np.sqrt(hist / hist.max())
            bw = span / len(h)
            for i, v in enumerate(h):
                if v > 0:
                    c.create_rectangle(p + i * bw, p + (1 - v) * span, p + (i + 1) * bw, p + span,
                                       fill=HIST_COLOR[ch], outline="")
        for k in (0.25, 0.5, 0.75):
            c.create_line(p + k * span, p, p + k * span, p + span, fill="#2c2c2c")
            c.create_line(p, p + k * span, p + span, p + k * span, fill="#2c2c2c")
        c.create_line(p, p + span, p + span, p, fill="#333333", dash=(2, 3))
        xs = np.linspace(0, 1, 129)
        ys = curve_values(self._points(), xs)
        coords = []
        for x, y in zip(xs, ys):
            coords.extend(self._to_screen(float(x), float(y)))
        c.create_line(*coords, fill=CHANNEL_COLOR[ch], width=2, smooth=False)
        r = self.app.px(4)
        for x, y in self._points():
            sx, sy = self._to_screen(x, y)
            c.create_oval(sx - r, sy - r, sx + r, sy + r, fill="#1a1a1a", outline=CHANNEL_COLOR[ch], width=2)
