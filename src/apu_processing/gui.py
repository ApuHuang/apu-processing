"""視窗介面，APU Astro 系列同一套暗房介面：頂部列、中間影像、右側可收合參數面板、底部狀態列。

主畫面是 ProcessingView(tk.Frame)，不自己建立 tk.Tk()；gui.main() 才建立視窗、選單列與快捷鍵。
將來 APU Astro 整合 App 可以把 ProcessingView 當成一個分頁，用 open_image(path) 開檔。
對外：open_image(path)、ask_open()、ask_save()、is_busy()、close()、rebuild()（換語言後重建）。

處理流程：開檔 → 背景執行緒跑建議設定（大圖先用縮圖出快速預覽，再換完整結果）→ 改任何設定停手 0.3 秒自動更新，
只算最新一次（舊的取消）；成品微調只重畫畫面，不重跑管線。
啟動：python -m apu_processing.gui [影像檔]
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np

from . import __version__, display, geometry, imageio, pipeline, stretch
from .canvas import ImageCanvas
from .curves import CurveEditor
from .darkroom import (IS_MAC, Darkroom, Fonts, MetricRow, PanelGroup, ParameterSlider, ParameterToggle,
                       Segmented, Tooltip, dark_title_bar, enable_dpi_awareness, setup_style)
from .detail import _block_mean
from .i18n import APP_NAME, LANGUAGES, get_language, set_language, tr
from .progress import Cancelled
from .settings import load_settings, save_settings

ASSETS = Path(__file__).parent / "assets"
ICON = ASSETS / "app.ico"
OPEN_SHORTCUT = "⌘O" if IS_MAC else "Ctrl+O"
DEBOUNCE_MS = 300
QUICK_LIMIT = 1600           # 長邊超過這個就先出快速預覽
OPEN_TYPES = [("影像", "*.fit *.fits *.fts *.tif *.tiff *.png"), ("FITS", "*.fit *.fits *.fts"), ("*", "*")]


class _Job:
    """背景工作：一次處理請求。cancel 設起來就停；generation 較舊的結果丟掉。"""

    def __init__(self, generation: int):
        self.generation = generation
        self.cancelled = threading.Event()


class ProcessingView(tk.Frame):
    def __init__(self, parent: tk.Misc, root: tk.Tk, *,
                 on_language: Callable[[str], None] | None = None, show_language: bool = True):
        """on_language：按下頂部列的語言切換時呼叫（由外面換語言、重建 View 與選單列）；
        沒給就自己換語言並 rebuild()。show_language=False 時頂部列不顯示語言切換。"""
        super().__init__(parent, bg=Darkroom.canvas)
        self.root = root
        self.on_language = on_language
        self.show_language = show_language
        self.path: Path | None = None
        self.header = None
        self.loaded: np.ndarray | None = None          # 原始檔（未裁切旋轉）
        self.loaded_display: np.ndarray | None = None
        self.geometry = geometry.Geometry()
        self.source: np.ndarray | None = None
        self.original_display: np.ndarray | None = None
        self.current: pipeline.Result | None = None
        self.current_is_full = False
        self.previous_display: np.ndarray | None = None
        self.processor = pipeline.Processor()
        self.quick = pipeline.Processor()
        self.quick_factor = 1
        self.generation = 0
        self.job: _Job | None = None
        self.events: queue.Queue = queue.Queue()
        self.popover: tk.Toplevel | None = None
        self.popover_owner: tk.Widget | None = None
        self._debounce: str | None = None
        self._status_key = ("gui.status.start", {})

        st = load_settings()
        self.panel_state: dict[str, bool] = dict(st.get("panel", {}))
        self.settings = pipeline.ProcessingSettings.from_dict(st.get("processing", {}))
        self.adjust = display.DisplayAdjustments.from_dict(st.get("display", {}))
        self._make_vars()

        self._scale = root.winfo_fpixels("1i") / 96.0
        self.fonts = Fonts(root)
        setup_style(root, self.px, self.fonts)
        # 這個 View 專用的事件標籤：加在自己底下每個元件上，不用 bind_all，
        # 跟別的畫面放在同一個視窗時（整合版的分頁）不會互相搶事件
        self._tag = f"ApuProcessingView{id(self)}"
        self.bind_class(self._tag, "<Button-1>", self._maybe_close_popover, add="+")
        self.bind_class(self._tag, "<Escape>", lambda _e: self.close_popover())
        self.bind_class(self._tag, "<MouseWheel>", self._scroll_panel, add="+")
        self._build()
        self._poll_job = self.after(50, self._poll)

    # ------------------------------------------------------------------ 共用（暗房元件會呼叫）

    def px(self, v: float) -> int:
        return int(round(v * self._scale))

    def last_dir(self) -> str:
        return str(self.path.parent) if self.path else load_settings().get("last_dir", "")

    def show_popover(self, owner: tk.Widget, text: str) -> None:
        D = Darkroom
        self.close_popover()
        top = tk.Toplevel(self.root)
        top.overrideredirect(True)
        top.configure(bg=D.separator)
        tk.Label(top, text=text, font=self.fonts.ui, fg=D.label, bg=D.group_header, justify="left",
                 wraplength=self.px(360), padx=self.px(16), pady=self.px(14)).pack(padx=1, pady=1)
        top.update_idletasks()
        x = owner.winfo_rootx() - top.winfo_reqwidth() - self.px(8)
        y = owner.winfo_rooty() - self.px(6)
        top.geometry(f"+{max(x, self.root.winfo_rootx())}+{y}")
        self.popover, self.popover_owner = top, owner

    def close_popover(self) -> None:
        if self.popover is not None:
            try:
                self.popover.destroy()
            except tk.TclError:
                pass
        self.popover, self.popover_owner = None, None

    def _maybe_close_popover(self, event: tk.Event) -> None:
        if self.popover is None or event.widget is self.popover_owner:
            return
        try:
            if str(event.widget).startswith(str(self.popover)):
                return
        except (tk.TclError, AttributeError):
            pass
        self.close_popover()

    # ------------------------------------------------------------------ 變數

    def _make_vars(self) -> None:
        s, a = self.settings, self.adjust
        self.lang_var = tk.StringVar(value=get_language())
        self.view_var = tk.StringVar(value="current")
        self.v = {
            "background_enabled": tk.BooleanVar(value=s.background_enabled),
            "color_enabled": tk.BooleanVar(value=s.color_enabled),
            "denoise_enabled": tk.BooleanVar(value=s.denoise_enabled),
            "denoise": tk.DoubleVar(value=round(s.denoise.amount * 100)),
            "detail_enabled": tk.BooleanVar(value=s.detail_enabled),
            "stars": tk.DoubleVar(value=round(s.detail.star_reduction * 100)),
            "sharpen": tk.DoubleVar(value=round(s.detail.sharpen * 100)),
            "stretch": tk.DoubleVar(value=round(s.stretch.strength * 100)),
            "sky": tk.DoubleVar(value=round(s.stretch.background * 1000) / 10),
            "linked": tk.BooleanVar(value=s.stretch.linked),
            "bg_color": tk.DoubleVar(value=round(s.stretch.background_color * 100)),
        }
        self.d = {
            "exposure": tk.DoubleVar(value=round(a.exposure * 100)),
            "contrast": tk.DoubleVar(value=round(a.contrast * 100)),
            "saturation": tk.DoubleVar(value=round(a.saturation * 100)),
            "green_removal": tk.DoubleVar(value=round(a.green_removal * 100)),
        }
        for var in self.v.values():
            var.trace_add("write", lambda *_: self._processing_changed())
        for var in self.d.values():
            var.trace_add("write", lambda *_: self._display_changed())
        self.view_var.trace_add("write", lambda *_: self._show_view())
        # 等點擊處理完才重建（切換鈕本身會被 destroy）
        self.lang_var.trace_add("write", lambda *_: self.root.after_idle(self._language_clicked))

    def _settings_from_vars(self) -> pipeline.ProcessingSettings:
        v, s = self.v, self.settings
        return replace(
            s,
            background_enabled=bool(v["background_enabled"].get()),
            color_enabled=bool(v["color_enabled"].get()),
            denoise_enabled=bool(v["denoise_enabled"].get()),
            denoise=replace(s.denoise, amount=float(v["denoise"].get()) / 100),
            detail_enabled=bool(v["detail_enabled"].get()),
            detail=replace(s.detail, star_reduction=float(v["stars"].get()) / 100,
                           sharpen=float(v["sharpen"].get()) / 100),
            stretch=replace(s.stretch, strength=float(v["stretch"].get()) / 100,
                            background=float(v["sky"].get()) / 100, linked=bool(v["linked"].get()),
                            background_color=float(v["bg_color"].get()) / 100),
        )

    def _adjust_from_vars(self) -> display.DisplayAdjustments:
        d = self.d
        return display.DisplayAdjustments(
            exposure=float(d["exposure"].get()) / 100, contrast=float(d["contrast"].get()) / 100,
            saturation=float(d["saturation"].get()) / 100, green_removal=float(d["green_removal"].get()) / 100,
            curves=self.adjust.curves)

    def _load_vars(self, s: pipeline.ProcessingSettings, a: display.DisplayAdjustments) -> None:
        """一次改很多變數時先暫停自動更新，改完再排一次。"""
        self._suspend = True
        try:
            v, d = self.v, self.d
            v["background_enabled"].set(s.background_enabled)
            v["color_enabled"].set(s.color_enabled)
            v["denoise_enabled"].set(s.denoise_enabled)
            v["denoise"].set(round(s.denoise.amount * 100))
            v["detail_enabled"].set(s.detail_enabled)
            v["stars"].set(round(s.detail.star_reduction * 100))
            v["sharpen"].set(round(s.detail.sharpen * 100))
            v["stretch"].set(round(s.stretch.strength * 100))
            v["sky"].set(round(s.stretch.background * 1000) / 10)
            v["linked"].set(s.stretch.linked)
            v["bg_color"].set(round(s.stretch.background_color * 100))
            d["exposure"].set(round(a.exposure * 100))
            d["contrast"].set(round(a.contrast * 100))
            d["saturation"].set(round(a.saturation * 100))
            d["green_removal"].set(round(a.green_removal * 100))
            self.adjust = a
            if hasattr(self, "curve_editor"):
                self.curve_editor.set_curves(a.curves)
        finally:
            self._suspend = False
        self._processing_changed()
        self._display_changed()

    _suspend = False

    # ------------------------------------------------------------------ 版面

    def _build(self) -> None:
        self._build_top_bar()
        self._build_status_bar()
        body = tk.Frame(self, bg=Darkroom.canvas)
        body.pack(fill="both", expand=True)
        self._build_panel(body)
        tk.Frame(body, bg=Darkroom.separator, width=1).pack(side="right", fill="y")
        area = tk.Frame(body, bg=Darkroom.canvas)
        area.pack(side="left", fill="both", expand=True)
        self.canvas = ImageCanvas(area, on_zoom=lambda _z: self._refresh_summary())
        self.canvas.on_crop_change = self._refresh_all
        self.canvas.pack(fill="both", expand=True)
        self.empty = tk.Frame(area, bg=Darkroom.canvas)
        tk.Label(self.empty, text=APP_NAME, font=self.fonts.hero, fg=Darkroom.label, bg=Darkroom.canvas).pack()
        tk.Label(self.empty, text=tr("gui.empty.hint", shortcut=OPEN_SHORTCUT), font=self.fonts.hero_sub,
                 fg=Darkroom.secondary, bg=Darkroom.canvas, justify="center").pack(pady=(self.px(8), 0))
        self._tag_widgets(self)
        self._refresh_all()

    def _tag_widgets(self, widget: tk.Misc) -> None:
        """把這個 View 的事件標籤加到自己和底下每個元件（放在 'all' 之前；重建介面後要再做一次）。"""
        tags = list(widget.bindtags())
        if self._tag not in tags:
            tags.insert(max(0, len(tags) - 1), self._tag)
            widget.bindtags(tuple(tags))
        for child in widget.winfo_children():
            if not isinstance(child, tk.Toplevel):
                self._tag_widgets(child)

    def _build_top_bar(self) -> None:
        D = Darkroom
        bar = tk.Frame(self, bg=D.chrome, height=self.px(D.top_bar_height))
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Frame(self, bg=D.separator, height=1).pack(fill="x")
        identity = tk.Frame(bar, bg=D.chrome)
        identity.pack(side="left", padx=(self.px(14), 0))
        tk.Label(identity, text=APP_NAME, font=self.fonts.title, fg=D.label, bg=D.chrome).pack(side="left")
        badge = tk.Frame(identity, bg=D.beta, padx=1, pady=1)
        badge.pack(side="left", padx=(self.px(7), 0))
        tk.Label(badge, text=f"v{__version__}", font=self.fonts.badge, fg=D.beta, bg=D.chrome,
                 padx=self.px(4)).pack()

        actions = tk.Frame(bar, bg=D.chrome)
        actions.pack(side="right", padx=(0, self.px(14)))
        if self.show_language:
            Segmented(actions, self, [("zh", "繁中"), ("en", "EN")], self.lang_var).pack(side="left")
            tk.Frame(actions, bg=D.separator, width=1, height=self.px(18)).pack(side="left", padx=self.px(10))
        self.open_btn = ttk.Button(actions, text=tr("gui.btn.open"), style="Dark.TButton", command=self.ask_open)
        self.open_btn.pack(side="left")
        Tooltip(self.open_btn, tr("gui.btn.open.help", shortcut=OPEN_SHORTCUT), self)
        self.save_btn = ttk.Button(actions, text=tr("gui.btn.save"), style="Dark.TButton", command=self.ask_save)
        self.save_btn.pack(side="left", padx=(self.px(6), 0))
        Tooltip(self.save_btn, tr("gui.btn.save.help"), self)
        self.auto_btn = ttk.Button(actions, text=tr("gui.btn.recommended"), style="Prominent.TButton",
                                   command=self.apply_recommended)
        self.auto_btn.pack(side="left", padx=(self.px(6), 0))
        Tooltip(self.auto_btn, tr("gui.btn.recommended.help"), self)

        # 中間：比較切換與檔名
        middle = tk.Frame(bar, bg=D.chrome)
        middle.pack(side="left", expand=True)
        self.view_seg = Segmented(middle, self, [("original", tr("gui.view.original")),
                                                 ("previous", tr("gui.view.previous")),
                                                 ("current", tr("gui.view.current"))], self.view_var)
        self.view_seg.pack(side="left")
        self.doc_title = tk.Label(middle, font=self.fonts.ui, fg=D.secondary, bg=D.chrome)
        self.doc_title.pack(side="left", padx=self.px(14))

    def _build_status_bar(self) -> None:
        D = Darkroom
        bar = tk.Frame(self, bg=D.chrome, height=self.px(D.status_bar_height))
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)
        tk.Frame(self, bg=D.separator, height=1).pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(bar, style="Dark.Horizontal.TProgressbar", length=self.px(120),
                                        mode="determinate", maximum=1.0)
        self.progress.pack(side="left", padx=(self.px(14), 0))
        self.summary = tk.Label(bar, font=self.fonts.small, fg=D.label, bg=D.chrome)
        self.summary.pack(side="right", padx=self.px(14))
        self.status = tk.Label(bar, font=self.fonts.small, fg=D.secondary, bg=D.chrome, anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=self.px(8))

    def _build_panel(self, body: tk.Frame) -> None:
        D = Darkroom
        outer = tk.Frame(body, bg=D.panel, width=self.px(D.panel_width))
        outer.pack(side="right", fill="y")
        outer.pack_propagate(False)
        canvas = tk.Canvas(outer, bg=D.panel, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        panel = tk.Frame(canvas, bg=D.panel)
        window = canvas.create_window((0, 0), window=panel, anchor="nw")
        self.panel_canvas = canvas

        def relayout(_event: object = None) -> None:
            canvas.itemconfigure(window, width=canvas.winfo_width())
            canvas.configure(scrollregion=(0, 0, 0, panel.winfo_reqheight()))
            if panel.winfo_reqheight() > canvas.winfo_height() > 1:
                scrollbar.pack(side="right", fill="y", before=canvas)
            else:
                scrollbar.pack_forget()
                canvas.yview_moveto(0)

        panel.bind("<Configure>", relayout)
        canvas.bind("<Configure>", relayout)
        self._relayout_panel = relayout
        pct = lambda v: f"{v:.0f}%"  # noqa: E731
        signed = lambda v: f"{v:+.0f}"  # noqa: E731

        group = PanelGroup(panel, self, "info", tr("gui.group.info"), tr("gui.group.info.info"))
        self.metrics = {k: MetricRow(group.body, self, tr(f"gui.metric.{k}")) for k in ("size", "noise", "color")}
        grid = tk.Frame(group.body, bg=D.panel)
        grid.pack(fill="x", pady=(self.px(8), 0))
        self.geometry_buttons = []
        for i, (label, cmd) in enumerate((("gui.btn.rotate_left", lambda: self.rotate(1)),
                                          ("gui.btn.rotate_right", lambda: self.rotate(-1)),
                                          ("gui.btn.flip_h", lambda: self.flip(True)),
                                          ("gui.btn.flip_v", lambda: self.flip(False)))):
            b = ttk.Button(grid, text=tr(label), style="Dark.TButton", command=cmd)
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0 if i % 2 == 0 else self.px(3), 0), pady=self.px(2))
            self.geometry_buttons.append(b)
        grid.columnconfigure(0, weight=1, uniform="g")
        grid.columnconfigure(1, weight=1, uniform="g")
        row = tk.Frame(group.body, bg=D.panel)
        row.pack(fill="x", pady=(self.px(4), 0))
        self.crop_btn = ttk.Button(row, text=tr("gui.btn.crop"), style="Dark.TButton", command=self.toggle_crop)
        self.crop_btn.pack(side="left")
        Tooltip(self.crop_btn, tr("gui.btn.crop.help"), self)
        self.apply_crop_btn = ttk.Button(row, text=tr("gui.btn.apply_crop"), style="Prominent.TButton",
                                         command=self.apply_crop)
        self.apply_crop_btn.pack(side="left", padx=(self.px(4), 0))
        self.reset_geo_btn = ttk.Button(row, text=tr("gui.btn.reset_geometry"), style="Dark.TButton",
                                        command=self.reset_geometry)
        self.reset_geo_btn.pack(side="right")
        Tooltip(self.reset_geo_btn, tr("gui.btn.reset_geometry.help"), self)

        group = PanelGroup(panel, self, "processing", tr("gui.group.processing"), tr("gui.group.processing.info"))
        ParameterToggle(group.body, self, tr("gui.toggle.background"), self.v["background_enabled"]).pack(
            fill="x", pady=self.px(2))
        ParameterToggle(group.body, self, tr("gui.toggle.color"), self.v["color_enabled"]).pack(fill="x", pady=self.px(2))
        ParameterToggle(group.body, self, tr("gui.toggle.denoise"), self.v["denoise_enabled"]).pack(
            fill="x", pady=(self.px(8), self.px(2)))
        self.denoise_slider = ParameterSlider(group.body, self, tr("gui.slider.denoise"), self.v["denoise"], 0, 100,
                                              pct, 1)
        self.denoise_slider.pack(fill="x", pady=(0, self.px(4)))
        ParameterToggle(group.body, self, tr("gui.toggle.detail"), self.v["detail_enabled"]).pack(
            fill="x", pady=(self.px(8), self.px(2)))
        self.stars_slider = ParameterSlider(group.body, self, tr("gui.slider.stars"), self.v["stars"], 0, 100, pct, 1)
        self.stars_slider.pack(fill="x", pady=(0, self.px(4)))
        self.sharpen_slider = ParameterSlider(group.body, self, tr("gui.slider.sharpen"), self.v["sharpen"], 0, 100,
                                              pct, 1)
        self.sharpen_slider.pack(fill="x")

        group = PanelGroup(panel, self, "stretch", tr("gui.group.stretch"), tr("gui.group.stretch.info"))
        ParameterSlider(group.body, self, tr("gui.slider.stretch"), self.v["stretch"], 0, 100, pct, 1).pack(
            fill="x", pady=(0, self.px(4)))
        ParameterSlider(group.body, self, tr("gui.slider.sky"), self.v["sky"], 2, 30, lambda v: f"{v:.1f}%", 0.5).pack(
            fill="x", pady=(0, self.px(6)))
        ParameterSlider(group.body, self, tr("gui.slider.bg_color"), self.v["bg_color"], 0, 100, pct, 1).pack(
            fill="x", pady=(0, self.px(6)))
        ParameterToggle(group.body, self, tr("gui.toggle.linked"), self.v["linked"]).pack(fill="x", pady=self.px(2))

        group = PanelGroup(panel, self, "finishing", tr("gui.group.finishing"), tr("gui.group.finishing.info"))
        for key, lo, hi, fmt in (("exposure", -100, 100, signed), ("contrast", -100, 100, signed),
                                 ("saturation", 0, 200, pct), ("green_removal", 0, 100, pct)):
            ParameterSlider(group.body, self, tr(f"gui.slider.{key}"), self.d[key], lo, hi, fmt, 1).pack(
                fill="x", pady=(0, self.px(4)))
        tk.Label(group.body, text=tr("gui.curves"), font=self.fonts.small, fg=D.secondary, bg=D.panel).pack(
            anchor="w", pady=(self.px(6), self.px(3)))
        self.curve_editor = CurveEditor(group.body, self, self._curves_changed)
        self.curve_editor.set_curves(self.adjust.curves)
        self.curve_editor.pack(fill="x")
        row = tk.Frame(group.body, bg=D.panel)
        row.pack(fill="x", pady=(self.px(6), 0))
        ttk.Button(row, text=tr("gui.btn.reset_curve"), style="Dark.TButton",
                   command=self.curve_editor.reset_channel).pack(side="left")
        ttk.Button(row, text=tr("gui.btn.reset_finishing"), style="Dark.TButton",
                   command=self.reset_finishing).pack(side="left", padx=(self.px(6), 0))

    def _scroll_panel(self, event: tk.Event) -> None:
        w = event.widget
        try:
            inside = str(w).startswith(str(self.panel_canvas))
        except (tk.TclError, AttributeError):
            return
        if not inside or not self.panel_canvas.winfo_exists():
            return
        first, last = self.panel_canvas.yview()
        if first <= 0 and last >= 1:
            return
        if IS_MAC:
            steps = -event.delta
        else:
            steps = -3 * (int(event.delta / 120) if abs(event.delta) >= 120 else (1 if event.delta > 0 else -1))
        if steps:
            self.panel_canvas.yview_scroll(steps, "units")

    def _language_clicked(self) -> None:
        lang = self.lang_var.get()
        if lang == get_language():
            return
        if self.on_language is not None:
            self.on_language(lang)
        else:
            set_language(lang)
            save_settings(language=lang)
            self.rebuild()

    def rebuild(self) -> None:
        """照目前語言重建介面；影像、設定與檢視位置不變。"""
        if self.lang_var.get() != get_language():
            self.lang_var.set(get_language())
        self.close_popover()
        view = (self.canvas.zoom, self.canvas.cx, self.canvas.cy)
        self.canvas.set_crop_mode(False)
        for child in self.winfo_children():
            child.destroy()
        self._build()
        self.canvas.zoom, self.canvas.cx, self.canvas.cy = view
        self._show_view()
        self.update_idletasks()
        self._relayout_panel()

    # ------------------------------------------------------------------ 狀態

    def set_status(self, key: str, **kw: object) -> None:
        self._status_key = (key, kw)
        try:
            self.status.configure(text=tr(key, **kw))
        except tk.TclError:
            pass

    def _refresh_summary(self) -> None:
        parts = []
        if self.canvas.full is not None:
            parts.append(tr("gui.summary.zoom", z=round(self.canvas.current_zoom() * 100)))
        try:
            self.summary.configure(text="  ".join(parts))
        except tk.TclError:
            pass

    def _refresh_all(self) -> None:
        has = self.source is not None
        busy = self.job is not None
        self.save_btn.state(["!disabled"] if has and self.current_is_full and not busy else ["disabled"])
        for b in self.geometry_buttons + [self.crop_btn]:
            b.state(["!disabled"] if has else ["disabled"])
        cropping = has and self.canvas.crop_mode
        self.crop_btn.configure(text=tr("gui.btn.cancel_crop" if cropping else "gui.btn.crop"))
        self.apply_crop_btn.state(["!disabled"] if cropping and self.canvas.crop_box is not None else ["disabled"])
        self.reset_geo_btn.state(["!disabled"] if has and not self.geometry.is_identity() else ["disabled"])
        self.auto_btn.state(["!disabled"] if has else ["disabled"])
        if has:
            self.empty.place_forget()
        else:
            self.empty.place(relx=0.5, rely=0.45, anchor="center")
        self.doc_title.configure(text=self.path.name if self.path else tr("gui.no_image"))
        self.denoise_slider.set_enabled(bool(self.v["denoise_enabled"].get()))
        on = bool(self.v["detail_enabled"].get())
        self.stars_slider.set_enabled(on)
        self.sharpen_slider.set_enabled(on)
        if self.source is not None:
            c, h, w = (1,) + self.source.shape if self.source.ndim == 2 else self.source.shape
            self.metrics["size"].set(f"{w} × {h}{'' if c == 3 else ' ' + tr('gui.mono')}")
        if self.current is not None:
            self.metrics["noise"].set(f"{self.current.noise.sigma:.2e}")
            b = self.current.balance
            self.metrics["color"].set("—" if b is None else " / ".join(f"{g:.2f}" for g in b.gains))
        self.set_status(*self._status_key[:1], **self._status_key[1])
        self._refresh_summary()

    # ------------------------------------------------------------------ 開檔

    def ask_open(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title=tr("gui.btn.open"), initialdir=self.last_dir(),
                                          filetypes=OPEN_TYPES)
        if path:
            self.open_image(path)

    def open_image(self, path: str | Path) -> None:
        """開一張影像並自動跑建議設定（整合 App 也呼叫這個）。"""
        path = Path(path)
        self._cancel_job()
        self.set_status("gui.status.opening", name=path.name)
        self.progress.configure(value=0)

        events = self.events  # 背景工作只抓需要的東西：View 關掉後不會在背景執行緒裡被釋放

        def work() -> None:
            try:
                image, header = imageio.load_image(path)
                original = stretch.apply(image, stretch.StretchSettings())
                events.put(("opened", path, image, header, original))
            except Exception as e:  # noqa: BLE001  顯示給使用者
                events.put(("error", tr("gui.error.open", name=path.name), e))

        threading.Thread(target=work, daemon=True).start()

    def _opened(self, path: Path, image: np.ndarray, header, original: np.ndarray) -> None:
        self.path, self.header = path, header
        self.loaded, self.loaded_display = image, original
        self.geometry = geometry.Geometry()
        save_settings(last_dir=str(path.parent))
        self._use_source(keep_view=False)

    def _use_source(self, keep_view: bool) -> None:
        """依目前的幾何（裁切、旋轉、翻轉）從原始檔產生要處理的影像，重新處理。"""
        self._cancel_job()
        image = geometry.apply(self.loaded, self.geometry)
        self.source = image
        self.original_display = geometry.apply(self.loaded_display, self.geometry)
        self.current, self.current_is_full, self.previous_display = None, False, None
        self.processor.set_source(image)
        self.quick_factor = pipeline.preview_factor(image.shape, QUICK_LIMIT)
        if self.quick_factor > 1:
            f = self.quick_factor
            small = np.stack([_block_mean(p, f) for p in (image[None] if image.ndim == 2 else image)])
            self.quick = pipeline.Processor(pixel_scale=f)
            self.quick.set_source(small[0] if image.ndim == 2 else small)
        self.canvas.set_crop_mode(False)
        self.canvas.set_image(self.original_display, keep_view=keep_view)
        self.view_var.set("current")
        self._refresh_all()
        self._start_processing(quick_first=True)

    # ------------------------------------------------------------------ 幾何

    def _set_geometry(self, g: geometry.Geometry) -> None:
        if self.source is None:
            return
        self.geometry = g
        self._use_source(keep_view=False)

    def rotate(self, turns: int) -> None:
        g = self.geometry
        # 旋轉疊在既有的翻轉之後：翻轉會讓轉向相反
        flipped = g.flip_h != g.flip_v
        self._set_geometry(replace(g, quarter_turns=(g.quarter_turns + (-turns if flipped else turns)) % 4))

    def flip(self, horizontal: bool) -> None:
        g = self.geometry
        self._set_geometry(replace(g, flip_h=not g.flip_h) if horizontal else replace(g, flip_v=not g.flip_v))

    def toggle_crop(self) -> None:
        if self.source is None:
            return
        on = not self.canvas.crop_mode
        self.canvas.set_crop_mode(on)
        if on:
            self.view_var.set("original")
            self.set_status("gui.status.crop")
        self._refresh_all()

    def apply_crop(self) -> None:
        box = self.canvas.crop_box
        if box is None or self.source is None:
            return
        src_box = geometry.displayed_box_to_source(box, self.geometry, (self.loaded.shape[-1], self.loaded.shape[-2]))
        self._set_geometry(replace(self.geometry, crop=src_box))

    def reset_geometry(self) -> None:
        if not self.geometry.is_identity():
            self._set_geometry(geometry.Geometry())

    # ------------------------------------------------------------------ 處理

    def apply_recommended(self) -> None:
        self._load_vars(pipeline.ProcessingSettings.recommended(), display.DisplayAdjustments())

    def reset_finishing(self) -> None:
        self._load_vars(self._settings_from_vars(), display.DisplayAdjustments())

    def _processing_changed(self) -> None:
        if self._suspend:
            return
        self.settings = self._settings_from_vars()
        save_settings(processing=self.settings.to_dict())
        try:
            self._refresh_all()
        except (tk.TclError, AttributeError):
            return
        if self.source is None:
            return
        if self._debounce is not None:
            self.after_cancel(self._debounce)
        self._debounce = self.after(DEBOUNCE_MS, lambda: self._start_processing(quick_first=True))

    def _display_changed(self) -> None:
        if self._suspend:
            return
        self.adjust = self._adjust_from_vars()
        save_settings(display=self.adjust.to_dict())
        if hasattr(self, "canvas"):
            self.canvas.set_adjustments(self.adjust if self.view_var.get() != "original" else
                                        display.DisplayAdjustments())

    def _curves_changed(self, curves: display.ToneCurves) -> None:
        self.adjust = replace(self._adjust_from_vars(), curves=curves)
        save_settings(display=self.adjust.to_dict())
        self.canvas.set_adjustments(self.adjust if self.view_var.get() != "original" else display.DisplayAdjustments())

    def _cancel_job(self) -> None:
        if self.job is not None:
            self.job.cancelled.set()
            self.job = None

    def _start_processing(self, quick_first: bool) -> None:
        self._debounce = None
        if self.source is None:
            return
        self._cancel_job()
        self.generation += 1
        job = _Job(self.generation)
        self.job = job
        settings = self.settings
        use_quick = quick_first and self.quick_factor > 1
        self.set_status("gui.status.processing")
        self._refresh_all()

        events, quick, processor = self.events, self.quick, self.processor  # 同上

        def progress(fraction: float, message: object) -> None:
            events.put(("progress", job.generation, fraction, str(message)))

        def work() -> None:
            try:
                t0 = time.perf_counter()
                if use_quick:
                    r = quick.run(settings, cancel=job.cancelled.is_set)
                    events.put(("result", job.generation, r, False, time.perf_counter() - t0))
                r = processor.run(settings, progress, job.cancelled.is_set)
                events.put(("result", job.generation, r, True, time.perf_counter() - t0))
            except Cancelled:
                pass
            except Exception as e:  # noqa: BLE001
                events.put(("error", tr("gui.error.process"), e))

        threading.Thread(target=work, daemon=True).start()

    def _result(self, generation: int, result: pipeline.Result, full: bool, elapsed: float) -> None:
        if generation != self.generation:
            return
        if full:
            if self.current is not None and self.current_is_full:
                self.previous_display = self.current.display
            self.job = None
            self.set_status("gui.status.done", s=f"{elapsed:.1f}")
            self.progress.configure(value=1.0)
        else:
            self.set_status("gui.status.quick")
        self.current, self.current_is_full = result, full
        self._show_view()
        self._refresh_all()

    def _show_view(self) -> None:
        mode = self.view_var.get()
        img, scale, adj = None, 1.0, self.adjust
        if mode == "original" or self.current is None:
            img, adj = self.original_display, display.DisplayAdjustments()
        elif mode == "previous" and self.previous_display is not None:
            img = self.previous_display
        else:
            img = self.current.display
            scale = 1.0 if self.current_is_full else float(self.quick_factor)
        if img is None:
            return
        self.canvas.adjust = adj
        self.canvas.set_image(img, keep_view=True, pixel_scale=scale)
        self.curve_editor.set_histogram(self.canvas.preview)

    def _poll(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "opened":
                    self._opened(*ev[1:])
                elif kind == "progress":
                    if ev[1] == self.generation:
                        self.progress.configure(value=ev[2])
                        self.status.configure(text=ev[3])
                elif kind == "result":
                    self._result(*ev[1:])
                elif kind == "saved":
                    self.set_status("gui.status.saved", name=ev[1])
                    self._refresh_all()
                elif kind == "error":
                    self.job = None
                    self.set_status("gui.status.error")
                    self._refresh_all()
                    messagebox.showerror(APP_NAME, f"{ev[1]}\n\n{ev[2]}", parent=self.root)
                    traceback.print_exception(ev[2])
        except queue.Empty:
            pass
        self._poll_job = self.after(50, self._poll)

    # ------------------------------------------------------------------ 儲存

    def ask_save(self) -> None:
        if self.current is None or not self.current_is_full or self.path is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title=tr("gui.btn.save"), initialdir=str(self.path.parent),
            initialfile=f"{self.path.stem}_APU.png", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF 16-bit", "*.tif"), (tr("gui.save.fits"), "*.fits")])
        if path:
            self.save(Path(path))

    def save(self, path: Path) -> None:
        result, adj, header = self.current, self.adjust, self.header
        self.set_status("gui.status.saving", name=path.name)

        events = self.events  # 同上

        def work() -> None:
            # 先寫暫存檔再換名：存到一半失敗不會留下壞掉的檔案
            tmp = path.with_name(f".{path.stem}.saving{path.suffix}")
            try:
                if path.suffix.lower() in imageio.FITS_SUFFIXES:
                    imageio.save_fits(tmp, result.linear, header)
                else:
                    imageio.save_display(tmp, display.apply(result.display, adj))
                os.replace(tmp, path)
                events.put(("saved", path.name))
            except Exception as e:  # noqa: BLE001
                tmp.unlink(missing_ok=True)
                events.put(("error", tr("gui.error.save", name=path.name), e))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ 結束

    def is_busy(self) -> bool:
        """有處理工作在跑（關閉視窗前要詢問）。"""
        return self.job is not None

    def close(self) -> None:
        """取消背景工作、停掉排程。呼叫端接著 destroy() 這個 View；整合版關掉分頁後請在主執行緒 gc.collect()，
        不然 View 的 Tk 變數可能在背景執行緒被 Python 的循環回收釋放（tkinter 會忽略，但不乾淨）。"""
        self._cancel_job()
        try:
            self.after_cancel(self._poll_job)
        except (tk.TclError, ValueError):
            pass


# ---------------------------------------------------------------------- 視窗


def _build_menubar(root: tk.Tk, view: ProcessingView) -> None:
    menubar = tk.Menu(root)
    if IS_MAC:
        app_menu = tk.Menu(menubar, name="apple", tearoff=False)
        app_menu.add_command(label=tr("gui.menu.about"), command=lambda: root.tk.call("::tk::mac::standardAboutPanel"))
        app_menu.add_separator()
        menubar.add_cascade(menu=app_menu)
    accel = "Command" if IS_MAC else "Ctrl"
    file_menu = tk.Menu(menubar, tearoff=False)
    file_menu.add_command(label=tr("gui.menu.open"), accelerator=f"{accel}-O", command=view.ask_open)
    file_menu.add_command(label=tr("gui.menu.save"), accelerator=f"{accel}-S", command=view.ask_save)
    menubar.add_cascade(label=tr("gui.menu.file"), menu=file_menu)
    view_menu = tk.Menu(menubar, tearoff=False)
    view_menu.add_command(label=tr("gui.menu.fit"), accelerator=f"{accel}-0", command=view.canvas.fit)
    view_menu.add_command(label=tr("gui.menu.actual"), accelerator=f"{accel}-1",
                          command=lambda: view.canvas.set_zoom(1.0))
    menubar.add_cascade(label=tr("gui.menu.view"), menu=view_menu)
    if IS_MAC:
        menubar.add_cascade(label=tr("gui.menu.window"), menu=tk.Menu(menubar, name="window"))
    root.configure(menu=menubar)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    lang = load_settings().get("language")
    if lang in LANGUAGES:
        set_language(lang)
    enable_dpi_awareness()
    root = tk.Tk()
    if IS_MAC:
        root.tk.call("tk", "scaling", 96 / 72)
    try:
        if IS_MAC and (ASSETS / "icon_128.png").is_file():
            root.iconphoto(True, tk.PhotoImage(file=str(ASSETS / "icon_128.png")))
        elif ICON.is_file():
            root.iconbitmap(default=str(ICON))
    except tk.TclError:
        pass
    scale = root.winfo_fpixels("1i") / 96.0
    width = min(int(1440 * scale), root.winfo_screenwidth() - int(40 * scale))
    height = min(int(920 * scale), root.winfo_screenheight() - int(110 * scale))
    root.geometry(f"{width}x{height}")
    root.minsize(int(1000 * scale), int(660 * scale))
    root.configure(bg=Darkroom.canvas)
    root.title(f"{APP_NAME} {__version__}")
    def change_language(lang: str) -> None:
        set_language(lang)
        save_settings(language=lang)
        view.rebuild()
        _build_menubar(root, view)        # 選單文字跟著換

    view = ProcessingView(root, root, on_language=change_language)
    view.pack(fill="both", expand=True)
    _build_menubar(root, view)
    mod = "Command" if IS_MAC else "Control"
    root.bind_all(f"<{mod}-o>", lambda _e: view.ask_open())
    root.bind_all(f"<{mod}-s>", lambda _e: view.ask_save())
    root.bind_all(f"<{mod}-Key-0>", lambda _e: view.canvas.fit())
    root.bind_all(f"<{mod}-Key-1>", lambda _e: view.canvas.set_zoom(1.0))

    def on_close() -> None:
        if view.is_busy() and not messagebox.askyesno(APP_NAME, tr("gui.confirm.quit"), parent=root):
            return
        view.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    if IS_MAC:
        root.createcommand("::tk::mac::Quit", on_close)
        root.createcommand("::tk::mac::OpenDocument", lambda *paths: paths and view.open_image(paths[0]))
    dark_title_bar(root)
    files = [a for a in argv if not a.startswith("-")]
    if files:
        root.after(200, lambda: view.open_image(files[0]))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
