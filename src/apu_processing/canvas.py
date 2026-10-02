"""影像畫布：縮放、拖曳平移、成品微調即時預覽。

每一幀只處理畫面上看得到的像素（第 1 步效能探路驗證過的做法，tools/probe_tk_preview.py）：
從預覽快取（長邊 2560）或完整尺寸的顯示快取裁出可見範圍 → 縮到畫布大小 → 套成品微調的查表 → PhotoImage。
同時只算一幀、只留最新請求（after_idle 合併），拖滑桿時不會排隊。

Mac 的 Tk（8.6 與 9 都是）在 Retina 上一個影像像素畫成 2×2 螢幕像素，看不到真正的 1:1；使用者選擇接受（2026-09-30）。
"""

from __future__ import annotations

import time
import tkinter as tk
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageTk

from . import display as display_mod
from .darkroom import IS_MAC, Darkroom

PREVIEW_LONG_SIDE = 2560
MIN_ZOOM, MAX_ZOOM = 0.02, 8.0


def area_downsample(image: np.ndarray, long_side: int) -> np.ndarray:
    """面積平均縮圖（PIL BOX，每個色版分開）。"""
    planes = image[None] if image.ndim == 2 else image
    h, w = planes.shape[1:]
    scale = long_side / max(h, w)
    if scale >= 1:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    out = np.stack([np.asarray(Image.fromarray(p, "F").resize(size, Image.Resampling.BOX)) for p in planes])
    return out[0] if image.ndim == 2 else out


def _resize(region: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    method = Image.Resampling.NEAREST if size[0] > region.shape[2] * 1.5 else Image.Resampling.BILINEAR
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(ch), "F").resize(size, method))
                     for ch in region])


class ImageCanvas(tk.Canvas):
    """顯示 0–1 的顯示影像 (3, H, W) 或 (H, W)。set_image 換圖；set_adjustments 換成品微調（只重畫）。"""

    def __init__(self, master: tk.Misc, on_zoom: Callable[[float], None] | None = None):
        super().__init__(master, bg=Darkroom.canvas, highlightthickness=0, bd=0)
        self.full: np.ndarray | None = None       # (3, H, W) float32
        self.preview: np.ndarray | None = None
        self.pixel_scale = 1.0                     # 這張圖一個像素＝完整尺寸幾個像素（快速預覽是縮圖）
        self.adjust = display_mod.DisplayAdjustments()
        self.zoom: float | None = None             # 螢幕像素 / 影像像素；None＝符合視窗
        self.cx = self.cy = 0.0
        self.on_zoom = on_zoom
        self.last_frame_ms = 0.0
        self._pending = False
        self._photo: ImageTk.PhotoImage | None = None
        self._item = self.create_image(0, 0, anchor="nw")
        self._drag: tuple[int, int, float, float] | None = None
        # 裁切模式：拖曳畫框（影像座標，完整尺寸）
        self.crop_mode = False
        self.crop_box: tuple[float, float, float, float] | None = None
        self.on_crop_change: Callable[[], None] | None = None
        self._crop_start: tuple[float, float] | None = None
        self._crop_item = self.create_rectangle(0, 0, 0, 0, outline="#ffcc44", width=2, dash=(6, 4), state="hidden")
        self.bind("<Configure>", lambda _e: self.request())
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.25))
        self.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 0.8))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", lambda _e: None if self.crop_mode else self.fit())

    # ------------------------------------------------------------------ 內容

    def set_image(self, image: np.ndarray | None, keep_view: bool = True, pixel_scale: float = 1.0) -> None:
        """換圖。縮放與位置以完整尺寸的像素為單位，所以快速預覽（縮圖）與完整結果之間切換時畫面不會跳。"""
        if image is not None and image.ndim == 2:
            image = np.repeat(image[None], 3, axis=0)
        old_size = self.image_size()
        self.full = image
        self.pixel_scale = pixel_scale
        self.preview = None if image is None else area_downsample(image, PREVIEW_LONG_SIDE)
        new_size = self.image_size()
        if image is None or not keep_view or abs(old_size[0] - new_size[0]) > 2 * pixel_scale \
                or abs(old_size[1] - new_size[1]) > 2 * pixel_scale:
            self.zoom = None
            self.cx, self.cy = new_size[0] / 2, new_size[1] / 2
        self.request()

    def set_adjustments(self, adjust: display_mod.DisplayAdjustments) -> None:
        self.adjust = adjust
        self.request()

    # ------------------------------------------------------------------ 檢視

    def image_size(self) -> tuple[int, int]:
        """完整尺寸的寬高（快速預覽時是縮圖尺寸 × pixel_scale）。"""
        if self.full is None:
            return (0, 0)
        return (round(self.full.shape[2] * self.pixel_scale), round(self.full.shape[1] * self.pixel_scale))

    def fit_zoom(self) -> float:
        w, h = self.image_size()
        if not w:
            return 1.0
        return min(max(1, self.winfo_width()) / w, max(1, self.winfo_height()) / h)

    def current_zoom(self) -> float:
        return self.zoom if self.zoom is not None else self.fit_zoom()

    def fit(self) -> None:
        self.zoom = None
        w, h = self.image_size()
        self.cx, self.cy = w / 2, h / 2
        self.request()
        if self.on_zoom:
            self.on_zoom(self.current_zoom())

    def set_zoom(self, zoom: float) -> None:
        self.zoom = min(MAX_ZOOM, max(MIN_ZOOM, zoom))
        self.request()
        if self.on_zoom:
            self.on_zoom(self.zoom)

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        if self.full is None:
            return
        z = self.current_zoom()
        cw, ch = self.winfo_width(), self.winfo_height()
        ix = self.cx + (x - cw / 2) / z
        iy = self.cy + (y - ch / 2) / z
        nz = min(MAX_ZOOM, max(min(self.fit_zoom(), 1.0) * 0.5, z * factor))
        self.cx = ix - (x - cw / 2) / nz
        self.cy = iy - (y - ch / 2) / nz
        self.zoom = nz
        self._clamp_center()
        self.request()
        if self.on_zoom:
            self.on_zoom(nz)

    def _wheel(self, event: tk.Event) -> None:
        # 觸控板（Mac 的 delta 小、連續）跟滑鼠滾輪都縮放；以游標為中心
        step = event.delta if IS_MAC else event.delta / 120
        if step:
            self.zoom_at(event.x, event.y, 1.1 ** max(-5, min(5, step)))

    def to_image(self, sx: float, sy: float) -> tuple[float, float]:
        """畫布座標 → 影像座標（完整尺寸像素）。"""
        z = self.current_zoom()
        return self.cx + (sx - self.winfo_width() / 2) / z, self.cy + (sy - self.winfo_height() / 2) / z

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        z = self.current_zoom()
        return self.winfo_width() / 2 + (ix - self.cx) * z, self.winfo_height() / 2 + (iy - self.cy) * z

    def set_crop_mode(self, on: bool) -> None:
        self.crop_mode = on
        self.crop_box = None
        self.configure(cursor="crosshair" if on else "")
        self._draw_crop()

    def set_crop_box(self, box: tuple[float, float, float, float] | None) -> None:
        """裁切模式下直接畫一個框（影像座標），例如依覆蓋率建議的範圍。"""
        self.crop_box = box
        self._draw_crop()
        if self.on_crop_change:
            self.on_crop_change()

    def _draw_crop(self) -> None:
        if not self.crop_mode or self.crop_box is None:
            self.itemconfigure(self._crop_item, state="hidden")
            return
        x0, y0 = self.to_screen(self.crop_box[0], self.crop_box[1])
        x1, y1 = self.to_screen(self.crop_box[2], self.crop_box[3])
        self.coords(self._crop_item, x0, y0, x1, y1)
        self.itemconfigure(self._crop_item, state="normal")
        self.tag_raise(self._crop_item)

    def _press(self, event: tk.Event) -> None:
        if self.crop_mode and self.full is not None:
            w, h = self.image_size()
            ix, iy = self.to_image(event.x, event.y)
            self._crop_start = (min(max(ix, 0), w), min(max(iy, 0), h))
            return
        self._drag = (event.x, event.y, self.cx, self.cy)

    def _motion(self, event: tk.Event) -> None:
        if self.crop_mode and self._crop_start is not None:
            w, h = self.image_size()
            ix, iy = self.to_image(event.x, event.y)
            ix, iy = min(max(ix, 0), w), min(max(iy, 0), h)
            x0, y0 = self._crop_start
            self.crop_box = (min(x0, ix), min(y0, iy), max(x0, ix), max(y0, iy))
            self._draw_crop()
            if self.on_crop_change:
                self.on_crop_change()
            return
        if self._drag is None or self.full is None:
            return
        x0, y0, cx0, cy0 = self._drag
        z = self.current_zoom()
        if self.zoom is None:
            self.zoom = z
        self.cx = cx0 - (event.x - x0) / z
        self.cy = cy0 - (event.y - y0) / z
        self._clamp_center()
        self.request()

    def _release(self, _event: tk.Event) -> None:
        self._drag = None
        self._crop_start = None

    def _clamp_center(self) -> None:
        w, h = self.image_size()
        self.cx = min(max(self.cx, 0), w)
        self.cy = min(max(self.cy, 0), h)

    # ------------------------------------------------------------------ 繪製

    def request(self) -> None:
        if not self._pending:
            self._pending = True
            self.after_idle(self._render)

    def _render(self) -> None:
        self._pending = False
        try:
            self.render()
        except tk.TclError:
            pass

    def render(self) -> None:
        if self.full is None:
            self.itemconfigure(self._item, image="")
            self._photo = None
            return
        t0 = time.perf_counter()
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        w, h = self.image_size()
        z = self.current_zoom()
        if self.zoom is None:
            self.cx, self.cy = w / 2, h / 2
        half_w, half_h = cw / 2 / z, ch / 2 / z
        x0, x1 = max(0.0, self.cx - half_w), min(float(w), self.cx + half_w)
        y0, y1 = max(0.0, self.cy - half_h), min(float(h), self.cy + half_h)
        out_w, out_h = max(1, round((x1 - x0) * z)), max(1, round((y1 - y0) * z))
        ps = self.preview.shape[2] / w           # 預覽像素／完整像素
        fs = self.full.shape[2] / w              # 這張圖的像素／完整像素（完整結果＝1，快速預覽＝1/pixel_scale）
        if z <= ps * 1.0001:
            src, s = self.preview, ps
        else:
            src, s = self.full, fs
        ya, yb = int(y0 * s), max(int(y0 * s) + 1, int(np.ceil(y1 * s)))
        xa, xb = int(x0 * s), max(int(x0 * s) + 1, int(np.ceil(x1 * s)))
        region = src[:, ya:yb, xa:xb]
        shown = display_mod.apply(_resize(region, (out_w, out_h)), self.adjust)
        rgb = np.ascontiguousarray((np.clip(np.moveaxis(shown, 0, -1), 0, 1) * 255 + 0.5).astype(np.uint8))
        img = Image.fromarray(rgb, "RGB")
        if self._photo is not None and self._photo.width() == out_w and self._photo.height() == out_h:
            self._photo.paste(img)
        else:
            self._photo = ImageTk.PhotoImage(img)
            self.itemconfigure(self._item, image=self._photo)
        # 可見範圍左上角在畫布上的位置（影像比畫布小時自然置中）
        self.coords(self._item, round(cw / 2 + (x0 - self.cx) * z), round(ch / 2 + (y0 - self.cy) * z))
        self._draw_crop()
        self.last_frame_ms = (time.perf_counter() - t0) * 1000
