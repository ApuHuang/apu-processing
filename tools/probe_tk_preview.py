"""第 1 步效能探路：Tk 能不能流暢顯示大張天文影像並即時調整。

開一張 RGB FITS → 自動拉伸 → 2560 px 預覽快取 → 放進 Tk Canvas；
拖「明暗」滑桿、滾輪縮放時，每一幀只處理「畫面上看得到的像素」：
從預覽快取（放大超過預覽解析度時改從完整尺寸）裁出可見範圍 → 縮放到畫布大小 → 套調整 → 8-bit → PhotoImage。

用法：
    python tools/probe_tk_preview.py 影像.fit            # 開視窗自己操作，狀態列顯示每幀耗時
    python tools/probe_tk_preview.py 影像.fit --bench    # 自動模擬拖滑桿與縮放，印出統計後結束
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from astropy.io import fits
from PIL import Image, ImageTk

PREVIEW_LONG_SIDE = 2560


def load_rgb(path: str) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        top_down = str(hdul[0].header.get("ROWORDER", "")).upper() == "TOP-DOWN"
    if data.ndim == 2:
        data = data[None]
    if not top_down:
        data = data[:, ::-1]
    return np.ascontiguousarray(data)


def block_downsample(image: np.ndarray, long_side: int) -> np.ndarray:
    """面積平均縮圖（每個通道用 PIL 的 BOX，C 實作，比 numpy reshape 快而且不用整除）。"""
    c, h, w = image.shape
    scale = long_side / max(h, w)
    if scale >= 1:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.stack([np.asarray(Image.fromarray(image[i], "F").resize(size, Image.Resampling.BOX))
                     for i in range(c)])


def simple_stretch_params(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """探路用的簡單自動拉伸（每通道天空中位數當黑點、MAD 定尺度、共用 asinh 強度）；正式版在第 2 步重做。"""
    sample = image[:, ::7, ::7].reshape(image.shape[0], -1)
    median = np.median(sample, axis=1)
    mad = np.median(np.abs(sample - median[:, None]), axis=1) * 1.4826
    black = median - 2.8 * mad
    white = np.percentile(sample, 99.95, axis=1)
    return black.astype(np.float32), (white - black).astype(np.float32), 60.0


def apply_stretch(region: np.ndarray, black: np.ndarray, span: np.ndarray, k: float) -> np.ndarray:
    x = (region - black[:, None, None]) / span[:, None, None]
    np.clip(x, 0, 1, out=x)
    return (np.arcsinh(k * x) / np.arcsinh(k)).astype(np.float32)


def resize_float(region: np.ndarray, size: tuple[int, int], upscale: bool) -> np.ndarray:
    method = Image.Resampling.NEAREST if upscale else Image.Resampling.BILINEAR
    return np.stack([np.asarray(Image.fromarray(ch, "F").resize(size, method)) for ch in region])


def adjust_to_rgb8(display: np.ndarray, brightness: float, saturation: float) -> np.ndarray:
    """明暗（中間調曲線，以 4096 階查表）＋飽和度，輸出 (h, w, 3) uint8。"""
    gamma = 2.0 ** (-brightness)
    lut = (np.linspace(0, 1, 4096, dtype=np.float32) ** gamma)
    idx = np.clip(display * 4095 + 0.5, 0, 4095).astype(np.uint16)
    y = lut[idx]
    if saturation != 1.0:
        lum = y.mean(axis=0, keepdims=True)
        y = lum + (y - lum) * saturation
        np.clip(y, 0, 1, out=y)
    return np.ascontiguousarray(np.moveaxis((y * 255 + 0.5).astype(np.uint8), 0, -1))


class Probe:
    def __init__(self, root: tk.Tk, path: str):
        self.root = root
        t0 = time.perf_counter()
        self.full = load_rgb(path)
        t1 = time.perf_counter()
        self.black, self.span, self.k = simple_stretch_params(self.full)
        t2 = time.perf_counter()
        self.preview = apply_stretch(block_downsample(self.full, PREVIEW_LONG_SIDE), self.black, self.span, self.k)
        t3 = time.perf_counter()
        # 完整尺寸的顯示快取（拉伸後、uint16），放大超過預覽解析度時從這裡裁，不用每幀重新拉伸。
        # 正式版在背景執行緒做；拉伸設定改變時重算，重算完成前先用預覽放大頂著。
        self.full_display = np.empty(self.full.shape, dtype=np.uint16)
        for i in range(self.full.shape[0]):
            ch = apply_stretch(self.full[i:i + 1], self.black[i:i + 1], self.span[i:i + 1], self.k)[0]
            np.multiply(ch, 65535, out=ch)
            ch += 0.5
            self.full_display[i] = ch
        t4 = time.perf_counter()
        _, self.h, self.w = self.full.shape
        self.preview_scale = self.preview.shape[2] / self.w  # 預覽像素 / 完整像素
        self.setup = {"load": t1 - t0, "stretch_params": t2 - t1, "preview": t3 - t2, "full_display": t4 - t3}

        self.zoom = None  # 螢幕像素 / 完整像素；None = 符合視窗
        self.cx, self.cy = self.w / 2, self.h / 2
        self.brightness = tk.DoubleVar(value=0.0)
        self.saturation = tk.DoubleVar(value=1.0)
        self.timings: list[dict[str, float]] = []
        self._pending = False
        self._photo: ImageTk.PhotoImage | None = None

        root.configure(bg="#000000")
        bar = tk.Frame(root, bg="#1c1c1c")
        bar.pack(side="bottom", fill="x")
        self.status = tk.Label(bar, fg="#949494", bg="#1c1c1c", anchor="w", font=("Menlo", 11))
        self.status.pack(fill="x", padx=8, pady=4)
        panel = tk.Frame(root, bg="#252525", width=320)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        for text, var, lo, hi in (("明暗", self.brightness, -2, 2), ("飽和度", self.saturation, 0, 2)):
            tk.Label(panel, text=text, fg="#ebebeb", bg="#252525").pack(anchor="w", padx=14, pady=(14, 0))
            ttk.Scale(panel, from_=lo, to=hi, variable=var, command=lambda _v: self.request()).pack(fill="x", padx=14)
        self.canvas = tk.Canvas(root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", lambda _e: self.request())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.25))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 0.8))

    # ---- 檢視

    def fit_zoom(self) -> float:
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        return min(cw / self.w, ch / self.h)

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, 1.25 if event.delta > 0 else 0.8)

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        z = self.zoom or self.fit_zoom()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        # 游標下的影像位置不動
        ix = self.cx + (x - cw / 2) / z
        iy = self.cy + (y - ch / 2) / z
        nz = min(8.0, max(self.fit_zoom(), z * factor))
        self.cx = ix - (x - cw / 2) / nz
        self.cy = iy - (y - ch / 2) / nz
        self.zoom = nz
        self.request()

    # ---- 只算最新一幀

    def request(self) -> None:
        if not self._pending:
            self._pending = True
            self.root.after_idle(self._render)

    def _render(self) -> None:
        self._pending = False
        self.render()

    def render(self) -> dict[str, float]:  # noqa: C901
        t = {"start": time.perf_counter()}
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        z = self.zoom or self.fit_zoom()
        if self.zoom is None:
            self.cx, self.cy = self.w / 2, self.h / 2
        # 可見範圍（完整像素座標），夾在影像內
        half_w, half_h = cw / 2 / z, ch / 2 / z
        x0, x1 = max(0.0, self.cx - half_w), min(float(self.w), self.cx + half_w)
        y0, y1 = max(0.0, self.cy - half_h), min(float(self.h), self.cy + half_h)
        out_w, out_h = max(1, round((x1 - x0) * z)), max(1, round((y1 - y0) * z))
        if z <= self.preview_scale * 1.0001:
            s = self.preview_scale
            region = self.preview[:, int(y0 * s):max(int(y0 * s) + 1, int(np.ceil(y1 * s))),
                                  int(x0 * s):max(int(x0 * s) + 1, int(np.ceil(x1 * s)))]
            source = "preview"
        else:
            raw = self.full_display[:, int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
            region = raw.astype(np.float32)
            region *= 1 / 65535
            source = "full"
        t["crop"] = time.perf_counter()
        display = resize_float(np.ascontiguousarray(region), (out_w, out_h), upscale=out_w > region.shape[2])
        t["resize"] = time.perf_counter()
        rgb = adjust_to_rgb8(display, float(self.brightness.get()), float(self.saturation.get()))
        t["adjust"] = time.perf_counter()
        img = Image.fromarray(rgb, "RGB")
        if self._photo is not None and self._photo.width() == out_w and self._photo.height() == out_h:
            self._photo.paste(img)
        else:
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfigure(self.image_item, image=self._photo)
        self.canvas.coords(self.image_item, (cw - out_w) / 2 if x1 - x0 < self.w or self.zoom is None else 0,
                           (ch - out_h) / 2)
        t["photo"] = time.perf_counter()
        self.root.update_idletasks()
        t["draw"] = time.perf_counter()
        row = {
            "crop": t["crop"] - t["start"], "resize": t["resize"] - t["crop"], "adjust": t["adjust"] - t["resize"],
            "photo": t["photo"] - t["adjust"], "draw": t["draw"] - t["photo"], "total": t["draw"] - t["start"],
            "pixels": out_w * out_h, "source": source,
        }
        self.timings.append(row)
        self.status.configure(text=f"{source}  {out_w}×{out_h}  zoom {z * 100:.0f}%  "
                                   f"每幀 {row['total'] * 1000:.0f} ms（縮放 {row['resize'] * 1000:.0f}、"
                                   f"調整 {row['adjust'] * 1000:.0f}、PhotoImage {row['photo'] * 1000:.0f}）")
        return row


def summarize(label: str, rows: list[dict]) -> None:
    if not rows:
        return
    ms = lambda key: [r[key] * 1000 for r in rows]  # noqa: E731
    total = ms("total")
    print(f"  {label:<28} n={len(rows):>3}  每幀 中位 {statistics.median(total):5.1f} ms  最慢 {max(total):5.1f} ms  "
          f"（裁切 {statistics.median(ms('crop')):.1f}／縮放 {statistics.median(ms('resize')):.1f}／調整 {statistics.median(ms('adjust')):.1f}／"
          f"PhotoImage {statistics.median(ms('photo')):.1f}／繪製 {statistics.median(ms('draw')):.1f}）"
          f"  {rows[0]['pixels'] / 1e6:.2f} MP")


def bench(probe: Probe, root: tk.Tk) -> None:
    root.update()
    print(f"影像 {probe.w}×{probe.h}  畫布 {probe.canvas.winfo_width()}×{probe.canvas.winfo_height()}  "
          f"預覽 {probe.preview.shape[2]}×{probe.preview.shape[1]}")
    print("  開檔準備：" + "、".join(f"{k} {v * 1000:.0f} ms" for k, v in probe.setup.items()))

    def drag(n: int = 40) -> list[dict]:
        probe.timings.clear()
        for i in range(n):
            probe.brightness.set(-1 + 2 * i / (n - 1))
            probe.saturation.set(0.6 + 0.8 * i / (n - 1))
            probe.render()
            root.update()
        return list(probe.timings)

    probe.zoom = None
    summarize("符合視窗：拖明暗＋飽和", drag())
    cw, ch = probe.canvas.winfo_width(), probe.canvas.winfo_height()
    for label, zoom in (("100%（完整像素）", 1.0), ("200%", 2.0), ("50%", 0.5)):
        probe.zoom = zoom
        probe.cx, probe.cy = probe.w * 0.4, probe.h * 0.45
        summarize(f"{label}：拖明暗＋飽和", drag(25))
    probe.zoom = None
    probe.timings.clear()
    for _ in range(12):
        probe._zoom_at(cw * 0.45, ch * 0.5, 1.25)
        probe.render()
        root.update()
    for _ in range(12):
        probe._zoom_at(cw * 0.55, ch * 0.5, 0.8)
        probe.render()
        root.update()
    summarize("滾輪縮放（進 12 次、出 12 次）", list(probe.timings))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--size", default="1400x900", help="視窗大小（預設 1400x900）")
    args = parser.parse_args()
    root = tk.Tk()
    if sys.platform == "darwin":
        root.tk.call("tk", "scaling", 96 / 72)
    root.geometry(args.size)
    root.title("APU Processing — Tk 效能探路")
    probe = Probe(root, args.path)
    if args.bench:
        root.after(300, lambda: (bench(probe, root), root.destroy()))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
