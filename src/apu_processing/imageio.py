"""讀寫影像。

記憶體中一律是 float32：單色 (H, W)、彩色 (3, H, W)，列方向 top-down（第 0 列是畫面最上面）。

- FITS：有 ROWORDER='TOP-DOWN' 就照原樣，沒有（或 BOTTOM-UP）就上下翻轉，與 Swift 版 FITSIO 相同；寫出時標 TOP-DOWN。
  整數資料正規化到 0–1；浮點資料若明顯超過 1（例如 0–65535 的疊圖），同樣縮到 0–1。
- TIFF：tifffile，支援 8／16-bit 與浮點。
- PNG／JPEG：Pillow，8-bit（Pillow 讀 16-bit RGB PNG 會降成 8-bit）。
- 成品輸出（PNG／JPEG／TIFF）吃的是 0–1 的顯示影像；FITS 輸出保留線性浮點。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from .i18n import Msg

FITS_SUFFIXES = {".fit", ".fits", ".fts"}
TIFF_SUFFIXES = {".tif", ".tiff"}
BITMAP_SUFFIXES = {".png", ".jpg", ".jpeg"}
IMAGE_SUFFIXES = FITS_SUFFIXES | TIFF_SUFFIXES | BITMAP_SUFFIXES

# 浮點資料的最大值超過這個數，就當作不是 0–1 的資料
_FLOAT_RANGE_LIMIT = 1.5


def is_image(path: Path) -> bool:
    """能處理的檔案；略過 macOS 在 exFAT 上留下的「._檔名」附屬檔。"""
    return path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and is_image(p))


def load_image(path: Path | str) -> tuple[np.ndarray, fits.Header]:
    """讀一張影像，回傳 (float32 影像, header)。非 FITS 的 header 只有 APUSRC 等少數欄位。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in FITS_SUFFIXES:
        return _load_fits(path)
    if suffix in TIFF_SUFFIXES:
        return _load_tiff(path)
    if suffix in BITMAP_SUFFIXES:
        return _load_bitmap(path)
    raise ValueError(Msg("msg.unsupported_format", name=path.name))


def _load_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
        if hdu is None:
            raise ValueError(Msg("msg.no_image_data", name=path.name))
        bitpix = int(hdu.header.get("BITPIX", -32))
        data = np.asarray(hdu.data)
        header = hdu.header.copy()
    if bitpix > 0:
        # astropy 已套用 BZERO／BSCALE；8-bit 是 0–255，16／32-bit 通常是無號範圍
        data = data.astype(np.float32) / np.float32(2 ** bitpix - 1)
    data = _to_planes(np.asarray(data, dtype=np.float32), path, planar=True)
    if str(header.get("ROWORDER", "")).strip().upper() != "TOP-DOWN":
        data = data[..., ::-1, :]
    data = _normalize_float(np.ascontiguousarray(data), header)
    header["APUSRC"] = ("FITS", "APU Processing: source format")
    return data, header


def _load_tiff(path: Path) -> tuple[np.ndarray, fits.Header]:
    import tifffile

    raw = tifffile.imread(path)
    header = fits.Header()
    data = _integer_to_unit(raw)
    data = _to_planes(data, path, planar=raw.ndim == 3 and raw.shape[0] in (3, 4) and raw.shape[-1] not in (3, 4))
    data = _normalize_float(np.ascontiguousarray(data), header)
    header["APUSRC"] = ("TIFF", "APU Processing: source format")
    return data, header


def _load_bitmap(path: Path) -> tuple[np.ndarray, fits.Header]:
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L"):
            raw = np.asarray(im, dtype=np.uint16)
        elif im.mode == "I":
            raw = np.asarray(im, dtype=np.int32).clip(0, 65535).astype(np.uint16)
        elif im.mode in ("L", "RGB"):
            raw = np.asarray(im)
        else:
            raw = np.asarray(im.convert("RGBA" if "A" in im.mode else "RGB"))
    header = fits.Header()
    data = _to_planes(_integer_to_unit(raw), path, planar=False)
    header["APUSRC"] = (path.suffix.lstrip(".").upper(), "APU Processing: source format")
    return np.ascontiguousarray(data), header


def _integer_to_unit(raw: np.ndarray) -> np.ndarray:
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype(np.float32) / np.float32(np.iinfo(raw.dtype).max)
    return raw.astype(np.float32)


def _to_planes(data: np.ndarray, path: Path, planar: bool) -> np.ndarray:
    """整理成 (H, W) 或 (3, H, W)。planar=True 表示色版在第 0 軸；否則在最後一軸（交錯存放）。"""
    data = np.squeeze(data)
    if data.ndim == 2:
        return data
    if data.ndim != 3:
        raise ValueError(Msg("msg.bad_shape", name=path.name, shape=tuple(data.shape)))
    if not planar:
        data = np.moveaxis(data, -1, 0)
    n = data.shape[0]
    if n == 4:  # RGBA：丟掉 alpha
        data = data[:3]
    elif n != 3:
        raise ValueError(Msg("msg.bad_channels", name=path.name, n=n))
    return data


def _normalize_float(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """非有限值改成 0；超出 0–1 很多的浮點資料縮回 0–1（例如 DSS 的 0–65535 浮點疊圖）。"""
    bad = ~np.isfinite(data)
    if bad.any():
        data[bad] = 0.0
    peak = float(data.max()) if data.size else 0.0
    if peak > _FLOAT_RANGE_LIMIT:
        scale = 65535.0 if peak <= 65535.0 * 1.001 else peak
        data /= np.float32(scale)
        header["APUSCALE"] = (scale, "APU Processing: input divided by this")
    return data


# ---------------------------------------------------------------------- 寫出


def save_fits(path: Path | str, data: np.ndarray, header: fits.Header | None = None) -> None:
    """線性浮點 FITS，標 ROWORDER='TOP-DOWN'。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    if header is not None:
        for card in header.cards:
            if card.keyword not in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND",
                                    "BZERO", "BSCALE", "ROWORDER", "APUSRC", "APUSCALE", ""):
                try:
                    hdr.append(card)
                except (ValueError, KeyError):
                    pass
    hdr["ROWORDER"] = ("TOP-DOWN", "Order of pixel rows stored in the image array")
    fits.PrimaryHDU(np.nan_to_num(data, nan=0.0).astype(np.float32), header=hdr).writeto(path, overwrite=True)


def display_to_uint(display: np.ndarray, bits: int) -> np.ndarray:
    """0–1 的顯示影像 → 交錯存放的 uint8／uint16（(H, W) 或 (H, W, 3)）。"""
    top = 255 if bits == 8 else 65535
    out = np.clip(display, 0.0, 1.0) * top + 0.5
    out = out.astype(np.uint8 if bits == 8 else np.uint16)
    return np.ascontiguousarray(np.moveaxis(out, 0, -1)) if out.ndim == 3 else out


def save_display(path: Path | str, display: np.ndarray, bits: int | None = None) -> None:
    """成品輸出。PNG、JPEG 為 8-bit；TIFF 預設 16-bit。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        import tifffile

        tifffile.imwrite(path, display_to_uint(display, bits or 16), photometric="rgb" if display.ndim == 3 else
                         "minisblack", compression="zlib")
        return
    if suffix not in BITMAP_SUFFIXES:
        raise ValueError(Msg("msg.unsupported_format", name=path.name))
    from PIL import Image

    image = Image.fromarray(display_to_uint(display, 8))
    if suffix == ".png":
        image.save(path, optimize=False, compress_level=6)
    else:
        image.save(path, quality=95, subsampling=0)
