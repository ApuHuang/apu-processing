"""命令列：apu-processing 輸入.fit -o 輸出.png [選項]

輸出副檔名決定格式：.png／.jpg／.tif 是成品（拉伸＋成品微調），.fit／.fits 是線性處理結果。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import __version__, display, imageio, pipeline
from .i18n import APP_NAME, LANGUAGES, set_language


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="apu-processing", description=f"{APP_NAME} {__version__}：一鍵後製天文影像")
    p.add_argument("input", help="未拉伸的線性影像（FIT／FITS／FTS、TIFF、PNG）")
    p.add_argument("-o", "--output", action="append", required=True,
                   help="輸出檔（可以給多次）；.png/.jpg/.tif＝成品，.fit/.fits＝線性結果")
    g = p.add_argument_group("處理（預設都開）")
    g.add_argument("--no-background", action="store_true", help="不去光梯度")
    g.add_argument("--no-color", action="store_true", help="不校色")
    g.add_argument("--no-denoise", action="store_true", help="不降噪")
    g.add_argument("--no-detail", action="store_true", help="不做細節與縮星")
    g.add_argument("--denoise", type=float, help="降噪強度 0–1（預設 0.5）")
    g.add_argument("--stars", type=float, help="縮星 0–1（預設 0.5 ≈ 星點 FWHM ×0.8）")
    g.add_argument("--sharpen", type=float, help="星雲細節 0–1（預設 0.5）")
    g = p.add_argument_group("拉伸與成品微調")
    g.add_argument("--stretch", type=float, help="拉伸強度 0–1（預設 0.5）")
    g.add_argument("--background-level", type=float, help="天空亮度（預設 0.13）")
    g.add_argument("--background-color", type=float, help="背景保留色彩 0–1（預設 0.5；0＝天空中性）")
    g.add_argument("--unlinked", action="store_true", help="三色各自拉伸（未校色的素材）")
    g.add_argument("--exposure", type=float, default=0.0, help="明暗 −1–1")
    g.add_argument("--contrast", type=float, default=0.0, help="對比 −1–1")
    g.add_argument("--saturation", type=float, default=1.0, help="飽和度 0–2")
    g.add_argument("--green-removal", type=float, default=0.0, help="去綠 0–1")
    p.add_argument("--lang", choices=sorted(LANGUAGES), default="zh")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return p


def settings_from_args(args: argparse.Namespace) -> pipeline.ProcessingSettings:
    s = pipeline.ProcessingSettings.recommended()
    s = replace(s, background_enabled=not args.no_background, color_enabled=not args.no_color,
                denoise_enabled=not args.no_denoise, detail_enabled=not args.no_detail)
    if args.denoise is not None:
        s = replace(s, denoise=replace(s.denoise, amount=args.denoise))
    if args.stars is not None or args.sharpen is not None:
        d = s.detail
        s = replace(s, detail=replace(d, star_reduction=d.star_reduction if args.stars is None else args.stars,
                                      sharpen=d.sharpen if args.sharpen is None else args.sharpen))
    st = s.stretch
    if args.stretch is not None:
        st = replace(st, strength=args.stretch)
    if args.background_level is not None:
        st = replace(st, background=args.background_level)
    if args.background_color is not None:
        st = replace(st, background_color=args.background_color)
    if args.unlinked:
        st = replace(st, linked=False)
    return replace(s, stretch=st)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_language(args.lang)
    image, header = imageio.load_image(Path(args.input))
    settings = settings_from_args(args)
    adjust = display.DisplayAdjustments(exposure=args.exposure, contrast=args.contrast,
                                        saturation=args.saturation, green_removal=args.green_removal)
    t = time.perf_counter()

    def progress(fraction: float, message: object) -> None:
        print(f"\r{int(fraction * 100):3d}%  {message}      ", end="", file=sys.stderr, flush=True)

    result = pipeline.process(image, settings, progress)
    print(file=sys.stderr)
    finished = display.apply(result.display, adjust)
    for out in args.output:
        path = Path(out)
        if path.suffix.lower() in imageio.FITS_SUFFIXES:
            imageio.save_fits(path, result.linear, header)
        else:
            imageio.save_display(path, finished)
        print(path)
    print(f"{time.perf_counter() - t:.1f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
