"""產生程式圖示與 Logo（APU Astro 系列同一家族：深藍圓角底、帶繞射芒的星點）：
APU Processing 多一條色調曲線（後製）與淡淡的直方圖。

- src/apu_processing/assets/app.ico      視窗與程式圖示
- src/apu_processing/assets/icon_128.png 視窗 Logo 用
- docs/logo.png                          README 用的 Logo 橫幅

python packaging/make_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "apu_processing" / "assets"
DOCS = ROOT / "docs"
SIZE = 1024
NAVY = (16, 28, 58, 255)
NAME = "APU Processing"
SUBTITLE = "Astrophotography Processing Utility"


def draw_icon() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=SIZE * 0.2, fill=NAVY)

    # 淡淡的直方圖（左低右長尾，像天文影像的亮度分布）
    hist = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hist)
    x0, x1, base = 0.16 * SIZE, 0.84 * SIZE, 0.84 * SIZE
    n = 28
    for i in range(n):
        t = (i + 0.5) / n
        h = 0.42 * SIZE * np.exp(-((t - 0.16) / 0.10) ** 2) + 0.05 * SIZE * np.exp(-t * 2.5)
        bx = x0 + (x1 - x0) * i / n
        hd.rectangle((bx + 3, base - h, bx + (x1 - x0) / n - 3, base), fill=(90, 120, 180, 70))
    img.alpha_composite(hist)

    # 色調曲線（拉伸）：從左下往上快速抬起、右上漸緩
    curve = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    cd = ImageDraw.Draw(curve)
    ts = np.linspace(0, 1, 200)
    ys = 1 - (1 + ts / 0.08) ** -0.9
    pts = [(x0 + (x1 - x0) * t, base - (base - 0.18 * SIZE) * y) for t, y in zip(ts, ys)]
    cd.line(pts, fill=(150, 200, 255, 235), width=int(SIZE * 0.028), joint="curve")
    img.alpha_composite(curve.filter(ImageFilter.GaussianBlur(1.5)))

    # 右上的亮星
    cx, cy = 0.70 * SIZE, 0.33 * SIZE
    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((cx - 170, cy - 170, cx + 170, cy + 170), fill=(120, 170, 255, 110))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(70)))
    spike = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spike)
    length, width = SIZE * 0.22, SIZE * 0.026
    sd.polygon([(cx, cy - length), (cx + width, cy), (cx, cy + length), (cx - width, cy)], fill=(235, 242, 255, 255))
    sd.polygon([(cx - length, cy), (cx, cy - width), (cx + length, cy), (cx, cy + width)], fill=(235, 242, 255, 255))
    img.alpha_composite(spike.filter(ImageFilter.GaussianBlur(3)))
    core = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(core).ellipse((cx - 62, cy - 62, cx + 62, cy + 62), fill=(255, 255, 255, 255))
    img.alpha_composite(core.filter(ImageFilter.GaussianBlur(9)))

    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=SIZE * 0.2, fill=255)
    img.putalpha(Image.composite(img.getchannel("A"), Image.new("L", (SIZE, SIZE), 0), mask))
    return img


def _font(names: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in [Path("C:/Windows/Fonts") / n for n in names] + [Path("/System/Library/Fonts/Helvetica.ttc")]:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def draw_logo(icon: Image.Image) -> Image.Image:
    w, h = 1640, 400
    scale = 2
    img = Image.new("RGBA", (w * scale, h * scale), NAVY)
    icon_size = 280 * scale
    img.alpha_composite(icon.resize((icon_size, icon_size), Image.LANCZOS), (60 * scale, 60 * scale))
    d = ImageDraw.Draw(img)
    title = _font(["segoeuib.ttf"], 140 * scale)
    sub = _font(["seguisb.ttf", "segoeui.ttf"], 56 * scale)
    x = (60 + 280 + 60) * scale
    d.text((x, 76 * scale), NAME, font=title, fill=(255, 255, 255, 255))
    d.text((x + 6 * scale, 262 * scale), SUBTITLE, font=sub, fill=(174, 187, 214, 255))
    return img.resize((w, h), Image.LANCZOS)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    icon = draw_icon()
    icon.resize((256, 256), Image.LANCZOS).save(
        ASSETS / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    icon.resize((128, 128), Image.LANCZOS).save(ASSETS / "icon_128.png", optimize=True)
    icon.save(ASSETS / "icon_1024.png", optimize=True)
    draw_logo(icon).convert("RGB").save(DOCS / "logo.png", optimize=True)
    print(f"已產生 {ASSETS / 'app.ico'}、{ASSETS / 'icon_128.png'}、{DOCS / 'logo.png'}")


if __name__ == "__main__":
    main()
