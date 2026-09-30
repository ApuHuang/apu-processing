"""去光梯度對參考去光的比對。

D = 我們的結果 − 參考去光（同一份原始資料、都是加減曲面，所以不用比倍率），扣掉中位數後 σ=48 模糊，
以參考去光圖的 σ₈ 噪聲為單位（1 ≈ 拉伸後 0.05 的亮度差）：
  殘留 = D 的 99 百分位（該扣沒扣）
  過扣 = −D 的 1 百分位（扣太多，通常是把星雲當光害）
另外存一張差異圖（亮度）方便目視：output/analysis/gradient/<key>_<方法>.png
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from apu_processing import background, stretch
from apu_processing.imageio import load_image
from tools.scorecard.data import Sample, splits
from tools.scorecard.metrics import blur

OUT = Path(__file__).resolve().parents[2] / "output/analysis/gradient"
SWIFT = Path.home() / "Claude/AstroSharp/.build-local/release/AstroSharpAnalyze"


def run_v098(sample: Sample) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "g.fits"
        subprocess.run([str(SWIFT), "--gradient", str(sample.raw_path), str(out)], check=True, capture_output=True)
        return load_image(out)[0]


METHODS = {
    "raw": lambda s, raw: raw,
    "v098": lambda s, raw: run_v098(s),
    "apu": lambda s, raw: background.correct(raw),
}


def diff_map(ours: np.ndarray, ref_gradient: np.ndarray, valid: np.ndarray, sigma8: float) -> np.ndarray:
    d = ours - ref_gradient
    d -= np.median(d[:, valid], axis=1)[:, None, None]
    return np.stack([blur(c, 48) for c in d]) / sigma8


def save_map(path: Path, dmap: np.ndarray, valid: np.ndarray, limit: float = 4.0) -> None:
    L = dmap.mean(0)
    f = max(1, L.shape[1] // 600)
    L = L[::f, ::f]
    v = valid[::f, ::f]
    x = np.clip(L / limit, -1, 1)
    rgb = np.zeros(x.shape + (3,), np.float32)
    rgb[..., 0] = np.where(x > 0, x, 0)          # 紅：比參考去光亮（殘留）
    rgb[..., 2] = np.where(x < 0, -x, 0)         # 藍：比參考去光暗（過扣）
    rgb[..., 1] = 0.15 * (1 - np.abs(x))
    rgb[~v] = 0.3
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("methods", nargs="+", choices=sorted(METHODS))
    p.add_argument("--only", default="")
    p.add_argument("--split", default="all")
    args = p.parse_args()
    g = splits()
    keys = g["tune"] + g["holdout"] if args.split == "all" else g[args.split]
    if args.only:
        keys = [k for k in keys if k in args.only.split(",")]
    table = {m: [] for m in args.methods}
    for k in keys:
        s = Sample(k)
        raw = s.load_raw()
        ref_gradient = load_image(s.stage_path("gradient"))[0]
        valid = s.valid_mask(ref_gradient.shape[1:])
        sig = stretch.measure(ref_gradient).sigma
        cells = []
        for m in args.methods:
            ours = METHODS[m](s, raw)
            dm = diff_map(ours, ref_gradient, valid, sig)
            lum = dm.mean(0)[valid]
            chroma = (dm - dm.mean(0, keepdims=True))[:, valid]
            res, over = float(np.quantile(lum, 0.99)), float(-np.quantile(lum, 0.01))
            col = float(np.quantile(np.abs(chroma), 0.99))
            table[m].append((res, over, col))
            save_map(OUT / f"{k}_{m}.png", dm, valid)
            cells.append(f"{m} 殘留 {res:5.1f} 過扣 {over:5.1f} 色偏 {col:4.1f}")
        print(f"{k:12} {'驗收' if k in g['holdout'] else '調參'}  " + "  |  ".join(cells), flush=True)
    for m, rows in table.items():
        a = np.array(rows)
        print(f"中位數 {m:5} 殘留 {np.median(a[:, 0]):.2f}  過扣 {np.median(a[:, 1]):.2f}  色偏 {np.median(a[:, 2]):.2f}   "
              f"最差 殘留 {a[:, 0].max():.1f} 過扣 {a[:, 1].max():.1f} 色偏 {a[:, 2].max():.1f}")


if __name__ == "__main__":
    main()
