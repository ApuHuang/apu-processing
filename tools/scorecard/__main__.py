"""評分：python -m tools.scorecard 方法 [方法…] [--split tune|holdout|all] [--only a,b] [--sheets]

方法見 methods.py（ref、v098、ref-simple、raw-simple、apu …）。結果存到 output/scorecard/。
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
from pathlib import Path

import numpy as np

from .data import Sample, splits
from .methods import METHODS, resolve
from .metrics import Regions, evaluate, find_regions, reference_data, summary
from .sheets import make_sheet

print = functools.partial(print, flush=True)  # noqa: A001  背景跑時可以邊跑邊看

OUT = Path(__file__).resolve().parents[2] / "output/scorecard"
CACHE = OUT / "cache"

COLUMNS = [
    ("detail_faint", "暗雲結構", "{:.2f}"),
    ("detail_bright", "亮雲結構", "{:.2f}"),
    ("fine_bright", "亮雲細部", "{:.2f}"),
    ("color_detail", "色彩細節", "{:.2f}"),
    ("blotch", "色塊", "{:.2f}"),
    ("sky_noise_fine", "天空顆粒", "{:.2f}"),
    ("sky_noise_large", "天空斑駁", "{:.2f}"),
    ("sky_noise_C", "天空彩噪", "{:.2f}"),
    ("holes", "亮星黑塊", "{:.3f}"),
    ("star_size", "星點大小", "{:.2f}"),
    ("star_profile", "星點剖面", "{:.3f}"),
    ("ring", "暗環", "{:+.3f}"),
    ("sky", "天空亮度", "{:.3f}"),
    ("tone_error", "色調差", "{:.3f}"),
    ("color_error", "色偏", "{:.3f}"),
]


def load_regions(sample: Sample, reference: np.ndarray) -> Regions:
    path = CACHE / f"{sample.key}_regions.npz"
    shape = reference.shape[1:]
    if path.exists():
        z = np.load(path)
        unpack = lambda k: np.unpackbits(z[k])[: shape[0] * shape[1]].reshape(shape).astype(bool)  # noqa: E731
        return Regions(unpack("valid"), unpack("star"), unpack("star_dilated"), unpack("sky"), unpack("faint"),
                       unpack("bright"), float(z["sky_level"]), [tuple(s) for s in z["stars"].tolist()])
    regions = find_regions(reference, sample.valid_mask(shape))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.packbits(getattr(regions, k)) for k in
                                 ("valid", "star", "star_dilated", "sky", "faint", "bright")},
                        sky_level=regions.sky_level, stars=np.array(regions.stars, dtype=np.float64).reshape(-1, 4))
    return regions


def fmt(key: str, value: float) -> str:
    spec = next(f for k, _, f in COLUMNS if k == key)
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else spec.format(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="+", help=f"{', '.join(sorted(METHODS))}，或 apu:設定改動（例如 apu:denoise.amount=0.6）")
    parser.add_argument("--split", default="all", choices=["tune", "holdout", "all"])
    parser.add_argument("--only", default="")
    parser.add_argument("--sheets", action="store_true")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    groups = splits()
    keys = groups["tune"] + groups["holdout"] if args.split == "all" else groups[args.split]
    if args.only:
        keys = [k for k in keys if k in args.only.split(",")]

    results: dict[str, dict[str, dict]] = {}
    for key in keys:
        t0 = time.perf_counter()
        sample = Sample(key)
        reference = sample.reference
        regions = load_regions(sample, reference)
        ref = reference_data(reference, regions)
        results[key] = {}
        shown = {}
        for method in args.methods:
            candidate = np.clip(resolve(method)(sample), 0, 1).astype(np.float32)
            if candidate.ndim == 2:
                candidate = np.repeat(candidate[None], 3, axis=0)
            results[key][method] = evaluate(candidate, reference, regions, ref)
            if args.sheets:
                shown[method] = candidate
        if args.sheets:
            shown["參考成品"] = reference
            make_sheet(OUT / "sheets" / f"{key}.jpg", f"{key}  （100% 裁切）", shown, regions)
        split = "驗收" if key in groups["holdout"] else "調參"
        print(f"\n{key}（{split}）  {time.perf_counter() - t0:.0f} s   "
              + "  ".join(f"{n}={v * 100:.0f}%" for n, v in results[key][args.methods[0]]["area"].items()))
        print("  " + f"{'':10}" + "".join(f"{label:>8}" for _, label, _ in COLUMNS))
        for method in args.methods:
            s = summary(results[key][method])
            print("  " + f"{method[:10]:10}" + "".join(f"{fmt(k, s[k]):>10}" for k, _, _ in COLUMNS))
        s = summary(results[key][args.methods[0]])
        print("  " + f"{'參考成品':6}" + "".join(
            f"{fmt(k, v):>10}" for k, v in [(c, {'detail_faint': 1, 'detail_bright': 1, 'fine_bright': 1, 'color_detail': 1,
                                                 'blotch': 0, 'sky_noise_fine': 1, 'sky_noise_large': 1,
                                                 'sky_noise_C': 1, 'holes': 0, 'star_size': 1, 'star_profile': 0, 'ring': s['ring_ref'],
                                                 'sky': s['ref_sky'], 'tone_error': 0, 'color_error': 0}[c])
                                            for c, _, _ in COLUMNS]))

    print("\n分組中位數")
    for split in ("tune", "holdout"):
        ks = [k for k in keys if k in groups[split]]
        if not ks:
            continue
        print(f"  {'調參組' if split == 'tune' else '驗收組'}（{len(ks)} 張）")
        for method in args.methods:
            rows = [summary(results[k][method]) for k in ks]
            med = {c: float(np.nanmedian([r[c] for r in rows])) for c, _, _ in COLUMNS}
            print("  " + f"{method[:10]:10}" + "".join(f"{fmt(k, med[k]):>10}" for k, _, _ in COLUMNS))
    for i, method in enumerate(args.methods):
        print(f"  方法 {i + 1}: {method}")

    OUT.mkdir(parents=True, exist_ok=True)
    name = args.name or time.strftime("%Y%m%d-%H%M%S")
    (OUT / f"results-{name}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n存檔：{OUT / f'results-{name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
