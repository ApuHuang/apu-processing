"""拆疊評分：python -m tools.truthscore 方法 [方法…] [--sets a,b] [--subs A4,A8] [--name 名稱]

方法：apu（建議設定）或「apu:設定改動」（同 tools/scorecard，例如 apu:denoise.amount=0.7）。
素材根目錄：環境變數 APU_TRUTH（預設 D:\\APU-truth），每組有 B、A4、A8 三個疊圖輸出資料夾（各有 done.json）。結果存到 output/truthscore/。

指標（每組、每份 A；「剩」越小越好、「留」越接近 1 越好）：
- 剩：誤差（扣掉 B 自己的噪聲後）/ 不降噪時的誤差。1＝跟沒處理一樣，0＝完全等於真值
- 留：訊號保留率。不降噪時應該 ≈ 1（量法的自我檢查，列在 留0）
- 細／中／大＝1–2、4–8、16–32 px；亮度 L、色差 C
- 星暈：星點周圍那一圈 處理後 − 答案 的中位數（天空像素噪聲倍數），負的＝暗環
- 暗星留：暗星（星心 5–50 倍噪聲）r ≤ 2 px 的光通量，處理後 ÷ 不降噪
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from tools.scorecard.methods import settings_with

from . import core

print = functools.partial(print, flush=True)  # noqa: A001

OUT = Path(__file__).resolve().parents[2] / "output/truthscore"
SETS = ("vdb152", "ngc1893", "lbn576", "m45", "ic405", "m8")
SHARED_PATTERN = {"m8"}   # A、B 共有固定圖樣（沒 dither）：降噪的「剩」不可信


def method_settings(name: str):
    if name == "apu":
        return settings_with("")
    if name.startswith("apu:"):
        return settings_with(name[4:])
    raise SystemExit(f"不認得的方法 {name}")


def score_set(root: Path, name: str, methods: list[str], subs: list[str]) -> dict:
    t0 = time.time()
    split = core.load_split(root, name, subs)
    out = {"frames": split.frames, "shape": list(split.B.shape), "methods": {}}
    fronts: dict = {}
    for m in methods:
        settings = method_settings(m)
        fkey = (settings.background_enabled, settings.background, settings.color_enabled)
        if fkey not in fronts:
            fronts.clear()
            fr, rep = core.front(split, settings)
            xb = fr.apply(split.B)
            xa = {s: fr.apply(a) for s, a in split.subs.items()}
            reg = core.regions(xb, xa[subs[0]], split.mask)
            fronts[fkey] = (fr, rep, xb, xa, reg, {})
        fr, rep, xb, xa, reg, qcache = fronts[fkey]
        nodenoise = replace(settings, denoise_enabled=False)
        dkey = (settings.detail_enabled, settings.detail)
        if dkey not in qcache:
            qcache[dkey] = {"Q": core.linear_stages(xb, nodenoise)}
        Q = qcache[dkey]["Q"]
        cq = {k: core.coefficients(Q, k) for k in ("L", "C")}
        res = {"front": rep}
        for s in subs:
            base_key = ("P0", s)
            if base_key not in qcache[dkey]:
                qcache[dkey][base_key] = core.linear_stages(xa[s], nodenoise)
            P0 = qcache[dkey][base_key]
            P = core.linear_stages(xa[s], settings) if settings.denoise_enabled else P0
            d = xa[s] - xb
            r = {}
            for kind in ("L", "C"):
                cp, cp0, cd = core.coefficients(P, kind), core.coefficients(P0, kind), core.coefficients(d, kind)
                for rname in ("sky", "faint", "bright"):
                    region = getattr(reg, rname)
                    if region.sum() < 5000:
                        continue
                    sm = core.sums(cp, cq[kind], cd, region)
                    s0 = core.sums(cp0, cq[kind], cd, region)
                    for k, v in core.summarize(sm, s0, split.frames[s], split.frames["B"]).items():
                        r[f"{kind}_{rname}_{k}"] = v
            for label, img in (("halo", P), ("halo0", P0)):
                diff = (img.mean(axis=0) - Q.mean(axis=0))[reg.halo]
                r[label] = float(np.median(diff) / reg.sky_noise) if diff.size else float("nan")
            ys, xs = reg.faint_stars
            if len(ys) >= 50:
                # 暗星光通量：處理後 ÷ 不降噪（同樣的細節處理）；兩邊同一份 A，噪聲大部分抵消
                f0 = core.star_flux(P0.mean(axis=0), ys, xs)
                f1 = core.star_flux(P.mean(axis=0), ys, xs)
                r["faint_star_keep"] = float(f1.sum() / max(f0.sum(), 1e-30))
                r["faint_stars"] = int(len(ys))
            res[s] = r
        out["methods"][m] = res
    out["region_pixels"] = {k: int(getattr(reg, k).sum()) for k in ("sky", "faint", "bright", "halo")}
    out["seconds"] = round(time.time() - t0)
    return out


COLUMNS = [
    ("L_sky_remain_fine", "天空L剩細"), ("L_sky_remain_mid", "天空L剩中"), ("L_sky_remain_large", "天空L剩大"),
    ("C_sky_remain_fine", "天空C剩細"), ("C_sky_remain_mid", "天空C剩中"),
    ("L_faint_remain_mid", "暗雲L剩中"), ("L_faint_keep_fine", "暗雲L留細"), ("L_faint_keep_mid", "暗雲L留中"),
    ("L_faint_keep_large", "暗雲L留大"),
    ("L_bright_keep_fine", "亮雲L留細"), ("L_bright_keep_mid", "亮雲L留中"), ("L_bright_remain_fine", "亮雲L剩細"),
    ("C_faint_keep_mid", "暗雲C留中"),
    ("L_faint_keep0_mid", "留0"), ("halo", "星暈"), ("halo0", "星暈0"), ("faint_star_keep", "暗星留"),
]


def table(results: dict, methods: list[str], subs: list[str]) -> str:
    lines = []
    head = f"{'組':10s}{'份':4s}" + "".join(f"{lab:>9s}" for _, lab in COLUMNS)
    for m in methods:
        lines += ["", f"== {m}", head]
        for name, r in results.items():
            for s in subs:
                row = r["methods"][m].get(s, {})
                mark = "*" if name in SHARED_PATTERN else " "
                cells = "".join(f"{row.get(k, float('nan')):9.2f}" for k, _ in COLUMNS)
                lines.append(f"{name + mark:10s}{s:4s}{cells}")
        for s in subs:
            vals = [[results[n]["methods"][m][s].get(k, np.nan) for k, _ in COLUMNS]
                    for n in results if n not in SHARED_PATTERN]
            med = np.nanmedian(np.array(vals, float), axis=0) if vals else []
            lines.append(f"{'中位數':9s}{s:4s}" + "".join(f"{v:9.2f}" for v in med))
    lines.append("")
    lines.append("* A、B 共有固定圖樣：不算進中位數")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.truthscore")
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--subs", default="A4,A8")
    ap.add_argument("--name", default=None, help="結果檔名（預設 results-時間）")
    a = ap.parse_args()
    root = Path(os.environ.get("APU_TRUTH", r"D:\APU-truth"))
    subs = a.subs.split(",")
    results = {}
    for name in a.sets.split(","):
        print(f"[{name}] …")
        results[name] = score_set(root, name, a.methods, subs)
        print(f"[{name}] {results[name]['seconds']} s")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = a.name or time.strftime("results-%Y%m%d-%H%M")
    (OUT / f"{stem}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    text = table(results, a.methods, subs)
    (OUT / f"{stem}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
