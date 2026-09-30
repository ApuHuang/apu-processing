"""降噪對參考降噪：同一個輸入（參考降噪前，參考細節後），比「各尺度噪聲留下多少、明亮星雲結構保留多少」，
以及我們的結果跟參考降噪的相似度（亮雲各尺度：投影到參考降噪的保留率、跟參考降噪無關的多餘量）。"""

import argparse
import time

import numpy as np

from apu_processing import denoise, stretch
from apu_processing.imageio import load_image
from tools.analysis.denoise_profile import band_vectors
from tools.scorecard.__main__ import load_regions
from tools.scorecard.data import Sample, splits
from tools.scorecard.metrics import BAND_LABELS, sample_indices


def profile(va, vb):
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
    skyL = [rms(va[0][i]["sky"]) / rms(vb[0][i]["sky"]) for i in range(6)]
    skyC = [rms(np.r_[va[1][i]["sky"], va[2][i]["sky"]]) / rms(np.r_[vb[1][i]["sky"], vb[2][i]["sky"]]) for i in range(6)]
    retL = [float(va[0][i]["bright"] @ vb[0][i]["bright"] / (vb[0][i]["bright"] @ vb[0][i]["bright"])) for i in range(6)]
    return skyL, skyC, retL


def similarity(vo, vn):
    """我們 vs 參考降噪（亮雲、亮度）：保留率與多餘量。"""
    ret, exc = [], []
    for i in range(6):
        o, n = vo[0][i]["bright"], vo[0][i]["bright"] * 0 + vn[0][i]["bright"]
        k = float(o @ n / (n @ n))
        ret.append(k)
        exc.append(float(np.sqrt(np.sum((o - k * n) ** 2) / (n @ n))))
    return ret, exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--split", default="tune")
    ap.add_argument("--amount", type=float, default=0.5)
    ap.add_argument("--kl", default="", help="亮度各尺度門檻，逗號分隔")
    ap.add_argument("--kc", default="", help="色差各尺度門檻，逗號分隔")
    ap.add_argument("--per-scale-shape", action="store_true", help="舊做法：每個尺度各自擬合噪聲隨亮度的變化")
    args = ap.parse_args()
    g = splits()
    keys = g["tune"] + g["holdout"] if args.split == "all" else g[args.split]
    if args.only:
        keys = args.only.split(",")
    agg = {k: [] for k in ("ref", "ours", "sim")}
    for k in keys:
        s = Sample(k)
        before = load_image(s.before("denoise"))[0]
        ref_denoised = s.load_linear_final()
        t = time.perf_counter()
        kw = {"amount": args.amount, "shared_shape": not args.per_scale_shape}
        if args.kl:
            kw["k_luminance"] = tuple(float(v) for v in args.kl.split(","))
        if args.kc:
            kw["k_chroma"] = tuple(float(v) for v in args.kc.split(","))
        ours = denoise.reduce(before, denoise.DenoiseSettings(**kw))
        dt = time.perf_counter() - t
        R = load_regions(s, s.reference)
        n = stretch.measure(ref_denoised)
        idx = sample_indices(R)
        vb = band_vectors(stretch.apply(before, noise=n), R, idx)
        vn = band_vectors(stretch.apply(ref_denoised, noise=n), R, idx)
        vo = band_vectors(stretch.apply(ours, noise=n), R, idx)
        pn, po = profile(vn, vb), profile(vo, vb)
        sim = similarity(vo, vn)
        agg["ref"].append(pn)
        agg["ours"].append(po)
        agg["sim"].append(sim)
        f = lambda v: " ".join(f"{x:.2f}" for x in v)  # noqa: E731
        print(f"{k:12} {dt:4.1f}s  天空L 我們 {f(po[0])} / 參考降噪 {f(pn[0])} | 亮雲保留 我們 {f(po[2])} / 參考降噪 {f(pn[2])}", flush=True)
    print("尺度                " + " ".join(f"{b:>4}" for b in BAND_LABELS))
    for name, label in (("ref", "參考降噪"), ("ours", "我們")):
        a = np.median(np.array(agg[name]), 0)
        print(f"{label:4} 天空亮度噪聲留下 " + " ".join(f"{x:4.2f}" for x in a[0]))
        print(f"{label:4} 天空彩色噪聲留下 " + " ".join(f"{x:4.2f}" for x in a[1]))
        print(f"{label:4} 亮雲亮度保留     " + " ".join(f"{x:4.2f}" for x in a[2]))
    a = np.median(np.array(agg["sim"]), 0)
    print("我們 vs 參考降噪亮雲保留 " + " ".join(f"{x:4.2f}" for x in a[0]))
    print("我們 vs 參考降噪多餘量   " + " ".join(f"{x:4.2f}" for x in a[1]))


if __name__ == "__main__":
    main()
