"""參考降噪在各尺度做了什麼：參考降噪前（參考細節後，沒有參考細節就用參考校色）→ 參考降噪後。

兩張用同一組拉伸參數（量參考降噪後的噪聲）轉成顯示值，再分尺度：
  天空：參考降噪後／前 的噪聲比（亮度、色差各一）→ 每個尺度壓掉多少
  明亮星雲：參考降噪後投影到參考降噪前的保留率 → 結構有沒有被磨掉
"""

import sys

import numpy as np

from apu_processing import stretch
from apu_processing.imageio import load_image
from tools.scorecard.__main__ import load_regions
from tools.scorecard.data import Sample, all_samples
from tools.scorecard.metrics import BAND_LABELS, bands, inpaint, sample_indices, to_opponent


def band_vectors(display, R, index):
    opp = to_opponent(inpaint(display, R.star_dilated))
    out = []
    for plane in opp:
        out.append([{n: b.ravel()[i].astype(np.float64) for n, i in index.items()} for b in bands(plane)])
    return out


def main():
    keys = sys.argv[1:] or [s.key for s in all_samples()]
    agg = {"skyL": [], "skyC": [], "retL": [], "retC": []}
    for k in keys:
        s = Sample(k)
        before_path = s.before("denoise")
        before = load_image(before_path)[0]
        after = s.load_linear_final()
        R = load_regions(s, s.reference)
        n = stretch.measure(after)
        a = stretch.apply(after, noise=n)
        b = stretch.apply(before, noise=n)
        idx = sample_indices(R)
        va, vb = band_vectors(a, R, idx), band_vectors(b, R, idx)
        rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
        skyL = [rms(va[0][i]["sky"]) / rms(vb[0][i]["sky"]) for i in range(6)]
        skyC = [rms(np.r_[va[1][i]["sky"], va[2][i]["sky"]]) / rms(np.r_[vb[1][i]["sky"], vb[2][i]["sky"]]) for i in range(6)]
        retL = [float(va[0][i]["bright"] @ vb[0][i]["bright"] / (vb[0][i]["bright"] @ vb[0][i]["bright"])) for i in range(6)]
        retC = [float((va[1][i]["bright"] @ vb[1][i]["bright"] + va[2][i]["bright"] @ vb[2][i]["bright"]) /
                      (vb[1][i]["bright"] @ vb[1][i]["bright"] + vb[2][i]["bright"] @ vb[2][i]["bright"])) for i in range(6)]
        for key, v in (("skyL", skyL), ("skyC", skyC), ("retL", retL), ("retC", retC)):
            agg[key].append(v)
        f = lambda v: " ".join(f"{x:.2f}" for x in v)  # noqa: E731
        print(f"{k:12} 天空噪聲留下 L {f(skyL)} | C {f(skyC)}  ||  亮雲保留 L {f(retL)} | C {f(retC)}", flush=True)
    print("尺度        " + " ".join(f"{b:>4}" for b in BAND_LABELS))
    for key, label in (("skyL", "天空亮度噪聲留下"), ("skyC", "天空彩色噪聲留下"), ("retL", "亮雲亮度保留"), ("retC", "亮雲色彩保留")):
        print(f"{label:10} " + " ".join(f"{x:4.2f}" for x in np.median(np.array(agg[key]), 0)))


if __name__ == "__main__":
    main()
