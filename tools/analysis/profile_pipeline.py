"""量管線每一段的時間：python -m tools.analysis.profile_pipeline [素材代號]"""
import sys
import time

from apu_processing import background, color, denoise, detail, stretch
from tools.scorecard.data import Sample

raw = Sample(sys.argv[1] if len(sys.argv) > 1 else "ic434").load_raw()
T = {}


def tm(name, fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    T[name] = time.perf_counter() - t
    return r


x = tm("background", background.correct, raw)
bal = tm("color.measure", color.measure_balance, x)
x = tm("color.apply", color.apply, x, bal)
x2 = tm("denoise", denoise.reduce, x)
st = tm("find_stars", detail.find_stars, x2.mean(0))
y, S = tm("reduce_stars", detail.reduce_stars, x2, st, detail.DetailSettings())
z = tm("sharpen", detail.sharpen, y, st.mask, detail.DetailSettings())
n = tm("stretch.measure", stretch.measure, z)
d = tm("stretch.apply", stretch.apply, z, stretch.StretchSettings(), n)
for k, v in T.items():
    print(f"{k:16} {v:5.2f} s")
print(f"合計 {sum(T.values()):.1f} s  （{raw.shape[2]}×{raw.shape[1]}）")
