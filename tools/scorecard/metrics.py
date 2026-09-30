"""成品對參考成品的評分。全部在顯示空間（0–1）上算。

區域都從參考成品找（每個候選版本用同一套區域）：
- 星點：小尺度的圓形亮點（細長的星雲纖維不算星）
- 天空：最暗四分之一的區塊裡、不是星的像素
- 暗淡星雲／明亮星雲：比天空亮 0.015 以上的非星像素，依亮度分兩半

比細節前先把候選版本每個色版的色調對到參考成品（分位數對應），拉伸亮暗不同不會被當成細節差異；
色調本身另外評（天空亮度、色調曲線差、顏色差）。

每個尺度（高斯差分，約 1、2、4、8、16、32 px）量兩件事：
- 保留率 retained = <C,M>/<M,M>：參考成品在這個尺度有的結構，候選版本留下幾成。磨糊 → 小於 1
- 多餘量 excess = |C − retained·M| / |M|：跟參考成品無關的東西（噪聲、色塊、假細節）
天空只看噪聲比 rms(C)/rms(M)。亮星另外量暗環深度。
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

SIGMAS = (1, 2, 4, 8, 16, 32)
BAND_LABELS = ("1px", "2px", "4px", "8px", "16px", "32px")
_POOL = ThreadPoolExecutor(max_workers=6)


# ---------------------------------------------------------------------- 基本運算


def block_mean(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape
    h2, w2 = h // f * f, w // f * f
    return img[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def upsample(small: np.ndarray, shape: tuple[int, int], f: int) -> np.ndarray:
    """block_mean 的反向：雙線性放大回原尺寸（像素中心對齊）。"""
    h, w = shape
    ys = (np.arange(h) + 0.5) / f - 0.5
    xs = (np.arange(w) + 0.5) / f - 0.5
    ys = np.clip(ys, 0, small.shape[0] - 1)
    xs = np.clip(xs, 0, small.shape[1] - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, small.shape[0] - 1)
    x1 = np.minimum(x0 + 1, small.shape[1] - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    top = small[y0][:, x0] * (1 - wx) + small[y0][:, x1] * wx
    bottom = small[y1][:, x0] * (1 - wx) + small[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """高斯模糊；σ 大時先縮圖再模糊，速度與 σ 無關、誤差對評分無影響。"""
    if sigma < 6:
        return ndimage.gaussian_filter(img, sigma, truncate=3.0, mode="reflect").astype(np.float32)
    f = 2 ** int(math.log2(sigma / 3))
    small = block_mean(img, f)
    rest = math.sqrt(max(0.25, (sigma / f) ** 2 - 1 / 12))
    small = ndimage.gaussian_filter(small, rest, truncate=3.0, mode="reflect")
    return upsample(small.astype(np.float32), img.shape, f)


def pmap(fn, items):
    return list(_POOL.map(fn, items))


# ---------------------------------------------------------------------- 色調對齊


def match_tone(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """每個色版各自做分位數對應，讓候選版本的亮度分布與參考成品相同。"""
    qs = np.linspace(0, 1, 2049)
    out = np.empty_like(candidate)
    for c in range(candidate.shape[0]):
        src = candidate[c][mask][::3]
        dst = reference[c][mask][::3]
        qc = np.quantile(src, qs)
        qm = np.quantile(dst, qs)
        qc = np.maximum.accumulate(qc) + np.arange(qs.size) * 1e-9  # np.interp 需要遞增
        out[c] = np.interp(candidate[c], qc, qm).astype(np.float32)
    return out


# ---------------------------------------------------------------------- 區域


@dataclass
class Regions:
    valid: np.ndarray
    star: np.ndarray          # 星點（未擴張）
    star_dilated: np.ndarray  # 評分時排除的範圍
    sky: np.ndarray
    faint: np.ndarray
    bright: np.ndarray
    sky_level: float
    stars: list[tuple[float, float, float, float]] = field(default_factory=list)  # (y, x, 半徑, 峰值)


def _robust_sigma(values: np.ndarray) -> float:
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med))) or 1e-6


def find_regions(reference: np.ndarray, valid: np.ndarray) -> Regions:
    L = reference.mean(axis=0)
    h, w = L.shape

    # 星點：兩個尺度的 top-hat（小星在原尺寸、大星在 1/4 尺寸），再用形狀排除細長的纖維
    small_top = L - ndimage.grey_opening(L, size=(11, 11))
    f = 4
    Ls = block_mean(L, f)
    big_top = upsample(Ls - ndimage.grey_opening(Ls, size=(11, 11)), (h, w), f)
    t_small = max(0.04, 6 * _robust_sigma(small_top[valid][::7]))
    t_big = max(0.05, 6 * _robust_sigma(big_top[valid][::7]))
    cand = ((small_top > t_small) | (big_top > t_big)) & valid
    labels, n = ndimage.label(cand)
    star = np.zeros_like(cand)
    stars: list[tuple[float, float, float, float]] = []
    if n:
        slices = ndimage.find_objects(labels)
        for i, sl in enumerate(slices, start=1):
            comp = labels[sl] == i
            area = int(comp.sum())
            if area < 3:
                continue
            ys, xs = np.nonzero(comp)
            if area >= 12:
                cov = np.cov(np.vstack([ys, xs]))
                ev = np.linalg.eigvalsh(cov)
                if ev[0] <= 0 or ev[1] / ev[0] > 4.0:  # 長短軸比 > 2：纖維，不是星
                    continue
            star[sl] |= comp
            radius = math.sqrt(area / math.pi)
            peak = float(L[sl][comp].max())
            stars.append((sl[0].start + ys.mean(), sl[1].start + xs.mean(), radius, peak))

    # 依大小擴張：半徑越大的星，暈越大
    dist = ndimage.distance_transform_edt(~star)
    radius_map = np.zeros((h, w), np.float32)
    if stars:
        lab, _ = ndimage.label(star)
        rads = np.zeros(lab.max() + 1, np.float32)
        for y, x, r, _p in stars:
            rads[lab[int(round(y)), int(round(x))]] = max(rads[lab[int(round(y)), int(round(x))]], r)
        _, (iy, ix) = ndimage.distance_transform_edt(~star, return_indices=True)
        radius_map = rads[lab[iy, ix]]
    star_dilated = star | (dist <= 2 + 1.5 * radius_map)

    # 天空：64 px 區塊的亮度中位數最暗的 25%
    tile = 64
    Lb = blur(L, 2)
    th, tw = h // tile, w // tile
    tiles = Lb[:th * tile, :tw * tile].reshape(th, tile, tw, tile)
    vt = valid[:th * tile, :tw * tile].reshape(th, tile, tw, tile).all(axis=(1, 3))
    med = np.median(tiles, axis=(1, 3))
    cutoff = np.quantile(med[vt], 0.25) if vt.any() else np.inf
    dark_tiles = vt & (med <= cutoff)
    sky = np.zeros((h, w), bool)
    sky[:th * tile, :tw * tile] = np.repeat(np.repeat(dark_tiles, tile, 0), tile, 1)
    sky &= valid & ~star_dilated
    sky_level = float(np.median(L[sky])) if sky.any() else float(np.quantile(L[valid], 0.1))

    # 星雲：大尺度亮度比天空亮 0.015 以上的非星像素，依亮度分兩半
    L_large = blur(inpaint(L[None], star_dilated)[0], 8)
    delta = L_large - sky_level
    structure = valid & ~star_dilated & (delta > 0.015)
    if structure.any():
        split = np.median(delta[structure])
        faint = structure & (delta <= split)
        bright = structure & (delta > split)
    else:
        faint = bright = structure
    return Regions(valid, star, star_dilated, sky, faint, bright, sky_level, stars)


def inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """遮罩內的像素用周圍的值補（正規化卷積）：先小範圍，補不到的（大星中心）再用更大的範圍。"""
    out = image.copy()
    known = (~mask).astype(np.float32)
    remaining = mask.copy()
    for sigma in (6, 24, 96):
        wgt = blur(known, sigma)
        fill_now = remaining & (wgt > 0.02)
        for c in range(out.shape[0]):
            val = blur(out[c] * known, sigma) / np.maximum(wgt, 1e-6)
            out[c][fill_now] = val[fill_now]
        known[fill_now] = 1.0
        remaining &= ~fill_now
        if not remaining.any():
            break
    for c in range(out.shape[0]):
        if remaining.any():
            out[c][remaining] = np.median(out[c][~mask])
    return out


# ---------------------------------------------------------------------- 分尺度


def to_opponent(image: np.ndarray) -> np.ndarray:
    """亮度與兩個色差：L=(R+G+B)/3、c1=(R−B)/√2、c2=(R+B−2G)/√6。"""
    r, g, b = image
    return np.stack([(r + g + b) / 3, (r - b) / math.sqrt(2), (r + b - 2 * g) / math.sqrt(6)]).astype(np.float32)


def bands(plane: np.ndarray) -> list[np.ndarray]:
    out = []
    prev = plane
    for s in SIGMAS:
        cur = blur(plane, s)
        out.append(prev - cur)
        prev = cur
    return out


# ---------------------------------------------------------------------- 亮星暗環


def pick_bright_stars(stars: list[tuple[float, float, float, float]], shape: tuple[int, int],
                      limit: int = 40) -> list[tuple[float, float, float]]:
    """最大的幾顆星（半徑 ≥ 2.5 px、離邊緣夠遠、彼此不重疊）。"""
    h, w = shape
    picked: list[tuple[float, float, float]] = []
    for y, x, r, _p in sorted(stars, key=lambda s: -s[2]):
        if r < 2.5:
            break
        R = max(r, 3.0)
        outer = 7 * R
        if y < outer or x < outer or y > h - outer or x > w - outer:
            continue
        if any((y - py) ** 2 + (x - px) ** 2 < (6 * max(R, pr)) ** 2 for py, px, pr in picked):
            continue
        picked.append((y, x, R))
        if len(picked) >= limit:
            break
    return picked


def pick_small_stars(stars, shape, limit: int = 400) -> list[tuple[float, float]]:
    """量星點大小用：半徑 1.5–6 px、彼此至少 15 px、離邊緣 20 px 以上。"""
    h, w = shape
    cand = [(y, x) for y, x, r, _p in stars if 1.5 <= r <= 6 and 20 < y < h - 20 and 20 < x < w - 20]
    if not cand:
        return []
    pts = np.array(cand)
    grid: dict[tuple[int, int], list[int]] = {}
    for i, (y, x) in enumerate(pts):
        grid.setdefault((int(y // 15), int(x // 15)), []).append(i)
    keep = []
    for i, (y, x) in enumerate(pts):
        gy, gx = int(y // 15), int(x // 15)
        near = [j for dy in (-1, 0, 1) for dx in (-1, 0, 1) for j in grid.get((gy + dy, gx + dx), []) if j != i]
        if all(np.hypot(*(pts[j] - pts[i])) >= 15 for j in near):
            keep.append((float(y), float(x)))
    step = max(1, len(keep) // limit)
    return keep[::step][:limit]


def star_fwhm(L: np.ndarray, y: float, x: float, r: int = 6) -> float:
    """二階矩半高寬（扣掉外圈中位數當背景）。"""
    yi, xi = int(round(y)), int(round(x))
    box = L[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(np.float64)
    outer = L[yi - r - 3:yi + r + 4, xi - r - 3:xi + r + 4]
    bg = np.median(np.r_[outer[:3].ravel(), outer[-3:].ravel(), outer[:, :3].ravel(), outer[:, -3:].ravel()])
    f = np.clip(box - bg, 0, None)
    tot = f.sum()
    if tot <= 0:
        return float("nan")
    yy, xx = np.indices(f.shape)
    cy, cx = (f * yy).sum() / tot, (f * xx).sum() / tot
    return float(2.3548 * np.sqrt((f * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / tot / 2))


def profile_violation(L: np.ndarray, y: float, x: float, r_max: float = 10.0) -> float:
    """星點徑向剖面（每 1 px 一圈取中位數）「往外反而變亮」的總量（顯示單位）。
    正常的星往外單調變暗 → 0；甜甜圈星點、星點遮罩的硬邊暗框 → 正值。"""
    yi, xi = int(round(y)), int(round(x))
    r0 = int(math.ceil(r_max)) + 1
    patch = L[yi - r0:yi + r0 + 1, xi - r0:xi + r0 + 1]
    if patch.shape != (2 * r0 + 1, 2 * r0 + 1):
        return float("nan")
    yy, xx = np.indices(patch.shape)
    rr = np.hypot(yy + yi - r0 - y, xx + xi - r0 - x)
    edges = np.arange(0, r_max + 0.01, 1.0)
    prof = [float(np.median(patch[(rr >= a) & (rr < b)])) for a, b in zip(edges[:-1], edges[1:])
            if ((rr >= a) & (rr < b)).any()]
    d = np.diff(prof)
    return float(np.clip(d, 0, None).sum())


def _local_dark(patch: np.ndarray, rr: np.ndarray, sel: np.ndarray) -> float:
    """每個半徑（0.25R 一圈）減掉該圈的中位數，取最暗 1% 的平均深度。"""
    rbin = np.floor(rr[sel] * 4).astype(int)
    vals = patch[sel].astype(np.float64)
    order = np.argsort(rbin, kind="stable")
    rbin, vals = rbin[order], vals[order]
    starts = np.flatnonzero(np.r_[True, rbin[1:] != rbin[:-1]])
    ends = np.r_[starts[1:], rbin.size]
    dev = np.empty_like(vals)
    for a, b in zip(starts, ends):
        dev[a:b] = vals[a:b] - np.median(vals[a:b])
    k = max(1, dev.size // 100)
    return float(-np.mean(np.partition(dev, k)[:k]))


def star_defects(Lc: np.ndarray, Lm: np.ndarray, stars: list[tuple[float, float, float]],
                 valid: np.ndarray) -> dict[str, list[float]]:
    """亮星周圍的兩種黑：
    - ring：一整圈的暗環（外側背景 − 1.3R～4R 環狀中位數的最低值；正數＝有暗環），候選與參考成品各量一次
    - holes：局部黑斑（1.2R～6R 內，比「同一圈」的中位數暗最多的 1% 像素平均暗多少，扣掉參考成品自己的值）。
      拿星點自己同半徑的亮度當基準，星暈大小不同不會被算成黑斑；斷掉的繞射芒、黑圈殘留會
    """
    out = {"ring": [], "ring_ref": [], "holes": []}
    for y, x, R in stars:
        r0 = int(math.ceil(7 * R))
        y0, x0 = int(y) - r0, int(x) - r0
        sl = (slice(y0, y0 + 2 * r0 + 1), slice(x0, x0 + 2 * r0 + 1))
        pc, pm, vp = Lc[sl], Lm[sl], valid[sl]
        yy, xx = np.indices(pc.shape)
        rr = np.hypot(yy + y0 - y, xx + x0 - x) / R
        bg_sel = (rr >= 5) & (rr <= 7) & vp
        if bg_sel.sum() < 20:
            continue
        for patch, key in ((pc, "ring"), (pm, "ring_ref")):
            background = float(np.median(patch[bg_sel]))
            edges = np.arange(1.3, 4.01, 0.3)
            prof = [float(np.median(patch[(rr >= a) & (rr < b) & vp])) for a, b in zip(edges[:-1], edges[1:])
                    if ((rr >= a) & (rr < b) & vp).sum() >= 8]
            out[key].append(background - min(prof) if prof else float("nan"))
        ann = (rr >= 1.2) & (rr <= 6) & vp
        if ann.sum() >= 50:
            out["holes"].append(_local_dark(pc, rr, ann) - _local_dark(pm, rr, ann))
    return out


# ---------------------------------------------------------------------- 總評


SAMPLES_PER_REGION = 400_000


def sample_indices(regions: Regions, seed: int = 0) -> dict[str, np.ndarray]:
    """每個區域固定抽一組像素（平面索引），所有候選版本都用同一組。"""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("sky", "faint", "bright"):
        idx = np.flatnonzero(getattr(regions, name))
        if idx.size > SAMPLES_PER_REGION:
            idx = np.sort(rng.choice(idx, SAMPLES_PER_REGION, replace=False))
        out[name] = idx
    return out


def band_samples(image: np.ndarray, regions: Regions, index: dict[str, np.ndarray]) -> list[list[dict[str, np.ndarray]]]:
    """[亮度, 色差1, 色差2] × 各尺度 × 各區域的抽樣值。一次只留一個色版的完整分尺度結果，記憶體才夠。"""
    opp = to_opponent(inpaint(image, regions.star_dilated))

    def one(plane: np.ndarray) -> list[dict[str, np.ndarray]]:
        flat_bands = [b.ravel() for b in bands(plane)]
        return [{name: fb[idx].astype(np.float64) for name, idx in index.items()} for fb in flat_bands]

    return pmap(one, list(opp))


def _stats(cv: np.ndarray, mv: np.ndarray) -> tuple[float, float, float]:
    mm = float(mv @ mv)
    if cv.size < 500 or mm <= 0:
        return float("nan"), float("nan"), float("nan")
    k = float(cv @ mv) / mm
    resid = cv - k * mv
    return k, math.sqrt(float(resid @ resid) / mm), math.sqrt(float(cv @ cv) / mm)


def reference_data(reference: np.ndarray, regions: Regions) -> dict:
    """同一張參考成品只算一次：抽樣位置、分尺度抽樣值、暗環。"""
    index = sample_indices(regions)
    return {"index": index, "bands": band_samples(reference, regions, index),
            "bright_stars": pick_bright_stars(regions.stars, regions.valid.shape),
            "small_stars": pick_small_stars(regions.stars, regions.valid.shape)}


def evaluate(candidate: np.ndarray, reference: np.ndarray, regions: Regions, ref: dict) -> dict:
    """candidate、reference：(3, H, W) 0–1 顯示影像；ref 是 reference_data() 的結果。"""
    R = regions
    matched = match_tone(candidate, reference, R.valid)
    cand = band_samples(matched, R, ref["index"])
    mref = ref["bands"]
    nb = len(SIGMAS)

    out: dict = {"bands": list(BAND_LABELS)}
    for name in ("faint", "bright"):
        L = [_stats(cand[0][b][name], mref[0][b][name]) for b in range(nb)]
        C = []
        for b in range(nb):
            cv = np.concatenate([cand[1][b][name], cand[2][b][name]])
            mv = np.concatenate([mref[1][b][name], mref[2][b][name]])
            C.append(_stats(cv, mv))
        for key, rows in (("L", L), ("C", C)):
            out[f"{name}_{key}"] = {"retained": [r[0] for r in rows], "excess": [r[1] for r in rows],
                                   "amplitude": [r[2] for r in rows]}
    sky_noise = {"L": [], "C": []}
    for b in range(nb):
        sky_noise["L"].append(_stats(cand[0][b]["sky"], mref[0][b]["sky"])[2])
        cv = np.concatenate([cand[1][b]["sky"], cand[2][b]["sky"]])
        mv = np.concatenate([mref[1][b]["sky"], mref[2][b]["sky"]])
        sky_noise["C"].append(_stats(cv, mv)[2])
    out["sky_noise"] = sky_noise

    # 亮星：用色調對齊後的亮度跟參考成品比
    d = star_defects(matched.mean(axis=0), reference.mean(axis=0), ref["bright_stars"], R.valid)
    nanmed = lambda v: float(np.nanmedian(v)) if v else float("nan")  # noqa: E731
    out["stars"] = {"n": len(d["holes"]), "ring": nanmed(d["ring"]), "ring_ref": nanmed(d["ring_ref"]),
                    "holes_median": nanmed(d["holes"]), "holes_max": float(np.max(d["holes"])) if d["holes"] else float("nan")}

    # 星點大小：孤立的中小星，候選／參考成品的 FWHM 比（色調對齊後量）
    Lmatch = matched.mean(axis=0)
    Lref = reference.mean(axis=0)
    ratios = []
    for y, x in ref["small_stars"]:
        a, b = star_fwhm(Lmatch, y, x), star_fwhm(Lref, y, x)
        if np.isfinite(a) and np.isfinite(b) and b > 0.5:
            ratios.append(a / b)
    out["stars"]["size_ratio"] = float(np.median(ratios)) if ratios else float("nan")
    out["stars"]["size_n"] = len(ratios)
    # 星點剖面：候選版本往外反而變亮的量，扣掉參考成品自己的（取 90 百分位：少數壞星也看得到）
    viol = []
    for y, x in ref["small_stars"][:200]:
        a, b = profile_violation(Lmatch, y, x), profile_violation(Lref, y, x)
        if np.isfinite(a) and np.isfinite(b):
            viol.append(a - b)
    out["stars"]["profile_p90"] = float(np.quantile(viol, 0.9)) if viol else float("nan")

    # 色調與顏色（未對齊的原始成品）
    Lraw = candidate.mean(axis=0)
    Lm = reference.mean(axis=0)
    qs = np.linspace(0.01, 0.999, 200)
    sel = R.valid & ~R.star_dilated
    tone = np.abs(np.quantile(Lraw[sel][::5], qs) - np.quantile(Lm[sel][::5], qs))
    structure = R.faint | R.bright
    out["tone"] = {
        "sky": float(np.median(Lraw[R.sky])) if R.sky.any() else float("nan"),
        "ref_sky": R.sky_level,
        "curve_error": float(tone.mean()),
        "structure": float(np.median(Lraw[structure])) if structure.any() else float("nan"),
        "ref_structure": float(np.median(Lm[structure])) if structure.any() else float("nan"),
    }
    color = []
    for mask in (R.sky, R.faint, R.bright):
        if mask.sum() > 500:
            dc = [float(np.median(candidate[c][mask]) - np.median(reference[c][mask])) for c in range(3)]
            mean = sum(dc) / 3
            color.append(math.sqrt(sum((x - mean) ** 2 for x in dc) / 3))  # 只看色偏，不看整體亮度
    out["color_error"] = float(np.mean(color)) if color else float("nan")
    out["area"] = {k: float(getattr(R, k).mean()) for k in ("sky", "faint", "bright", "star_dilated")}
    return out


def summary(result: dict) -> dict[str, float]:
    """每張圖濃縮成幾個數字（給表格與分組統計用）。

    尺度索引：0=1px 1=2px 2=4px 3=8px 4=16px 5=32px。
    最細的 1–2 px 在暗處幾乎都是噪聲（而且參考成品裡留著同一份資料的噪聲，會跟原始檔相關），
    所以「細節」看 4–32 px 的結構；1–2 px 只在明亮星雲看（那裡訊號遠大於噪聲）。
    """
    def avg(values, idx):
        vals = [values[i] for i in idx if not math.isnan(values[i])]
        return float(np.mean(vals)) if vals else float("nan")

    def both(key, stat, idx):
        return avg([(a + b) / 2 for a, b in zip(result[f"faint_{key}"][stat], result[f"bright_{key}"][stat])], idx)

    structure, fine = (2, 3, 4, 5), (0, 1)
    return {
        "detail_faint": avg(result["faint_L"]["retained"], structure),
        "detail_bright": avg(result["bright_L"]["retained"], structure),
        "fine_bright": avg(result["bright_L"]["retained"], fine),
        "color_detail": both("C", "retained", (1, 2, 3, 4)),
        "blotch": both("C", "excess", (2, 3, 4)),
        "sky_noise_fine": avg(result["sky_noise"]["L"], (0, 1, 2)),
        "sky_noise_large": avg(result["sky_noise"]["L"], (3, 4, 5)),
        "sky_noise_C": avg(result["sky_noise"]["C"], (1, 2, 3, 4)),
        "holes": result["stars"]["holes_max"],
        "star_size": result["stars"].get("size_ratio", float("nan")),
        "star_profile": result["stars"].get("profile_p90", float("nan")),
        "ring": result["stars"]["ring"],
        "ring_ref": result["stars"]["ring_ref"],
        "sky": result["tone"]["sky"],
        "ref_sky": result["tone"]["ref_sky"],
        "tone_error": result["tone"]["curve_error"],
        "color_error": result["color_error"],
    }
