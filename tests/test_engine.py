"""引擎各階段在合成圖上的關鍵性質。合成圖都帶真正隨機的噪聲。"""

import math

import numpy as np

from apu_processing import background, color, denoise, detail, pipeline

H, W = 600, 900


def _noise(shape, sigma, seed):
    return np.random.default_rng(seed).normal(0, sigma, shape).astype(np.float32)


def _add_star(img, y, x, flux, fwhm, ratios=(1.0, 1.0, 1.0)):
    s = fwhm / 2.3548
    r = int(4 * s) + 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-((yy + (int(y) - y)) ** 2 + (xx + (int(x) - x)) ** 2) / (2 * s * s))
    g = (flux * g / g.sum()).astype(np.float32)
    for c in range(img.shape[0]):
        img[c, int(y) - r:int(y) + r + 1, int(x) - r:int(x) + r + 1] += g * ratios[c]


def _nebula():
    yy, xx = np.mgrid[0:H, 0:W]
    return (0.004 * np.exp(-((yy - 300) ** 2 + (xx - 450) ** 2) / (2 * 60 ** 2))).astype(np.float32)


def test_background_removes_gradient_keeps_nebula():
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]
    gradient = (0.003 * xx + 0.002 * yy ** 2).astype(np.float32)
    neb = _nebula()
    img = np.stack([0.02 + gradient + neb + _noise((H, W), 2e-4, c) for c in range(3)])
    out = background.correct(img)
    sky = out[:, :, :150].mean(axis=0)            # 左邊沒有星雲
    right = out[:, :, -150:].mean(axis=0)
    assert abs(float(np.median(right)) - float(np.median(sky))) < 0.0004   # 原本差 0.0025
    center = out[:, 280:320, 430:470].mean() - np.median(out[:, :, :150])
    assert center > 0.0030                         # 星雲 0.004 大部分留著


def test_color_balance_recovers_star_ratios():
    rng = np.random.default_rng(3)
    img = np.stack([np.full((H, W), 0.01, np.float32) + _noise((H, W), 1e-4, c) for c in range(3)])
    for _ in range(150):
        _add_star(img, rng.uniform(20, H - 20), rng.uniform(20, W - 20), rng.uniform(0.2, 2.0), 3.0, (0.5, 1.0, 0.8))
    bal = color.measure_balance(img)
    r, b = bal.star_ratios
    assert abs(r - 0.5) < 0.05 and abs(b - 0.8) < 0.05
    fixed = color.apply(img, bal)
    r2, b2, _ = color.star_colors(fixed)
    assert abs(r2 - color.TARGET_RG) < 0.06 and abs(b2 - color.TARGET_BG) < 0.06


def test_denoise_halves_fine_noise_and_keeps_structure():
    neb = _nebula() * 10
    clean = np.stack([0.02 + neb] * 3)
    noisy = clean + np.stack([_noise((H, W), 1e-3, c) for c in range(3)])
    out = denoise.reduce(noisy)
    # 亮度留一半左右（混回原圖）；色彩噪聲全開（chroma_boost），所以每個色版會降得比一半多
    before = np.std((noisy - clean)[:, :100, :100].mean(axis=0))
    after = np.std((out - clean)[:, :100, :100].mean(axis=0))
    assert 0.35 < after / before < 0.75
    assert np.std(out[:, :100, :100] - clean[:, :100, :100]) < 0.75 * np.std(noisy[:, :100, :100] - clean[:, :100, :100])
    peak_before = noisy[:, 290:310, 440:460].mean() - clean[:, :50, :50].mean()
    peak_after = out[:, 290:310, 440:460].mean() - clean[:, :50, :50].mean()
    assert abs(peak_after / peak_before - 1) < 0.02
    assert out.dtype == np.float32 and out.shape == noisy.shape


def test_octave_scale_noise_ignores_faint_large_structure():
    """疊圖的噪聲（相鄰像素相關）＋到處都有的淡大尺度結構（像暗塵）：直接量的大尺度噪聲會把結構算進去，
    "octave" 只量 1、2 px 再往上推，要接近真正的噪聲。"""
    from scipy import ndimage
    k = np.array([0.5, 1.0, 0.5], np.float32)
    noise = ndimage.convolve1d(ndimage.convolve1d(_noise((H, W), 1e-3, 7), k, axis=0), k, axis=1)
    texture = ndimage.gaussian_filter(_noise((H, W), 1.0, 8), 12)
    texture *= np.float32(1.5e-3 / texture.std())
    img = 0.02 + noise + texture
    level = denoise._atrous(denoise._atrous(img, 0), 1)
    coeffs = denoise.starlet(img)[0]
    a0, _, sky = denoise.noise_model(coeffs[0], level)
    truth = [math.sqrt(denoise.sky_noise(w, level, sky)) for w in denoise.starlet(noise + 0.02)[0]]
    measured = denoise.scale_noise(coeffs, level, a0, sky, "measured")
    octave = denoise.scale_noise(coeffs, level, a0, sky, "octave")
    for j in range(2, len(coeffs)):
        assert 0.75 < octave[j] / truth[j] < 1.3, (j, octave[j] / truth[j])
    assert measured[-1] / truth[-1] > 1.5          # 直接量的被淡結構拉高
    assert octave[0] == measured[0]                # 1 px 兩種一樣


def test_neighbour_shrink_keeps_faint_stars():
    """很多暗星（星心約 4 倍噪聲）＋噪聲：看周圍能量的收縮要比逐係數的留下更多暗星光通量，天空一樣乾淨。"""
    from dataclasses import replace
    rng = np.random.default_rng(11)
    clean = np.full((1, H, W), 0.02, np.float32)
    pos = [(rng.uniform(15, H - 15), rng.uniform(15, W - 15)) for _ in range(400)]
    for y, x in pos:
        _add_star(clean, y, x, 0.004 * 2 * math.pi * (2.5 / 2.3548) ** 2, 2.5)
    noisy = clean[0] + _noise((H, W), 1e-3, 12)
    out = {}
    for mode in ("coefficient", "neighbour"):
        out[mode] = denoise.reduce(noisy, replace(denoise.DenoiseSettings(), amount=1.0, shrink=mode))

    def star_flux(img):
        return sum(float(img[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].sum() - 25 * 0.02) for y, x in pos)
    truth = star_flux(clean[0])
    kept = {m: star_flux(o) / truth for m, o in out.items()}
    assert kept["neighbour"] > kept["coefficient"] + 0.05, kept
    empty = np.ones((H, W), bool)
    for y, x in pos:
        empty[max(0, int(y) - 8):int(y) + 9, max(0, int(x) - 8):int(x) + 9] = False
    residual = {m: float(np.std((o - clean[0])[empty])) for m, o in out.items()}
    assert residual["neighbour"] < 0.5e-3, residual        # 原本噪聲 1e-3


def test_chroma_boost_reduces_colour_noise_more():
    from dataclasses import replace
    neb = _nebula() * 10
    clean = np.stack([0.02 + neb] * 3)
    noisy = clean + np.stack([_noise((H, W), 1e-3, c + 20) for c in range(3)])
    out = denoise.reduce(noisy, replace(denoise.DenoiseSettings(), chroma_boost=2.0))
    sky = np.s_[:, :100, :100]
    lum = lambda a: a.mean(axis=0)                     # noqa: E731
    chroma = lambda a: a[0] - a[2]                     # noqa: E731
    l_ratio = np.std(lum(out[sky] - clean[sky])) / np.std(lum(noisy[sky] - clean[sky]))
    c_ratio = np.std(chroma(out[sky] - clean[sky])) / np.std(chroma(noisy[sky] - clean[sky]))
    assert c_ratio < 0.3 and l_ratio > c_ratio + 0.1, (l_ratio, c_ratio)


def test_elongated_stars_become_round():
    """整張星點往同一個方向拉長（像追蹤誤差）：縮星後要變圓，光通量保住；圓形的模糊（round_stars=0）反而更扁。"""
    from dataclasses import replace
    rng = np.random.default_rng(31)
    img = np.full((1, H, W), 0.02, np.float32)
    ang = math.radians(30)
    cov = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) @ np.diag([2.0 ** 2, 1.4 ** 2])
    cov = cov @ np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
    inv = np.linalg.inv(cov)
    yy, xx = np.mgrid[-10:11, -10:11]
    g = np.exp(-0.5 * (inv[0, 0] * yy ** 2 + 2 * inv[0, 1] * yy * xx + inv[1, 1] * xx ** 2)).astype(np.float32)
    g /= g.sum()
    pos = [(int(rng.uniform(20, H - 20)), int(rng.uniform(20, W - 20))) for _ in range(250)]
    for y, x in pos:
        img[0, y - 10:y + 11, x - 10:x + 11] += g * rng.uniform(0.05, 0.3)
    img[0] += _noise((H, W), 2e-4, 32)

    def ellipticity(a):
        s = detail.measure_shape(a, detail.find_stars(a))
        ev = np.linalg.eigvalsh(s)
        return 1 - math.sqrt(ev[0] / ev[1])

    e0 = ellipticity(img[0])
    new = detail.process(img[0], detail.DetailSettings())
    old = detail.process(img[0], replace(detail.DetailSettings(), round_stars=0.0))
    assert e0 > 0.2
    assert ellipticity(new) < 0.5 * e0, (e0, ellipticity(new))
    assert ellipticity(old) > e0 - 0.02, (e0, ellipticity(old))
    flux = lambda a: sum(float(a[y - 8:y + 9, x - 8:x + 9].sum() - 289 * 0.02) for y, x in pos)  # noqa: E731
    assert abs(flux(new) / flux(img[0]) - 1) < 0.05


def _star_field(seed: int, n: int = 200):
    rng = np.random.default_rng(seed)
    img = np.stack([np.full((H, W), 0.02, np.float32) for _ in range(3)])
    pos = [(rng.uniform(20, H - 20), rng.uniform(20, W - 20)) for _ in range(n)]
    for y, x in pos:
        _add_star(img, y, x, rng.uniform(0.02, 0.3), 3.0, (1.0, 0.9, 0.7))
    clean = img.copy()
    return clean, clean + np.stack([_noise((H, W), 5e-4, seed + c) for c in range(3)]), pos


def test_layers_add_back_to_original():
    _, noisy, _ = _star_field(41)
    layers = detail.split(noisy)
    assert np.allclose(layers.background + layers.stars_layer, noisy, atol=1e-6)
    assert len(layers.stars.y) > 100 and layers.stars.noise > 0


def test_clean_layer_removes_edge_noise_keeps_stars():
    clean, noisy, pos = _star_field(42)
    layers = detail.split(noisy)
    S = layers.stars_layer.mean(axis=0)
    C = detail.clean_layer(layers.stars_layer, layers.stars.noise).mean(axis=0)
    truth = (clean - detail.split(clean).background).mean(axis=0)    # 沒有噪聲時的星點層（近似）
    edge = layers.stars.mask & (layers.stars.weight < 0.5)
    assert np.std((C - truth)[edge]) < 0.6 * np.std((S - truth)[edge])
    peaks = [(int(round(y)), int(round(x))) for y, x in pos]
    kept = sum(C[y, x] for y, x in peaks) / sum(S[y, x] for y, x in peaks)
    assert kept > 0.95


def test_tighten_keeps_peak_and_monotonic_profile():
    img = np.full((1, 81, 81), 0.0, np.float32)
    _add_star(img, 40, 40, 1.0, 5.0)
    out = detail.tighten_layer(img, detail.DetailSettings(tighten=1.0))[0]
    prof_in, prof_out = img[0, 40, 40:], out[40, 40:]
    assert abs(prof_out[0] - prof_in[0]) < 1e-6                    # 峰值不變
    assert np.all(np.diff(prof_out[:12]) <= 1e-9)                   # 單調
    core = prof_in >= prof_in[0] * detail.TIGHTEN_LEVEL
    assert np.allclose(prof_out[core], prof_in[core])               # 峰值一半以上不動
    wing = ~core & (prof_in > prof_in[0] * 1e-3)
    assert np.all(prof_out[wing] < 0.9 * prof_in[wing])              # 星暈壓低


def test_star_reduction_shrinks_keeps_flux_no_dark_ring():
    base = 0.02 + _nebula()
    img = np.stack([base + _noise((H, W), 1e-5, c) for c in range(3)])
    stars = [(100.5, 100.5), (100.5, 400.5), (400.5, 700.5), (500.5, 200.5), (250.5, 250.5), (200.5, 600.5)]
    for y, x in stars:
        _add_star(img, y, x, 0.5, 4.0)
    out = detail.process(img, detail.DetailSettings(star_reduction=0.5, sharpen=0.0))
    for y, x in stars:
        yi, xi = int(y), int(x)
        box_in = img[:, yi - 12:yi + 13, xi - 12:xi + 13].mean(0) - base[yi - 12:yi + 13, xi - 12:xi + 13]
        box_out = out[:, yi - 12:yi + 13, xi - 12:xi + 13].mean(0) - base[yi - 12:yi + 13, xi - 12:xi + 13]
        assert abs(box_out.sum() / box_in.sum() - 1) < 0.1                      # 光通量保住
        yy, xx = np.indices(box_in.shape)
        def fwhm(b):
            f = np.clip(b, 0, None)
            t = f.sum()
            cy, cx = (f * yy).sum() / t, (f * xx).sum() / t
            return 2.3548 * math.sqrt((f * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / t / 2)
        assert fwhm(box_out) < 0.9 * fwhm(box_in)                                # 變小
        assert box_out.min() > -5e-4                                              # 周圍沒有比背景暗的環


def test_pipeline_runs_and_settings_round_trip():
    img = np.stack([0.02 + _nebula() + _noise((H, W), 2e-4, c) for c in range(3)])
    for y, x in ((100.5, 100.5), (400.5, 700.5)):
        _add_star(img, y, x, 0.3, 3.0)
    result = pipeline.process(img)
    assert result.display.shape == img.shape
    assert 0 <= result.display.min() and result.display.max() <= 1
    s = pipeline.ProcessingSettings.recommended()
    assert pipeline.ProcessingSettings.from_dict(s.to_dict()) == s
    old = s.to_dict()
    del old["detail"]["sharpen"]                   # 舊設定檔缺欄位
    old["unknown"] = 1                             # 不認得的欄位
    assert pipeline.ProcessingSettings.from_dict(old).detail.sharpen == s.detail.sharpen


def test_processor_reuses_earlier_stages():
    from dataclasses import replace
    img = np.stack([0.02 + _nebula() + _noise((H, W), 2e-4, c) for c in range(3)])
    _add_star(img, 200.5, 300.5, 0.3, 3.0)
    p = pipeline.Processor()
    p.set_source(img)
    s = pipeline.ProcessingSettings.recommended()
    first = p.run(s)
    assert first.computed == pipeline.STAGES
    again = p.run(replace(s, stretch=replace(s.stretch, strength=0.7)))
    assert again.computed == ("stretch",)
    stars = p.run(replace(s, detail=replace(s.detail, star_reduction=0.2)))
    assert stars.computed == ("detail", "stretch")
    same = p.run(s)
    assert set(same.computed) == {"detail", "stretch"}     # 只存最新一份：切回來要重算後兩段
    assert np.array_equal(same.display, first.display)
    assert pipeline.preview_factor((3, 4000, 6000)) == 4 and pipeline.preview_factor((3, 1000, 1500)) == 1


def _add_winged_star(img, y, x, flux, fwhm):
    """小望遠鏡常見的星點：高斯核心＋寬星暈（三成光通量在 4 倍寬的星暈裡）。"""
    _add_star(img, y, x, 0.7 * flux, fwhm)
    _add_star(img, y, x, 0.3 * flux, 4 * fwhm)


def test_deconvolution_psf_stays_narrower_than_star_cores():
    """小核心＋寬星暈、很多暗星：二階矩被星暈與噪聲拉大，縮星的 PSF 不能因此比星點核心還寬（會挖出暗環）；
    一般乾淨的星點不受上限影響。"""
    rng = np.random.default_rng(5)
    img = np.stack([np.full((H, W), 0.02, np.float32) + _noise((H, W), 1e-4, c) for c in range(3)])
    for _ in range(150):
        _add_winged_star(img, rng.uniform(30, H - 30), rng.uniform(30, W - 30), rng.uniform(0.01, 0.06), 2.7)
    L = img.mean(0)
    stars = detail.find_stars(L)
    core = detail.measure_core_fwhm(L, stars)
    assert abs(core / 2.7 - 1) < 0.15
    assert detail.measure_fwhm(L, stars) > 1.25 * core
    assert detail.deconvolution_fwhm(L, stars, 0.8) <= detail.PSF_CORE_LIMIT * core + 1e-9

    clean = np.stack([np.full((H, W), 0.02, np.float32) + _noise((H, W), 1e-5, c) for c in range(3)])
    for _ in range(80):
        _add_star(clean, rng.uniform(30, H - 30), rng.uniform(30, W - 30), rng.uniform(0.2, 1.0), 4.0)
    L = clean.mean(0)
    stars = detail.find_stars(L)
    assert math.isclose(detail.deconvolution_fwhm(L, stars, 0.8), 0.6 * detail.measure_fwhm(L, stars), rel_tol=1e-9)


def test_star_with_halo_stays_monotonic():
    """亮星＋寬星暈縮星後，徑向剖面要從中心往外單調下降（不能有暗環或硬邊暗框）。"""
    base = 0.02
    img = np.stack([np.full((H, W), base, np.float32) + _noise((H, W), 1e-5, c) for c in range(3)])
    _add_star(img, 300.5, 450.5, 2.0, 4.0)       # 星心
    _add_star(img, 300.5, 450.5, 1.0, 16.0)      # 寬星暈
    _add_star(img, 300.5, 470.5, 0.4, 4.0)       # 旁邊一顆
    out = detail.process(img, detail.DetailSettings(star_reduction=0.5, sharpen=0.0))
    L = out.mean(0)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(yy - 300.5, xx - 450.5)
    left = np.hypot(yy - 300.5, xx - 470.5) > 12   # 避開旁邊那顆
    prof = [float(np.median(L[(rr >= r) & (rr < r + 1) & left])) for r in range(2, 40)]
    assert all(b <= a + 2e-5 for a, b in zip(prof, prof[1:])), np.round(np.array(prof) - base, 5)
    assert L.min() > base - 1e-4
