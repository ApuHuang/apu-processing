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
    before = np.std(noisy[:, :100, :100] - clean[:, :100, :100])
    after = np.std(out[:, :100, :100] - clean[:, :100, :100])
    assert 0.35 < after / before < 0.75
    peak_before = noisy[:, 290:310, 440:460].mean() - clean[:, :50, :50].mean()
    peak_after = out[:, 290:310, 440:460].mean() - clean[:, :50, :50].mean()
    assert abs(peak_after / peak_before - 1) < 0.02
    assert out.dtype == np.float32 and out.shape == noisy.shape


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
