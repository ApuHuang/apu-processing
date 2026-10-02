"""多濾鏡合成：濾鏡對應、噪聲對齊、強度、尺寸檢查。"""

import numpy as np
import pytest

from apu_processing import compose, pipeline, stretch

from .test_engine import H, W, _nebula, _noise


def _plane(sky, sigma, signal, seed):
    return (sky + signal * _nebula() / 0.004 + _noise((H, W), sigma, seed)).astype(np.float32)


def test_recommended_follows_filter_names():
    s = compose.ComposeSettings.recommended(["OIII", " ha ", "SII"])
    assert s.preset == "SHO"
    assert [(c.role, c.source) for c in s.channels] == [("SII", 2), ("Ha", 1), ("OIII", 0)]
    s = compose.ComposeSettings.recommended(["oiii", "Ha"])
    assert s.preset == "HOO" and [(c.role, c.source) for c in s.channels] == [("Ha", 1), ("OIII", 0)]
    s = compose.ComposeSettings.recommended(["B", "R", "G"])
    assert s.preset == "RGB" and [c.source for c in s.channels] == [1, 2, 0]
    assert compose.ComposeSettings.recommended(["R", "G", "B"]).narrowband is False


def test_unmatched_names_are_assigned_in_order():
    # 拼法不同（H-alpha）不自動當成 Ha：退回 HOO、照順序指定，讓使用者自己改
    s = compose.ComposeSettings.recommended(["H-alpha", "O3"])
    assert s.preset == "HOO" and [c.source for c in s.channels] == [0, 1]
    # 兩張開成 SHO：缺的那個角色沒有來源
    s = compose.ComposeSettings.for_preset("SHO", ["Ha", "OIII"])
    assert [c.source for c in s.channels] == [-1, 0, 1]


def test_noise_alignment_and_strength():
    # Ha 亮又乾淨、OIII 暗又吵，單位還不一樣（讀檔時各自除以自己的峰值）
    ha = _plane(0.012, 0.0009, 0.010, 1)
    oiii = _plane(0.020, 0.0016, 0.002, 2) * 1.09
    c = compose.Composer([ha, oiii], ["ha.fits", "oiii.fits"])
    rgb = c.compose(compose.ComposeSettings.recommended(["Ha", "OIII"]))
    assert rgb.shape == (3, H, W) and rgb.dtype == np.float32
    assert np.array_equal(rgb[1], rgb[2])                       # OIII → G、B
    n = [stretch.measure(p) for p in rgb]
    # 天空對齊到同一個位置、噪聲一樣大
    assert abs(n[0].sky[0] - n[1].sky[0]) < 0.2 * n[0].sigma
    assert n[1].sigma == pytest.approx(n[0].sigma, rel=0.08)
    # OIII 強度 ×2：天空以上的部分剛好兩倍
    s = compose.ComposeSettings.recommended(["Ha", "OIII"]).with_strength(1, 2.0)
    doubled = c.compose(s)
    sky = n[1].sky[0]
    np.testing.assert_allclose(doubled[1] - sky, 2 * (rgb[1] - sky), atol=1e-6)
    np.testing.assert_array_equal(doubled[0], rgb[0])


def test_input_units_do_not_matter():
    ha, oiii = _plane(0.012, 0.0009, 0.010, 3), _plane(0.020, 0.0016, 0.002, 4)
    s = compose.ComposeSettings.recommended(["Ha", "OIII"])
    a = compose.Composer([ha, oiii], ["a", "b"]).compose(s)
    b = compose.Composer([ha, oiii * 0.8], ["a", "b"]).compose(s)
    # 只差共同的天空底（三色同一個常數，後面拉伸會扣掉）
    np.testing.assert_allclose(a - a.mean(), b - b.mean(), atol=2e-5)


def test_unused_channel_and_rejected_inputs():
    ha, oiii = _plane(0.012, 0.0009, 0.010, 5), _plane(0.020, 0.0016, 0.002, 6)
    c = compose.Composer([ha, oiii], ["a", "b"])
    rgb = c.compose(compose.ComposeSettings.for_preset("SHO", ["Ha", "OIII"]))   # 沒有 SII
    assert float(np.ptp(rgb[0])) == 0.0                                          # 紅色只剩天空
    with pytest.raises(ValueError, match="尺寸不同"):
        compose.Composer([ha, oiii[:-10]], ["a.fits", "b.fits"])
    with pytest.raises(ValueError, match="彩色"):
        compose.Composer([ha, np.stack([oiii] * 3)], ["a.fits", "rgb.fits"])
    # 疊圖失敗輸出整張 0 的 master：合成前就擋下來，不要到去光才當掉
    with pytest.raises(ValueError, match="b.fits 整張都是同一個值"):
        compose.Composer([ha, np.zeros_like(oiii)], ["a.fits", "b.fits"])


def test_composite_runs_through_pipeline():
    ha, oiii = _plane(0.012, 0.0009, 0.010, 7), _plane(0.020, 0.0016, 0.002, 8)
    rgb = compose.Composer([ha, oiii], ["a", "b"]).compose(compose.ComposeSettings.recommended(["Ha", "OIII"]))
    from dataclasses import replace
    r = pipeline.process(rgb, replace(pipeline.ProcessingSettings.recommended(), color_enabled=False))
    assert r.display.shape == (3, H, W) and np.isfinite(r.display).all()
