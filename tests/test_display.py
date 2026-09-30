import numpy as np

from apu_processing import display
from apu_processing.display import DisplayAdjustments, ToneCurves


def _img(seed=0):
    return np.random.default_rng(seed).random((3, 40, 60), dtype=np.float32)


def test_neutral_is_identity():
    img = _img()
    assert display.apply(img, DisplayAdjustments()) is img


def test_curve_passes_through_points_and_is_monotonic():
    pts = [[0, 0], [0.3, 0.5], [0.7, 0.8], [1, 1]]
    x = np.linspace(0, 1, 1001)
    y = display.curve_values(pts, x)
    assert np.all(np.diff(y) >= -1e-12)
    for px, py in pts:
        assert abs(display.curve_values(pts, np.array([px]))[0] - py) < 1e-9


def test_exposure_contrast_saturation_directions():
    img = _img()
    brighter = display.apply(img, DisplayAdjustments(exposure=0.5))
    assert brighter.mean() > img.mean()
    contrast = display.apply(img, DisplayAdjustments(contrast=1.0))
    assert contrast.std() > img.std()
    gray = display.apply(img, DisplayAdjustments(saturation=0.0))
    assert np.allclose(gray[0], gray[1], atol=1e-6) and np.allclose(gray[1], gray[2], atol=1e-6)


def test_green_removal_only_touches_green_excess():
    img = np.zeros((3, 2, 2), np.float32)
    img[:, 0, 0] = (0.2, 0.6, 0.2)   # 偏綠
    img[:, 0, 1] = (0.6, 0.2, 0.4)   # 紅紫，綠不多
    out = display.apply(img, DisplayAdjustments(green_removal=1.0))
    assert abs(out[1, 0, 0] - 0.2) < 1e-3
    assert np.allclose(out[:, 0, 1], img[:, 0, 1], atol=1e-3)


def test_channel_curve_changes_only_that_channel():
    img = _img()
    curves = ToneCurves(blue=[[0, 0], [0.5, 0.3], [1, 1]])
    out = display.apply(img, DisplayAdjustments(curves=curves))
    assert np.allclose(out[0], img[0], atol=1 / 4096) and np.allclose(out[1], img[1], atol=1 / 4096)
    assert out[2].mean() < img[2].mean()


def test_round_trip_dict():
    adj = DisplayAdjustments(exposure=0.2, curves=ToneCurves(master=[[0, 0.05], [1, 1]]))
    back = DisplayAdjustments.from_dict(adj.to_dict())
    assert back == adj
    assert DisplayAdjustments.from_dict({"exposure": 0.1}).saturation == 1.0
