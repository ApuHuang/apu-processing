import numpy as np

from apu_processing import stretch


def _field(h=512, w=768, sky=(0.02, 0.03, 0.04), sigma=0.001, seed=1):
    """帶真正隨機噪聲的合成天空（亂數要混合充分：相鄰差值固定的「噪聲」會被量成沒有噪聲）。"""
    rng = np.random.default_rng(seed)
    img = np.stack([np.full((h, w), s, np.float32) for s in sky])
    img += rng.normal(0, sigma, img.shape).astype(np.float32)
    return img


def test_white_noise_measures_same_at_8px():
    img = _field()
    n = stretch.measure(img)
    # 亮度是三色平均，白噪聲 σ/√3
    assert abs(n.sigma / (0.001 / np.sqrt(3)) - 1) < 0.1
    assert np.allclose(n.sky, (0.02, 0.03, 0.04), atol=2e-4)


def test_sky_lands_on_background_and_is_neutral():
    img = _field()
    out = stretch.apply(img, stretch.StretchSettings(background=0.13))
    med = np.median(out.reshape(3, -1), axis=1)
    assert np.all(np.abs(med - 0.13) < 0.01)


def test_correlated_noise_stretches_less():
    """同樣的像素噪聲，但噪聲是大塊的（降噪後的斑駁）：8 px 尺度的噪聲比較大，拉得比較淺。"""
    rng = np.random.default_rng(2)
    h, w = 512, 768
    white = rng.normal(0, 0.001, (h, w)).astype(np.float32)
    blotchy = np.kron(rng.normal(0, 0.001, (h // 8, w // 8)), np.ones((8, 8))).astype(np.float32)
    blotchy += rng.normal(0, 0.0003, (h, w)).astype(np.float32)
    signal = np.zeros((h, w), np.float32)
    signal[200:300, 300:500] = 0.01
    a = stretch.apply(np.stack([0.02 + white + signal] * 3))
    b = stretch.apply(np.stack([0.02 + blotchy + signal] * 3))
    assert a[0, 250, 400] > b[0, 250, 400] + 0.05


def test_strength_and_monotonic_curve():
    u = np.linspace(-20, 5000, 2000).astype(np.float32)
    y = stretch.curve(u, 0.13)
    assert np.all(np.diff(y) >= 0) and y.min() >= 0 and y.max() <= 1
    img = _field()
    img[:, 100:150, 100:150] += 0.005
    soft = stretch.apply(img, stretch.StretchSettings(strength=0.25))
    hard = stretch.apply(img, stretch.StretchSettings(strength=0.75))
    assert hard[0, 120, 120] > soft[0, 120, 120]


def test_mono_image():
    img = _field()[0]
    out = stretch.apply(img)
    assert out.shape == img.shape


def test_background_color_follows_faint_nebula():
    """底色偏綠（光害）＋一半區域有微弱的紅色星雲：0% 天空中性；100% 天空偏紅、亮度不變；沒有星雲時幾乎不變。"""
    rng = np.random.default_rng(7)
    h, w = 768, 1024
    pedestal = (0.02, 0.04, 0.03)
    emission = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    emission += 0.0006 * (0.5 + 0.5 * np.sin(xx / 90.0) * np.cos(yy / 70.0))   # 整片起伏的微弱 Hα
    img = np.stack([np.full((h, w), p, np.float32) for p in pedestal])
    img[0] += emission
    img[1] += 0.1 * emission
    img[2] += 0.1 * emission
    img += rng.normal(0, 3e-4, img.shape).astype(np.float32)
    neutral = stretch.apply(img, stretch.StretchSettings(background_color=0.0))
    tinted = stretch.apply(img, stretch.StretchSettings(background_color=1.0))
    darkest = emission < np.quantile(emission, 0.2)                       # 星雲最少的地方＝天空
    sky = lambda d: np.array([np.median(c[darkest]) for c in d])  # noqa: E731
    n, t = sky(neutral), sky(tinted)
    assert np.ptp(n) < 0.02
    assert t[0] - t[1] > 0.03 and abs(t.mean() - n.mean()) < 0.01
    clean = img.copy()
    clean[0] -= emission
    clean[1] -= 0.1 * emission
    clean[2] -= 0.1 * emission
    c = sky(stretch.apply(clean, stretch.StretchSettings(background_color=1.0)))
    assert np.ptp(c) < 0.02
