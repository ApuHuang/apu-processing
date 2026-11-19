"""給 APU Astro Studio 的公開介面（api.py），以及 0.2.4 為了它修的兩件事。"""

import numpy as np

from apu_processing import api, detail, stretch


def _field(h=600, w=900, seed=0):
    rng = np.random.default_rng(seed)
    img = np.stack([np.full((h, w), s, np.float32) for s in (0.10, 0.12, 0.14)])
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return img


def test_api_names_exist():
    for name in api.__all__:
        assert getattr(api, name) is not None, name


def test_measure_odd_tile():
    """縮圖倍率 3（區塊 21 px）、2.2（29 px）：以前 _faint_signal 的 reshape 丟 ValueError。"""
    img = _field()
    for scale in (3.0, 2.2, 64 / 21):
        n = stretch.measure(img, scale)
        assert np.isfinite(n.sigma) and n.sigma > 0
        assert len(n.faint_color) == 3


def test_measure_even_tile_unchanged():
    """偶數區塊（Processing 自己用的倍率 1、2、4、8）抽到的像素跟以前完全一樣。"""
    img = _field()
    planes, tile, step = img, 32, 2
    th, tw = img.shape[1] // tile, img.shape[2] // tile
    for p in planes:
        old = p[:th * tile:step, :tw * tile:step].reshape(th, tile // step, tw, tile // step)
        new = p[:th * tile, :tw * tile].reshape(th, tile, tw, tile)[:, ::step, :, ::step]
        assert np.array_equal(old, new)


def test_fine_noise_matches_find_stars():
    img = _field()
    L = img.mean(axis=0)
    rng = np.random.default_rng(3)
    for y, x in rng.integers(20, 580, size=(20, 2)):
        L[y - 2:y + 3, x - 2:x + 3] += 0.5
    assert detail.fine_noise(L) == detail.find_stars(L).noise
