import itertools

import numpy as np

from apu_processing import geometry
from apu_processing.geometry import Geometry


def test_flips_and_rotation():
    img = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    assert geometry.rotate(img, 1).shape == (2, 4, 3)
    assert np.array_equal(geometry.rotate(geometry.rotate(img, 1), 3), img)
    assert np.array_equal(geometry.flip_horizontal(geometry.flip_horizontal(img)), img)
    assert np.array_equal(geometry.flip_vertical(img)[:, 0], img[:, -1])


def test_displayed_box_maps_back_to_source():
    rng = np.random.default_rng(1)
    src = rng.random((3, 40, 60)).astype(np.float32)
    for crop, turns, fh, fv in itertools.product((None, (5, 3, 50, 37)), range(4), (False, True), (False, True)):
        g = Geometry(crop, turns, fh, fv)
        shown = geometry.apply(src, g)
        h, w = shown.shape[1:]
        box = (2, 3, w - 4, h - 5)
        expected = shown[:, box[1]:box[3], box[0]:box[2]]
        back = geometry.displayed_box_to_source(box, g, (60, 40))
        again = geometry.apply(src, Geometry(back, turns, fh, fv))
        assert np.array_equal(again, expected), (crop, turns, fh, fv)
