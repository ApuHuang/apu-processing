"""疊圖覆蓋率圖與疊圖紀錄（*.recipe.json）。"""

import json

import numpy as np
import pytest
from astropy.io import fits

from apu_processing import background, coverage, imageio, recipe

from .test_engine import H, W, _noise


def _coverage_map(top: int = 20, edge: int = 60) -> np.ndarray:
    """中間每張都蓋到，左邊與下面一條只有幾張（沒裁切的疊圖邊緣）。"""
    c = np.full((H, W), top, np.uint16)
    c[:, :edge] = 3
    c[H - edge // 2:, :] = 5
    return c


def _write_coverage(path, data: np.ndarray, row_order: str = "TOP-DOWN") -> None:
    hdu = fits.PrimaryHDU(data)
    hdu.header["ROWORDER"] = row_order
    hdu.writeto(path, overwrite=True)


def test_find_and_load_coverage(tmp_path):
    master = tmp_path / "m_Ha.fits"
    imageio.save_fits(master, np.zeros((H, W), np.float32))
    assert coverage.find_for(master) is None
    _write_coverage(tmp_path / "m_Ha_coverage.fits", _coverage_map())
    found = coverage.find_for(master)
    assert found == tmp_path / "m_Ha_coverage.fits"
    frac = coverage.load_fraction(found)
    assert frac.shape == (H, W) and frac.max() == 1.0 and frac[0, 0] == pytest.approx(3 / 20)
    # 方向跟 master 一樣照 ROWORDER：沒寫的要上下翻
    _write_coverage(tmp_path / "b.fits", _coverage_map()[::-1].copy(), row_order="BOTTOM-UP")
    np.testing.assert_array_equal(coverage.load_fraction(tmp_path / "b.fits"), frac)


def test_combine_takes_the_lowest_and_needs_every_map():
    a, b = np.full((H, W), 1.0, np.float32), np.full((H, W), 1.0, np.float32)
    b[:10] = 0.5
    assert float(coverage.combine([a, b], (H, W))[:10].max()) == 0.5
    assert coverage.combine([a, None], (H, W)) is None              # 少一張就不用
    assert coverage.combine([a, b[:-1]], (H, W)) is None            # 尺寸不對就不用


def test_suggest_crop_finds_the_well_covered_rectangle():
    frac = _coverage_map().astype(np.float32) / 20
    box = coverage.suggest_crop(frac)
    x0, y0, x1, y1 = box
    assert coverage.good_mask(frac)[y0:y1, x0:x1].all()
    assert 60 <= x0 <= 68 and y0 == 0 and x1 == W and H - 38 <= y1 <= H - 30
    assert coverage.suggest_crop(np.ones((H, W), np.float32)) is None          # 整張都夠：不用裁


def test_background_ignores_poorly_covered_edges():
    xx = np.mgrid[0:H, 0:W][1] / W
    noise = _noise((H, W), 2e-4, 1)
    sky = (0.02 + 0.004 * xx).astype(np.float32) + noise
    mask = np.ones((H, W), bool)
    mask[:, :96] = False
    ramp = np.clip((96 - np.arange(W)) / 96, 0, 1).astype(np.float32)[None, :]
    inner = (slice(50, H - 50), slice(150, W - 50))

    def leftover(corrected):     # 好的範圍裡，扣完曲面後剩下的不平（噪聲以外）
        d = corrected[inner] - noise[inner]
        return float(np.std(d))

    for sign in (-1, 1):
        # 覆蓋不足的邊緣往外逐漸變暗或變亮：曲面被它帶歪，連中間都扣錯
        bad = sky + sign * 0.001 * ramp
        plain = leftover(background.correct(bad))
        masked = leftover(background.correct(bad, sample_mask=mask))
        assert masked < 1e-5 and masked < 0.1 * plain, (sign, plain, masked)
    # 沒有可取樣的格子（整張 0）不會當掉
    assert np.array_equal(background.correct(np.zeros((H, W), np.float32)), np.zeros((H, W), np.float32))


def _recipe(tmp_path, outputs):
    path = tmp_path / "NGC1.recipe.json"
    path.write_text(json.dumps({"schema": "x/1", "target": "NGC1", "outputs": outputs}), encoding="utf-8")
    return path


def test_recipe_lists_existing_masters_by_group(tmp_path):
    for name in ("NGC1_Ha.fits", "NGC1_OIII.fits", "NGC1_Ha_coverage.fits", "NGC1_L.fits"):
        (tmp_path / name).write_bytes(b"x")
    path = _recipe(tmp_path, [
        {"align_group": "A", "filter": "Ha", "frames": 20, "file": "NGC1_Ha.fits", "coverage": "NGC1_Ha_coverage.fits"},
        {"align_group": "A", "filter": "OIII", "frames": 12, "file": "NGC1_OIII.fits", "coverage": "missing.fits"},
        {"align_group": "B", "filter": "L", "frames": 30, "file": "NGC1_L.fits"},
        {"align_group": "A", "filter": "SII", "file": "not_there.fits"},
        "garbage",
    ])
    assert recipe.is_recipe(path) and not recipe.is_recipe(tmp_path / "NGC1_Ha.fits")
    project = recipe.load(path)
    assert project.target == "NGC1" and len(project.outputs) == 3
    group, outputs = project.main_group()
    assert group == "A" and [o.filter for o in outputs] == ["Ha", "OIII"]
    assert outputs[0].coverage == tmp_path / "NGC1_Ha_coverage.fits" and outputs[1].coverage is None


def test_unusable_recipes_are_reported(tmp_path):
    with pytest.raises(ValueError, match="找不到"):
        recipe.load(_recipe(tmp_path, [{"file": "gone.fits"}]))
    bad = tmp_path / "x.recipe.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="看得懂"):
        recipe.load(bad)
