import numpy as np
import pytest
import tifffile
from astropy.io import fits
from PIL import Image

from apu_processing import imageio


def _rgb(h=24, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((3, h, w), dtype=np.float32)


def test_fits_round_trip_keeps_orientation_and_header(tmp_path):
    data = _rgb()
    header = fits.Header()
    header["OBJECT"] = "M33"
    path = tmp_path / "a.fits"
    imageio.save_fits(path, data, header)
    assert fits.getheader(path)["ROWORDER"] == "TOP-DOWN"
    loaded, hdr = imageio.load_image(path)
    assert loaded.dtype == np.float32 and loaded.shape == (3, 24, 32)
    np.testing.assert_array_equal(loaded, data)
    assert hdr["OBJECT"] == "M33"


def test_fits_without_roworder_is_bottom_up(tmp_path):
    data = _rgb()
    path = tmp_path / "b.fit"
    fits.PrimaryHDU(data).writeto(path)
    loaded, _ = imageio.load_image(path)
    np.testing.assert_array_equal(loaded, data[:, ::-1, :])


def test_mono_fits(tmp_path):
    data = _rgb()[0]
    path = tmp_path / "m.fts"
    imageio.save_fits(path, data)
    loaded, _ = imageio.load_image(path)
    assert loaded.shape == (24, 32)
    np.testing.assert_array_equal(loaded, data)


def test_uint16_fits_is_normalized(tmp_path):
    raw = np.array([[0, 32768], [65535, 1000]], dtype=np.uint16)
    path = tmp_path / "u.fits"
    hdu = fits.PrimaryHDU(raw)
    hdu.header["ROWORDER"] = "TOP-DOWN"
    hdu.writeto(path)
    loaded, _ = imageio.load_image(path)
    np.testing.assert_allclose(loaded, raw / 65535.0, rtol=1e-6)


def test_float_0_65535_stack_is_scaled_and_nan_cleared(tmp_path):
    data = _rgb() * 60000
    data[1, 3, 4] = np.nan
    path = tmp_path / "f.fits"
    hdu = fits.PrimaryHDU(data)
    hdu.header["ROWORDER"] = "TOP-DOWN"
    hdu.writeto(path)
    loaded, hdr = imageio.load_image(path)
    assert hdr["APUSCALE"] == 65535.0
    assert loaded[1, 3, 4] == 0.0
    expected = np.nan_to_num(data) / 65535.0
    np.testing.assert_allclose(loaded, expected, rtol=1e-6)


@pytest.mark.parametrize("planar", [False, True])
def test_tiff_16bit_and_float(tmp_path, planar):
    data = _rgb()
    u16 = (data * 65535 + 0.5).astype(np.uint16)
    path = tmp_path / "t.tif"
    tifffile.imwrite(path, u16 if planar else np.moveaxis(u16, 0, -1),
                     photometric="rgb", planarconfig="separate" if planar else "contig")
    loaded, _ = imageio.load_image(path)
    assert loaded.shape == (3, 24, 32)
    np.testing.assert_allclose(loaded, u16 / 65535.0, rtol=1e-6)

    path = tmp_path / "f.tiff"
    tifffile.imwrite(path, data if planar else np.moveaxis(data, 0, -1),
                     photometric="rgb", planarconfig="separate" if planar else "contig")
    loaded, _ = imageio.load_image(path)
    np.testing.assert_array_equal(loaded, data)


def test_png_rgba_drops_alpha(tmp_path):
    rgba = np.zeros((10, 12, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 128
    path = tmp_path / "p.png"
    Image.fromarray(rgba, "RGBA").save(path)
    loaded, hdr = imageio.load_image(path)
    assert loaded.shape == (3, 10, 12)
    assert loaded[0].min() == 1.0 and loaded[1].max() == 0.0
    assert hdr["APUSRC"] == "PNG"


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".tif"])
def test_save_display_round_trip(tmp_path, suffix):
    display = _rgb(40, 50)
    path = tmp_path / f"out{suffix}"
    imageio.save_display(path, display)
    loaded, _ = imageio.load_image(path)
    assert loaded.shape == display.shape
    tolerance = {".png": 0.5 / 255 + 1e-6, ".tif": 0.5 / 65535 + 1e-6, ".jpg": 0.2}[suffix]
    assert np.abs(loaded - display).max() <= tolerance


def test_list_images_skips_appledouble(tmp_path):
    for name in ("a.fit", "._a.fit", "b.PNG", "c.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in imageio.list_images(tmp_path)] == ["a.fit", "b.PNG"]


def test_unsupported_format(tmp_path):
    with pytest.raises(ValueError):
        imageio.load_image(tmp_path / "x.xisf")
