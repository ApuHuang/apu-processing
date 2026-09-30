"""視窗流程測試：不點滑鼠，直接呼叫方法。

Tk 在同一個行程只建一次（反覆建立、關閉 Tk 在 Windows 偶爾失敗、Mac 雲端機會卡死）。
"""

import time
import tkinter as tk

import numpy as np
import pytest

from apu_processing import gui, imageio
from apu_processing.i18n import set_language, tr

from .test_engine import H, W, _add_star, _nebula, _noise


@pytest.fixture(scope="session")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("沒有顯示環境")
    root.geometry("1200x800")
    yield root
    root.destroy()


@pytest.fixture
def view(tk_root, tmp_path, monkeypatch):
    monkeypatch.setenv("APU_PROCESSING_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: pytest.fail(f"error dialog: {a}"))
    set_language("zh")
    v = gui.ProcessingView(tk_root, tk_root)
    v.pack(fill="both", expand=True)
    tk_root.update()
    yield v
    v.close()
    v.destroy()
    set_language("zh")


def _fits(tmp_path):
    img = np.stack([0.02 + _nebula() + _noise((H, W), 2e-4, c) for c in range(3)])
    rng = np.random.default_rng(5)
    for _ in range(40):
        _add_star(img, rng.uniform(20, H - 20), rng.uniform(20, W - 20), rng.uniform(0.05, 0.5), 3.0)
    path = tmp_path / "m42.fits"
    imageio.save_fits(path, img)
    return path


def _wait(root, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return
        time.sleep(0.02)
    pytest.fail("timeout")


def test_open_process_compare_and_save(view, tk_root, tmp_path):
    view.open_image(_fits(tmp_path))
    _wait(tk_root, lambda: view.current is not None and view.current_is_full and view.job is None)
    assert view.canvas.full is not None and view.canvas.full.shape == (3, H, W)
    assert "m42.fits" in view.doc_title.cget("text")
    view.view_var.set("original")
    tk_root.update()
    assert view.canvas.full is view.original_display
    view.view_var.set("current")
    # 改拉伸：只重算拉伸
    view.v["stretch"].set(70)
    _wait(tk_root, lambda: view.job is None and view.current.computed == ("stretch",) and view._debounce is None)
    assert view.previous_display is not None
    # 成品微調不重跑處理
    gen = view.generation
    view.d["saturation"].set(150)
    tk_root.update()
    assert view.generation == gen and view.canvas.adjust.saturation == 1.5
    out = tmp_path / "out.png"
    view.save(out)
    _wait(tk_root, lambda: view.status.cget("text") == tr("gui.status.saved", name="out.png"))
    assert not list(tmp_path.glob(".*saving*"))
    loaded, _ = imageio.load_image(out)
    assert loaded.shape == (3, H, W)


def test_language_switch_keeps_image(view, tk_root, tmp_path):
    view.open_image(_fits(tmp_path))
    _wait(tk_root, lambda: view.current is not None and view.current_is_full and view.job is None)
    view.lang_var.set("en")
    _wait(tk_root, lambda: view.open_btn.cget("text") == "Open", 5)
    assert view.canvas.full is not None
    assert view.auto_btn.cget("text") == "Recommended"


def test_rotate_flip_crop_and_reset(view, tk_root, tmp_path):
    view.open_image(_fits(tmp_path))
    _wait(tk_root, lambda: view.current is not None and view.current_is_full and view.job is None)
    view.rotate(1)
    _wait(tk_root, lambda: view.current is not None and view.current_is_full and view.job is None)
    assert view.source.shape == (3, W, H)
    assert np.array_equal(view.source, np.rot90(view.loaded, 1, axes=(1, 2)))
    view.flip(True)
    _wait(tk_root, lambda: view.current is not None and view.current_is_full and view.job is None)
    view.toggle_crop()
    assert view.canvas.crop_mode
    view.canvas.crop_box = (10, 20, 210, 320)     # 顯示中影像的座標
    view.apply_crop()
    _wait(tk_root, lambda: view.current is not None and view.current_is_full and view.job is None)
    assert view.source.shape == (3, 300, 200)
    shown = np.rot90(view.loaded, 1, axes=(1, 2))[:, :, ::-1][:, 20:320, 10:210]
    assert np.array_equal(view.source, shown)
    view.reset_geometry()
    _wait(tk_root, lambda: view.current is not None and view.current_is_full and view.job is None)
    assert view.source.shape == (3, H, W)
