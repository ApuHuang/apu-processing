"""視窗流程測試：不點滑鼠，直接呼叫方法。

Tk 在同一個行程只建一次（反覆建立、關閉 Tk 在 Windows 偶爾失敗、Mac 雲端機會卡死）。
"""

import gc
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
    del v
    gc.collect()   # 在主執行緒回收：不然 Tk 變數可能在背景執行緒被釋放（tkinter 的已知狀況）
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


def test_view_keeps_events_to_itself(view, tk_root, tmp_path):
    """整合版會把好幾個畫面放進同一個視窗：View 不能用 bind_all 搶全視窗的事件。"""
    for event in ("<Button-1>", "<Escape>", "<MouseWheel>"):
        assert tk_root.bind_all(event) == ""
    # 點 View 裡的元件會關掉說明氣泡
    view.show_popover(view.open_btn, "說明")
    assert view.popover is not None
    view.status.event_generate("<Button-1>", x=1, y=1)
    tk_root.update()
    assert view.popover is None
    # 切換語言重建介面後，新的元件也有標籤
    view.lang_var.set("en")
    _wait(tk_root, lambda: view.open_btn.cget("text") == "Open", 5)
    assert view._tag in view.status.bindtags() and view._tag in view.canvas.bindtags()
    # is_busy：開檔處理中為真、做完為假
    assert not view.is_busy()
    view.open_image(_fits(tmp_path))
    _wait(tk_root, lambda: view.is_busy(), 10)
    _wait(tk_root, lambda: view.current is not None and view.current_is_full and not view.is_busy())


def _masters(tmp_path):
    """同一套器材疊出來的 Ha、OIII 單色 master（FILTER 寫在 header）。"""
    from astropy.io import fits

    paths = []
    for i, (name, sky, sigma, gain) in enumerate((("Ha", 0.012, 2e-4, 1.0), ("OIII", 0.020, 4e-4, 0.3))):
        img = np.stack([sky + gain * _nebula() + _noise((H, W), sigma, 10 + i)])
        rng = np.random.default_rng(7)
        for _ in range(40):
            _add_star(img, rng.uniform(20, H - 20), rng.uniform(20, W - 20), rng.uniform(0.05, 0.5), 3.0)
        header = fits.Header()
        header["FILTER"] = name
        path = tmp_path / f"ngc_{name}.fits"
        imageio.save_fits(path, img[0], header)
        paths.append(path)
    return paths


def test_open_several_masters_combines_them(view, tk_root, tmp_path):
    from astropy.io import fits

    done = lambda: view.current is not None and view.current_is_full and view.job is None  # noqa: E731
    view.open_images(_masters(tmp_path))
    _wait(tk_root, done)
    assert view.composer is not None and view.compose_settings.preset == "HOO"
    assert view.source.shape == (3, H, W)
    assert "HOO" in view.doc_title.cget("text")
    # 窄帶預設不校色，而且不改到單張影像的設定
    assert view.current.balance is None and view.v["color_enabled"].get()
    # 調 OIII 強度：重新合成、從頭處理
    before = view.loaded
    view.compose_strength_vars[1].set(200)
    _wait(tk_root, lambda: view.loaded is not before and done() and view._compose_debounce is None)
    assert view.compose_settings.channels[1].strength == 2.0
    assert view.compose_palette_var.get() == 0 and "palette" not in view.current.computed
    # 換成 SHO：哈伯色調預設 100%
    view.compose_preset_var.set("SHO")
    _wait(tk_root, lambda: view.compose_settings.preset == "SHO" and done() and "palette" in view.current.computed)
    assert view.compose_palette_var.get() == 100
    # 換成 RGB：預設校色、沒有哈伯色調、面板重建
    view.compose_preset_var.set("RGB")
    _wait(tk_root, lambda: view.compose_settings.preset == "RGB" and done() and view.compose_color_var.get())
    assert len(view.compose_strength_vars) == 3 and view.compose_palette_var.get() == 0
    view.compose_preset_var.set("HOO")
    _wait(tk_root, lambda: view.compose_settings.preset == "HOO" and done())
    out = tmp_path / "hoo.fits"
    view.save(out)
    _wait(tk_root, lambda: view.status.cget("text") == tr("gui.status.saved", name="hoo.fits"))
    header = fits.getheader(out)
    assert header["APUCOMP"] == "HOO" and "FILTER" not in header
    # 再開一張一般影像：離開合成模式
    view.open_image(_fits(tmp_path))
    _wait(tk_root, lambda: view.composer is None and done())
    assert view.current.balance is not None


def test_masters_of_different_size_are_refused(view, tk_root, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    ha, oiii = _masters(tmp_path)
    data, header = imageio.load_image(oiii)
    imageio.save_fits(oiii, data[:-20], header)
    view.open_images([ha, oiii])
    _wait(tk_root, lambda: errors, 30)
    assert "尺寸不同" in errors[0][1] and view.composer is None


def test_language_callback_and_hidden_switch(tk_root, tmp_path, monkeypatch):
    """整合版的外殼傳 on_language 自己換語言；show_language=False 時頂部列不顯示切換。"""
    from apu_processing.darkroom import Segmented

    monkeypatch.setenv("APU_PROCESSING_SETTINGS", str(tmp_path / "settings.json"))
    set_language("zh")
    calls = []

    def shell(lang):
        calls.append(lang)
        set_language(lang)
        v.rebuild()

    v = gui.ProcessingView(tk_root, tk_root, on_language=shell)
    v.pack(fill="both", expand=True)
    tk_root.update()
    try:
        v.lang_var.set("en")
        _wait(tk_root, lambda: v.open_btn.cget("text") == "Open", 5)
        assert calls == ["en"]
        assert not (tmp_path / "settings.json").exists()      # 存不存語言由外殼決定
        # 外殼直接換語言再 rebuild：切換鈕跟著換，不會再呼叫 callback
        set_language("zh")
        v.rebuild()
        tk_root.update()
        assert v.lang_var.get() == "zh" and v.open_btn.cget("text") == tr("gui.btn.open")
        assert calls == ["en"]
    finally:
        v.close()
        v.destroy()
    del v, shell
    gc.collect()

    hidden = gui.ProcessingView(tk_root, tk_root, show_language=False)
    hidden.pack(fill="both", expand=True)
    tk_root.update()

    def segments(w):
        found = [w] if isinstance(w, Segmented) else []
        for c in w.winfo_children():
            found += segments(c)
        return found

    try:
        assert segments(hidden) and not [w for w in segments(hidden) if w.var is hidden.lang_var]
    finally:
        hidden.close()
        hidden.destroy()
    del hidden
    gc.collect()
    set_language("zh")
