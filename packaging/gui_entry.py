"""APU Processing 打包用的進入點（PyInstaller）。

multiprocessing.freeze_support() 先呼叫（目前只用執行緒，將來加多行程時才不會每個子行程多開一個視窗）。

--smoke-test <結果檔>：用合成星場確認打包好的程式能完整處理（去光、校色、降噪、縮星、拉伸、成品微調、
讀寫 FITS／PNG／TIFF），build_exe.py 打包完會自動跑。
"""

import multiprocessing
import sys


def smoke_test(out: str) -> int:
    import tempfile
    import traceback
    from pathlib import Path

    result = Path(out)
    try:
        import numpy as np

        from apu_processing import display, gui, imageio, pipeline  # noqa: F401  視窗介面用的東西都要有打包進來

        rng = np.random.default_rng(0)
        h, w = 600, 900
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.stack([0.02 + 0.003 * xx / w + 0.004 * np.exp(-((yy - 300) ** 2 + (xx - 450) ** 2) / 7200)
                        + rng.normal(0, 2e-4, (h, w)) for _ in range(3)]).astype(np.float32)
        for _ in range(60):
            y, x = rng.uniform(20, h - 20), rng.uniform(20, w - 20)
            img[:, int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4] += rng.uniform(0.01, 0.1) * np.exp(
                -((np.arange(7)[:, None] - 3) ** 2 + (np.arange(7)[None] - 3) ** 2) / 3.0)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "smoke.fits"
            imageio.save_fits(src, img)
            data, header = imageio.load_image(src)
            r = pipeline.process(data)
            fin = display.apply(r.display, display.DisplayAdjustments(saturation=1.2, green_removal=0.5))
            for name in ("out.png", "out.tif", "out.fits"):
                p = Path(tmp) / name
                if name.endswith(".fits"):
                    imageio.save_fits(p, r.linear, header)
                else:
                    imageio.save_display(p, fin)
                assert imageio.load_image(p)[0].shape == img.shape
        result.write_text(f"ok stages={','.join(r.computed)} sky={r.noise.sky[1]:.4f}\n", encoding="utf-8")
        return 0
    except Exception:  # noqa: BLE001
        result.write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == "--smoke-test":
        sys.exit(smoke_test(sys.argv[2]))
    from apu_processing.gui import main

    sys.exit(main())
