"""用 Swift v0.9.8 的 AstroSharpAnalyze --oneclick-display 產生 17 組一鍵成品（建議預設＋成品微調），存成 16-bit TIFF 當對照組。"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from apu_processing.imageio import load_image, save_display
from tools.scorecard.data import all_samples

TOOL = Path.home() / "Claude/AstroSharp/.build-local/release/AstroSharpAnalyze"
OUT = Path(__file__).resolve().parents[1] / "output/baselines/v098"

only = set(sys.argv[1:])
OUT.mkdir(parents=True, exist_ok=True)
for s in all_samples():
    if only and s.key not in only:
        continue
    target = OUT / f"{s.key}.tif"
    if target.exists():
        continue
    t = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        fits_out = Path(tmp) / "out.fits"
        subprocess.run([str(TOOL), "--oneclick-display", str(s.raw_path), str(fits_out)], check=True,
                       capture_output=True)
        display, _ = load_image(fits_out)
    save_display(target, display.clip(0, 1))
    print(f"{s.key:12} {time.perf_counter() - t:5.1f} s  {target.stat().st_size / 1e6:.0f} MB", flush=True)
