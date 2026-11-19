"""給 APU Astro Studio（整合版）用的公開介面。

Studio 只透過這個模組引用 APU Processing。這裡列出的名稱、參數與行為改了，Studio 的契約測試會失敗
（Studio repo 的 tests/contract；APU Processing 每次 push 時雲端會用新版跑一次）。
其他模組裡的函式是 Processing 自己的內部實作，可以自由改。
"""

from __future__ import annotations

from . import i18n
from .color import ColorBalance, measure_balance
from .compose import PRESETS as COMPOSE_PRESETS
from .compose import Composer, ComposeSettings
from .coverage import combine as combine_coverage
from .coverage import find_for as find_coverage
from .coverage import good_mask as coverage_good_mask
from .coverage import load_fraction as load_coverage_fraction
from .coverage import suggest_crop
from .denoise import DenoiseSettings, starlet
from .denoise import reduce as reduce_noise
from .detail import PSF_CORE_LIMIT, Stars, find_stars, fine_noise, measure_core_fwhm
from .display import IDENTITY as IDENTITY_CURVE
from .display import DisplayAdjustments, ToneCurves, curve_values
from .display import apply as apply_display
from .imageio import FITS_SUFFIXES, load_image, save_fits
from .progress import Cancelled
from .recipe import is_recipe
from .recipe import load as load_recipe
from .stretch import StretchNoise, StretchSettings
from .stretch import apply as apply_stretch
from .stretch import measure as measure_stretch_noise

__all__ = [
    # 讀寫檔
    "FITS_SUFFIXES", "load_image", "save_fits",
    # 疊圖紀錄、覆蓋率、多張單色合成
    "is_recipe", "load_recipe",
    "find_coverage", "load_coverage_fraction", "combine_coverage", "coverage_good_mask", "suggest_crop",
    "COMPOSE_PRESETS", "Composer", "ComposeSettings",
    # 找星、星點形狀、噪聲
    "Stars", "find_stars", "fine_noise", "measure_core_fwhm", "PSF_CORE_LIMIT",
    # 降噪
    "starlet", "DenoiseSettings", "reduce_noise",
    # 拉伸、校色
    "StretchNoise", "StretchSettings", "measure_stretch_noise", "apply_stretch",
    "ColorBalance", "measure_balance",
    # 顯示微調（曲線）
    "IDENTITY_CURVE", "ToneCurves", "DisplayAdjustments", "curve_values", "apply_display",
    # 其他
    "Cancelled", "i18n",
]
