"""打包成可直接執行的程式，再壓成 zip 方便分享。

    pip install -e .[exe] matplotlib   # matplotlib 只給 astropy 的打包 hook 掃描用，不會打包進去
    python packaging/build_exe.py

- Windows：dist/APUProcessing/APUProcessing.exe → dist/APUProcessing-<版本>-win64.zip
- macOS：  dist/APUProcessing.app                → dist/APUProcessing-<版本>-macos-<arm64|x86_64>.zip
  （PyInstaller 不能跨平台：Mac 版在 Mac 打包，Windows 版在 Windows 桌機打包）

打包完用合成星場實際跑一次打包好的程式，正常才產生 zip。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from apu_processing import __version__  # noqa: E402

NAME = "APUProcessing"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
ASSETS = ROOT / "src" / "apu_processing" / "assets"
IS_MAC = sys.platform == "darwin"
EXCLUDES = ["pytest", "IPython", "PyQt5", "PyQt6", "PySide2", "PySide6", "notebook", "sphinx", "matplotlib"]


def platform_tag() -> str:
    if sys.platform == "win32":
        return "win64"
    if IS_MAC:
        return f"macos-{platform.machine()}"
    return f"{sys.platform}-{platform.machine()}"


def build() -> Path:
    import PyInstaller.__main__

    args = [
        str(ROOT / "packaging" / "gui_entry.py"),
        "--name", NAME,
        "--windowed",
        "--noconfirm", "--clean",
        "--icon", str(ASSETS / ("icon_1024.png" if IS_MAC else "app.ico")),
        "--paths", str(ROOT / "src"),
        "--add-data", f"{ASSETS}{os.pathsep}apu_processing/assets",
        "--collect-submodules", "apu_processing",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
    ]
    if IS_MAC:
        args += ["--osx-bundle-identifier", "tw.apu-astrophotography.apu-processing"]
    for mod in EXCLUDES:
        args += ["--exclude-module", mod]
    PyInstaller.__main__.run(args)
    if IS_MAC:
        app = DIST / f"{NAME}.app"
        set_bundle_info(app)
        return app / "Contents" / "MacOS" / NAME
    return DIST / NAME / f"{NAME}.exe"


def set_bundle_info(app: Path) -> None:
    """補齊 Info.plist 後重新簽章。

    - 版本：命令列打包的 PyInstaller 一律填 0.0.0
    - 系統元件跟著系統語言：沒宣告的話，選檔視窗、確認對話框、選單的「隱藏」「結束」都是英文
    - 可以把 FITS／TIFF 拖到程式圖示上開啟
    - 改了 Info.plist 原本的簽章就失效，Apple 晶片的 Mac 會說 app「已損毀」，所以要重新做 ad-hoc 簽章
    """
    import plistlib

    plist = app / "Contents" / "Info.plist"
    info = plistlib.loads(plist.read_bytes())
    info["CFBundleName"] = "APU Processing"          # 選單列顯示的名稱（程式檔名沒有空格）
    info["CFBundleDisplayName"] = "APU Processing"
    info["CFBundleShortVersionString"] = __version__
    info["CFBundleVersion"] = __version__
    info["CFBundleAllowMixedLocalizations"] = True
    info["CFBundleLocalizations"] = ["en", "zh-Hant"]
    info["CFBundleDocumentTypes"] = [{
        "CFBundleTypeName": "Astronomical image",
        "CFBundleTypeRole": "Viewer",
        "CFBundleTypeExtensions": ["fit", "fits", "fts", "tif", "tiff", "png"],
        "LSHandlerRank": "Alternate",
    }]
    plist.write_bytes(plistlib.dumps(info))
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)


def smoke_test(exe: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.txt"
        proc = subprocess.run([str(exe), "--smoke-test", str(out)], timeout=600)
        text = out.read_text(encoding="utf-8") if out.exists() else "(沒有輸出)"
        if proc.returncode != 0 or not text.startswith("ok "):
            raise SystemExit(f"打包好的程式測試失敗（exit {proc.returncode}）：\n{text}")
        print(f"打包好的程式測試通過：{text.strip()}")


def archive() -> tuple[Path, Path]:
    """zip 裡放一個資料夾：程式＋Quick-Start.txt（使用說明）。"""
    base = DIST / f"{NAME}-{__version__}-{platform_tag()}"
    stage = DIST / f"{NAME}-{__version__}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    shutil.copy2(ROOT / "docs" / "Quick-Start.txt", stage / "Quick-Start.txt")
    if IS_MAC:
        app = DIST / f"{NAME}.app"
        # 用 ditto 複製與壓縮才會保留 .app 裡的符號連結、執行權限與簽章
        subprocess.run(["ditto", str(app), str(stage / app.name)], check=True)
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(stage), f"{base}.zip"], check=True)
        shutil.rmtree(stage)
        return app, Path(f"{base}.zip")
    shutil.copytree(DIST / NAME, stage / NAME)
    zip_path = Path(shutil.make_archive(str(base), "zip", root_dir=DIST, base_dir=stage.name))
    shutil.rmtree(stage)
    return DIST / NAME, zip_path


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def main() -> None:
    exe = build()
    smoke_test(exe)
    bundle, zip_path = archive()
    print(f"\n程式：{bundle}（{folder_size(bundle) / 2**20:.0f} MB）")
    print(f"分享用：{zip_path}（{zip_path.stat().st_size / 2**20:.0f} MB）")


if __name__ == "__main__":
    main()
