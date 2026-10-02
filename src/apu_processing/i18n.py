"""介面文字：繁體中文（預設）與英文，用詞與 APU Astro 系列一致。

程式裡只放代號，顯示時才用 tr() 依目前語言轉成文字。兩份的代號必須一致（tests/test_i18n.py 會檢查）。

引擎的警告與錯誤存成 Msg（代號＋參數），顯示時才轉成文字：介面換語言後，已經產生的訊息也跟著換。
"""

from __future__ import annotations

APP_NAME = "APU Processing"
APP_SUBTITLE = "Astrophotography Processing Utility"

LANGUAGES = {"zh": "繁體中文", "en": "English"}
DEFAULT_LANGUAGE = "zh"
_language = DEFAULT_LANGUAGE


def set_language(lang: str) -> None:
    global _language
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang}")
    _language = lang


def get_language() -> str:
    return _language


def tr(key: str, **kw: object) -> str:
    text = _CATALOG[_language].get(key)
    if text is None:
        text = _CATALOG[DEFAULT_LANGUAGE][key]
    return text.format(**kw) if kw else text


class Msg:
    """延後翻譯的訊息：str() 時才依目前語言產生文字。參數本身也可以是 Msg。"""

    __slots__ = ("key", "kw")

    def __init__(self, key: str, **kw: object):
        self.key, self.kw = key, kw

    def __str__(self) -> str:
        return tr(self.key, **self.kw)

    def __repr__(self) -> str:
        return f"Msg({self.key!r}, {self.kw!r})"

    def __contains__(self, text: str) -> bool:
        return text in str(self)


_ZH: dict[str, str] = {
    # 讀寫
    "msg.unsupported_format": "不支援的檔案格式：{name}",
    "msg.no_image_data": "{name} 裡沒有影像資料",
    "msg.bad_shape": "{name} 的影像維度不支援：{shape}（要單色或 RGB）",
    "msg.bad_channels": "{name} 有 {n} 個色版，只支援 1 或 3 個",
    # 合成
    "msg.compose_empty": "沒有要合成的影像",
    "msg.compose_not_mono": "{name} 是彩色影像；合成只接受單色的 master（每個濾鏡一張）",
    "msg.compose_blank": "{name} 整張都是同一個值（例如全黑），沒有影像內容；請重新疊圖",
    "msg.compose_size": "{name}（{size}）跟 {first}（{first_size}）尺寸不同，可能不是同一套器材一起疊出來的；"
                        "合成需要已經對齊、尺寸相同的影像",
    # 介面共用
    "gui.details": "說明",
    "gui.no_image": "尚未開啟影像",
    "gui.mono": "（單色）",
    "gui.empty.hint": "開啟一張未拉伸的線性影像（{shortcut}）\nFIT／FITS／FTS、TIFF、PNG",
    # 頂部列
    "gui.btn.open": "開啟",
    "gui.btn.open.help": "開啟未拉伸的線性影像（{shortcut}）；開啟後自動用建議設定處理。"
                         "一次選好幾張單色 master（Ha、OIII、SII 或 R、G、B）就會合成成彩色",
    "gui.btn.save": "儲存",
    "gui.btn.save.help": "PNG／JPEG／TIFF 存成品（含成品微調）；FITS 存線性處理結果",
    "gui.btn.recommended": "建議設定",
    "gui.btn.recommended.help": "所有處理、拉伸與成品微調恢復建議值，重新處理",
    "gui.view.original": "原圖",
    "gui.view.previous": "上次結果",
    "gui.view.current": "目前結果",
    # 選單
    "gui.menu.about": "關於 APU Processing",
    "gui.menu.file": "檔案",
    "gui.menu.open": "開啟…",
    "gui.menu.save": "儲存…",
    "gui.menu.view": "顯示",
    "gui.menu.fit": "符合視窗",
    "gui.menu.actual": "100%",
    "gui.menu.window": "視窗",
    "gui.confirm.quit": "還在處理中，確定要結束嗎？",
    # 面板
    "gui.group.info": "影像",
    "gui.group.info.info": "尺寸：影像的寬 × 高。\n噪聲：拉伸用的 8 px 尺度噪聲（處理後量）。\n校色倍率：紅／綠／藍各乘了多少（讓星點平均顏色接近白色）。",
    "gui.metric.size": "尺寸",
    "gui.btn.rotate_left": "左轉 90°",
    "gui.btn.rotate_right": "右轉 90°",
    "gui.btn.flip_h": "左右翻轉",
    "gui.btn.flip_v": "上下翻轉",
    "gui.btn.crop": "裁切",
    "gui.btn.crop.help": "在畫面上拖出要保留的範圍，再按「套用」；疊圖邊緣的壞邊可以先裁掉，去光與拉伸只分析保留的範圍",
    "gui.btn.cancel_crop": "取消裁切",
    "gui.btn.apply_crop": "套用",
    "gui.btn.reset_geometry": "還原",
    "gui.btn.reset_geometry.help": "取消所有裁切、旋轉與翻轉，回到原始檔（原始檔本來就不會被修改）",
    "gui.status.crop": "拖出要保留的範圍，按「套用」",
    "gui.metric.noise": "噪聲",
    "gui.metric.color": "校色倍率",
    "gui.group.compose": "合成",
    "gui.group.compose.info": "把幾張單色 master 組成一張彩色影像，再照下面的步驟處理。\nHOO：Ha → 紅，OIII → 綠與藍。SHO：SII → 紅，Ha → 綠，OIII → 藍。RGB：R、G、B 各一張。\n開檔時依每張的濾鏡（FILTER）自動對應；對錯了可以在下拉選單改。\n各通道先扣自己的天空、再把噪聲調成一樣大才組合，所以天空顆粒一致，弱的通道不會被放大成一片噪聲。想讓某個通道更明顯，就把它的強度調高（100% 是預設）。\n窄帶合成的星點顏色不是真實顏色，所以 HOO、SHO 預設不校色。",
    "gui.compose.none": "（不使用）",
    "gui.slider.compose_strength": "強度",
    "gui.group.processing": "處理",
    "gui.group.processing.info": "去光梯度：扣掉光害與暗角造成的大範圍明暗，星雲不會被當成光害扣掉。\n校色：用星點的平均顏色做白平衡（不需要星表）。\n降噪：壓掉細顆粒與色彩噪聲，保留星雲結構；約保留一半原本的顆粒，看起來自然、不會變成塑膠感。\n細節與縮星：縮小星點（光通量不變、不會有黑圈），並加強星雲的細部。",
    "gui.toggle.background": "去光梯度",
    "gui.toggle.color": "校色",
    "gui.toggle.denoise": "降噪",
    "gui.slider.denoise": "降噪強度",
    "gui.toggle.detail": "細節與縮星",
    "gui.slider.stars": "縮星",
    "gui.slider.sharpen": "星雲細節",
    "gui.group.stretch": "拉伸",
    "gui.group.stretch.info": "依畫面剩下的噪聲自動決定拉多深：乾淨的素材拉得深，噪聲多的拉得淺，噪聲不會被拉出來。\n拉伸強度：在自動的基礎上加減；50% 是預設（以 17 組實拍的人工後製成品校準，星雲亮度與看得到的噪聲都跟成品差不多）。\n天空亮度：乾淨天空在成品上的亮度。\n背景保留色彩：背景帶上暗處微弱星雲的顏色（滿版 Hα 的暗紅、反射星雲的藍），亮度不變；乾淨的天空幾乎不受影響。預設 50%；0% 把最暗的天空校成中性。程式分不出暗處是光害還是微弱的星雲，覺得背景太紅或太藍就往下調。\n三色連動：三色共用同一個拉伸尺度，顏色比例最忠實；未校色的素材可以關掉。",
    "gui.slider.stretch": "拉伸強度",
    "gui.slider.sky": "天空亮度",
    "gui.slider.bg_color": "背景保留色彩",
    "gui.toggle.linked": "三色連動",
    "gui.group.finishing": "成品微調",
    "gui.group.finishing.info": "只影響畫面與 PNG／JPEG／TIFF 成品，FITS 仍是線性資料；調整時不用重新處理。\n去綠：把比紅藍平均還綠的部分拉回來，紅色或藍色為主的地方不受影響。\n曲線：拖曳控制點；雙擊空白處新增、雙擊控制點刪除。",
    "gui.slider.exposure": "明暗",
    "gui.slider.contrast": "對比",
    "gui.slider.saturation": "飽和度",
    "gui.slider.green_removal": "去綠",
    "gui.curves": "曲線",
    "gui.curve.master": "主控",
    "gui.curve.red": "紅",
    "gui.curve.green": "綠",
    "gui.curve.blue": "藍",
    "gui.btn.reset_curve": "重設這條曲線",
    "gui.btn.reset_finishing": "重設成品微調",
    # 狀態列
    "gui.status.start": "開啟一張影像開始",
    "gui.status.opening": "正在開啟 {name}…",
    "gui.status.opening_many": "正在開啟 {n} 張影像並合成…",
    "gui.status.composing": "合成中…",
    "gui.status.processing": "處理中…",
    "gui.status.quick": "快速預覽（完整結果處理中…）",
    "gui.status.done": "完成（{s} 秒）",
    "gui.status.saving": "正在儲存 {name}…",
    "gui.status.saved": "已儲存 {name}",
    "gui.status.error": "發生錯誤",
    "gui.summary.zoom": "縮放 {z}%",
    "gui.error.open": "無法開啟 {name}",
    "gui.error.compose": "無法合成這幾張影像",
    "gui.error.process": "處理時發生錯誤",
    "gui.error.save": "無法儲存 {name}",
    "gui.save.fits": "FITS（線性）",
    # 處理階段
    "stage.background": "去光梯度…",
    "stage.color": "校色…",
    "stage.denoise": "降噪…",
    "stage.detail": "細節與星點…",
    "stage.stretch": "拉伸…",
    "stage.done": "完成",
}

_EN: dict[str, str] = {
    "msg.unsupported_format": "Unsupported file format: {name}",
    "msg.no_image_data": "{name} contains no image data",
    "msg.bad_shape": "Unsupported image dimensions in {name}: {shape} (mono or RGB only)",
    "msg.bad_channels": "{name} has {n} channels; only 1 or 3 are supported",
    "msg.compose_empty": "No images to combine",
    "msg.compose_not_mono": "{name} is a color image; combining takes mono masters (one per filter)",
    "msg.compose_blank": "{name} has the same value everywhere (for example all black) and no image content; "
                         "please stack it again",
    "msg.compose_size": "{name} ({size}) and {first} ({first_size}) differ in size and were probably not stacked "
                        "together with the same equipment; combining needs aligned images of the same size",
    "gui.details": "Details",
    "gui.no_image": "No image open",
    "gui.mono": "(mono)",
    "gui.empty.hint": "Open an unstretched linear image ({shortcut})\nFIT/FITS/FTS, TIFF, PNG",
    "gui.btn.open": "Open",
    "gui.btn.open.help": "Open an unstretched linear image ({shortcut}); it is processed with the recommended settings. "
                         "Select several mono masters (Ha, OIII, SII or R, G, B) at once to combine them into color",
    "gui.btn.save": "Save",
    "gui.btn.save.help": "PNG/JPEG/TIFF save the finished image (with finishing); FITS saves the linear result",
    "gui.btn.recommended": "Recommended",
    "gui.btn.recommended.help": "Reset processing, stretch and finishing to the recommended values and reprocess",
    "gui.view.original": "Original",
    "gui.view.previous": "Previous",
    "gui.view.current": "Current",
    "gui.menu.about": "About APU Processing",
    "gui.menu.file": "File",
    "gui.menu.open": "Open…",
    "gui.menu.save": "Save…",
    "gui.menu.view": "View",
    "gui.menu.fit": "Fit to Window",
    "gui.menu.actual": "100%",
    "gui.menu.window": "Window",
    "gui.confirm.quit": "Processing is still running. Quit anyway?",
    "gui.group.info": "Image",
    "gui.group.info.info": "Size: width × height.\nNoise: the 8 px noise the stretch uses (measured after processing).\nColor gains: how much red, green and blue were scaled so that the average star is white.",
    "gui.metric.size": "Size",
    "gui.btn.rotate_left": "Rotate left",
    "gui.btn.rotate_right": "Rotate right",
    "gui.btn.flip_h": "Flip horizontal",
    "gui.btn.flip_v": "Flip vertical",
    "gui.btn.crop": "Crop",
    "gui.btn.crop.help": "Drag the area to keep, then press Apply; cropping bad stacking edges first means gradient removal and the stretch only analyze what is kept",
    "gui.btn.cancel_crop": "Cancel crop",
    "gui.btn.apply_crop": "Apply",
    "gui.btn.reset_geometry": "Reset",
    "gui.btn.reset_geometry.help": "Undo every crop, rotation and flip and go back to the file as loaded (the file itself is never modified)",
    "gui.status.crop": "Drag the area to keep, then press Apply",
    "gui.metric.noise": "Noise",
    "gui.metric.color": "Color gains",
    "gui.group.compose": "Combine",
    "gui.group.compose.info": "Combines several mono masters into one color image, which is then processed by the steps below.\nHOO: Ha → red, OIII → green and blue. SHO: SII → red, Ha → green, OIII → blue. RGB: one each for R, G and B.\nEach file is matched by its filter (FILTER) when opened; change it in the drop-down if it is wrong.\nEach channel has its own sky removed and its noise brought to the same level before combining, so the sky grain is even and a weak channel is not blown up into noise. Raise a channel's strength to make it stand out more (100% is the default).\nStar colors in a narrowband combination are not real colors, so HOO and SHO are not color calibrated by default.",
    "gui.compose.none": "(not used)",
    "gui.slider.compose_strength": "Strength",
    "gui.group.processing": "Processing",
    "gui.group.processing.info": "Gradients: removes light pollution and vignetting without taking nebulae for them.\nColor: white balance from the average star color (no catalog needed).\nDenoise: removes fine grain and color noise while keeping nebular structure; about half of the original grain is kept so the result looks natural rather than plastic.\nDetail and stars: makes stars smaller (flux kept, no dark rings) and sharpens nebular detail.",
    "gui.toggle.background": "Remove gradients",
    "gui.toggle.color": "Calibrate color",
    "gui.toggle.denoise": "Denoise",
    "gui.slider.denoise": "Denoise amount",
    "gui.toggle.detail": "Detail and stars",
    "gui.slider.stars": "Star reduction",
    "gui.slider.sharpen": "Nebula detail",
    "gui.group.stretch": "Stretch",
    "gui.group.stretch.info": "The stretch reads the remaining noise: clean data is stretched deeper, noisy data less, so noise is not pulled up.\nStrength: adjusts the automatic stretch; 50% is the default, calibrated on hand-processed results of 17 real image sets for similar nebula brightness and visible noise.\nSky level: where clean sky lands in the result.\nBackground color: tints the background with the color of the faint nebulosity found in the dark areas (deep red for full-frame H-alpha, blue for reflection nebulae) without changing its brightness; clean skies barely change. Default 50%; 0% makes the darkest sky neutral. The app cannot tell light pollution from faint nebulosity, so lower it if the background looks too red or too blue.\nLinked: one stretch for all three channels keeps color ratios; turn off for uncalibrated data.",
    "gui.slider.stretch": "Strength",
    "gui.slider.sky": "Sky level",
    "gui.slider.bg_color": "Background color",
    "gui.toggle.linked": "Linked channels",
    "gui.group.finishing": "Finishing",
    "gui.group.finishing.info": "Affects only the screen and PNG/JPEG/TIFF output; FITS stays linear, and nothing is reprocessed.\nRemove green: pulls back green that exceeds the red/blue average; red or blue subjects are untouched.\nCurves: drag points; double-click empty space to add, a point to delete.",
    "gui.slider.exposure": "Brightness",
    "gui.slider.contrast": "Contrast",
    "gui.slider.saturation": "Saturation",
    "gui.slider.green_removal": "Remove green",
    "gui.curves": "Curves",
    "gui.curve.master": "RGB",
    "gui.curve.red": "Red",
    "gui.curve.green": "Green",
    "gui.curve.blue": "Blue",
    "gui.btn.reset_curve": "Reset curve",
    "gui.btn.reset_finishing": "Reset finishing",
    "gui.status.start": "Open an image to begin",
    "gui.status.opening": "Opening {name}…",
    "gui.status.opening_many": "Opening and combining {n} images…",
    "gui.status.composing": "Combining…",
    "gui.status.processing": "Processing…",
    "gui.status.quick": "Quick preview (full result on the way…)",
    "gui.status.done": "Done ({s} s)",
    "gui.status.saving": "Saving {name}…",
    "gui.status.saved": "Saved {name}",
    "gui.status.error": "Something went wrong",
    "gui.summary.zoom": "Zoom {z}%",
    "gui.error.open": "Could not open {name}",
    "gui.error.compose": "Could not combine these images",
    "gui.error.process": "Processing failed",
    "gui.error.save": "Could not save {name}",
    "gui.save.fits": "FITS (linear)",
    "stage.background": "Removing gradients…",
    "stage.color": "Calibrating color…",
    "stage.denoise": "Reducing noise…",
    "stage.detail": "Detail and stars…",
    "stage.stretch": "Stretching…",
    "stage.done": "Done",
}

_CATALOG = {"zh": _ZH, "en": _EN}
