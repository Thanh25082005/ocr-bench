"""Font dùng cho tài liệu tổng hợp, tải từ kho Google Fonts (giấy phép OFL / Apache).

Font được tải một lần vào ~/.cache/ocrbench/fonts để mọi máy dựng ra ảnh giống nhau.
Font giả chữ viết tay (hand_*) chỉ là xấp xỉ: Aref Ruqaa mô phỏng kiểu Ruq'ah, kiểu chữ tay
thông dụng nhất của tiếng Ả Rập. Nó không thay được chữ viết tay thật.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/google/fonts/main/"

# family -> (đường dẫn trong repo google/fonts, có phải font biến thiên độ đậm không)
FONT_FILES = {
    "Noto Naskh Arabic": ("ofl/notonaskharabic/NotoNaskhArabic[wght].ttf", True),
    "Amiri": ("ofl/amiri/Amiri-Regular.ttf", False),
    "Cairo": ("ofl/cairo/Cairo[slnt,wght].ttf", True),
    "Tajawal": ("ofl/tajawal/Tajawal-Regular.ttf", False),
    "Source Serif 4": ("ofl/sourceserif4/SourceSerif4[opsz,wght].ttf", True),
    "Lato": ("ofl/lato/Lato-Regular.ttf", False),
    "Libre Baskerville": ("ofl/librebaskerville/LibreBaskerville[wght].ttf", True),
    "Aref Ruqaa": ("ofl/arefruqaa/ArefRuqaa-Regular.ttf", False),
    "Aref Ruqaa Bold": ("ofl/arefruqaa/ArefRuqaa-Bold.ttf", False),
    "Marhey": ("ofl/marhey/Marhey[wght].ttf", True),
    "Caveat": ("ofl/caveat/Caveat[wght].ttf", True),
    "Homemade Apple": ("apache/homemadeapple/HomemadeApple-Regular.ttf", False),
    "Indie Flower": ("ofl/indieflower/IndieFlower-Regular.ttf", False),
    "Shadows Into Light": ("ofl/shadowsintolight/ShadowsIntoLight.ttf", False),
}

FONTS = {
    "print_ar": ["Noto Naskh Arabic", "Amiri", "Cairo", "Tajawal"],
    "print_en": ["Source Serif 4", "Lato", "Libre Baskerville"],
    # Không dùng Aref Ruqaa Ink: đó là font màu, tự tô đỏ và bỏ qua màu mực
    "hand_ar": ["Aref Ruqaa", "Aref Ruqaa Bold", "Marhey"],
    "hand_en": ["Caveat", "Homemade Apple", "Indie Flower", "Shadows Into Light"],
}


def cache_dir() -> Path:
    return Path(os.environ.get("OCRBENCH_FONT_DIR", Path.home() / ".cache" / "ocrbench" / "fonts"))


def ensure_fonts() -> dict[str, Path]:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for family, (repo_path, _) in FONT_FILES.items():
        dest = d / Path(repo_path).name
        if not dest.exists() or dest.stat().st_size == 0:
            url = RAW + urllib.parse.quote(repo_path)
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
        paths[family] = dest
    return paths


def font_face_css(paths: dict[str, Path]) -> str:
    rules = []
    for family, path in paths.items():
        variable = FONT_FILES[family][1]
        weight = "font-weight:100 900;" if variable else ""
        rules.append(f"@font-face {{ font-family:'{family}'; src:url('{path.as_uri()}'); {weight} }}")
    return "\n".join(rules)
