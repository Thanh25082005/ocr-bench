"""Nhận dạng phông chữ Ả Rập của tài liệu từ ẢNH + chữ OCR, chọn trong thư viện phông mở (SIL OFL, được nhúng).

Vì sao: phông Ả Rập khác nhau rất xa (cùng một câu, Amiri và Kufi chênh ~50% bề rộng, hình nét khác hẳn), nên dựng
DOCX "giữ nguyên bố cục" bằng một phông cố định thì không thể khớp. Cách làm theo hướng nhận dạng phông Ả Rập bằng
so mẫu (KAFD / APTI): OCR đã biết nội dung từng dòng → dựng lại dòng đó bằng từng phông ứng viên, co về đúng hộp
mực gốc, so độ trùng nét (chịu lệch 1 px) → phông có điểm trung bình cao nhất trên cả tài liệu.

Phông tải một lần từ kho google/fonts vào ~/.cache/ocrbench/fonts/arabic (OCRBENCH_FONT_DIR); phông biến thiên
được cắt thành bản tĩnh Regular (400) / Bold (700) bằng fontTools để nhúng vào DOCX được.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl/"

# family -> (thư mục trong google/fonts/ofl, file Regular, file Bold | None). File có "[" = phông biến thiên.
LIBRARY: dict[str, tuple[str, str, str | None]] = {
    # Sans (gần Arial / Tahoma / Segoe UI tiếng Ả Rập)
    "Noto Sans Arabic": ("notosansarabic", "NotoSansArabic[wdth,wght].ttf", None),
    "IBM Plex Sans Arabic": ("ibmplexsansarabic", "IBMPlexSansArabic-Regular.ttf", "IBMPlexSansArabic-Bold.ttf"),
    "Tajawal": ("tajawal", "Tajawal-Regular.ttf", "Tajawal-Bold.ttf"),
    "Almarai": ("almarai", "Almarai-Regular.ttf", "Almarai-Bold.ttf"),
    "Cairo": ("cairo", "Cairo[slnt,wght].ttf", None),
    "Vazirmatn": ("vazirmatn", "Vazirmatn[wght].ttf", None),
    "Readex Pro": ("readexpro", "ReadexPro[HEXP,wght].ttf", None),
    "Mada": ("mada", "Mada[wght].ttf", None),
    "Alexandria": ("alexandria", "Alexandria[wght].ttf", None),
    "Rubik": ("rubik", "Rubik[wght].ttf", None),
    # Naskh (gần Traditional Arabic / Simplified Arabic / Times New Roman tiếng Ả Rập)
    "Noto Naskh Arabic": ("notonaskharabic", "NotoNaskhArabic[wght].ttf", None),
    "Amiri": ("amiri", "Amiri-Regular.ttf", "Amiri-Bold.ttf"),
    "Scheherazade New": ("scheherazadenew", "ScheherazadeNew-Regular.ttf", "ScheherazadeNew-Bold.ttf"),
    "Lateef": ("lateef", "Lateef-Regular.ttf", "Lateef-Bold.ttf"),
    "Harmattan": ("harmattan", "Harmattan-Regular.ttf", "Harmattan-Bold.ttf"),
    "Markazi Text": ("markazitext", "MarkaziText[wght].ttf", None),
    # Kufi / tiêu đề
    "Noto Kufi Arabic": ("notokufiarabic", "NotoKufiArabic[wght].ttf", None),
    "Reem Kufi": ("reemkufi", "ReemKufi[wght].ttf", None),
    "Changa": ("changa", "Changa[wght].ttf", None),
    "El Messiri": ("elmessiri", "ElMessiri[wght].ttf", None),
    "Kufam": ("kufam", "Kufam[wght].ttf", None),
}
DEFAULT = "Noto Sans Arabic"


def cache_dir() -> Path:
    return Path(os.environ.get("OCRBENCH_FONT_DIR", Path.home() / ".cache" / "ocrbench" / "fonts")) / "arabic"


def _static(var_path: Path, weight: int, dest: Path) -> Path:
    """Cắt phông biến thiên thành bản tĩnh (các trục khác giữ mặc định) — DOCX chỉ nhúng được phông tĩnh."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    f = TTFont(var_path)
    axes = {a.axisTag: a for a in f["fvar"].axes}
    loc = {}
    if "wght" in axes:
        a = axes["wght"]
        loc["wght"] = max(a.minValue, min(a.maxValue, weight))
    st = instancer.instantiateVariableFont(f, loc, updateFontNames=False)
    if "fvar" in st:  # còn trục khác (wdth, slnt, HEXP) → cố định ở giá trị mặc định
        st = instancer.instantiateVariableFont(st, {a.axisTag: a.defaultValue for a in st["fvar"].axes})
    tmp = dest.with_suffix(".part")
    st.save(tmp)
    tmp.rename(dest)
    return dest


def ensure_library(families=None, quiet: bool = True) -> dict[str, tuple[Path, Path | None]]:
    """Tải (một lần) + cắt bản tĩnh → {family: (Regular, Bold | None)}. Phông nào lỗi thì bỏ qua."""
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for fam in families or LIBRARY:
        folder, reg, bold = LIBRARY[fam]
        try:
            files = []
            for name, weight in ((reg, 400), (bold or (reg if "[" in reg else None), 700)):
                if name is None:
                    files.append(None)
                    continue
                stem = fam.replace(" ", "") + ("-Bold" if weight == 700 else "-Regular")
                dest = d / f"{stem}.ttf"
                if not dest.exists() or dest.stat().st_size == 0:
                    src = d / name
                    if not src.exists() or src.stat().st_size == 0:
                        tmp = src.with_suffix(".part")
                        urllib.request.urlretrieve(RAW + folder + "/" + urllib.parse.quote(name), tmp)
                        tmp.rename(src)
                    if "[" in name:
                        _static(src, weight, dest)
                    else:
                        src.replace(dest)
                files.append(dest)
            out[fam] = (files[0], files[1])
        except Exception as e:  # mạng / file hỏng → bỏ phông này
            if not quiet:
                print(f"   ⚠ {fam}: {type(e).__name__}: {e}")
    return out


@lru_cache(maxsize=1)
def available() -> dict[str, tuple[Path, Path | None]]:
    """Phông đã có trong bộ nhớ đệm (KHÔNG tải). Thiếu → hãy chạy `python -m ocrbench.arabic_fonts`."""
    d = cache_dir()
    out = {}
    for fam in LIBRARY:
        stem = fam.replace(" ", "")
        reg, bold = d / f"{stem}-Regular.ttf", d / f"{stem}-Bold.ttf"
        if reg.exists() and reg.stat().st_size:
            out[fam] = (reg, bold if bold.exists() and bold.stat().st_size else None)
    return out


def font_files(family: str) -> tuple[Path, Path | None] | None:
    return available().get(family)


# ------------------------------------------------------------------ nhận dạng

def _crop(m: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@lru_cache(maxsize=4096)
def _render(text: str, path: str) -> np.ndarray | None:
    f = ImageFont.truetype(path, 64, layout_engine=ImageFont.Layout.RAQM)
    w = int(f.getlength(text, direction="rtl", language="ar")) + 64
    im = Image.new("L", (max(w, 64), 180), 255)
    ImageDraw.Draw(im).text((32, 40), text, font=f, fill=0, direction="rtl", language="ar")
    return _crop(np.asarray(im) < 128)


def _dilate(m: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0


def _norm(m: np.ndarray, w: int, h: int) -> np.ndarray:
    im = Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.BOX).filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(im, dtype=np.float32)
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def line_score(orig: np.ndarray, rend: np.ndarray, h: int = 28) -> float:
    """Độ giống hình nét (0..1): co cả hai về cùng chiều cao thấp (h px, khử khác biệt độ phân giải / độ mờ / độ
    đậm do ngưỡng), làm mờ nhẹ, tương quan; phạt tỉ lệ khung (rộng/cao) sai."""
    H, W = orig.shape
    w = max(8, int(round(W * h / H)))
    c = float((_norm(orig, w, h) * _norm(rend, w, h)).sum())
    ar = (rend.shape[1] / rend.shape[0]) / (W / H)
    return max(0.0, c) * min(ar, 1 / ar) ** 0.5


def identify(lines: list[tuple[np.ndarray, str]], families=None, max_lines: int = 12) -> tuple[str | None, dict]:
    """lines: [(mặt nạ mực của MỘT dòng chữ Ả Rập, chữ OCR của dòng đó)] → (phông tốt nhất, {phông: điểm TB})."""
    lib = {k: v for k, v in available().items() if not families or k in families}
    picked = []
    for m, t in lines:
        o = _crop(m)
        t = (t or "").strip()
        if o is None or o.shape[0] < 8 or o.shape[1] < 40 or len(t) < 4:
            continue
        picked.append((o, t))
    picked.sort(key=lambda x: -len(x[1]))  # dòng dài nhiều thông tin hơn
    picked = picked[:max_lines]
    if not picked or not lib:
        return None, {}
    scores = {}
    for fam, (reg, _) in lib.items():
        vals = []
        for o, t in picked:
            r = _render(t, str(reg))
            vals.append(line_score(o, r) if r is not None else 0.0)
        scores[fam] = round(float(np.mean(vals)), 4)
    best = max(scores, key=scores.get)
    return best, dict(sorted(scores.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":  # tải + chuẩn bị thư viện phông (chạy một lần trên server)
    got = ensure_library(quiet=False)
    print(f"✔ {len(got)}/{len(LIBRARY)} phông Ả Rập trong {cache_dir()}")
    for k, (r, b) in got.items():
        print(f"   {k:22s} {r.name} | {b.name if b else '— (không có bản đậm)'}")
