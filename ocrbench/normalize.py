"""Chuẩn hóa văn bản trước khi chấm điểm.

Mỗi model được chấm theo hai cách:
- raw: gần như nguyên bản (chỉ NFC, bỏ ký tự điều khiển bidi, gộp khoảng trắng).
- norm: áp cùng một bộ quy tắc cố định cho cả đáp án lẫn kết quả model, để không phạt
  những khác biệt không quan trọng (Markdown, dấu tashkeel, chữ số Ả Rập-Ấn...).

Quy tắc norm cấu hình được trong file YAML (mục `normalization`). Đổi quy tắc thì chỉ cần
chạy lại `ocrbench score`, không phải chạy lại model.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, fields

# Dấu tashkeel và các dấu Quran nhỏ
_AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_TATWEEL = "ـ"
# Zero-width, đánh dấu hướng chữ, BOM
_BIDI = re.compile(r"[​-‏‪-‮⁦-⁩﻿؜]")
# chữ số Ả Rập-Ấn và Ba Tư -> ASCII; dấu thập phân ٫ -> . và phân cách hàng nghìn ٬ -> ,
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "0123456789" * 2 + ".,")
_ALEF = re.compile("[آأإٱ]")  # آ أ إ ٱ -> ا
_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٪": "%", "«": '"', "»": '"'})

_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
_MD_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?$")


@dataclass
class NormConfig:
    nfkc: bool = True  # gộp dạng trình bày của chữ Ả Rập (ﻻ -> لا), chữ full-width...
    strip_markdown: bool = True  # bỏ #, **, |, thẻ HTML... để so nội dung chữ
    strip_tatweel: bool = True
    strip_diacritics: bool = True  # bỏ tashkeel
    unify_alef: bool = False  # أ إ آ -> ا
    unify_yeh: bool = False  # ى -> ي
    unify_digits: bool = True  # ٠١٢ -> 012, ١٬٢٥٠٫٥ -> 1,250.5
    unify_punct: bool = False  # ، ؛ ؟ -> , ; ?
    lowercase: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "NormConfig":
        d = d or {}
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"normalization: không có tùy chọn {sorted(unknown)}; các tùy chọn hợp lệ: {sorted(names)}")
        return cls(**d)


def to_plain_text(s: str) -> str:
    """Bỏ định dạng Markdown/HTML, giữ lại nội dung chữ theo thứ tự đọc."""
    if "<" in s and _HTML_TAG.search(s):
        s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
        s = re.sub(r"</(?:p|div|tr|li|h[1-6]|table|caption)>", "\n", s, flags=re.I)
        s = re.sub(r"</t[dh]>", " ", s, flags=re.I)
        s = _HTML_TAG.sub(" ", s)
        s = html.unescape(s)
    out = []
    for line in s.splitlines():
        st = line.strip()
        if st.startswith("```") or _MD_TABLE_SEP.match(st):
            continue
        if st.startswith("|") or st.endswith("|"):
            st = re.sub(r"\s*\|\s*", " ", st.strip("|"))
        st = re.sub(r"^#{1,6}\s+", "", st)
        st = re.sub(r"^[-*+]\s+", "", st)
        st = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", st)
        st = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", st)
        st = st.replace("**", "").replace("__", "")
        out.append(st)
    return "\n".join(out)


def normalize(text: str, cfg: NormConfig) -> str:
    s = unicodedata.normalize("NFKC" if cfg.nfkc else "NFC", text)
    s = _BIDI.sub("", s)
    if cfg.strip_markdown:
        s = to_plain_text(s)
    if cfg.strip_tatweel:
        s = s.replace(_TATWEEL, "")
    if cfg.strip_diacritics:
        s = _AR_DIACRITICS.sub("", s)
    if cfg.unify_alef:
        s = _ALEF.sub("ا", s)
    if cfg.unify_yeh:
        s = s.replace("ى", "ي")
    if cfg.unify_digits:
        s = s.translate(_DIGITS)
    if cfg.unify_punct:
        s = s.translate(_PUNCT)
    if cfg.lowercase:
        s = s.lower()
    return re.sub(r"\s+", " ", s).strip()


def normalize_raw(text: str) -> str:
    s = unicodedata.normalize("NFC", text)
    s = _BIDI.sub("", s)
    return re.sub(r"\s+", " ", s).strip()
