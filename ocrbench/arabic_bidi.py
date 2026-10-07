"""Thứ tự hiển thị số / chữ Latin trong dòng tiếng Ả Rập: khôi phục hướng viết bị mất khi OCR.

Bản gốc thường đánh dấu hướng bằng thứ vô hình (dir="ltr", ký tự LRM/LRE, thuộc tính run của Word) — ví dụ số
điện thoại "+٩٧٤ ٨١٨ ٦٧٤٣" gõ trái → phải. OCR chỉ trả về chữ theo thứ tự đọc, mất dấu hướng → Word / LibreOffice
chạy thuật toán bidi (UAX #9) xếp các nhóm số theo chiều phải → trái, khác bản gốc.

Cách làm: tìm các cụm "trái → phải" có ký tự trung tính bên trong (khoảng trắng, / - . : @ ...) — chỉ những cụm này
mới có thể hiển thị khác nhau; dựng thử dòng chữ với từng lựa chọn (giữ nguyên / bọc LRE…PDF = ép trái → phải) và so
với nét mực trên ảnh gốc → giữ lựa chọn khớp nhất. LRE/PDF (U+202A / U+202C) là ký tự điều khiển chuẩn Unicode mà
Word, LibreOffice, trình duyệt đều hiểu; bề rộng bằng 0.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageDraw

LRE, PDF = "‪", "‬"
_DIGIT = r"0-9٠-٩۰-۹"
_LTR_CH = rf"A-Za-z{_DIGIT}"
# cụm bắt đầu và kết thúc bằng chữ Latin / chữ số (hoặc + ở đầu), bên trong được có ký tự trung tính
_SEG = re.compile(rf"\+?[{_LTR_CH}](?:[{_LTR_CH}]|[ /\-.:@_,+()٫٬](?=[{_LTR_CH}+(]))*[{_LTR_CH})]?")
_NEUTRAL_INSIDE = re.compile(r"[ /\-.:@_,()٫٬+]")


def _has_inner_neutral(s: str) -> bool:
    return len(s) >= 3 and bool(_NEUTRAL_INSIDE.search(s[1:-1]))


def candidates(line: str) -> list[tuple[int, int]]:
    """Các cụm (đầu, cuối) có thể hiển thị khác nhau tuỳ hướng: có ký tự trung tính ở GIỮA cụm.
    Cụm lớn gộp cả số lẫn chữ (vd. "+971 56 512 3883 contact@firm.co.uk") được tách thêm thành cụm SỐ và cụm CHỮ
    (bản gốc thường chỉ ép riêng số điện thoại) — cụm nhỏ đứng trước, cụm lớn sau."""
    out: list[tuple[int, int]] = []
    for m in _SEG.finditer(line):
        s = m.group(0).strip()
        if not _has_inner_neutral(s):
            continue
        a = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
        # tách theo khoảng trắng, gộp các từ liền nhau cùng loại (có chữ cái / chỉ số)
        toks = [(t.start() + a, t.end() + a, bool(re.search(r"[A-Za-z]", t.group(0))))
                for t in re.finditer(r"\S+", s)]
        groups: list[list] = []
        for t0, t1, alpha in toks:
            if groups and groups[-1][2] == alpha:
                groups[-1][1] = t1
            else:
                groups.append([t0, t1, alpha])
        if len(groups) > 1:
            out += [(g0, g1) for g0, g1, _ in groups if _has_inner_neutral(line[g0:g1])]
        out.append((a, a + len(s)))
    return list(dict.fromkeys(out))


def apply(line: str, segs: list[tuple[int, int]]) -> str:
    """Bọc các cụm bằng LRE…PDF (ép hiển thị trái → phải)."""
    for a, b in sorted(segs, reverse=True):
        line = line[:a] + LRE + line[a:b] + PDF + line[b:]
    return line


def _render(text: str, font) -> np.ndarray | None:
    w = int(font.getlength(text, direction="rtl", language="ar")) + 64
    im = Image.new("L", (max(64, w), int(font.size * 2.2)), 255)
    ImageDraw.Draw(im).text((32, int(font.size * 0.4)), text, font=font, fill=0, direction="rtl", language="ar")
    m = np.asarray(im) < 128
    ys, xs = np.nonzero(m)
    return None if len(xs) == 0 else m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _score(orig: np.ndarray, rend: np.ndarray | None, h: int = 32, bins: int = 160) -> float:
    """Độ khớp THỨ TỰ các cụm trong dòng: tương quan phân bố mực theo chiều ngang (ít phụ thuộc hình nét — phông
    dựng thử thường khác phông gốc) + tương quan hình 2D, sau khi co về cùng khung."""
    if rend is None:
        return -1.0
    H, W = orig.shape
    w = max(16, int(round(W * h / H)))

    def norm(a):
        a = a.astype(np.float32) - a.mean()
        return a / (np.linalg.norm(a) + 1e-6)

    def img2d(m):
        return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.BOX), dtype=np.float32)

    def profile(m):
        p = Image.fromarray((m.sum(axis=0, keepdims=True) * (255.0 / max(1, m.shape[0]))).astype(np.uint8))
        p = np.asarray(p.resize((bins, 1), Image.BOX), dtype=np.float32)[0]
        k = np.ones(5) / 5  # làm mượt: chịu lệch vài ký tự
        return np.convolve(p, k, mode="same")

    prof = float((norm(profile(orig)) * norm(profile(rend))).sum())
    shape = float((norm(img2d(orig)) * norm(img2d(rend))).sum())
    return 0.7 * prof + 0.3 * shape


def fix_line(line: str, ink: np.ndarray | None, font, max_try: int = 6) -> tuple[str, list[str]]:
    """→ (dòng đã thêm dấu hướng nếu khớp ảnh hơn, [các cụm đã ép trái → phải]).

    ink: mặt nạ mực của đúng dòng này trên ảnh gốc (đã cắt sát); font: phông PIL (Raqm) của tài liệu."""
    segs = candidates(line)
    if getattr(font, "layout_engine", None) != 1:  # cần Raqm (ImageFont.Layout.RAQM) để dựng thử bidi
        return line, []
    if not segs or ink is None or ink.size == 0 or not ink.any():
        return line, []
    ys, xs = np.nonzero(ink)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    chosen: list[tuple[int, int]] = []
    best = _score(ink, _render(line, font))
    for seg in segs[:max_try]:  # tham lam từng cụm: mỗi cụm ảnh hưởng chủ yếu vùng của nó
        if any(seg[0] < b and a < seg[1] for a, b in chosen):  # không chồng lên cụm đã chọn
            continue
        trial = chosen + [seg]
        s = _score(ink, _render(apply(line, trial), font))
        if s > best + 0.02:
            best, chosen = s, trial
    return apply(line, chosen), [line[a:b] for a, b in chosen]
