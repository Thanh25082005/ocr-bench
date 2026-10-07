"""DOCX "giữ nguyên bố cục" (kiểu Exact Copy của ABBYY): mỗi khối của trang đặt ĐÚNG TOẠ ĐỘ trên trang.

Đầu vào mỗi trang: ảnh trang + các khối {category, bbox [x1,y1,x2,y2] (px trên ảnh đó), text} — từ dots.ocr/dots.mocr.

- Khổ trang = tỉ lệ ảnh, rộng 210 mm (scan A4 → đúng A4); lề 0 → toạ độ trang = toạ độ ảnh × hệ số.
- Mỗi khối = một hình neo theo TRANG, không chiếm chỗ dòng chảy (wrapNone) → không khối nào đẩy khối nào:
  · chữ / tiêu đề / danh sách / đầu-chân trang → hộp chữ (wps:txbx) đúng bbox; cỡ chữ ước theo khoảng cách dòng
    đếm trên ảnh; CHÈN SẴN chỗ xuống dòng (đo bằng Liberation Sans = cùng thông số Arial) để Word không dàn lại;
  · bảng → bảng trong hộp chữ, độ rộng cột đo từ ảnh (khe trắng giữa cột), chiều cao dòng theo dải chữ trên ảnh;
  · Picture (logo, chữ ký, con dấu) / Formula → ẢNH CẮT từ trang gốc, đúng kích thước.
- Trang không có khối (vd. lấy từ lớp chữ PDF) → đặt cả ảnh trang (giữ nguyên hình, không sửa được chữ).

Sửa được chữ TRONG từng khung; không dành cho soạn thảo lại nhiều (dùng DOCX "sửa được" cho việc đó).
"""

from __future__ import annotations

import html as _html
import io
import re
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

from .docx_export import _ARABIC, _STRONG, _cell_text, is_rtl, para_rtl

PAGE_W_MM = 210.0
EMU_PER_MM = 36000
EMU_PER_PT = 12700
TWIP_PER_PT = 20
FONT = "Arial"  # tên ghi vào DOCX: Word / Google Docs có sẵn; LibreOffice thay bằng Liberation Sans (cùng thông số)
# → đo bằng Liberation Sans = hiển thị ở mọi nơi; không cần nhúng phông Latin
FONT_AR = "Noto Sans Arabic"  # chữ Ả Rập (w:cs), cũng nhúng (SIL OFL)
_FONT_FILES = ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
               "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
               "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"]
_FONT_BOLD = ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"]
IMAGE_CATEGORIES = {"Picture", "Formula"}
HEADING_CATEGORIES = {"Title", "Section-header"}


# ------------------------------------------------------------------ đo chữ

_FONT_AR = ["/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]
_FONT_AR_BOLD = ["/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"]


# phông Ả Rập của tài liệu đang dựng (nhận dạng từ ảnh — ocrbench.arabic_fonts); None = FONT_AR mặc định
_AR_FAMILY: ContextVar[str | None] = ContextVar("ar_family", default=None)


def ar_family() -> str:
    return _AR_FAMILY.get() or FONT_AR


def _ar_files(family: str) -> tuple[list[str], list[str]]:
    from .arabic_fonts import font_files

    ff = font_files(family) if family != FONT_AR or not Path(_FONT_AR[0]).exists() else None
    if ff:
        return [str(ff[0])], [str(ff[1] or ff[0])]
    return _FONT_AR, _FONT_AR_BOLD


@lru_cache(maxsize=None)
def _font(bold: bool, size100: int, arabic: bool = False, family: str | None = None):
    """Phông để ĐO: chữ Latin = Liberation Sans (cùng thông số Arial); chữ Ả Rập = phông Ả Rập của tài liệu
    (nhận dạng từ ảnh) + ghép nét (Raqm)."""
    if arabic:
        reg, bd = _ar_files(family or FONT_AR)
        files = bd if bold else reg
    else:
        files = _FONT_BOLD if bold else _FONT_FILES
    engine = ImageFont.Layout.RAQM if arabic and _raqm() else ImageFont.Layout.BASIC
    for p in files:
        if Path(p).exists():
            return ImageFont.truetype(p, size100, layout_engine=engine)
    return ImageFont.load_default(size=size100)


@lru_cache(maxsize=None)
def _raqm() -> bool:
    from PIL import features

    return bool(features.check("raqm"))


def _font_for(text: str, bold: bool):
    arabic = para_rtl(text or "")
    return _font(bold, 100, arabic, ar_family() if arabic else None)


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Bề rộng chuỗi (cùng đơn vị với size) theo thông số Arial/Liberation Sans (chữ Ả Rập: phông Ả Rập)."""
    f = _font_for(text, bold)
    return f.getlength(text) * size / 100.0


def wrap(text: str, width: float, size: float, bold: bool = False) -> list[str]:
    """Ngắt dòng tham lam theo từ (như Word) — trả về các dòng."""
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            cand = w if not cur else cur + " " + w
            if not cur or text_width(cand, size, bold) <= width:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


# ------------------------------------------------------------------ đọc ảnh

def ink_mask(g: np.ndarray) -> np.ndarray:
    """Mực trong một vùng xám: ngưỡng THÍCH NGHI = giữa màu nền và màu mực đậm nhất của vùng (ảnh scan mờ / nén
    nét chỉ còn xám 100–140 → ngưỡng cố định 150 mất ~80% nét). Vùng không có tương phản → không có mực."""
    if g.size == 0:
        return np.zeros(g.shape, bool)
    bg = float(np.percentile(g, 75))
    dark = float(np.percentile(g, 0.5))
    if bg - dark < 40:
        return np.zeros(g.shape, bool)
    return g < min(200.0, (bg + dark) / 2)


def _bg_rows(a: np.ndarray) -> np.ndarray:
    """Màu nền từng hàng điểm ảnh (H×3): nền chung của vùng (màu phổ biến nhất); hàng nào phần lớn KHÔNG phải nền
    chung (dải màu: hàng tiêu đề nền đỏ…) thì lấy màu trung vị của chính hàng đó."""
    q = (a // 16).reshape(-1, 3).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    mode = np.bincount(keys).argmax()
    sel = keys == mode
    bg = np.median(a.reshape(-1, 3)[sel], axis=0) if sel.any() else np.array([255.0, 255.0, 255.0])
    af = a.astype(np.float32)
    near = np.linalg.norm(af - bg, axis=2) < 40
    rows = np.repeat(bg[None, :], a.shape[0], axis=0).astype(np.float32)
    med = np.median(af, axis=1)  # H×3
    uniform = (np.linalg.norm(af - med[:, None, :], axis=2) < 40).mean(axis=1) > 0.6
    cand = (near.mean(axis=1) < 0.35) & uniform
    # dải nền phải ĐỦ DÀY (≥ 8 px liền): dòng chân chữ Ả Rập (nét nối gần kín bề ngang) không phải dải nền
    need = min(8, max(2, a.shape[0] // 2))
    band = np.zeros_like(cand)
    y = 0
    while y < len(cand):
        if cand[y]:
            e = y
            while e < len(cand) and cand[e]:
                e += 1
            if e - y >= need:
                band[y:e] = True
            y = e
        else:
            y += 1
    if band.any():
        rows[band] = med[band]
    return rows


def ink_rgb(a: np.ndarray) -> np.ndarray:
    """Mực = điểm ảnh KHÁC MÀU NỀN rõ rệt (không phải "tối hơn ngưỡng"): chữ trắng trên nền đỏ, chữ màu, nét mờ
    trên giấy xám đều đúng; dải nền màu không bị coi là mực."""
    if a.size == 0:
        return np.zeros(a.shape[:2], bool)
    rows = _bg_rows(a)
    d = np.linalg.norm(a.astype(np.float32) - rows[:, None, :], axis=2)
    hi = float(np.percentile(d, 99.5))
    if hi < 50:  # không có tương phản
        return np.zeros(a.shape[:2], bool)
    return d > max(45.0, 0.45 * hi)


def _ink(img: Image.Image, box) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    a = np.asarray(img.convert("RGB").crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))), dtype=np.uint8)
    return ink_rgb(a)


def ink_colors(img: Image.Image, box) -> tuple[str | None, str | None]:
    """(màu chữ, màu nền) của vùng dạng hex — None khi gần đen / gần trắng (mặc định)."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    a = np.asarray(img.convert("RGB").crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))), dtype=np.uint8)
    if a.size == 0:
        return None, None
    m = ink_rgb(a)
    rows = _bg_rows(a)
    bg = np.median(rows, axis=0)
    fill = None if np.linalg.norm(bg - 255) < 12 else "".join(f"{int(v):02X}" for v in bg)  # giữ cả nền xám nhạt
    color = None
    if m.sum() > 20:
        d = np.linalg.norm(a.astype(np.float32) - rows[:, None, :], axis=2)
        core = m & (d >= np.percentile(d[m], 60))  # lõi nét (bỏ viền khử răng cưa pha màu nền)
        c = np.median(a[core], axis=0)
        sat = (c.max() - c.min()) / max(1.0, c.max())
        if c.max() > 90 and (sat > 0.25 or c.min() > 170):  # có màu rõ, hoặc chữ sáng (trắng) trên nền tối
            color = "".join(f"{int(v):02X}" for v in c)
    return color, fill


def text_bands(img: Image.Image, box) -> list[tuple[int, int]]:
    """Các dải dòng chữ (theo chiều dọc, toạ độ trong khối): [(đầu, cuối)]. Bỏ đường kẻ ngang dài."""
    ink = _ink(img, box)
    if ink.size == 0:
        return []
    w = ink.shape[1]
    rows = ink.sum(axis=1)
    rows = np.where(rows > 0.85 * w, 0, rows)  # đường kẻ ngang của bảng / gạch chân dài
    on = rows > 0
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, len(on)])
    merged = []
    for b in bands:  # nối các mảnh của cùng một dòng (dấu chấm, dấu phụ tiếng Ả Rập)
        if merged and b[0] - merged[-1][1] <= max(1, 0.15 * (b[1] - b[0])):
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    if not merged:
        return []
    hmax = max(b[1] - b[0] for b in merged)
    return [tuple(b) for b in merged if b[1] - b[0] >= max(2, 0.3 * hmax)]


def column_bounds(img: Image.Image, box, n_cols: int) -> list[float] | None:
    """Ranh giới cột của bảng (toạ độ x trong khối) từ các khe trắng dọc; None nếu không tìm đủ."""
    if n_cols <= 1:
        return None
    ink = _ink(img, box)
    h, w = ink.shape
    cols = ink.sum(axis=0)
    # bảng có kẻ dọc (kể cả xám nhạt): ranh giới = ĐÚNG đường kẻ (khi đủ số cột)
    inner_v = [x for x in table_rules(img, box)["v"] if 0.01 * w < x < 0.99 * w]
    if len(inner_v) == n_cols - 1:
        return [0.0] + inner_v + [float(w)]
    rule_x = np.nonzero(cols > 0.85 * h)[0]
    rules = []
    for x in rule_x:
        if rules and x - rules[-1][-1] <= 2:
            rules[-1].append(x)
        else:
            rules.append([x])
    inner = [float(np.mean(r)) for r in rules if 0.01 * w < np.mean(r) < 0.99 * w]
    if len(inner) == n_cols - 1:
        return [0.0] + inner + [float(w)]
    cols = np.where(cols > 0.85 * h, 0, cols)  # đường kẻ dọc → coi như trắng (là ranh giới)
    empty = cols == 0
    gaps, start = [], None
    for x, v in enumerate(empty):
        if v and start is None:
            start = x
        elif not v and start is not None:
            gaps.append((start, x))
            start = None
    edge = max(3, 0.02 * len(empty))  # bỏ khe ở mép (lề trong bảng, viền ngoài)
    gaps = [g for g in gaps if g[0] > edge and g[1] < len(empty) - edge]
    if len(gaps) < n_cols - 1:
        return None
    best = sorted(sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[: n_cols - 1])
    return [0.0] + [(a + b) / 2 for a, b in best] + [float(ink.shape[1])]


# ------------------------------------------------------------------ XML

def _esc(s: str) -> str:
    return _html.escape(s, quote=False)


def _run(text: str, size_pt: float, bold=False, rtl=False, color: str | None = None, sx: float = 1.0) -> str:
    """Cỡ chữ Word chỉ có bước 0,5 pt (~4% ở cỡ 12) → làm tròn cỡ chữ, phần chênh bề ngang bù bằng co giãn ngang
    w:w (bước 1%). Thứ tự phần tử theo lược đồ CT_RPr (Word khắt khe hơn LibreOffice)."""
    sz = max(2, int(round(size_pt * 2)))
    scale = int(round(100 * sx * size_pt * 2 / sz))  # bù luôn phần làm tròn cỡ chữ
    scale = min(600, max(1, scale))
    # đoạn tiếng Ả Rập: MỘT phông cho cả đoạn (kể cả số, chữ Latin) — trộn phông thì mỗi chương trình chia chiều cao
    # dòng theo phông khác nhau (LibreOffice lấy tỉ lệ của phông Latin) → nét chữ lệch dọc
    ar = ar_family()
    fa = ar if rtl else FONT
    rpr = (f'<w:rPr><w:rFonts w:ascii="{fa}" w:hAnsi="{fa}" w:cs="{ar}" w:eastAsia="{fa}"/>'
           f'{"<w:b/><w:bCs/>" if bold else ""}{f"<w:color w:val={chr(34)}{color}{chr(34)}/>" if color else ""}'
           f'{f"<w:w w:val={chr(34)}{scale}{chr(34)}/>" if scale != 100 else ""}'
           f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>{"<w:rtl/>" if rtl else ""}</w:rPr>')
    return f'<w:r>{rpr}<w:t xml:space="preserve">{_esc(text)}</w:t></w:r>'


def _br(size_pt: float) -> str:
    return f'<w:r><w:rPr><w:sz w:val="{int(round(size_pt * 2))}"/></w:rPr><w:br/></w:r>'


def _para(lines: list[list[tuple[str, bool]]], size_pt: float, line_pt: float, rtl: bool, align: str,
          ind_left: int = 0, ind_right: int = 0, color: str | None = None, sx: float = 1.0) -> str:
    """Một đoạn; mỗi dòng là danh sách (chữ, đậm); giữa các dòng chèn ngắt dòng cứng.
    ind_left / ind_right: thụt lề (twip) ở mép TRÁI / PHẢI thật trên trang. Đoạn bidi: w:left là mép ĐẦU dòng (= phải),
    w:right là mép cuối (= trái) — ECMA-376 §17.3.1.12, đã thử trên LibreOffice → đổi chỗ khi ghi."""
    jc = align  # đoạn bidi ("distribute": căn đều kể cả dòng cuối): "start" = phải, "end" = trái (LibreOffice hiểu "right" của đoạn bidi thành trái)
    if rtl and align == "right":
        jc = "start"
    elif rtl and align == "left":
        jc = "end"
    ppr = (f'<w:pPr>{"<w:bidi/>" if rtl else ""}<w:spacing w:before="0" w:after="0" '
           f'w:line="{max(20, int(round(line_pt * TWIP_PER_PT)))}" w:lineRule="exact"/>'
           f'<w:ind w:left="{max(0, ind_right if rtl else ind_left)}" w:right="{max(0, ind_left if rtl else ind_right)}"/>'
           f'<w:jc w:val="{jc}"/></w:pPr>')
    body = []
    for i, line in enumerate(lines):
        if i:
            body.append(_br(size_pt))
        body += [_run(t, size_pt, b, rtl, color, sx) for t, b in line if t]
    return f"<w:p>{ppr}{''.join(body)}</w:p>"


def font_metrics(text: str, bold: bool = False) -> tuple[float, float, float]:
    """(ascent, descent, khoảng từ đường ascent tới mép trên nét mực của chuỗi) — đơn vị em (× cỡ chữ)."""
    f = _font_for(text, bold)
    asc, desc = f.getmetrics()
    top = f.getbbox(text or "H")[1]
    return asc / 100.0, desc / 100.0, top / 100.0


def _line_paras(lines: list[list[tuple[str, bool]]], size_pt: float, pitch_pt: float, rtl: bool, align: str,
                ind_left: int = 0, ind_right: int = 0, natural_pt: float | None = None, justify: bool = False,
                color: str | None = None, sx: float = 1.0) -> str:
    """Mỗi dòng một đoạn, chiều cao dòng = chiều cao TỰ NHIÊN của phông (không có phần thừa để chương trình chia
    lên/xuống), khoảng cách dòng tạo bằng spacing-before → vị trí nét chữ như nhau trên Word, LibreOffice."""
    nat = natural_pt or size_pt * 1.117
    gap = pitch_pt - nat
    out = []
    for i, line in enumerate(lines):
        # đoạn căn đều: mọi dòng trừ dòng cuối giãn đủ bề rộng ("distribute" = căn đều cả dòng đơn)
        a = "distribute" if justify and i < len(lines) - 1 else align
        h = nat if (i == 0 or gap >= 0) else pitch_pt  # dòng gốc sát hơn chiều cao tự nhiên → dùng đúng khoảng gốc
        para = _para([line], size_pt, h, rtl, a, ind_left, ind_right, color, sx)
        if i and gap > 0:
            para = para.replace('w:before="0"', f'w:before="{int(round(gap * TWIP_PER_PT))}"', 1)
        out.append(para)
    return "".join(out)


def _segments(line: str) -> list[tuple[str, bool]]:
    """'**đậm** thường' → [('đậm', True), (' thường', False)]"""
    out = []
    for i, part in enumerate(re.split(r"\*\*", line)):
        if part:
            out.append((part, i % 2 == 1))
    return out or [("", False)]


def _anchor(inner_graphic: str, x, y, cx, cy, ident: int, z: int, name: str, behind: bool = False) -> str:
    return (
        f'<w:r><w:drawing><wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0" relativeHeight="{z}" '
        f'behindDoc="{1 if behind else 0}" locked="0" layoutInCell="1" allowOverlap="1"><wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{int(x)}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{int(y)}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{int(cx)}" cy="{int(cy)}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:wrapNone/>'
        f'<wp:docPr id="{ident}" name="{_esc(name)}"/><wp:cNvGraphicFramePr/>'
        f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">{inner_graphic}</a:graphic>'
        f'</wp:anchor></w:drawing></w:r>')


def _textbox_graphic(cx, cy, content_xml: str) -> str:
    return (
        '<a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"><wps:wsp>'
        '<wps:cNvSpPr txBox="1"/><wps:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{int(cx)}" cy="{int(cy)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></wps:spPr>'
        f'<wps:txbx><w:txbxContent>{content_xml}</w:txbxContent></wps:txbx>'
        '<wps:bodyPr rot="0" vert="horz" wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t" anchorCtr="0">'
        '<a:noAutofit/></wps:bodyPr></wps:wsp></a:graphicData>')


def _picture_graphic(rid: str, cx, cy, ident: int) -> str:
    return (
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{ident}" name="anh{ident}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{int(cx)}" cy="{int(cy)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData>')


# ------------------------------------------------------------------ nội dung khối

def plain_text(md: str) -> str:
    """Markdown/HTML của một khối chữ → chữ (giữ ** để in đậm, bỏ #, thẻ HTML)."""
    t = re.sub(r"<br\s*/?>", "\n", md or "", flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = _html.unescape(t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    lines = [" ".join(line.split()) for line in t.split("\n")]
    return "\n".join(line for line in lines if line)


def ink_ratio(text: str, bold: bool = False) -> float:
    """Chiều cao nét mực / cỡ chữ của chính chuỗi này (có chữ có đuôi g, y, p hay không → khác nhau)."""
    f = _font_for(text, bold)
    box = f.getbbox(text.replace("**", "") or "Hg")
    return max(0.3, (box[3] - box[1]) / 100.0)


def size_from_length(text: str, bold: bool, avail_w: float, n_lines: int, last_w: float) -> float:
    """Cỡ chữ (px) để đoạn văn chia đúng n dòng trong bề rộng avail_w và dòng cuối dài last_w (đo trên ảnh):
    tổng chiều dài chữ ≈ (n−1)·avail_w + last_w."""
    plain = text.replace("**", "").replace("\n", " ")
    w100 = max(1.0, _font_for(plain, bold).getlength(plain))
    return ((n_lines - 1) * avail_w + last_w) / w100 * 100.0


def size_from_ink(text: str, bold: bool, band_h: float, ink_w: float | None, n_lines: int) -> float:
    """Cỡ chữ (px ảnh) từ nét mực: dòng đơn → khớp CHIỀU DÀI dòng (quyết định chữ có tràn không),
    kẹp trong ±30% so với ước theo chiều cao; nhiều dòng → theo chiều cao nét."""
    plain = text.replace("**", "").replace("\n", " ")
    size_h = band_h / ink_ratio(plain, bold)
    if n_lines == 1 and ink_w:  # dòng đơn: luôn khớp chiều dài dòng (kẹp 0,5–1,2 lần ước theo chiều cao)
        f = _font_for(plain, bold)
        b = f.getbbox(plain or "x")
        w100 = max(1.0, b[2] - b[0])
        size_w = ink_w / w100 * 100.0
        return min(max(size_w, 0.5 * size_h), 1.6 * size_h)
    return size_h


def fit_text(text: str, box_w: float, box_h: float, n_lines: int, pitch: float | None, bold: bool,
             size0: float | None = None):
    """Chọn cỡ chữ (đơn vị px ảnh) + các dòng sao cho chữ nằm trong khối, đúng số dòng trên ảnh nếu được.
    → (cỡ chữ, khoảng cách dòng, các dòng)"""
    n_lines = max(1, n_lines)
    line_h = pitch if pitch else box_h / n_lines
    size = size0 if size0 else min(line_h / 1.15, box_h / 1.15)
    plain = text.replace("**", "")
    for _ in range(40):
        lines = wrap(plain, box_w, size, bold)
        if len(lines) <= n_lines and len(lines) * min(line_h, size * 1.25) <= box_h * 1.06:
            break
        size *= 0.96
    lines = wrap(plain, box_w, size, bold)
    spacing = line_h if len(lines) > 1 and len(lines) == n_lines else size * 1.15
    spacing = min(spacing, box_h / max(1, len(lines))) if len(lines) > 1 else max(spacing, size * 1.05)
    return size, spacing, lines


def _reapply_bold(original: str, lines: list[str]) -> list[list[tuple[str, bool]]]:
    """Giữ in đậm (**...**) sau khi đã ngắt dòng trên chữ không có dấu **."""
    flags = []
    bold = False
    i = 0
    while i < len(original):
        if original.startswith("**", i):
            bold = not bold
            i += 2
            continue
        flags.append((original[i], bold))
        i += 1
    plain = "".join(c for c, _ in flags)
    out, pos = [], 0
    for line in lines:
        j = plain.find(line, pos) if line else pos
        if j < 0:
            out.append([(line, False)])
            continue
        segs, cur, cur_b = [], "", None
        for c, b in flags[j:j + len(line)]:
            if cur_b is None or b == cur_b:
                cur += c
                cur_b = b
            else:
                segs.append((cur, cur_b))
                cur, cur_b = c, b
        if cur:
            segs.append((cur, bool(cur_b)))
        out.append(segs)
        pos = j + len(line)
    return out


def table_rows(table_html: str):
    """HTML bảng → ma trận ô [(chữ, colspan, rowspan)] theo từng dòng."""
    from lxml import html as lh

    doc = lh.document_fromstring(f"<html><body>{table_html}</body></html>")
    rows = []
    for tr in doc.body.xpath(".//tr"):
        cells = []
        for td in tr.xpath("./td|./th"):
            def num(attr):
                try:
                    return max(1, int(td.get(attr, 1)))
                except ValueError:
                    return 1
            cells.append((_cell_text(td), num("colspan"), num("rowspan"), td.tag == "th"))
        if cells:
            rows.append(cells)
    return rows


def _grid(rows):
    """Đặt ô vào lưới có rowspan/colspan → (số cột, [(dòng, cột, chữ, colspan, rowspan, tiêu đề)], ô bị gộp dọc)"""
    taken, placed = set(), []
    for r, cells in enumerate(rows):
        c = 0
        for text, cs, rs, th in cells:
            while (r, c) in taken:
                c += 1
            placed.append((r, c, text, cs, rs, th))
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            c += cs
    n_cols = max((c + cs for _, c, _, cs, _, _ in placed), default=1)
    return n_cols, placed, taken


def _ink_extent_x(img, box) -> tuple[float, float] | None:
    """(trái, phải) của mực trong vùng (toạ độ trong vùng), bỏ đường kẻ dọc / ngang dài."""
    ink = _ink(img, box)
    if ink.size == 0:
        return None
    h, w = ink.shape
    ink = ink & ~(ink.sum(axis=1, keepdims=True) > 0.85 * w) & ~(ink.sum(axis=0, keepdims=True) > 0.85 * h)
    xs = np.nonzero(ink.any(axis=0))[0]
    if len(xs) == 0:
        return None
    return float(xs.min()), float(xs.max() + 1)


def table_rules(img, box) -> dict:
    """Đường kẻ của bảng, kể cả kẻ XÁM NHẠT (ngưỡng mực chữ bỏ qua): vị trí ngang / dọc (toạ độ trong khối), màu, độ dày."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    a = np.asarray(img.convert("RGB").crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))), dtype=np.float32)
    if a.size == 0:
        return {"h": [], "v": [], "color": None, "px": 1}
    g = a.mean(axis=2)
    bg = float(np.percentile(g, 90))
    dark = g < bg - 25
    h, w = dark.shape

    def group(idx, cover):
        out = []
        for i in idx:
            if out and i - out[-1][-1] <= 1:
                out[-1].append(i)
            else:
                out.append([i])
        return [(float(np.mean(gp)), len(gp)) for gp in out]

    # đường kẻ phải MẢNH: dải nền tối dày (hàng tiêu đề) không phải đường kẻ
    hl = [(p, n) for p, n in group(np.nonzero(dark.mean(axis=1) > 0.6)[0], None) if n <= max(6, 0.01 * h)]
    vl = [(p, n) for p, n in group(np.nonzero(dark.mean(axis=0) > 0.6)[0], None) if n <= max(6, 0.01 * w)]
    color, px = None, 1
    pix = []
    for y, n in hl[:4]:
        pix.append(a[int(y), :][dark[int(y), :]])
    for x, n in vl[:4]:
        pix.append(a[:, int(x)][dark[:, int(x)]])
    pix = [p for p in pix if len(p)]
    if pix:
        c = np.median(np.concatenate(pix), axis=0)
        color = "".join(f"{int(v):02X}" for v in c)
        px = int(np.median([n for _, n in hl + vl]))
    return {"h": [y for y, _ in hl], "v": [x for x, _ in vl], "color": color, "px": max(1, px), "size": (w, h)}


def _row_cuts(img, box, bands, n_rows: int) -> list[float]:
    """Ranh giới các dòng bảng (toạ độ trong khối): đường kẻ ngang nếu đủ, không thì khe trắng LỚN NHẤT giữa các dải
    chữ (ô nhiều dòng vẫn đúng), cuối cùng mới chia đều."""
    x1, y1, x2, y2 = box
    bh = y2 - y1
    inner = [m for m in table_rules(img, box)["h"] if 0.02 * bh < m < 0.98 * bh]
    if len(inner) == n_rows - 1:  # kẻ ngang giữa các dòng
        return [0.0] + inner + [float(bh)]
    if len(bands) >= n_rows and n_rows > 1:
        gaps = sorted(range(len(bands) - 1), key=lambda k: -(bands[k + 1][0] - bands[k][1]))[:n_rows - 1]
        cuts = sorted((bands[k][1] + bands[k + 1][0]) / 2 for k in gaps)
        return [0.0] + cuts + [float(bh)]
    return [bh * k / n_rows for k in range(n_rows + 1)]


def _ink_box(img, box) -> tuple[float, float, float, float] | None:
    """Hộp mực (toạ độ trang) trong vùng, bỏ đường kẻ ngang / dọc dài (viền ô)."""
    ink = _ink(img, box)
    if ink.size == 0:
        return None
    h, w = ink.shape

    def near_lines(cover, n):  # đường kẻ (+ 2 px hai bên: viền khử răng cưa của đường kẻ)
        m = cover > 0.5 * n
        for k in (1, 2):
            m = m | np.roll(m, k) | np.roll(m, -k)
        return m

    ink = ink & ~near_lines(ink.sum(axis=1), w)[:, None] & ~near_lines(ink.sum(axis=0), h)[None, :]
    ys, xs = np.nonzero(ink)
    if len(xs) < 4:
        return None
    x0, y0 = int(round(box[0])), int(round(box[1]))
    return (x0 + float(xs.min()), y0 + float(ys.min()), x0 + float(xs.max() + 1), y0 + float(ys.max() + 1))


def table_spec(img, box, table_html: str) -> dict:
    """Mọi thông số đặt một bảng (px ảnh): cột theo THỨ TỰ HIỂN THỊ (trái → phải), dòng, cỡ chữ, từng ô (vùng, hộp mực
    gốc, kiểu căn). Bảng phải → trái (bidiVisual): ô đầu tiên của dòng trong HTML là cột NGOÀI CÙNG BÊN PHẢI."""
    rows = table_rows(table_html)
    x1, y1, x2, y2 = box
    if not rows:
        return {"kind": "table", "box": box, "cells": [], "empty": plain_text(table_html)}
    n_rows = len(rows)
    n_cols, placed, _ = _grid(rows)
    bw, bh = x2 - x1, y2 - y1
    bounds = column_bounds(img, box, n_cols) or [bw * k / n_cols for k in range(n_cols + 1)]
    col_w = [bounds[k + 1] - bounds[k] for k in range(n_cols)]  # theo thứ tự hiển thị
    bands = text_bands(img, box)
    cuts = _row_cuts(img, box, bands, n_rows)
    row_h = [cuts[k + 1] - cuts[k] for k in range(n_rows)]
    texts = " ".join(t for _, _, t, _, _, _ in placed)
    size = min(row_h) / 1.6  # tạm, tính lại từ mực từng ô bên dưới

    def vis(c, cs, rtl):  # cột logic → cột hiển thị bắt đầu (trái)
        return n_cols - c - cs if rtl else c

    def cell_boxes(rtl):
        out = []
        for r, c, text, cs, rs, th in placed:
            v = vis(c, cs, rtl)
            region = (x1 + bounds[v], y1 + cuts[r], x1 + bounds[v + cs], y1 + cuts[min(n_rows, r + rs)])
            out.append((region, _ink_box(img, region)))
        return out

    # hướng bảng: mặc định theo chữ (Ả Rập → phải → trái), kiểm chứng bằng ảnh: bề rộng chữ dự đoán ↔ bề rộng mực đo
    def agree(rtl):
        err = 0.0
        for (r, c, text, cs, rs, th), (_, ib) in zip(placed, cell_boxes(rtl)):
            t = text.split("\n")[0].strip()
            if t and ib:
                err += abs(np.log(max(1.0, text_width(t, size, th)) / max(1.0, ib[2] - ib[0])))
            elif t or ib:
                err += 1.0
        return err

    rtl = is_rtl(texts)

    def size_from_cells(rtl_):  # trung vị cỡ chữ ước từ chiều cao mực của các ô một dòng
        est = []
        for (r, c, text, cs, rs, th), (_, ib) in zip(placed, cell_boxes(rtl_)):
            t = text.strip()
            if t and ib and "\n" not in t:
                est.append((ib[3] - ib[1]) / ink_ratio(t, th))
        return float(np.median(est)) if est else size

    size = min(size_from_cells(rtl), min(row_h) / 1.05)
    if n_cols > 1 and agree(not rtl) < 0.8 * agree(rtl):
        rtl = not rtl
        size = min(size_from_cells(rtl), min(row_h) / 1.05)
    cells = []
    for (r, c, text, cs, rs, th), (region, ib) in zip(placed, cell_boxes(rtl)):
        cell_rtl = para_rtl(text) if _STRONG.search(text) else rtl
        cw = region[2] - region[0]
        align = "right" if cell_rtl else "left"
        if ib and text.strip():
            lg, rg = ib[0] - region[0], region[2] - ib[2]
            if lg > 0.06 * cw and rg > 0.06 * cw and abs(lg - rg) < 0.3 * max(lg, rg):
                align = "center"
            elif rg < lg:
                align = "right"
            else:
                align = "left"
        bidi = {}
        if cell_rtl and ib and text.strip() and "\n" not in text:
            from .arabic_bidi import fix_line

            g = np.asarray(img.convert("L").crop(tuple(int(round(v)) for v in ib)))
            fixed, forced = fix_line(text, ink_mask(g), _font(th, 64, True, ar_family()))
            if forced:
                bidi[text] = fixed
        color, fill = ink_colors(img, region)
        cells.append({"kind": "cell", "r": r, "c": c, "cs": cs, "rs": rs, "text": text, "th": th, "rtl": cell_rtl,
                      "bidi": bidi, "color": color, "fill": fill,
                      "box": region, "ink": ib if text.strip() else None, "align": align, "fs": 1.0, "dx": 0.0,
                      "dy": 0.0, "justify": False, "segs": [[(text.split("\n")[0], th)]]})
    # cỡ chữ chung: mọi ô vừa bề rộng cột (cho phép xuống dòng nhưng không quá chiều cao dòng)
    for _ in range(30):
        ok = True
        for cl in cells:
            w = cl["box"][2] - cl["box"][0]
            h = cl["box"][3] - cl["box"][1]
            n = sum(len(wrap(line, w * 0.98, size, cl["th"])) for line in (cl["text"].split("\n") or [""]))
            a_, d_, _ = font_metrics(cl["text"] or "H", cl["th"])
            if n * size * (a_ + d_) > h * 1.02:
                ok = False
                break
        if ok:
            break
        size *= 0.95
    for cl in cells:  # số dòng chữ THẬT trong ô (giữ đúng số dòng khi dựng, kể cả khi co giãn ngang)
        cl["n_lines"] = max(1, len(text_bands(img, cl["box"]))) if cl["ink"] else 1
    # chữ sát mép trên ô: dời ranh giới dòng lên (dòng > 0) hoặc thêm lề trên cho cả bảng (dòng 0) — khoảng trên
    # của đoạn không thể âm
    def need_top(cl):
        t = cl["text"].split("\n")[0] or "H"
        return cl["ink"][1] - font_metrics(t, cl["th"])[2] * size - 1
    pad_top = 0.0
    for r in range(n_rows):
        starts = [cl for cl in cells if cl["r"] == r and cl["ink"]]
        if not starts:
            continue
        need = min(need_top(cl) for cl in starts) - y1  # toạ độ trong khối
        if r == 0:
            pad_top = max(0.0, -need)
        elif need < cuts[r]:
            above = [cl["ink"][3] - y1 for cl in cells if cl["r"] + cl["rs"] == r and cl["ink"]]
            cuts[r] = max(need, (max(above) + 1) if above else cuts[r - 1] + 1)
    row_h = [cuts[k + 1] - cuts[k] for k in range(n_rows)]
    for cl in cells:  # vùng ô theo ranh giới dòng mới
        bx1, _, bx2, _ = cl["box"]
        cl["box"] = (bx1, y1 + cuts[cl["r"]] - (pad_top if cl["r"] == 0 else 0), bx2,
                     y1 + cuts[min(n_rows, cl["r"] + cl["rs"])])
    if pad_top:
        row_h[0] += pad_top
    rl = table_rules(img, box)
    near = lambda vals, at, tol: any(abs(v - at) <= tol for v in vals)  # noqa: E731
    tol_y, tol_x = max(4.0, 0.02 * bh), max(4.0, 0.01 * bw)
    inner_h = [v for v in rl["h"] if tol_y < v < bh - tol_y]
    inner_v = [v for v in rl["v"] if tol_x < v < bw - tol_x]
    borders = {"top": near(rl["h"], 0, tol_y), "bottom": near(rl["h"], bh, tol_y),
               "left": near(rl["v"], 0, tol_x), "right": near(rl["v"], bw, tol_x),
               "insideH": n_rows > 1 and len(inner_h) >= (n_rows - 1) / 2,
               "insideV": n_cols > 1 and len(inner_v) >= (n_cols - 1) / 2,
               "color": rl["color"] or "000000", "px": rl["px"]}
    return {"kind": "table", "box": box, "n_rows": n_rows, "n_cols": n_cols, "rtl": rtl, "col_w": col_w,
            "row_h": row_h, "size": size, "borders": borders, "cells": cells, "pad_top": pad_top}


def _cell_xml(cl: dict, size: float, scale_pt: float, w_px: float, color: str | None) -> str:
    """Nội dung một ô: các dòng chữ, căn + thụt lề sao cho mực rơi đúng toạ độ gốc; khoảng trên = đúng vị trí dọc."""
    tw = lambda px: int(round(max(0.0, px) * scale_pt * TWIP_PER_PT))  # noqa: E731
    th, rtl = cl["th"], cl["rtl"]
    sz = size * cl["fs"]
    x1, y1, x2, y2 = cl["box"]
    ib = cl["ink"]
    if not cl["text"].strip():
        return _para([[("", False)]], sz * scale_pt, sz * scale_pt, rtl, "left")
    al = cl["align"]
    il = ir = 0.0
    if ib:
        if al == "left":
            il = ib[0] - x1 + cl["dx"]
        elif al == "right":
            ir = x2 - ib[2] - cl["dx"]
        else:  # giữa: lệch tâm = (il - ir) / 2
            off = ((ib[0] + ib[2]) / 2 - (x1 + x2) / 2 + cl["dx"]) * 2
            il, ir = max(0.0, off), max(0.0, -off)
    avail = max(4.0, w_px - il - ir)
    lines = []
    sx = cl.get("sx", 1.0)
    if cl.get("n_lines", 1) == 1 and "\n" not in cl["text"]:
        lines = [cl["text"]]  # ô một dòng trên ảnh → một dòng (co giãn ngang không được làm xuống dòng)
    else:
        for line in cl["text"].split("\n"):
            lines += wrap(line, avail * 1.02 / sx, sz, th)
    if rtl:
        lines = [cl.get("bidi", {}).get(ln, ln) for ln in lines]
    asc, desc, top = font_metrics(lines[0] if lines else "H", th)
    nat = (asc + desc) * sz
    before = 0.0
    if ib:
        before = ib[1] - y1 - top * sz + cl["dy"]  # mép trên nét chữ = mép trên mực gốc trong ô
    segs = [[(ln, th)] for ln in lines]
    xml = _line_paras(segs, sz * scale_pt, nat * scale_pt, rtl, al, tw(il), tw(ir), nat * scale_pt, False, color, sx)
    if before > 0:
        xml = xml.replace('w:before="0"', f'w:before="{tw(before)}"', 1)
    return xml


def table_xml_spec(spec: dict, scale_pt: float, colors: dict | None = None,
                   float_at: tuple[float, float] | None = None) -> str:
    """Bảng (trong hộp chữ) từ kế hoạch: lưới cột cố định, dòng cao ĐÚNG như ảnh, mỗi ô đặt chữ theo mực gốc."""
    if not spec["cells"]:
        return _para([[(spec.get("empty", ""), False)]], 8, 9, False, "left")
    tw = lambda px: int(round(max(0.0, px) * scale_pt * TWIP_PER_PT))  # noqa: E731
    n_cols, rtl = spec["n_cols"], spec["rtl"]
    col_w = spec["col_w"]
    logical_w = col_w[::-1] if rtl else col_w  # bidiVisual: cột lưới đầu tiên là cột bên PHẢI
    b = spec["borders"]
    if not isinstance(b, dict):  # kế hoạch cũ (True/False)
        b = {k: bool(b) for k in ("top", "left", "bottom", "right", "insideH", "insideV")} | {"color": "000000", "px": 1}
    eighths = max(2, min(96, int(round(b["px"] * scale_pt * 8))))  # độ dày theo 1/8 pt
    tbl_borders = "".join(
        f'<w:{x} w:val="single" w:sz="{eighths}" w:space="0" w:color="{b["color"]}"/>' if b.get(x) else f'<w:{x} w:val="nil"/>'
        for x in ("top", "left", "bottom", "right", "insideH", "insideV"))
    out = [f'<w:tbl><w:tblPr>{_tblp(*float_at, scale_pt) if float_at else ""}'
           f'{"<w:bidiVisual/>" if rtl else ""}<w:tblW w:w="{tw(sum(col_w))}" w:type="dxa"/>'
           f'<w:tblBorders>{tbl_borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
           '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar></w:tblPr>'
           "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{tw(w)}"/>' for w in logical_w) + "</w:tblGrid>"]
    by_row: dict[int, list] = {}
    for cl in spec["cells"]:
        by_row.setdefault(cl["r"], []).append(cl)
    size = spec["size"]
    empty = _para([[("", False)]], size * scale_pt, size * scale_pt, False, "left")
    for r in range(spec["n_rows"]):
        out.append(f'<w:tr><w:trPr><w:trHeight w:val="{tw(spec["row_h"][r])}" w:hRule="exact"/></w:trPr>')
        c_expect = 0
        for cl in sorted(by_row.get(r, []), key=lambda x: x["c"]):
            c, cs, rs = cl["c"], cl["cs"], cl["rs"]
            while c_expect < c:  # ô bị gộp dọc từ dòng trên
                out.append(f'<w:tc><w:tcPr><w:tcW w:w="{tw(logical_w[c_expect])}" w:type="dxa"/><w:vMerge/></w:tcPr>'
                           f"{empty}</w:tc>")
                c_expect += 1
            w = sum(logical_w[c:c + cs])
            tcpr = (f'<w:tcPr><w:tcW w:w="{tw(w)}" w:type="dxa"/>'
                    f'{f"<w:gridSpan w:val={chr(34)}{cs}{chr(34)}/>" if cs > 1 else ""}'
                    f'{"<w:vMerge w:val=" + chr(34) + "restart" + chr(34) + "/>" if rs > 1 else ""}'
                    '<w:vAlign w:val="top"/></w:tcPr>')
            if colors is None and cl.get("fill"):  # nền ô (không dùng khi hiệu chỉnh: lẫn với màu đo)
                tcpr = tcpr.replace('<w:vAlign', f'<w:shd w:val="clear" w:color="auto" w:fill="{cl["fill"]}"/><w:vAlign')
            color = colors.get(id(cl)) if colors else cl.get("color")
            out.append(f"<w:tc>{tcpr}{_cell_xml(cl, size, scale_pt, w, color)}</w:tc>")
            c_expect = c + cs
        while c_expect < n_cols:
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{tw(logical_w[c_expect])}" w:type="dxa"/><w:vMerge/></w:tcPr>'
                       f"{empty}</w:tc>")
            c_expect += 1
        out.append("</w:tr>")
    out.append("</w:tbl>")
    if not float_at:
        out.append(_para([[("", False)]], 1, 1, False, "left"))  # Word cần một đoạn sau bảng trong hộp chữ
    return "".join(out)


def table_xml(table_html: str, img, box, scale_pt: float) -> str:
    """Bảng trong hộp chữ (không hiệu chỉnh) — giữ cho mã cũ / thử nghiệm."""
    return table_xml_spec(table_spec(img, box, table_html), scale_pt)


# ------------------------------------------------------------------ nhúng phông

_EMBED = {FONT: ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
          FONT_AR: ("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
                    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf")}


def _embed_set() -> dict:
    """Phông nhúng cho tài liệu đang dựng: Liberation Sans + phông Ả Rập của tài liệu."""
    fam = ar_family()
    reg, bd = _ar_files(fam)
    return {fam: (next((p for p in reg if Path(p).exists()), reg[0]),
                  next((p for p in bd if Path(p).exists()), bd[0]))}


def _obfuscate(data: bytes, guid: str) -> bytes:
    """ECMA-376 §17.8.1: XOR 32 byte đầu của phông với khoá lấy từ GUID (đảo thứ tự byte)."""
    hexs = guid.strip("{}").replace("-", "")
    key = bytes(int(hexs[i:i + 2], 16) for i in range(0, 32, 2))[::-1]
    head = bytes(b ^ key[i % 16] for i, b in enumerate(data[:32]))
    return head + data[32:]


# CT_Settings: các phần tử ĐỨNG TRƯỚC embedTrueTypeFonts (ECMA-376 §17.15.1.78). Sai thứ tự → Word báo file hỏng
# (LibreOffice thì bỏ qua) — đã kiểm bằng Open XML SDK (OpenXmlValidator, Office 2010 / 2019).
_SETTINGS_BEFORE_EMBED = ("writeProtection", "view", "zoom", "removePersonalInformation", "removeDateAndTime",
                          "doNotDisplayPageBoundaries", "displayBackgroundShape", "printPostScriptOverText",
                          "printFractionalCharacterWidth", "printFormsData")


def _settings_embed(xml: bytes) -> bytes:
    """Thêm <w:embedTrueTypeFonts/> vào settings.xml ĐÚNG VỊ TRÍ theo lược đồ."""
    from lxml import etree

    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    root = etree.fromstring(xml)
    if root.find(f"{{{W}}}embedTrueTypeFonts") is not None:
        return xml
    el = etree.Element(f"{{{W}}}embedTrueTypeFonts")
    pos = 0
    for i, child in enumerate(root):
        if isinstance(child.tag, str) and etree.QName(child).localname in _SETTINGS_BEFORE_EMBED:
            pos = i + 1
    root.insert(pos, el)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def embed_fonts(docx_path, fonts: dict | None = None) -> list[str]:
    """Nhúng phông TrueType vào DOCX (Word, LibreOffice đọc được) → hiển thị đúng thông số đã dùng để đo/ngắt dòng."""
    import uuid
    import zipfile

    fonts = fonts or _EMBED
    src = Path(docx_path)
    with zipfile.ZipFile(src) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    font_xml, rels, ctypes, done, k = [], [], [], [], 0
    for name, files in fonts.items():
        entries = []
        for kind, path in zip(("embedRegular", "embedBold"), files):
            if not Path(path).exists():
                continue
            k += 1
            guid = "{" + str(uuid.uuid4()).upper() + "}"
            parts[f"word/fonts/font{k}.odttf"] = _obfuscate(Path(path).read_bytes(), guid)
            rels.append(f'<Relationship Id="rIdF{k}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                        f'relationships/font" Target="fonts/font{k}.odttf"/>')
            entries.append(f'<w:{kind} r:id="rIdF{k}" w:fontKey="{guid}"/>')
        if entries:
            font_xml.append(f'<w:font w:name="{name}"><w:charset w:val="00"/><w:family w:val="swiss"/>'
                            f'<w:pitch w:val="variable"/>{"".join(entries)}</w:font>')
            done.append(name)
    if not done:
        return []
    ft = parts["word/fontTable.xml"].decode("utf-8")
    parts["word/fontTable.xml"] = ft.replace("</w:fonts>", "".join(font_xml) + "</w:fonts>").encode("utf-8")
    parts["word/_rels/fontTable.xml.rels"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/'
        'package/2006/relationships">' + "".join(rels) + "</Relationships>").encode("utf-8")
    ct = parts["[Content_Types].xml"].decode("utf-8")
    if 'Extension="odttf"' not in ct:
        ct = ct.replace("<Default ", '<Default Extension="odttf" ContentType="application/vnd.openxmlformats-'
                        'officedocument.obfuscatedFont"/><Default ', 1)
    parts["[Content_Types].xml"] = ct.encode("utf-8")
    parts["word/settings.xml"] = _settings_embed(parts["word/settings.xml"])
    tmp = src.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in parts.items():
            z.writestr(n, data)
    tmp.replace(src)
    return done


# ------------------------------------------------------------------ khối chữ: kế hoạch + XML

def text_spec(img, box, text: str, cat: str) -> dict:
    """Mọi thông số đặt một khối chữ (đơn vị px ảnh). Hiệu chỉnh chỉ sửa fs (hệ số cỡ chữ), dx, dy."""
    x1, y1, x2, y2 = box
    original = plain_text(text)
    bold = cat in HEADING_CATEGORIES
    rtl = para_rtl(original.replace("**", ""))
    bands = text_bands(img, box)
    n = len(bands) or 1
    pitch = (bands[-1][0] - bands[0][0]) / (n - 1) if n > 1 else None
    ext = _ink_extent_x(img, box)
    band_h = float(np.median([b1 - b0 for b0, b1 in bands])) if bands else (y2 - y1) * 0.7
    size0 = size_from_ink(original, bold, band_h, (ext[1] - ext[0]) if ext else None, n)
    if n > 1 and bands and ext:  # nhiều dòng: theo tổng chiều dài chữ (dòng cuối đo trên ảnh)
        last = _ink_extent_x(img, (x1, y1 + bands[-1][0], x2, y1 + bands[-1][1]))
        if last:
            s_len = size_from_length(original, bold, ext[1] - ext[0], n, last[1] - last[0])
            size_h = band_h / ink_ratio(original.replace("**", ""), bold)
            size0 = min(max(s_len, 0.6 * size_h), 1.6 * size_h)
    bw = x2 - x1
    lg, rg = (ext[0], bw - ext[1]) if ext else (0.0, 0.0)
    centered = bool(ext and lg > 0.06 * bw and rg > 0.06 * bw and abs(lg - rg) < 0.3 * max(lg, rg))
    if centered:
        align, box_w = "center", bw
    elif ext and (rg <= lg + 0.02 * bw if rtl else rg < lg - 0.02 * bw):
        # chữ dồn phải; chữ phủ kín bề ngang → theo hướng viết (Ả Rập: phải, Latin: trái)
        align, box_w = ("start" if rtl else "right"), bw - rg
    else:
        align, box_w = ("end" if rtl else "left"), bw - lg
    size, spacing, lines = fit_text(original, box_w, y2 - y1, n, pitch, bold, size0)
    bidi_forced = []
    if rtl:
        from .arabic_bidi import fix_line

        fixed = []
        gray = None
        for i, t in enumerate(lines):
            t = t.replace("**", "")
            if len(lines) == n and bands:  # dòng i ↔ dải mực i trên ảnh → chọn thứ tự số / Latin khớp ảnh
                if gray is None:
                    gray = np.asarray(img.convert("L"))
                b0, b1 = bands[i]
                ink = ink_mask(gray[int(y1 + b0):int(y1 + b1) + 1, int(x1):int(round(x2))])
                t, forced = fix_line(t, ink, _font(bold, 64, True, ar_family()))
                bidi_forced += forced
            fixed.append(t)
        segs = [[(t, bold)] for t in fixed]
    else:
        segs = _reapply_bold(original if not bold else original.replace("**", ""), lines)
        if bold:
            segs = [[(t, True) for t, _ in line] for line in segs]
    first = "".join(t for t, _ in segs[0]) if segs else original
    asc, desc, top = font_metrics(first, bold)
    justify = False
    if n > 1 and len(lines) == n and bands and not centered:
        edges = [_ink_extent_x(img, (x1, y1 + b0, x2, y1 + b1)) for b0, b1 in bands[:-1]]
        if all(edges):
            full = [e[1] for e in edges] if not rtl else [e[0] for e in edges]
            ref = max(full) if not rtl else min(full)
            justify = all(abs(v - ref) < 0.012 * bw for v in full)
    color, _ = ink_colors(img, box)
    ink = None  # hộp mực gốc (toạ độ trang) — đích để hiệu chỉnh
    if ext and bands:
        ink = (x1 + ext[0], y1 + bands[0][0], x1 + ext[1], y1 + bands[-1][1])
    # mỗi dòng phải VỪA chỗ chứa (đo từng đoạn theo đúng đậm / thường): dòng dài hơn chỗ chứa thì chương trình tự
    # xuống dòng (vd. dòng căn đều có phần **đậm** — đo cả dòng bằng chữ thường thì thiếu) → co ngang cho vừa
    margin = 0.04 * bw
    if centered:
        avail = bw + 2 * margin
    elif justify:
        avail = bw - lg - rg
    else:
        avail = bw + margin - (rg if align in ("right", "start") else lg)
    need = max((sum(text_width(t, size, b) for t, b in line if t) for line in segs), default=0.0)
    sx0 = min(1.0, 0.995 * avail / need) if need > 0 else 1.0
    return {"box": (x1, y1, x2, y2), "segs": segs, "size": size, "em": asc + desc, "top": top, "n_src": n, "sx": sx0,
            "pitch": pitch if (pitch and len(lines) == n) else None, "spacing": spacing, "align": align, "rtl": rtl,
            "justify": justify, "lg": lg, "rg": rg, "ink": ink, "fs": 1.0, "dx": 0.0, "dy": 0.0,
            "ink_top": y1 + (bands[0][0] if bands else 0), "bidi_forced": bidi_forced, "color": color}


_NIL_BORDERS = "".join(f'<w:{x} w:val="nil"/>' for x in ("top", "left", "bottom", "right", "insideH", "insideV"))
_ZERO_MAR = ('<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
             '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>')
# đoạn rỗng tí hon ngăn cách các bảng nổi (hai bảng liền nhau sẽ bị gộp thành một) — cao 1 pt
_SPACER = ('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/>'
           '<w:rPr><w:sz w:val="2"/><w:szCs w:val="2"/></w:rPr></w:pPr></w:p>')


def _tblp(x_px: float, y_px: float, scale_pt: float) -> str:
    """Vị trí bảng nổi theo TRANG (twip, được âm). Google Docs (từ 2023), Word, LibreOffice đều giữ toạ độ này —
    khác khung chữ (text box) mà Google Docs bỏ mất."""
    t = lambda px: int(round(px * scale_pt * TWIP_PER_PT))  # noqa: E731
    return (f'<w:tblpPr w:leftFromText="0" w:rightFromText="0" w:topFromText="0" w:bottomFromText="0" '
            f'w:vertAnchor="page" w:horzAnchor="page" w:tblpX="{t(x_px)}" w:tblpY="{t(y_px)}"/>'
            '<w:tblOverlap w:val="overlap"/>')


def float_cell(x_px: float, y_px: float, w_px: float, h_px: float, content: str, scale_pt: float) -> str:
    """Một khối = bảng nổi MỘT ô, không viền, lề ô 0, đặt đúng toạ độ trang."""
    t = lambda px: int(round(max(0.0, px) * scale_pt * TWIP_PER_PT))  # noqa: E731
    return (f'<w:tbl><w:tblPr>{_tblp(x_px, y_px, scale_pt)}<w:tblW w:w="{t(w_px)}" w:type="dxa"/>'
            f'<w:tblBorders>{_NIL_BORDERS}</w:tblBorders><w:tblLayout w:type="fixed"/>{_ZERO_MAR}</w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{t(w_px)}"/></w:tblGrid>'
            f'<w:tr><w:trPr><w:trHeight w:val="{t(h_px)}" w:hRule="atLeast"/></w:trPr>'
            f'<w:tc><w:tcPr><w:tcW w:w="{t(w_px)}" w:type="dxa"/><w:vAlign w:val="top"/></w:tcPr>{content}</w:tc>'
            '</w:tr></w:tbl>')


def text_xml(spec: dict, k: float, scale_pt: float, color: str | None = None) -> str:
    x1, y1, x2, y2 = spec["box"]
    bw = x2 - x1
    size = spec["size"] * spec["fs"]
    natural = spec["em"] * size
    n = len(spec["segs"])
    pitch = spec["pitch"] or max(spec["spacing"] * spec["fs"], natural)
    margin = 0.04 * bw  # khung rộng thêm hai bên: chữ hơi rộng hơn dự đoán cũng không bị khung cắt / ngắt dòng
    tw = lambda px: int(round(max(0.0, px) * scale_pt * TWIP_PER_PT))  # noqa: E731
    al = spec["align"]
    if al == "center":
        il = ir = 0
    elif spec["justify"]:  # căn đều: giãn đúng tới hai mép GỐC (không nới), nếu không dòng bị kéo quá mép phải
        il, ir = tw(spec["lg"] + margin), tw(spec["rg"] + margin)
    elif al in ("right", "start"):  # lề an toàn ở phía KHÔNG căn
        il, ir = 0, tw(spec["rg"] + margin)
    else:
        il, ir = tw(spec["lg"] + margin), 0
    content = _line_paras(spec["segs"], size * scale_pt, pitch * scale_pt, spec["rtl"], al, il, ir,
                          natural * scale_pt, spec["justify"], color or spec.get("color"), spec.get("sx", 1.0))
    box_y = spec["ink_top"] - spec["top"] * size + spec["dy"]  # mép trên nét chữ dòng đầu trùng ảnh gốc
    box_h = (n - 1) * pitch + natural + 0.3 * size
    return float_cell(x1 - margin + spec["dx"], box_y, bw + 2 * margin, box_h, content, scale_pt)


# ------------------------------------------------------------------ hiệu chỉnh bằng LibreOffice

def find_soffice(hint: str | None = None) -> str | None:
    """LibreOffice để dựng thử: tham số, biến OCRBENCH_SOFFICE, PATH, hoặc bản AppImage đã giải nén."""
    import glob
    import os
    import shutil

    for c in (hint, os.environ.get("OCRBENCH_SOFFICE"), shutil.which("soffice"), shutil.which("libreoffice")):
        if c and c != "auto" and Path(c).exists():
            return c
    for pat in ("/kaggle/working/libreoffice/squashfs-root/opt/libreoffice*/program/soffice",
                "/opt/libreoffice*/program/soffice"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def _palette(n: int) -> list[tuple[str, float]]:
    """n màu bão hoà, sắc độ cách xa nhau (tỉ lệ vàng) → (hex, sắc độ 0..1)."""
    import colorsys

    out = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 0.72)
        out.append((f"{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}", h))
    return out


def _render_pages(soffice: str, docx: Path, sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    import os
    import subprocess
    import tempfile

    import pypdfium2 as pdfium

    import shutil

    outdir = Path(tempfile.mkdtemp(prefix="exact_cal_"))  # hồ sơ LibreOffice riêng → chạy song song không đụng nhau
    try:
        subprocess.run([soffice, f"-env:UserInstallation=file://{outdir}/lo_profile", "--headless", "--convert-to",
                        "pdf", "--outdir", str(outdir), str(docx)], check=True, capture_output=True, timeout=300,
                       env={**os.environ, "HOME": str(outdir)})
        pdf = pdfium.PdfDocument(str(outdir / (docx.stem + ".pdf")))
        pages = []
        for i, (w, h) in enumerate(sizes):
            if i >= len(pdf):
                break
            pw, _ = pdf[i].get_size()
            im = pdf[i].render(scale=w / pw).to_pil().convert("RGB").resize((w, h))
            pages.append(np.asarray(im))
        pdf.close()
        return pages
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def _items(specs: list[dict]) -> list[dict]:
    """Những thứ được đo / sửa khi hiệu chỉnh: khối chữ + từng ô bảng có chữ."""
    out = []
    for s in specs:
        if s.get("kind") == "table":
            out += [c for c in s["cells"] if c["ink"]]
        else:
            out.append(s)
    return out


def _measure(rgb: np.ndarray, specs: list[dict], pal: dict) -> dict:
    """Hộp mực đã dựng của từng khối, nhận theo MÀU riêng của khối (không lẫn khối bên cạnh)."""
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    ink = (sat > 0.35) & (mn < 0.85)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    out = {}
    H, W = ink.shape

    def region(s):
        x1, y1, x2, y2 = s["box"]
        if s.get("kind") == "cell":  # ô bảng: lệch ít, vùng hẹp
            pad_x, pad_y = 0.1 * (x2 - x1) + 6, 0.3 * (y2 - y1) + 4
        else:
            pad_x, pad_y = 0.06 * (x2 - x1) + 12, 0.35 * (y2 - y1) + 8  # vừa đủ cho lệch, không trùm sang dòng khác
        return (int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y)), int(min(W, x2 + pad_x)), int(min(H, y2 + pad_y)))

    regs = {id(s): region(s) for s in specs}
    for s in specs:
        _, h0 = pal[id(s)]
        X1, Y1, X2, Y2 = regs[id(s)]
        near = [abs(pal[id(o)][1] - h0) for o in specs if o is not s and not (
            regs[id(o)][2] <= X1 or regs[id(o)][0] >= X2 or regs[id(o)][3] <= Y1 or regs[id(o)][1] >= Y2)]
        near = [min(d, 1 - d) for d in near]
        tol = min(0.02, max(0.004, 0.45 * min(near))) if near else 0.02  # màu lân cận gần → dung sai hẹp lại
        dh = np.abs(hue[Y1:Y2, X1:X2] - h0)
        dh = np.minimum(dh, 1 - dh)
        m = ink[Y1:Y2, X1:X2] & (dh < tol)
        ys, xs = np.nonzero(m)
        if len(xs) > 5:
            out[id(s)] = (X1 + xs.min(), Y1 + ys.min(), X1 + xs.max() + 1, Y1 + ys.max() + 1)
    return out


def _correct(s: dict, got, apply: bool = True) -> float:
    """Ghi độ lệch của trạng thái hiện tại (fs, dx, dy) vào lịch sử, rồi (nếu apply) sửa theo hộp mực đã dựng.
    → độ lệch lớn nhất (px) của trạng thái trước khi sửa."""
    o = s["ink"]
    if not o or not got:
        return 0.0
    dev = max(abs(o[i] - got[i]) for i in range(4))
    s.setdefault("hist", []).append((dev, s.get("sx", 1.0), s["dx"], s["dy"]))
    if not apply:
        return dev
    ow, gw = o[2] - o[0], got[2] - got[0]
    if gw > 4 and not s["justify"]:
        lo, hi = (0.85, 1.18) if s.get("kind") == "cell" else (0.7, 1.4)
        ratio = min(hi, max(lo, ow / gw))
        if abs(ratio - 1) > 0.004:
            s["sx"] = min(1.6, max(0.6, s.get("sx", 1.0) * ratio))
    al = s["align"]
    if al == "center":
        s["dx"] += (o[0] + o[2]) / 2 - (got[0] + got[2]) / 2
    elif al in ("right", "start"):
        s["dx"] += o[2] - got[2]
    else:
        s["dx"] += o[0] - got[0]
    s["dy"] += o[1] - got[1]
    return dev


# ------------------------------------------------------------------ dựng tài liệu

def _components(m: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Các mảnh liền (8 hướng) của mặt nạ → [(x1, y1, x2, y2, toạ độ điểm)] (không cần scipy)."""
    H, W = m.shape
    seen = np.zeros_like(m, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(m)):
        if seen[y0, x0]:
            continue
        stack, pts = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        p = np.array(pts)
        out.append((p[:, 1].min(), p[:, 0].min(), p[:, 1].max() + 1, p[:, 0].max() + 1, p))
    return out


def residual_layer(img: Image.Image, boxes: list, text_boxes: list | None = None, table_boxes: list | None = None
                   ) -> tuple[Image.Image, tuple[int, int]] | None:
    """Nét mực NẰM NGOÀI mọi khối của model (đường gạch ký tên, kẻ chấm, khung, họa tiết, thứ model bỏ sót) → một
    ảnh trong suốt (RGBA) đặt dưới chữ, đúng toạ độ.
    - ngưỡng riêng (khác nền > 60): nhận cả nét xám nhạt mà ngưỡng mực chữ bỏ qua;
    - lọc nhiễu theo MẬT ĐỘ (giữ đường chấm, bỏ chấm lẻ);
    - bỏ mảnh nét DÍNH mép một khối chữ (phần chữ bị khung model cắt — chữ đã được dựng lại bằng phông)."""
    from PIL import ImageFilter

    a = np.asarray(img.convert("RGB"))
    rows = _bg_rows(a)
    d = np.linalg.norm(a.astype(np.float32) - rows[:, None, :], axis=2)
    m = d > 60
    full = m.copy()
    H, W = m.shape
    for x1, y1, x2, y2 in boxes:
        m[max(0, int(y1) - 2):min(H, int(y2) + 3), max(0, int(x1) - 2):min(W, int(x2) + 3)] = False

    def thin_runs(cov, thr, max_t=4):  # nhóm hàng (cột) liền nhau có độ phủ cao nhưng MẢNH → đường kẻ
        keep = np.zeros(len(cov), bool)
        i = 0
        while i < len(cov):
            if cov[i] > thr:
                e = i
                while e < len(cov) and cov[e] > thr:
                    e += 1
                # mảnh VÀ tách biệt: các hàng (cột) ngay trên / dưới gần như trống (nét ngang của chữ thì không)
                around = [cov[k] for k in (i - 3, i - 2, e + 1, e + 2) if 0 <= k < len(cov)]
                if e - i <= max_t and (not around or max(around) < 0.35 * float(np.mean(cov[i:e]))):
                    keep[i:e] = True
                i = e
            else:
                i += 1
        return keep

    def longest_run(v, gap=6):  # đoạn liền dài nhất, coi khe ≤ gap px là liền (đường kẻ chấm)
        best = cur = last = 0
        started = False
        for i, x in enumerate(v):
            if x:
                cur = cur + (i - last) if started and i - last <= gap + 1 else 1
                started, last = True, i
                best = max(best, cur)
        return best

    def rule_mask(f, axis):  # hàng (cột) là đường kẻ: mảnh, tách biệt VÀ liền mạch ≥ 15% chiều dài trang
        cov = f.mean(axis=1 - axis) if axis == 0 else f.mean(axis=0)
        cand = thin_runs(cov, 0.15)
        out = np.zeros(len(cov), bool)
        n = f.shape[1] if axis == 0 else f.shape[0]
        for i in np.nonzero(cand)[0]:
            v = f[i, :] if axis == 0 else f[:, i]
            if longest_run(v) >= 0.15 * n:  # mảnh chữ thẳng hàng (đầu dòng Ả Rập) cách nhau cả khoảng dòng → loại
                out[i] = True
        return out

    # đường kẻ mảnh chạy dài (kể cả kẻ chấm) giữ NGUYÊN, cả đoạn nằm dưới khung khối của model
    lines = np.zeros_like(m)
    lines[rule_mask(full, 0), :] = True
    lines[:, rule_mask(full, 1)] = True
    m |= full & lines
    for x1, y1, x2, y2 in table_boxes or []:  # viền bảng do bảng DOCX tự vẽ → không giữ đường kẻ gốc (kẻ đôi)
        m[max(0, int(y1) - 3):min(H, int(y2) + 4), max(0, int(x1) - 3):min(W, int(x2) + 4)] = False
    dens = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.BoxBlur(2)))
    m &= dens >= 30  # ≥ 3 điểm mực trong ô 5×5
    if m.sum() < 60:
        return None
    for x1, y1, x2, y2, pts in _components(m & ~lines):  # điểm thuộc đường kẻ không bao giờ là "mảnh chữ"
        for bx1, by1, bx2, by2 in text_boxes or []:
            bh = by2 - by1
            tol = max(4.0, 0.3 * bh)  # đuôi chữ (g, y, nét tay) thò ra dưới / trên khung tới ~1/3 chiều cao khối
            touch = (x1 <= bx2 + tol and x2 >= bx1 - tol and y1 <= by2 + tol and y2 >= by1 - tol)
            small = (y2 - y1) <= 1.3 * bh and (x2 - x1) <= 2.0 * bh  # mảnh chữ có thể cao bằng cả dòng
            if touch and small:
                m[pts[:, 0], pts[:, 1]] = False
                break
    if m.sum() < 60:
        return None
    ys, xs = np.nonzero(m)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.zeros((y2 - y1, x2 - x1, 4), np.uint8)
    rgba[..., :3] = a[y1:y2, x1:x2]
    rgba[..., 3] = m[y1:y2, x1:x2] * 255
    return Image.fromarray(rgba, "RGBA"), (int(x1), int(y1))


def detect_arabic_font(pages: list[tuple[Image.Image, list[dict]]]) -> tuple[str | None, dict]:
    """Phông Ả Rập của tài liệu: lấy các khối chữ Ả Rập MỘT dòng (chữ OCR ↔ đúng một dải mực), so mẫu với thư viện."""
    from .arabic_fonts import identify

    lines = []
    for img, blocks in pages:
        rgb = img.convert("RGB")
        gray = None
        for b in blocks or []:
            cat, bbox = b.get("category") or "Text", b.get("bbox")
            if cat in IMAGE_CATEGORIES or cat == "Table" or not bbox or len(bbox) != 4:
                continue
            t = plain_text(b.get("text") or "").replace("**", "").strip()
            if "\n" in t or not (_ARABIC.search(t) and is_rtl(t) and para_rtl(t)):
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bands = text_bands(rgb, (x1, y1, x2, y2))
            if len(bands) != 1:
                continue
            if gray is None:
                gray = np.asarray(rgb.convert("L"))
            m = ink_mask(gray[int(y1 + bands[0][0]):int(y1 + bands[0][1]) + 1, int(x1):int(round(x2))])
            lines.append((m, t))
    return identify(lines)


def build_exact_docx(pages: list[tuple[Image.Image, list[dict]]], out_path, title: str | None = None,
                     calibrate: str | bool | None = "auto", rounds: int = 2, arabic_font: str | None = "auto") -> dict:
    """pages: [(ảnh trang, [khối {category, bbox, text}])] → ghi DOCX; trả về thống kê.

    calibrate: "auto" = có LibreOffice thì hiệu chỉnh (dựng thử, đo từng khối theo màu riêng, sửa cỡ chữ + vị trí,
    lặp `rounds` vòng); None/False = không; đường dẫn soffice = dùng bản đó.
    arabic_font: "auto" = nhận dạng phông Ả Rập từ ảnh (ocrbench.arabic_fonts); tên phông = dùng phông đó;
    None = phông mặc định (FONT_AR)."""
    fam, scores = (detect_arabic_font(pages) if arabic_font == "auto" else (arabic_font, {}))
    token = _AR_FAMILY.set(fam)
    try:
        stats = _build_exact(pages, out_path, title, calibrate, rounds)
    finally:
        _AR_FAMILY.reset(token)
    stats["arabic_font"] = fam or FONT_AR
    if scores:
        stats["arabic_font_scores"] = dict(list(scores.items())[:3])
    return stats


def _build_exact(pages, out_path, title, calibrate, rounds) -> dict:
    out_path = Path(out_path)
    soffice = find_soffice(calibrate if isinstance(calibrate, str) else None) if calibrate else None
    specs_by_page: list[list[dict]] = []
    stats = _build_once(pages, out_path, title, None, specs_by_page)
    if soffice and any(specs_by_page):
        sizes = [im.size for im, _ in pages]
        items_by_page = [_items(ps) for ps in specs_by_page]
        all_items = [s for ps in items_by_page for s in ps]
        pal = {id(s): c for s, c in zip(all_items, _palette(len(all_items)))}
        history = []
        for _ in range(rounds + 1):
            tmp = out_path.with_suffix(".cal.docx")
            _build_once(pages, tmp, title, {k: v[0] for k, v in pal.items()}, specs_by_page)
            try:
                rendered = _render_pages(soffice, tmp, sizes)
            except Exception as e:  # không dựng thử được → giữ bản chưa hiệu chỉnh
                stats["calibration"] = f"lỗi: {type(e).__name__}: {e}"[:200]
                break
            finally:
                tmp.unlink(missing_ok=True)
            devs, worst = [], []
            for pi, ps in enumerate(items_by_page):
                if pi >= len(rendered):
                    continue
                got = _measure(rendered[pi], ps, pal)
                mm = PAGE_W_MM / sizes[pi][0]
                for s in ps:
                    d = _correct(s, got.get(id(s)), apply=len(history) < rounds) * mm
                    devs.append(d)
            history.append(round(float(max(devs)), 2) if devs else 0.0)
        # mỗi khối giữ trạng thái TỐT NHẤT đã đo (hiệu chỉnh không bao giờ làm khối nào tệ hơn ban đầu)
        for pi, ps in enumerate(items_by_page):
            mm = PAGE_W_MM / sizes[pi][0]
            for s in ps:
                if s.get("hist"):
                    dev, s["sx"], s["dx"], s["dy"] = min(s["hist"], key=lambda h: h[0])
                    worst.append((round(float(dev) * mm, 2), pi + 1, "".join(t for t, _ in s["segs"][0])[:30]))
        worst.sort(reverse=True)
        stats["calibration"] = {"soffice": True, "max_dev_mm_by_round": history,
                                "max_dev_mm_final": worst[0][0] if worst else 0.0, "worst": worst[:3]}
        stats.update({k: v for k, v in _build_once(pages, out_path, title, None, specs_by_page).items()
                      if k != "calibration"})
    elif calibrate:
        stats["calibration"] = "không có LibreOffice — chưa hiệu chỉnh"
    return stats


def _build_once(pages, out_path, title, colors, specs_by_page: list) -> dict:
    """Dựng DOCX một lần. specs_by_page rỗng → lập kế hoạch khối chữ; có sẵn → dùng lại (đã hiệu chỉnh)."""
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    from docx.shared import Emu

    doc = Document()
    if title:
        doc.core_properties.title = title
    ns = nsdecls("w", "wp", "a", "r", "pic") + ' xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
    stats = {"pages": 0, "text": 0, "table": 0, "image": 0, "whole_page_image": 0}
    ident = 1
    reuse = bool(specs_by_page)
    body_par = doc.paragraphs[0] if doc.paragraphs else doc.add_paragraph()
    for pi, (img, blocks) in enumerate(pages):
        img = img.convert("RGB")
        W, H = img.size
        page_w = PAGE_W_MM * EMU_PER_MM
        k = page_w / W  # EMU / px
        scale_pt = k / EMU_PER_PT  # pt / px
        section = doc.sections[0] if pi == 0 else doc.add_section(WD_SECTION.NEW_PAGE)
        section.page_width, section.page_height = Emu(int(page_w)), Emu(int(H * k))
        for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin", "header_distance",
                     "footer_distance", "gutter"):
            setattr(section, attr, Emu(0))
        par = body_par if pi == 0 else doc.add_paragraph()
        z = 251658240
        if reuse:
            specs = specs_by_page[pi]
            spec_iter = iter(specs)
        else:
            specs = []
            specs_by_page.append(specs)
        if not blocks:
            blocks = [{"category": "Picture", "bbox": [0, 0, W, H], "text": ""}]
            stats["whole_page_image"] += 1
        elif colors is None:
            boxes = [b["bbox"] for b in blocks if b.get("bbox") and len(b["bbox"]) == 4]
            tboxes = [b["bbox"] for b in blocks if b.get("bbox") and len(b["bbox"]) == 4
                      and (b.get("category") or "Text") not in IMAGE_CATEGORIES and (b.get("text") or "").strip()]
            tables = [b["bbox"] for b in blocks if b.get("bbox") and len(b["bbox"]) == 4
                      and ((b.get("category") or "") == "Table" or "<table" in (b.get("text") or "").lower())]
            res = residual_layer(img, boxes, tboxes, tables)
            if res is not None:
                rim, (rx, ry) = res
                buf = io.BytesIO()
                rim.save(buf, format="PNG", optimize=True)
                buf.seek(0)
                rid, _ = doc.part.get_or_add_image(buf)
                ident += 1
                graphic = _picture_graphic(rid, rim.width * k, rim.height * k, ident)
                xml = _anchor(graphic, rx * k, ry * k, rim.width * k, rim.height * k, ident, z, "Nét ngoài khối",
                              behind=True)
                par._p.append(parse_xml(xml.replace("<w:r>", f"<w:r {ns}>", 1)))
                stats["residual"] = stats.get("residual", 0) + 1
        for b in blocks:
            bbox = b.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            cat = b.get("category") or "Text"
            text = b.get("text") or ""
            ident += 1
            z += 1
            if cat in IMAGE_CATEGORIES or (not text.strip() and cat != "Table"):
                buf = io.BytesIO()
                img.crop((int(x1), int(y1), int(round(x2)), int(round(y2)))).save(buf, format="PNG")
                buf.seek(0)
                rid, _ = doc.part.get_or_add_image(buf)
                graphic = _picture_graphic(rid, (x2 - x1) * k, (y2 - y1) * k, ident)
                stats["image"] += 1
                xml = _anchor(graphic, x1 * k, y1 * k, (x2 - x1) * k, (y2 - y1) * k, ident, z, f"Ảnh {ident}")
            elif cat == "Table" or "<table" in text.lower():
                if reuse:
                    spec = next(spec_iter)
                else:
                    m = re.search(r"<table.*?</table>", text, flags=re.S | re.I)
                    spec = table_spec(img, (x1, y1, x2, y2), m.group(0) if m else text)
                    specs.append(spec)
                stats["table"] += 1
                pt = spec.get("pad_top", 0.0)
                xml = table_xml_spec(spec, scale_pt, colors, float_at=(x1, y1 - pt))
            else:
                if reuse:
                    spec = next(spec_iter)
                else:
                    spec = text_spec(img, (x1, y1, x2, y2), text, cat)
                    spec["kind"] = "text"
                    specs.append(spec)
                spec.update(ident=ident, z=z)
                stats["text"] += 1
                xml = text_xml(spec, k, scale_pt, color=colors.get(id(spec)) if colors else None)
            if xml.startswith("<w:tbl>"):  # bảng nổi: phần tử thân văn bản, trước đoạn neo của trang + đoạn ngăn
                par._p.addprevious(parse_xml(xml.replace("<w:tbl>", f"<w:tbl {ns}>", 1)))
                par._p.addprevious(parse_xml(_SPACER.replace("<w:p>", f"<w:p {ns}>", 1)))
            else:
                par._p.append(parse_xml(xml.replace("<w:r>", f"<w:r {ns}>", 1)))
        stats["pages"] += 1
    doc.save(out_path)
    stats["fonts"] = embed_fonts(out_path, _embed_set())
    return stats
