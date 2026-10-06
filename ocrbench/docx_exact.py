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
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

from .docx_export import _ARABIC, _cell_text, is_rtl

PAGE_W_MM = 210.0
EMU_PER_MM = 36000
EMU_PER_PT = 12700
TWIP_PER_PT = 20
FONT = "Liberation Sans"  # NHÚNG vào DOCX (SIL OFL) → đo = hiển thị trên mọi máy (cùng thông số Arial)
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


@lru_cache(maxsize=None)
def _font(bold: bool, size100: int, arabic: bool = False):
    """Phông để ĐO: chữ Latin = Liberation Sans (cùng thông số Arial); chữ Ả Rập = phông có chữ Ả Rập + ghép nét (Raqm)."""
    files = (_FONT_AR_BOLD if bold else _FONT_AR) if arabic else (_FONT_BOLD if bold else _FONT_FILES)
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
    return _font(bold, 100, bool(_ARABIC.search(text or "")) and is_rtl(text or ""))


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

def _ink(img: Image.Image, box) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    g = np.asarray(img.convert("L").crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))), dtype=np.uint8)
    return g < 150


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
    h = ink.shape[0]
    cols = ink.sum(axis=0)
    cols = np.where(cols > 0.85 * h, 0, cols)  # đường kẻ dọc → coi như trắng (là ranh giới)
    empty = cols == 0
    gaps, start = [], None
    for x, v in enumerate(empty):
        if v and start is None:
            start = x
        elif not v and start is not None:
            gaps.append((start, x))
            start = None
    gaps = [g for g in gaps if g[0] > 0 and g[1] < len(empty)]  # bỏ khe ở mép
    if len(gaps) < n_cols - 1:
        return None
    best = sorted(sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[: n_cols - 1])
    return [0.0] + [(a + b) / 2 for a, b in best] + [float(ink.shape[1])]


# ------------------------------------------------------------------ XML

def _esc(s: str) -> str:
    return _html.escape(s, quote=False)


def _run(text: str, size_pt: float, bold=False, rtl=False) -> str:
    sz = max(2, int(round(size_pt * 2)))
    # đoạn tiếng Ả Rập: MỘT phông cho cả đoạn (kể cả số, chữ Latin) — trộn phông thì mỗi chương trình chia chiều cao
    # dòng theo phông khác nhau (LibreOffice lấy tỉ lệ của phông Latin) → nét chữ lệch dọc
    fa = FONT_AR if rtl else FONT
    rpr = (f'<w:rPr><w:rFonts w:ascii="{fa}" w:hAnsi="{fa}" w:cs="{FONT_AR}" w:eastAsia="{fa}"/>'
           f'{"<w:b/><w:bCs/>" if bold else ""}{"<w:rtl/>" if rtl else ""}'
           f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>')
    return f'<w:r>{rpr}<w:t xml:space="preserve">{_esc(text)}</w:t></w:r>'


def _br(size_pt: float) -> str:
    return f'<w:r><w:rPr><w:sz w:val="{int(round(size_pt * 2))}"/></w:rPr><w:br/></w:r>'


def _para(lines: list[list[tuple[str, bool]]], size_pt: float, line_pt: float, rtl: bool, align: str,
          ind_left: int = 0, ind_right: int = 0) -> str:
    """Một đoạn; mỗi dòng là danh sách (chữ, đậm); giữa các dòng chèn ngắt dòng cứng. ind_*: thụt lề (twip)."""
    jc = align  # đoạn bidi ("distribute": căn đều kể cả dòng cuối): "start" = phải, "end" = trái (LibreOffice hiểu "right" của đoạn bidi thành trái)
    if rtl and align == "right":
        jc = "start"
    elif rtl and align == "left":
        jc = "end"
    ppr = (f'<w:pPr>{"<w:bidi/>" if rtl else ""}<w:spacing w:before="0" w:after="0" '
           f'w:line="{max(20, int(round(line_pt * TWIP_PER_PT)))}" w:lineRule="exact"/>'
           f'<w:ind w:left="{max(0, ind_left)}" w:right="{max(0, ind_right)}"/><w:jc w:val="{jc}"/></w:pPr>')
    body = []
    for i, line in enumerate(lines):
        if i:
            body.append(_br(size_pt))
        body += [_run(t, size_pt, b, rtl) for t, b in line if t]
    return f"<w:p>{ppr}{''.join(body)}</w:p>"


def font_metrics(text: str, bold: bool = False) -> tuple[float, float, float]:
    """(ascent, descent, khoảng từ đường ascent tới mép trên nét mực của chuỗi) — đơn vị em (× cỡ chữ)."""
    f = _font_for(text, bold)
    asc, desc = f.getmetrics()
    top = f.getbbox(text or "H")[1]
    return asc / 100.0, desc / 100.0, top / 100.0


def _line_paras(lines: list[list[tuple[str, bool]]], size_pt: float, pitch_pt: float, rtl: bool, align: str,
                ind_left: int = 0, ind_right: int = 0, natural_pt: float | None = None, justify: bool = False) -> str:
    """Mỗi dòng một đoạn, chiều cao dòng = chiều cao TỰ NHIÊN của phông (không có phần thừa để chương trình chia
    lên/xuống), khoảng cách dòng tạo bằng spacing-before → vị trí nét chữ như nhau trên Word, LibreOffice."""
    nat = natural_pt or size_pt * 1.117
    gap = pitch_pt - nat
    out = []
    for i, line in enumerate(lines):
        # đoạn căn đều: mọi dòng trừ dòng cuối giãn đủ bề rộng ("distribute" = căn đều cả dòng đơn)
        a = "distribute" if justify and i < len(lines) - 1 else align
        h = nat if (i == 0 or gap >= 0) else pitch_pt  # dòng gốc sát hơn chiều cao tự nhiên → dùng đúng khoảng gốc
        para = _para([line], size_pt, h, rtl, a, ind_left, ind_right)
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


def _anchor(inner_graphic: str, x, y, cx, cy, ident: int, z: int, name: str) -> str:
    return (
        f'<w:r><w:drawing><wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0" relativeHeight="{z}" '
        f'behindDoc="0" locked="0" layoutInCell="1" allowOverlap="1"><wp:simplePos x="0" y="0"/>'
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


def table_xml(table_html: str, img, box, scale_pt: float) -> str:
    """Bảng trong hộp chữ: độ rộng cột từ ảnh, chiều cao dòng theo dải chữ, cỡ chữ vừa ô."""
    rows = table_rows(table_html)
    if not rows:
        return _para([[(plain_text(table_html), False)]], 8, 9, False, "left")
    n_rows = len(rows)
    n_cols, placed, _ = _grid(rows)
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    bounds = column_bounds(img, box, n_cols) or [bw * k / n_cols for k in range(n_cols + 1)]
    col_w = [bounds[k + 1] - bounds[k] for k in range(n_cols)]
    bands = text_bands(img, box)
    if len(bands) == n_rows:  # mỗi dải chữ là một dòng bảng → ranh giới dòng = giữa hai dải
        cuts = [0.0] + [(bands[k][1] + bands[k + 1][0]) / 2 for k in range(n_rows - 1)] + [float(bh)]
    else:
        cuts = [bh * k / n_rows for k in range(n_rows + 1)]
    row_h = [cuts[k + 1] - cuts[k] for k in range(n_rows)]
    glyph = np.median([b[1] - b[0] for b in bands]) if bands else min(row_h) * 0.6
    size = min(glyph / 0.78, min(row_h) / 1.25)
    pad = 0.06 * size
    for _ in range(30):  # mọi ô vừa bề rộng cột (cho phép xuống dòng nhưng không quá chiều cao dòng)
        ok = True
        for r, c, text, cs, rs, th in placed:
            w = sum(col_w[c:c + cs]) - 2 * pad
            h = sum(row_h[r:r + rs])
            n = sum(len(wrap(line, w, size, th)) for line in (text.split("\n") or [""]))
            a_, d_, _ = font_metrics(text or "H", th)
            if n * size * (a_ + d_) > h * 1.02:
                ok = False
                break
        if ok:
            break
        size *= 0.95
    tw = lambda px: int(round(px * scale_pt * TWIP_PER_PT))  # noqa: E731
    size_pt = size * scale_pt
    borders = any(len(_ink(img, (x1, y1 + int(cuts[k]) - 2, x2, y1 + int(cuts[k]) + 2)).shape) and
                  _ink(img, (x1, y1 + int(cuts[k]) - 2, x2, y1 + int(cuts[k]) + 2)).sum(axis=1).max() > 0.6 * bw
                  for k in range(1, n_rows))
    bd = '<w:{s} w:val="single" w:sz="4" w:space="0" w:color="000000"/>' if borders else '<w:{s} w:val="nil"/>'
    tbl_borders = "".join(bd.format(s=s) for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    rtl = is_rtl(" ".join(t for _, _, t, _, _, _ in placed))
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{tw(bw)}" w:type="dxa"/>{"<w:bidiVisual/>" if rtl else ""}'
           f'<w:tblLayout w:type="fixed"/><w:tblBorders>{tbl_borders}</w:tblBorders>'
           f'<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="{tw(pad)}" w:type="dxa"/>'
           f'<w:bottom w:w="0" w:type="dxa"/><w:right w:w="{tw(pad)}" w:type="dxa"/></w:tblCellMar></w:tblPr>'
           "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{tw(w)}"/>' for w in col_w) + "</w:tblGrid>"]
    by_row: dict[int, list] = {}
    for p in placed:
        by_row.setdefault(p[0], []).append(p)
    for r in range(n_rows):
        out.append(f'<w:tr><w:trPr><w:trHeight w:val="{tw(row_h[r])}" w:hRule="atLeast"/></w:trPr>')
        cells = sorted(by_row.get(r, []), key=lambda p: p[1])
        c_expect = 0
        for _, c, text, cs, rs, th in cells:
            while c_expect < c:  # ô bị gộp dọc từ dòng trên
                out.append(f'<w:tc><w:tcPr><w:tcW w:w="{tw(col_w[c_expect])}" w:type="dxa"/><w:vMerge/></w:tcPr>'
                           f"{_para([[('', False)]], size_pt, size_pt * 1.15, False, 'left')}</w:tc>")
                c_expect += 1
            w = sum(col_w[c:c + cs])
            lines = []
            for line in text.split("\n"):
                lines += wrap(line, w - 2 * pad, size, th)
            cell_rtl = bool(_ARABIC.search(text))
            # vị trí chữ THẬT trong ô (đo mực trên ảnh) → thụt lề để chữ bắt đầu / kết thúc đúng toạ độ gốc
            cx1, cx2 = x1 + bounds[c], x1 + bounds[c + cs]
            ext = _ink_extent_x(img, (cx1, y1 + cuts[r], cx2, y1 + cuts[min(n_rows, r + rs)]))
            ind_l = ind_r = 0
            if ext and text.strip():
                left_gap, right_gap = ext[0] - pad, (cx2 - cx1) - ext[1] - pad
                if cell_rtl or right_gap < left_gap * 0.5:  # chữ dồn về phải (số, tiếng Ả Rập)
                    align, ind_r = "right", tw(max(0.0, right_gap))
                else:
                    align, ind_l = "left", tw(max(0.0, left_gap))
                if ind_l and lines and max(text_width(ln, size, th) for ln in lines) + ind_l / scale_pt / TWIP_PER_PT > w - 2 * pad:
                    ind_l = 0  # không đủ chỗ thì thôi thụt lề (tránh xuống dòng thừa)
            else:
                align = "left"
            tcpr = (f'<w:tcPr><w:tcW w:w="{tw(w)}" w:type="dxa"/>{f"<w:gridSpan w:val={chr(34)}{cs}{chr(34)}/>" if cs > 1 else ""}'
                    f'{"<w:vMerge w:val=" + chr(34) + "restart" + chr(34) + "/>" if rs > 1 else ""}<w:vAlign w:val="center"/></w:tcPr>')
            a_, d_, _ = font_metrics(text or "H", th)
            nat_pt = size_pt * (a_ + d_)  # chiều cao dòng tự nhiên của phông thật trong ô
            out.append(f"<w:tc>{tcpr}{_para([[(ln, th)] for ln in lines] or [[('', False)]], size_pt, nat_pt, cell_rtl, align, ind_l, ind_r)}</w:tc>")
            c_expect = c + cs
        while c_expect < n_cols:
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{tw(col_w[c_expect])}" w:type="dxa"/><w:vMerge/></w:tcPr>'
                       f"{_para([[('', False)]], size_pt, size_pt * 1.15, False, 'left')}</w:tc>")
            c_expect += 1
        out.append("</w:tr>")
    out.append("</w:tbl>")
    out.append(_para([[("", False)]], 1, 1, False, "left"))  # Word cần một đoạn sau bảng trong hộp chữ
    return "".join(out)


# ------------------------------------------------------------------ nhúng phông

_EMBED = {FONT: ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
          FONT_AR: ("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
                    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf")}


def _obfuscate(data: bytes, guid: str) -> bytes:
    """ECMA-376 §17.8.1: XOR 32 byte đầu của phông với khoá lấy từ GUID (đảo thứ tự byte)."""
    hexs = guid.strip("{}").replace("-", "")
    key = bytes(int(hexs[i:i + 2], 16) for i in range(0, 32, 2))[::-1]
    head = bytes(b ^ key[i % 16] for i, b in enumerate(data[:32]))
    return head + data[32:]


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
    st = parts["word/settings.xml"].decode("utf-8")
    if "embedTrueTypeFonts" not in st:
        st = re.sub(r"(<w:settings[^>]*>)", r"\1<w:embedTrueTypeFonts/>", st, count=1)
    parts["word/settings.xml"] = st.encode("utf-8")
    tmp = src.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in parts.items():
            z.writestr(n, data)
    tmp.replace(src)
    return done


# ------------------------------------------------------------------ dựng tài liệu

def build_exact_docx(pages: list[tuple[Image.Image, list[dict]]], out_path, title: str | None = None) -> dict:
    """pages: [(ảnh trang, [khối {category, bbox, text}])] → ghi DOCX; trả về thống kê."""
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
        if not blocks:
            blocks = [{"category": "Picture", "bbox": [0, 0, W, H], "text": ""}]
            stats["whole_page_image"] += 1
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
                m = re.search(r"<table.*?</table>", text, flags=re.S | re.I)
                content = table_xml(m.group(0) if m else text, img, (x1, y1, x2, y2), scale_pt)
                stats["table"] += 1
                xml = _anchor(_textbox_graphic((x2 - x1) * k, (y2 - y1) * k * 1.04, content),
                              x1 * k, y1 * k, (x2 - x1) * k, (y2 - y1) * k * 1.04, ident, z, f"Bảng {ident}")
            else:
                original = plain_text(text)
                bold = cat in HEADING_CATEGORIES
                rtl = bool(_ARABIC.search(original)) and is_rtl(original)
                bands = text_bands(img, (x1, y1, x2, y2))
                n = len(bands) or 1
                pitch = (bands[-1][0] - bands[0][0]) / (n - 1) if n > 1 else None
                ext = _ink_extent_x(img, (x1, y1, x2, y2))
                band_h = float(np.median([b1 - b0 for b0, b1 in bands])) if bands else (y2 - y1) * 0.7
                size0 = size_from_ink(original, bold, band_h, (ext[1] - ext[0]) if ext else None, n)
                if n > 1 and bands and ext:  # nhiều dòng: theo tổng chiều dài chữ (dòng cuối đo trên ảnh)
                    last = _ink_extent_x(img, (x1, y1 + bands[-1][0], x2, y1 + bands[-1][1]))
                    if last:
                        s_len = size_from_length(original, bold, ext[1] - ext[0], n, last[1] - last[0])
                        size_h = band_h / ink_ratio(original.replace("**", ""), bold)
                        size0 = min(max(s_len, 0.6 * size_h), 1.6 * size_h)
                box_w = (x2 - x1) * 1.0
                # kiểu căn đo trên ảnh: khoảng trống trái / phải của mực trong khối
                bw_px = x2 - x1
                lg, rg = (ext[0], bw_px - ext[1]) if ext else (0.0, 0.0)
                centered = ext and lg > 0.06 * bw_px and rg > 0.06 * bw_px and abs(lg - rg) < 0.3 * max(lg, rg)
                ind_l = ind_r = 0
                if centered:
                    align = "center"
                elif ext and rg < lg:  # chữ dồn phải
                    align, ind_r = ("start" if rtl else "right"), int(round(max(0.0, rg) * scale_pt * TWIP_PER_PT))
                    box_w = (x2 - x1) - rg
                else:
                    align, ind_l = ("end" if rtl else "left"), int(round(max(0.0, lg) * scale_pt * TWIP_PER_PT))
                    box_w = (x2 - x1) - lg
                size, spacing, lines = fit_text(original, box_w, y2 - y1, n, pitch, bold, size0)
                if rtl:  # chữ Ả Rập: ngắt dòng đã đo bằng phông Ả Rập + ghép nét; căn đầu dòng (= phải)
                    segs = [[(t.replace("**", ""), bold)] for t in lines]
                else:
                    segs = _reapply_bold(original if not bold else original.replace("**", ""), lines)
                    if bold:
                        segs = [[(t, True) for t, _ in line] for line in segs]
                # khoảng cách dòng = đúng khoảng cách đo trên ảnh (nếu số dòng khớp), dòng cao tự nhiên theo phông
                first = "".join(t for t, _ in segs[0]) if segs else original
                asc, desc, top = font_metrics(first, bold)
                natural = (asc + desc) * size
                pitch_px = pitch if (pitch and len(lines) == n) else max(spacing, natural)
                justify = False
                if n > 1 and len(lines) == n and bands and not centered:
                    edges = [_ink_extent_x(img, (x1, y1 + b0, x2, y1 + b1)) for b0, b1 in bands[:-1]]
                    if all(edges):
                        full = [e[1] for e in edges] if not rtl else [e[0] for e in edges]
                        ref = max(full) if not rtl else min(full)
                        justify = all(abs(v - ref) < 0.012 * bw_px for v in full)
                margin = 0.04 * bw_px  # khung rộng thêm hai bên: chữ hơi rộng hơn dự đoán cũng không bị khung cắt
                m_tw = int(round(margin * scale_pt * TWIP_PER_PT))
                # lề an toàn nằm ở phía KHÔNG căn → chữ hơi rộng hơn dự đoán cũng không bị ngắt dòng, đầu dòng giữ nguyên
                if centered:
                    il, ir = 0, 0
                elif ind_r or align in ("right", "start"):
                    il, ir = 0, ind_r + m_tw
                else:
                    il, ir = ind_l + m_tw, 0
                content = _line_paras(segs, size * scale_pt, pitch_px * scale_pt, rtl, align, il, ir,
                                      natural * scale_pt, justify)
                # đặt khung sao cho mép trên nét chữ dòng đầu trùng ảnh gốc
                ink_top = y1 + (bands[0][0] if bands else 0)
                box_y = ink_top - top * size
                box_h = (len(lines) - 1) * pitch_px + natural + 0.3 * size
                stats["text"] += 1
                box_wk = (x2 - x1 + 2 * margin) * k
                xml = _anchor(_textbox_graphic(box_wk, box_h * k, content),
                              (x1 - margin) * k, box_y * k, box_wk, box_h * k, ident, z, f"Khối {ident}")
            par._p.append(parse_xml(xml.replace("<w:r>", f"<w:r {ns}>", 1)))
        stats["pages"] += 1
    doc.save(out_path)
    stats["fonts"] = embed_fonts(out_path)
    return stats
