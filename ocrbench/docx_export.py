"""Chuyển kết quả OCR (văn bản thường / Markdown / bảng HTML) thành file DOCX.

- Có khối bố cục (dots: category + bbox + text) và ảnh trang → dựng theo THỨ TỰ ĐỌC từng khối; khối Picture /
  Formula (logo, chữ ký, con dấu, công thức) → ẢNH CẮT từ trang gốc chèn đúng chỗ trong dòng chảy, đúng cỡ thật,
  căn trái / giữa / phải theo vị trí trên trang; nhiều ảnh cùng hàng (2 chữ ký + con dấu) → chung một dòng.

- Tiêu đề Markdown (#) → Heading; **đậm** → chữ đậm; dòng gạch đầu dòng giữ nguyên.
- Bảng HTML (<table>, có colspan) và bảng Markdown (| a | b |) → bảng Word.
- Đoạn chủ yếu là chữ Ả Rập → căn phải, chiều phải-sang-trái (bidi); bảng tiếng Ả Rập → cột từ phải sang trái.
- Mỗi trang nguồn là một trang Word, đầu trang ghi nguồn (lớp chữ PDF hay OCR) để người duyệt biết trang nào cần soát.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Pt, RGBColor

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
_MD_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?$")


@dataclass
class Block:
    kind: str  # "heading" | "para" | "table"
    text: str = ""
    level: int = 0
    rows: list | None = None  # bảng: list[list[(text, colspan)]]


_STRONG = re.compile(r"[A-Za-z\u00C0-\u024F\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]")


def para_rtl(text: str) -> bool:
    """Hướng đoạn theo UAX #9 (quy tắc P2): chữ CÓ HƯỚNG đầu tiên là Ả Rập / Do Thái → phải → trái.
    ("الهاتف: +971 56 512 3883 contact@firm.co.uk" nhiều chữ Latin hơn nhưng vẫn là đoạn phải → trái.)"""
    m = _STRONG.search(text or "")
    return bool(m) and m.group(0) >= "\u0590"


def is_rtl(text: str) -> bool:
    return len(_ARABIC.findall(text)) > len(_LATIN.findall(text))


def _list_marker(kind: str, n: int) -> str:
    """Số thứ tự của <li> theo <ol type>: 1 → 1. · a → a. · A → A. · i → i. · I → I. · ul → •"""
    if kind == "ul":
        return "•"
    if kind in ("a", "A"):
        s, x = "", n
        while x > 0:
            x, r = divmod(x - 1, 26)
            s = chr(97 + r) + s
        return (s if kind == "a" else s.upper()) + "."
    if kind in ("i", "I"):
        vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
                (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
        s, x = "", n
        for v, sym in vals:
            while x >= v:
                s, x = s + sym, x - v
        return (s if kind == "i" else s.upper()) + "."
    return f"{n}."


def _cell_text(el) -> str:
    """Chữ trong một ô bảng HTML, GIỮ xuống dòng (<br>, <p>) và số thứ tự danh sách (<ol type="a"> → a. b. c.,
    danh sách lồng thì thụt lề). Trước đây dùng text_content() → các mục dính liền, mất 1. 2. / a. b."""
    out: list[str] = []

    def nl():
        if out and not out[-1].endswith("\n"):
            out.append("\n")

    def walk(node, depth):
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if tag == "br":
            nl()
        elif tag in ("ol", "ul"):
            kind = "ul" if tag == "ul" else (node.get("type") or "1")
            try:
                n = int(node.get("start") or 1)
            except ValueError:
                n = 1
            if node.text and node.text.strip():
                out.append(node.text)
            for li in node:
                if not isinstance(li.tag, str) or li.tag.lower() != "li":
                    continue
                nl()
                out.append("    " * depth + _list_marker(kind, n) + " ")
                if li.text:
                    out.append(li.text)
                for ch in li:
                    walk(ch, depth + 1)
                n += 1
            nl()
        else:
            block = tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6")
            if block:
                nl()
            if node.text:
                out.append(node.text)
            for ch in node:
                walk(ch, depth)
            if block:
                nl()
        if node.tail:
            out.append(node.tail)

    if el.text:
        out.append(el.text)
    for ch in el:
        walk(ch, 0)
    lines = []
    for line in "".join(out).split("\n"):
        indent = len(line) - len(line.lstrip(" "))
        body = " ".join(line.split())
        if body:
            lines.append(" " * indent + body)
    return "\n".join(lines)


def _html_table_rows(table_html: str) -> list[list[tuple[str, int]]]:
    from lxml import html as lh

    doc = lh.document_fromstring(f"<html><body>{table_html}</body></html>")
    rows = []
    for tr in doc.body.xpath(".//tr"):
        cells = []
        for td in tr.xpath("./td|./th"):
            try:
                span = max(1, int(td.get("colspan", 1)))
            except ValueError:
                span = 1
            cells.append((_cell_text(td), span))
        if cells:
            rows.append(cells)
    return rows


def _text_blocks(segment: str) -> list[Block]:
    """Phần không phải bảng HTML: tách tiêu đề, bảng Markdown, đoạn văn."""
    segment = re.sub(r"<br\s*/?>", "\n", segment, flags=re.I)
    segment = re.sub(r"</(?:p|div|h[1-6]|li)>", "\n\n", segment, flags=re.I)
    segment = html.unescape(_TAG.sub("", segment))
    blocks, para, md_rows = [], [], []

    def flush_para():
        if para:
            blocks.append(Block("para", "\n".join(para)))
            para.clear()

    def flush_table():
        if md_rows:
            blocks.append(Block("table", rows=[[(c.strip(), 1) for c in r] for r in md_rows]))
            md_rows.clear()

    for raw in segment.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            continue
        if line.startswith("|") and line.count("|") >= 2:
            flush_para()
            if not _MD_SEP.match(line):
                md_rows.append(line.strip("|").split("|"))
            continue
        flush_table()
        if not line:
            flush_para()
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para()
            blocks.append(Block("heading", m.group(2).strip(), level=min(len(m.group(1)), 3)))
            continue
        para.append(re.sub(r"^[*+]\s+", "• ", re.sub(r"^-\s+", "• ", line)))
    flush_para()
    flush_table()
    return blocks


def parse_blocks(text: str) -> list[Block]:
    blocks, pos = [], 0
    for m in _HTML_TABLE.finditer(text):
        blocks += _text_blocks(text[pos:m.start()])
        rows = _html_table_rows(m.group(0))
        if rows:
            blocks.append(Block("table", rows=rows))
        pos = m.end()
    blocks += _text_blocks(text[pos:])
    return blocks


# --- ghi DOCX ------------------------------------------------------------------------


# CT_PPrBase: các phần tử đứng SAU w:bidi (chèn bidi trước phần tử đầu tiên trong số này; sai thứ tự → Word báo hỏng)
_AFTER_BIDI = ("adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents",
               "suppressOverlap", "jc", "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId",
               "cnfStyle", "rPr", "sectPr", "pPrChange")


def _set_bidi(paragraph) -> None:
    """Đoạn phải → trái, căn ĐẦU dòng (= phải). jc="start" chứ không "right": trong đoạn bidi, left/right bị hiểu
    theo chiều logic (LibreOffice / Word căn "right" thành trái)."""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT  # tạo w:jc đúng vị trí, rồi đổi giá trị
    ppr = paragraph._p.get_or_add_pPr()
    ppr.find(qn("w:jc")).set(qn("w:val"), "start")
    if ppr.find(qn("w:bidi")) is None:
        el = OxmlElement("w:bidi")
        nxt = next((c for c in ppr if isinstance(c.tag, str) and c.tag.split("}")[-1] in _AFTER_BIDI), None)
        if nxt is not None:
            nxt.addprevious(el)
        else:
            ppr.append(el)


def _add_runs(paragraph, text: str) -> None:
    """**đậm** → run đậm; xuống dòng trong đoạn → ngắt dòng mềm."""
    for li, line in enumerate(text.split("\n")):
        if li:
            paragraph.add_run().add_break()
        for part in re.split(r"(\*\*[^*]+\*\*)", line):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                paragraph.add_run(part[2:-2]).bold = True
            else:
                paragraph.add_run(part.replace("**", ""))


def _add_table(doc, rows) -> None:
    n_cols = max(sum(span for _, span in r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"
    rtl = is_rtl(" ".join(t for r in rows for t, _ in r))
    if rtl:  # CT_TblPr: bidiVisual đứng sau tblStyle / tblpPr / tblOverlap, TRƯỚC tblW (sai → Word báo hỏng)
        tblpr = table._tbl.tblPr
        el = OxmlElement("w:bidiVisual")
        nxt = next((c for c in tblpr if isinstance(c.tag, str) and c.tag.split("}")[-1] not in
                    ("tblStyle", "tblpPr", "tblOverlap")), None)
        if nxt is not None:
            nxt.addprevious(el)
        else:
            tblpr.append(el)
    for i, r in enumerate(rows):
        j = 0
        for text, span in r:
            if j >= n_cols:
                break
            cell = table.cell(i, j)
            if span > 1 and j + span - 1 < n_cols:
                cell = cell.merge(table.cell(i, j + span - 1))
            p = cell.paragraphs[0]
            _add_runs(p, text)
            if i == 0 and all(not ch.isdigit() for ch in text):
                for run in p.runs:
                    run.bold = True
            if para_rtl(text):
                _set_bidi(p)
            j += span
    doc.add_paragraph()


def _setup_styles(doc) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(11)
    rpr = normal.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), "Arial")  # w:cs = font cho chữ Ả Rập (complex script)


_IMAGE_CATEGORIES = {"Picture", "Signature", "Formula"}


def _picture_groups(blocks: list) -> list:
    """Theo thứ tự đọc: [("text", khối)] hoặc [("pics", [khối ảnh cùng hàng])] — các ảnh LIỀN NHAU trong thứ tự
    đọc và chồng nhau theo chiều dọc (≥ 50% chiều cao ảnh thấp hơn) gộp thành một dòng."""
    out = []
    for b in blocks:
        bbox = b.get("bbox")
        is_pic = (b.get("category") in _IMAGE_CATEGORIES or not (b.get("text") or "").strip()) \
            and b.get("category") != "Table"
        if not is_pic or not bbox or len(bbox) != 4:
            out.append(("text", b))
            continue
        if out and out[-1][0] == "pics":
            prev = out[-1][1][-1]["bbox"]
            overlap = min(prev[3], bbox[3]) - max(prev[1], bbox[1])
            if overlap >= 0.5 * min(prev[3] - prev[1], bbox[3] - bbox[1]):
                out[-1][1].append(b)
                continue
        out.append(("pics", [b]))
    return out


def _add_pictures(doc, group: list, image) -> None:
    """Một dòng ảnh cắt từ trang gốc: cỡ thật (theo tỉ lệ khổ trang), trái → phải, căn theo vị trí trên trang."""
    import io

    W = image.width
    sec = doc.sections[-1]
    page_w, text_w = sec.page_width, sec.page_width - sec.left_margin - sec.right_margin
    group = sorted(group, key=lambda b: b["bbox"][0])
    widths = [max(1.0, b["bbox"][2] - b["bbox"][0]) / W * page_w for b in group]
    k = min(1.0, 0.95 * text_w / sum(widths))  # cả dòng phải vừa bề rộng chữ
    p = doc.add_paragraph()
    for i, (b, w) in enumerate(zip(group, widths)):
        x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(image.height, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        buf = io.BytesIO()
        image.crop((x1, y1, x2, y2)).convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        if i:
            p.add_run("    ")
        p.add_run().add_picture(buf, width=Emu(int(w * k)))
    cx = (min(b["bbox"][0] for b in group) + max(b["bbox"][2] for b in group)) / 2 / W
    if len(group) > 1 or 0.4 <= cx <= 0.6:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if cx < 0.4 else WD_ALIGN_PARAGRAPH.RIGHT


def _bidi_from_image(block: dict, image, cache: dict | None = None, blocks: list | None = None) -> str:
    """Khối chữ Ả Rập MỘT dòng có số / chữ Latin: chọn thứ tự hiển thị khớp ảnh gốc (ocrbench.arabic_bidi), vì dấu
    hướng (vd. số điện thoại gõ trái → phải) mất khi OCR. Khối khác: giữ nguyên."""
    text = block.get("text") or ""
    bbox = block.get("bbox")
    if not bbox or len(bbox) != 4 or "\n" in text.strip() or "|" in text or "<" in text or not para_rtl(text):
        return text
    from .arabic_bidi import candidates, fix_line

    plain = text.replace("**", "").strip()
    if plain != text.strip() or not candidates(plain):
        return text
    try:
        import numpy as np

        from .docx_exact import _font, detect_arabic_font, ink_mask, text_bands

        x1, y1, x2, y2 = [float(v) for v in bbox]
        bands = text_bands(image, (x1, y1, x2, y2))
        if len(bands) != 1:
            return text
        g = np.asarray(image.convert("L"))
        ink = ink_mask(g[int(y1 + bands[0][0]):int(y1 + bands[0][1]) + 1, int(x1):int(round(x2))])
        cache = {} if cache is None else cache
        if "font" not in cache:  # phông Ả Rập của trang (nhận dạng một lần, chỉ khi có dòng cần kiểm)
            cache["font"] = detect_arabic_font([(image, blocks or [block])])[0] or "Noto Sans Arabic"
        fixed, _ = fix_line(plain, ink, _font(False, 64, True, cache["font"]))
        return fixed
    except Exception:  # không đo được → giữ nguyên chữ OCR
        return text


def _add_markdown(doc, text: str) -> int:
    n = 0
    for b in parse_blocks(text):
        n += 1
        if b.kind == "table":
            _add_table(doc, b.rows)
            continue
        p = doc.add_heading(level=b.level) if b.kind == "heading" else doc.add_paragraph()
        _add_runs(p, b.text)
        if para_rtl(b.text):
            _set_bidi(p)
    return n


def add_page(doc, text: str, header: str | None = None, first: bool = False, blocks: list | None = None,
             image=None) -> None:
    """Một trang nguồn. Có `blocks` (khối dots có chữ) + `image` (ảnh trang) → dựng theo từng khối, chèn ảnh cắt
    cho logo / chữ ký / con dấu; không có → dựng từ Markdown của cả trang như trước."""
    if not first:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    if header:
        hp = doc.add_paragraph()
        run = hp.add_run(header)
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    if blocks and image is not None and any("text" in b for b in blocks):
        n, cache = 0, {}
        for kind, item in _picture_groups([b for b in blocks if "text" in b or b.get("category") in _IMAGE_CATEGORIES]):
            if kind == "pics":
                _add_pictures(doc, item, image)
                n += 1
            else:
                n += _add_markdown(doc, _bidi_from_image(item, image, cache, blocks))
        if not n:
            doc.add_paragraph("(Không đọc được chữ nào trên trang này)")
        return
    if not _add_markdown(doc, text):
        doc.add_paragraph("(Không đọc được chữ nào trên trang này)")


def new_document(title: str | None = None):
    doc = Document()
    _setup_styles(doc)
    if title:
        doc.core_properties.title = title
    return doc
