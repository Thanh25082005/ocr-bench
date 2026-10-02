"""Chuyển kết quả OCR (văn bản thường / Markdown / bảng HTML) thành file DOCX.

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
from docx.shared import Pt, RGBColor

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


def is_rtl(text: str) -> bool:
    return len(_ARABIC.findall(text)) > len(_LATIN.findall(text))


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
            cells.append((" ".join(td.text_content().split()), span))
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


def _set_bidi(paragraph) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    if ppr.find(qn("w:bidi")) is None:
        ppr.insert(0, OxmlElement("w:bidi"))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT


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
    if rtl:
        tblpr = table._tbl.tblPr
        tblpr.append(OxmlElement("w:bidiVisual"))
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
            if is_rtl(text):
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


def add_page(doc, text: str, header: str | None = None, first: bool = False) -> None:
    if not first:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    if header:
        hp = doc.add_paragraph()
        run = hp.add_run(header)
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    blocks = parse_blocks(text)
    if not blocks:
        doc.add_paragraph("(Không đọc được chữ nào trên trang này)")
    for b in blocks:
        if b.kind == "table":
            _add_table(doc, b.rows)
            continue
        p = doc.add_heading(level=b.level) if b.kind == "heading" else doc.add_paragraph()
        _add_runs(p, b.text)
        if is_rtl(b.text):
            _set_bidi(p)


def new_document(title: str | None = None):
    doc = Document()
    _setup_styles(doc)
    if title:
        doc.core_properties.title = title
    return doc
