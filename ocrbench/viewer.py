"""Hiển thị kết quả OCR (Markdown + bảng HTML) thành HTML an toàn để xem cạnh ảnh trang gốc.

Kết quả của model là chữ do model sinh ra → có thể chứa thẻ lạ. Chỉ giữ một danh sách thẻ/thuộc tính an toàn
(tiêu đề, đoạn, danh sách, bảng, in đậm/nghiêng...), bỏ hết script, style, sự kiện on*, link javascript.
Mỗi đoạn tự chọn chiều chữ theo nội dung (unicode-bidi: plaintext) → tiếng Ả Rập phải-sang-trái.
"""

from __future__ import annotations

import html
from html.parser import HTMLParser

ALLOWED_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "hr", "ul", "ol", "li", "b", "strong", "i", "em",
                "u", "s", "sub", "sup", "code", "pre", "blockquote", "span", "div", "table", "thead", "tbody",
                "tfoot", "tr", "td", "th", "caption"}
ALLOWED_ATTRS = {"colspan", "rowspan", "dir", "align", "type", "start"}  # type/start: <ol type="a"> = a, b, c
VOID = {"br", "hr"}
DROP_CONTENT = {"script", "style", "iframe", "object", "embed", "svg", "math", "img", "video", "audio", "form"}


class _Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in DROP_CONTENT:
            if tag not in ("img",):
                self.skip += 1
            return
        if self.skip or tag not in ALLOWED_TAGS:
            return
        keep = "".join(f' {k}="{html.escape(v or "", quote=True)}"' for k, v in attrs
                       if k in ALLOWED_ATTRS and (v or "").replace(" ", "").isalnum())
        self.out.append(f"<{tag}{keep}>")

    def handle_startendtag(self, tag, attrs):
        if tag in VOID and not self.skip:
            self.out.append(f"<{tag}>")

    def handle_endtag(self, tag):
        if tag in DROP_CONTENT and tag != "img":
            self.skip = max(0, self.skip - 1)
            return
        if self.skip or tag not in ALLOWED_TAGS or tag in VOID:
            return
        self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(html.escape(data, quote=False))


def sanitize(fragment: str) -> str:
    s = _Sanitizer()
    s.feed(fragment)
    s.close()
    return "".join(s.out)


def ocr_html(text: str) -> str:
    """Markdown (kể cả bảng Markdown và bảng HTML lồng trong) → HTML an toàn."""
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = sanitize(md.render(text or ""))
    return f'<div class="ocr-page">{body or "<p><i>(trang không có chữ)</i></p>"}</div>'


CSS = """
.ocr-page {font-size: 15px; line-height: 1.55; padding: 4px 10px; max-height: 82vh; overflow: auto;}
.ocr-page p, .ocr-page li, .ocr-page td, .ocr-page th, .ocr-page h1, .ocr-page h2, .ocr-page h3, .ocr-page h4
  {unicode-bidi: plaintext; text-align: start;}
.ocr-page table {border-collapse: collapse; margin: 8px 0; width: 100%;}
.ocr-page td, .ocr-page th {border: 1px solid #9ca3af; padding: 3px 6px; vertical-align: top;}
.ocr-page th, .ocr-page thead td {font-weight: 600; background: rgba(127,127,127,.12);}
.ocr-page h1 {font-size: 1.4em;} .ocr-page h2 {font-size: 1.2em;} .ocr-page h3 {font-size: 1.05em;}
"""
