"""Kết quả dạng BỐ CỤC: danh sách khối {bbox, loại khối, nội dung} theo thứ tự đọc (kiểu dots.ocr / dots.mocr).

Model được huấn luyện cho việc này (prompt "layout") trả về JSON:
    [{"bbox": [x1, y1, x2, y2], "category": "Table", "text": "<table>...</table>"}, ...]
- bbox tính trên ảnh SAU KHI processor đổi kích thước (bội số 28, giới hạn số điểm ảnh) → đổi về ảnh gốc.
- bảng: HTML · công thức: LaTeX · còn lại: Markdown.

Từ khối sinh ra:
- văn bản để chấm điểm / dựng DOCX (`blocks_to_markdown`): tiêu đề thành heading, bảng giữ HTML;
- ảnh tô khung từng khối (`draw_blocks`) để người duyệt đối chiếu vùng ảnh gốc.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

CATEGORIES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture",
              "Section-header", "Table", "Text", "Title"]

# prompt gốc của dots.ocr / dots.mocr (dots_ocr/utils/prompts.py) — model được huấn luyện với đúng chuỗi này
DOTS_PROMPTS = {
    "layout": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",
    "text": "Extract the text content from this image.",
}


@dataclass
class Block:
    category: str
    text: str
    bbox: list[int] | None = None  # toạ độ trên ảnh GỐC

    def to_dict(self, with_text: bool = True) -> dict:
        d = asdict(self)
        if not with_text:
            d.pop("text")
        return d


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 3136,
                 max_pixels: int = 11289600) -> tuple[int, int]:
    """Kích thước ảnh mà processor kiểu Qwen2-VL thực sự đưa vào model (chép logic của qwen_vl_utils)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


_DICT_RE = re.compile(r'\{[^{}]*?"bbox"\s*:\s*\[[^\]]*?\][^{}]*?\}', re.S)


def _salvage(text: str) -> list[dict]:
    """JSON hỏng (bị cắt giữa chừng, thiếu dấu phẩy, lặp): nhặt từng khối hoàn chỉnh theo thứ tự."""
    out = []
    for m in _DICT_RE.finditer(text):
        frag = m.group(0)
        try:
            out.append(json.loads(frag))
        except json.JSONDecodeError:
            try:  # chuỗi bị cắt trong trường text: bỏ khối này
                out.append(json.loads(re.sub(r",\s*}$", "}", frag)))
            except json.JSONDecodeError:
                continue
    return out


def parse_layout(raw: str, image_size: tuple[int, int] | None = None, min_pixels: int = 3136,
                 max_pixels: int = 11289600) -> tuple[list[Block], str]:
    """raw: chuỗi model trả về. image_size: (rộng, cao) ảnh GỐC để đổi bbox về ảnh gốc.
    Trả về (các khối, trạng thái): "ok" | "repaired" (JSON hỏng, đã nhặt lại) | "failed" (không có khối nào)."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    status = "ok"
    try:
        data = json.loads(s)
        if isinstance(data, dict):  # {"layout": [...]} hoặc một khối duy nhất
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        if not isinstance(data, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        data, status = _salvage(s), "repaired"
    blocks: list[Block] = []
    seen: set[tuple] = set()
    for d in data:
        if not isinstance(d, dict):
            continue
        cat = str(d.get("category") or "Text")
        text = d.get("text") or ""
        if not isinstance(text, str):
            text = str(text)
        bbox = d.get("bbox")
        bbox = [int(float(v)) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None
        key = (cat, text, tuple(bbox) if bbox else None)
        if key in seen:  # model lặp lại nguyên khối → bỏ bản trùng
            status = "repaired"
            continue
        seen.add(key)
        blocks.append(Block(cat, text, bbox))
    if image_size and blocks:
        w, h = image_size
        ih, iw = smart_resize(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
        sx, sy = w / iw, h / ih
        for b in blocks:
            if b.bbox:
                x1, y1, x2, y2 = b.bbox
                b.bbox = [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]
    if not blocks:
        status = "failed"
    return blocks, status


_HEADING = {"Title": "# ", "Section-header": "## "}


def blocks_to_markdown(blocks: list[Block], drop: tuple[str, ...] = ()) -> str:
    """Nối các khối theo thứ tự đọc. Tiêu đề → heading Markdown (nếu model chưa thêm #), bảng giữ HTML,
    Picture không có chữ thì bỏ. `drop`: loại khối bỏ qua (vd. Page-header, Page-footer)."""
    parts = []
    for b in blocks:
        t = b.text.strip()
        if b.category in drop or not t:
            continue
        if b.category in _HEADING and not t.startswith("#"):
            t = _HEADING[b.category] + t
        elif b.category == "List-item" and not re.match(r"^([-*•]|\d+[.)])\s", t):
            t = "- " + t
        parts.append(t)
    return "\n\n".join(parts)


COLORS = {"Title": (220, 38, 38), "Section-header": (8, 145, 178), "Text": (22, 163, 74), "List-item": (37, 99, 235),
          "Table": (219, 39, 119), "Picture": (147, 51, 234), "Caption": (234, 88, 12), "Formula": (100, 116, 139),
          "Footnote": (101, 163, 13), "Page-header": (120, 113, 108), "Page-footer": (120, 113, 108)}


def draw_blocks(image, blocks: list[Block]):
    """Ảnh trang có khung + số thứ tự đọc của từng khối (để người duyệt đối chiếu)."""
    from PIL import ImageDraw

    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    width = max(2, img.width // 500)
    for i, b in enumerate(blocks, 1):
        if not b.bbox:
            continue
        c = COLORS.get(b.category, (22, 163, 74))
        d.rectangle(b.bbox, outline=c, width=width)
        d.text((b.bbox[0] + 4, max(0, b.bbox[1] - 14)), f"{i} {b.category}", fill=c)
    return img
