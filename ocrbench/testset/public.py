"""Lấy mẫu từ các bộ dữ liệu OCR công khai trên Hugging Face.

Mỗi nguồn chỉ tải đúng số mẫu cần (streaming), không tải cả bộ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _body_html(code: str) -> str:
    """Lấy phần nội dung hiển thị (body) của trang HTML, bỏ style/script."""
    from lxml import html as lh

    doc = lh.document_fromstring(code)
    for bad in doc.xpath("//style|//script|//head"):
        bad.getparent().remove(bad)
    body = doc.body
    parts = [body.text or ""] + [lh.tostring(ch, encoding="unicode") for ch in body]
    return "".join(parts).strip()


def _first_table(html_text: str) -> str:
    import re

    m = re.search(r"<table\b.*?</table>", html_text, flags=re.S | re.I)
    return m.group(0) if m else html_text


@dataclass
class PublicSource:
    key: str
    hf_id: str
    split: str
    category: str
    lang: str
    license: str
    note: str
    image_field: str
    gt: Callable[[dict], tuple[str, str]]  # row -> (đáp án, gt_type)
    uid: Callable[[dict, int], str] = lambda row, i: str(i)
    granularity: str = "page"
    keep: Callable[[dict], bool] = lambda row: True  # lọc bỏ mẫu không dùng được


SOURCES = [
    PublicSource(
        key="misraj_dococr", hf_id="Misraj/Misraj-DocOCR", split="train",
        category="pub_printed_ar", lang="ar", license="Apache-2.0",
        note="Trang sách/tạp chí/biểu mẫu tiếng Ả Rập thật, đáp án Markdown đã được chuyên gia duyệt. "
             "Model Baseer do cùng nhóm Misraj làm, nên có thể đã gặp dữ liệu cùng nguồn.",
        image_field="image", gt=lambda r: (r["markdown"], "text"), uid=lambda r, i: r["uuid"],
    ),
    PublicSource(
        key="kitab_tables", hf_id="ahmedheakl/arocrbench_tables", split="train",
        category="pub_tables_ar", lang="ar", license="chưa ghi rõ (KITAB-Bench, ACL 2025)",
        note="Bảng tiếng Ả Rập dựng từ HTML (dữ liệu tổng hợp của KITAB-Bench). Đáp án lấy từ chính mã HTML đã dựng ảnh; "
             "bỏ các bảng vẽ bằng code Python (plotly) vì không có đáp án HTML đáng tin.",
        image_field="image", gt=lambda r: (_body_html(r["code"]), "table_html"), uid=lambda r, i: r["uid"],
        keep=lambda r: '"HTMLTablePipeline"' in r["metadata"] and "<table" in r["code"].lower(),
    ),
    PublicSource(
        key="pubtabnet", hf_id="apoidea/pubtabnet-html", split="validation",
        category="pub_tables_en", lang="en", license="CDLA-Permissive-1.0",
        note="Bảng tiếng Anh cắt từ bài báo khoa học (PubTabNet, tập validation).",
        image_field="image", gt=lambda r: (_first_table(r["html_table"]), "table_html"),
        uid=lambda r, i: str(r["imgid"]),
    ),
    PublicSource(
        key="khatt_lines", hf_id="ahmedheakl/arocrbench_khatt", split="train",
        category="pub_handwriting_ar", lang="ar", license="MIT (theo thẻ dataset; KHATT gốc có điều khoản riêng)",
        note="Dòng chữ viết tay tiếng Ả Rập hiện đại (KHATT, bản dùng trong KITAB-Bench).",
        image_field="image", gt=lambda r: (r["text"], "text"), granularity="line",
    ),
    PublicSource(
        key="khatt_paragraphs", hf_id="ahmedheakl/arocrbench_khattparagraph", split="train",
        category="pub_handwriting_ar", lang="ar", license="chưa ghi rõ (KITAB-Bench)",
        note="Đoạn văn viết tay tiếng Ả Rập (KHATT, cấp đoạn).",
        image_field="image", gt=lambda r: (r["answer"], "text"), granularity="paragraph",
    ),
    PublicSource(
        key="iam_lines", hf_id="Teklia/IAM-line", split="test",
        category="pub_handwriting_en", lang="en",
        license="MIT (theo thẻ dataset); IAM gốc chỉ cho phép nghiên cứu phi thương mại",
        note="Dòng chữ viết tay tiếng Anh (IAM, tập test, người viết không trùng với tập train).",
        image_field="image", gt=lambda r: (r["text"], "text"), granularity="line",
    ),
]


def iter_samples(src: PublicSource, n: int, seed: int):
    """Sinh (uid, ảnh PIL, đáp án, gt_type) cho n mẫu xáo trộn ngẫu nhiên theo seed."""
    from datasets import load_dataset

    ds = load_dataset(src.hf_id, split=src.split, streaming=True).shuffle(seed=seed, buffer_size=1000)
    ds = ds.filter(src.keep)
    for i, row in enumerate(ds.take(n)):
        gt, gt_type = src.gt(row)
        yield src.uid(row, i), row[src.image_field], gt, gt_type
