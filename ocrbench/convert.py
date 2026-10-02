"""PDF / ảnh → DOCX bằng model OCR đã chọn trong config.

- Trang PDF có sẵn lớp chữ (PDF xuất từ Word, hệ thống...) → lấy chữ trực tiếp: nhanh, chính xác, không cần OCR.
  (Lớp chữ không giữ cấu trúc bảng; muốn giữ bảng thì bật `force_ocr`.)
  NGOẠI TRỪ trang có chữ Ả Rập: thư viện đọc PDF trả chữ Ả Rập theo thứ tự hiển thị, đảo không nhất quán
  (đo thử: CER 74%, sắp lại bằng thuật toán bidi vẫn 42%), nên trang đó luôn đưa qua OCR.
- Trang scan / ảnh → dựng ảnh (mặc định 200 dpi) → model OCR → văn bản / Markdown / bảng HTML.
- Ghép mọi trang thành một file DOCX (ocrbench/docx_export.py).
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .adapters import create_adapter
from .config import load_config
from .dataset import Item
from .docx_export import _ARABIC, add_page, new_document

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DOC_TYPES = {"text": "text", "table": "table_html"}  # loại tài liệu → gt_type (chọn prompt trong config)


@dataclass
class PageResult:
    index: int  # bắt đầu từ 1
    source: str  # "lớp chữ PDF" | "OCR" | "lỗi"
    text: str
    seconds: float
    error: str | None = None


@dataclass
class FileResult:
    path: Path
    pages: list[PageResult] = field(default_factory=list)
    docx: Path | None = None


def parse_pages(spec: str | None, n_pages: int) -> list[int]:
    """'1-3,5' → [1, 2, 3, 5] (đánh số từ 1); rỗng → mọi trang."""
    if not spec or not spec.strip():
        return list(range(1, n_pages + 1))
    pages = set()
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        elif part:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= n_pages)


class Converter:
    def __init__(self, config_path: str | Path, model: str, dpi: int = 200, min_text_chars: int = 30):
        cfg = load_config(config_path)
        self.spec = cfg.model(model)
        self.model = model
        self.dpi = dpi
        self.min_text_chars = min_text_chars
        self.adapter = create_adapter(self.spec.adapter, self.spec.params)
        self.max_side = self.spec.params.get("max_image_side")
        t = time.perf_counter()
        self.adapter.load()
        self.load_s = time.perf_counter() - t

    # --- đọc trang ---
    def _open(self, path: Path, pages: str | None):
        """Trả về ([(số trang, hàm dựng ảnh, chữ của lớp chữ PDF)], hàm đóng file).
        File PDF phải mở suốt lúc dựng ảnh các trang, nên người gọi đóng sau khi xong."""
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            def load_img():
                return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            return [(1, load_img, "")], lambda: None
        if ext not in PDF_EXTS:
            raise ValueError(f"Không hỗ trợ định dạng {ext} (chỉ nhận PDF hoặc ảnh)")
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        entries = []
        for n in parse_pages(pages, len(pdf)):
            page = pdf[n - 1]
            textpage = page.get_textpage()
            text_layer = textpage.get_text_range() or ""
            textpage.close()

            def render(page=page):
                return page.render(scale=self.dpi / 72).to_pil().convert("RGB")
            entries.append((n, render, text_layer))
        return entries, pdf.close

    def _ocr(self, image: Image.Image, path: Path, n: int, doc_type: str) -> str:
        if self.max_side and max(image.size) > self.max_side:
            image.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        gt_type = DOC_TYPES.get(doc_type, "text")
        item = Item(id=f"{path.name}#p{n}", image=path, category=f"app_{doc_type}", gt="", gt_type=gt_type)
        return self.adapter.predict(image, item).text

    def convert_file(self, path: str | Path, out_dir: str | Path, force_ocr: bool = False, pages: str | None = None,
                     doc_type: str = "text", progress=None) -> FileResult:
        path, out_dir = Path(path), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = FileResult(path)
        page_list, close = self._open(path, pages)
        try:
            self._convert_pages(path, page_list, result, force_ocr, doc_type, progress)
        finally:
            close()
        doc = new_document(title=path.stem)
        for i, pr in enumerate(result.pages):
            src = pr.source if pr.source != "OCR" else f"OCR · model {self.model}"
            header = f"Trang {pr.index} — nguồn: {src}" + (f" — LỖI: {pr.error}" if pr.error else "")
            add_page(doc, pr.text, header=header, first=(i == 0))
        result.docx = out_dir / f"{path.stem}.docx"
        doc.save(result.docx)
        return result

    def _convert_pages(self, path, page_list, result, force_ocr, doc_type, progress):
        for k, (n, render, text_layer) in enumerate(page_list):
            if progress:
                progress(k, len(page_list), f"{path.name}: trang {n}")
            t = time.perf_counter()
            layer = unicodedata.normalize("NFKC", text_layer).strip()  # gộp dạng trình bày của chữ Ả Rập
            try:
                if not force_ocr and len(layer) >= self.min_text_chars and not _ARABIC.search(layer):
                    result.pages.append(PageResult(n, "lớp chữ PDF", layer, time.perf_counter() - t))
                else:
                    text = self._ocr(render(), path, n, doc_type)
                    result.pages.append(PageResult(n, "OCR", text, time.perf_counter() - t))
            except Exception as e:  # một trang lỗi không làm hỏng cả file
                result.pages.append(PageResult(n, "lỗi", "", time.perf_counter() - t, f"{type(e).__name__}: {e}"))
