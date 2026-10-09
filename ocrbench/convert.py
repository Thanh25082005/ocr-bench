"""PDF / ảnh → DOCX bằng model OCR đã chọn trong config.

- Trang PDF có sẵn lớp chữ (PDF xuất từ Word, hệ thống...) → lấy chữ trực tiếp: nhanh, chính xác, không cần OCR.
  (Lớp chữ không giữ cấu trúc bảng; muốn giữ bảng thì bật `force_ocr`.)
  NGOẠI TRỪ trang có chữ Ả Rập: thư viện đọc PDF trả chữ Ả Rập theo thứ tự hiển thị, đảo không nhất quán
  (đo thử: CER 74%, sắp lại bằng thuật toán bidi vẫn 42%), nên trang đó luôn đưa qua OCR.
- Trang scan / ảnh → dựng ảnh (mặc định 200 dpi) → model OCR → văn bản / Markdown / bảng HTML.
- Ghép mọi trang thành một file DOCX (ocrbench/docx_export.py).
"""

from __future__ import annotations

import queue
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .adapters import create_adapter
from .config import load_config
from .dataset import Item
from .docx_export import _ARABIC, add_page, new_document
from .preprocess import config_of as preprocess_config
from .preprocess import preprocess

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
    note: str | None = None  # vd. "bị cắt", "model bị lặp" → người duyệt cần soát trang này
    blocks: list | None = None  # chế độ bố cục: [{category, bbox}] theo thứ tự đọc
    layout_image: Path | None = None  # ảnh trang có khung từng khối (để đối chiếu)
    page_image: Path | None = None  # ảnh trang gốc (save_pages=True) — để xem song song bản gốc / bản OCR
    image: object = field(default=None, repr=False)  # ảnh model đã đọc (toạ độ khối theo ảnh này); dùng tạm khi dựng DOCX


@dataclass
class FileResult:
    path: Path
    pages: list[PageResult] = field(default_factory=list)
    docx: Path | None = None
    docx_exact: Path | None = None  # DOCX giữ nguyên bố cục (mỗi khối đúng toạ độ) — chỉ khi model trả khối có chữ
    docx_blocks: Path | None = None  # DOCX danh sách khối: loại + nội dung từng khối theo thứ tự đọc (docx_blocks.py)


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
    """Nạp model (một bản trên MỖI GPU nếu model vừa 1 GPU) và chuyển file.

    params: ghi đè tham số model trong config, vd. {"batch_size": 4, "stop_on_loop": True, "max_new_tokens": 4096}.
    gpus: "auto" = mọi GPU thấy được; "0" / "0,1" = chỉ các GPU này; None = để model tự chọn (1 bản).
    """

    def __init__(self, config_path: str | Path, model: str, dpi: int = 200, min_text_chars: int = 30,
                 params: dict | None = None, gpus: str | None = "auto"):
        cfg = load_config(config_path)
        self.spec = cfg.model(model)
        self.model = model
        self.dpi = dpi
        self.min_text_chars = min_text_chars
        self.params = merge_params(self.spec.params, params or {})
        self.batch = max(1, int(self.params.get("batch_size", 1)))
        self.max_side = self.params.get("max_image_side")
        self.pre_cfg = preprocess_config(self.params.get("preprocess"))  # giống hệt lúc chạy benchmark
        # chế độ đọc (kiểu dots.ocr): params.modes = {tên: {prompt, output_format, ...}}; mục đầu là mặc định
        self.modes = dict(self.params.get("modes") or {})
        devices = self._devices(gpus)
        if self.spec.adapter == "dots":  # cần chữ từng khối để dựng DOCX giữ nguyên bố cục
            self.params = merge_params(self.params, {"blocks_with_text": True})
        if len(devices) > 1:
            self.adapters = [create_adapter(self.spec.adapter, {**self.params, "device_map": {"": f"cuda:{d}"}})
                             for d in devices]
        else:
            self.adapters = [create_adapter(self.spec.adapter, self.params)]
        self.devices = devices
        self.fallbacks = 0  # số batch lỗi (vd. hết VRAM) phải chạy lại từng trang
        t = time.perf_counter()
        for a in self.adapters:
            a.load()
        self.load_s = time.perf_counter() - t
        self._base = [dict(a.params) for a in self.adapters]

    def set_mode(self, mode: str | None) -> None:
        """Đổi prompt / kiểu kết quả cho các lần chuyển sau (không nạp lại model)."""
        if mode and mode not in self.modes:
            raise ValueError(f"không có chế độ '{mode}'. Có: {list(self.modes)}")
        over = self.modes.get(mode, {}) if mode else {}
        for a, base in zip(self.adapters, self._base):
            a.params = merge_params(base, over)

    @property
    def adapter(self):
        return self.adapters[0]

    def _devices(self, gpus) -> list[int]:
        if gpus is None or self.spec.adapter not in ("hf_vlm", "dots") or self.spec.gpus > 1 or self.params.get("quantization"):
            return []
        try:
            import torch
        except ImportError:
            return []
        n = torch.cuda.device_count()
        if n == 0:
            return []
        if gpus == "auto":
            return list(range(n))
        return [int(x) for x in str(gpus).split(",") if x.strip() != "" and int(x) < n]

    def close(self):
        for a in self.adapters:
            a.close()

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

    def _prepare(self, image: Image.Image, path: Path, n: int, doc_type: str):
        if self.max_side and max(image.size) > self.max_side:
            image.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        if self.pre_cfg:
            image, _ = preprocess(image, self.pre_cfg)
        gt_type = DOC_TYPES.get(doc_type, "text")
        return image, Item(id=f"{path.name}#p{n}", image=path, category=f"app_{doc_type}", gt="", gt_type=gt_type)

    def convert_file(self, path: str | Path, out_dir: str | Path, force_ocr: bool = False, pages: str | None = None,
                     doc_type: str = "text", progress=None, mode: str | None = None,
                     save_pages: bool = False, exact: bool = True, blocks: bool = True) -> FileResult:
        path, out_dir = Path(path), Path(out_dir)
        if self.modes or mode:
            self.set_mode(mode)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = FileResult(path)
        page_list, close = self._open(path, pages)
        try:
            self._convert_pages(path, page_list, result, force_ocr, doc_type, progress, out_dir, save_pages, exact)
        finally:
            close()
        doc = new_document(title=path.stem)
        for i, pr in enumerate(result.pages):
            src = pr.source if pr.source != "OCR" else f"OCR · model {self.model}"
            header = (f"Trang {pr.index} — nguồn: {src}" + (f" — LỖI: {pr.error}" if pr.error else "")
                      + (f" — ⚠ CẦN SOÁT: {pr.note}" if pr.note else ""))
            add_page(doc, pr.text, header=header, first=(i == 0), blocks=pr.blocks, image=pr.image)
        result.docx = out_dir / f"{path.stem}.docx"
        doc.save(result.docx)
        if exact and any(pr.blocks and any("text" in b for b in pr.blocks) for pr in result.pages):
            from .docx_exact import build_exact_docx

            pages = [(pr.image, [b for b in (pr.blocks or []) if "text" in b]) for pr in result.pages
                     if pr.image is not None]
            result.docx_exact = out_dir / f"{path.stem}_bo_cuc.docx"
            build_exact_docx(pages, result.docx_exact, title=path.stem)
        if blocks:
            from .docx_blocks import build_blocks_docx

            result.docx_blocks = build_blocks_docx(result.pages, out_dir / f"{path.stem}_khoi.docx", title=path.stem)
        for pr in result.pages:
            pr.image = None  # không giữ ảnh trong bộ nhớ sau khi dựng xong
        return result

    def _save_page(self, image, out_dir, path, n) -> Path:
        im = image.convert("RGB")
        if max(im.size) > 2000:
            im = im.copy()
            im.thumbnail((2000, 2000), Image.LANCZOS)
        p = Path(out_dir) / "trang" / f"{path.stem}_trang{n}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p, quality=88)
        return p

    def _convert_pages(self, path, page_list, result, force_ocr, doc_type, progress, out_dir=None, save_pages=False,
                       keep_images=False):
        """Trang có lớp chữ dùng được → lấy luôn; trang còn lại gom thành batch, chia cho các GPU chạy song song."""
        slots: list[PageResult | None] = [None] * len(page_list)
        todo = []  # (vị trí, số trang, hàm dựng ảnh)
        for k, (n, render, text_layer) in enumerate(page_list):
            layer = unicodedata.normalize("NFKC", text_layer).strip()  # gộp dạng trình bày của chữ Ả Rập
            if not force_ocr and len(layer) >= self.min_text_chars and not _ARABIC.search(layer):
                slots[k] = PageResult(n, "lớp chữ PDF", layer, 0.0)
                if (save_pages and out_dir is not None) or keep_images:
                    try:
                        im = render()
                        if save_pages and out_dir is not None:
                            slots[k].page_image = self._save_page(im, out_dir, path, n)
                        if keep_images:
                            slots[k].image = im  # DOCX giữ nguyên bố cục: trang không có khối → đặt cả ảnh trang
                    except Exception:  # ảnh xem trước hỏng không được làm hỏng kết quả chữ
                        pass
            else:
                todo.append((k, n, render))
        done = len(page_list) - len(todo)
        window = len(self.adapters) * self.batch * 2  # dựng ảnh theo từng đợt để không giữ cả file trong RAM
        for w in range(0, len(todo), window):
            part = todo[w:w + window]
            if progress:
                progress(done, len(page_list), f"{path.name}: OCR trang {part[0][1]}–{part[-1][1]}")
            ready = []
            for k, n, render in part:  # pdfium không an toàn đa luồng: dựng ảnh ở luồng chính
                try:
                    img, item = self._prepare(render(), path, n, doc_type)
                    ready.append((k, n, img, item))
                except Exception as e:
                    slots[k] = PageResult(n, "lỗi", "", 0.0, f"{type(e).__name__}: {e}")
            batches = [ready[i:i + self.batch] for i in range(0, len(ready), self.batch)]
            images = {k: img for k, n, img, item in ready}
            for k, res in self._run_batches(batches):
                if k in images:
                    res.image = images[k]
                if save_pages and out_dir is not None and k in images:
                    res.page_image = self._save_page(images[k], out_dir, path, res.index)
                if res.blocks and out_dir is not None:
                    from .layout import Block, draw_blocks

                    res.layout_image = Path(out_dir) / f"{path.stem}_trang{res.index}_bocuc.jpg"
                    draw_blocks(images[k], [Block(b["category"], "", b.get("bbox"), b.get("polygon"))
                                             for b in res.blocks]).save(
                        res.layout_image, quality=85)
                slots[k] = res
            done += len(part)
        result.pages.extend(slots)

    def _run_batches(self, batches):
        """Mỗi GPU một luồng, lấy batch từ hàng đợi chung. Batch lỗi thì thử lại từng trang để cô lập trang hỏng."""
        jobs: queue.Queue = queue.Queue()
        for b in batches:
            jobs.put(b)
        out, lock = [], threading.Lock()

        def worker(adapter):
            while True:
                try:
                    b = jobs.get_nowait()
                except queue.Empty:
                    return
                t = time.perf_counter()
                try:
                    preds = adapter.predict_batch([x[2] for x in b], [x[3] for x in b])
                    per = (time.perf_counter() - t) / len(b)
                    rows = [(k, _page_result(n, p, per)) for (k, n, _, _), p in zip(b, preds)]
                except Exception as batch_err:
                    with lock:
                        self.fallbacks += 1
                        if self.fallbacks == 1:
                            print(f"⚠ batch {len(b)} trang lỗi ({type(batch_err).__name__}: {str(batch_err)[:200]}); "
                                  "chạy lại từng trang. Nếu là hết VRAM, giảm batch_size.", flush=True)
                    rows = []
                    for k, n, img, item in b:
                        t1 = time.perf_counter()
                        try:
                            rows.append((k, _page_result(n, adapter.predict(img, item), time.perf_counter() - t1)))
                        except Exception as e:
                            rows.append((k, PageResult(n, "lỗi", "", time.perf_counter() - t1,
                                                       f"{type(e).__name__}: {e}")))
                with lock:
                    out.extend(rows)

        threads = [threading.Thread(target=worker, args=(a,), daemon=True) for a in self.adapters]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return out


def _page_result(n: int, pred, seconds: float) -> PageResult:
    notes = []
    if pred.extra.get("stopped_loop"):
        notes.append("model bị lặp, đã dừng sớm")
    if pred.extra.get("hit_max_tokens"):
        notes.append("bị cắt do hết max_new_tokens")
    if pred.extra.get("hit_max_time"):
        notes.append("bị cắt do quá thời gian cho phép mỗi trang")
    if pred.extra.get("truncated_repaired"):
        notes.append("khối cuối bị cắt dở, đã bỏ phần lặp và giữ phần đọc được")
    if pred.extra.get("copied_cells"):
        notes.append(f"{pred.extra['copied_cells']} khối nghi bị chép chữ / số từ khối khác (chữ viết tay khó đọc)")
    if pred.extra.get("layout") == "repaired":
        notes.append("JSON bố cục bị hỏng, đã nhặt lại các khối đọc được")
    elif pred.extra.get("layout") == "failed":
        notes.append("không đọc được bố cục, giữ nguyên kết quả thô")
    return PageResult(n, "OCR", pred.text, seconds, note="; ".join(notes) or None, blocks=pred.extra.get("blocks"))


def merge_params(base: dict, override: dict) -> dict:
    """Ghi đè tham số; riêng các dict con (processor_kwargs, generation_kwargs...) thì gộp."""
    out = dict(base)
    for k, v in override.items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
