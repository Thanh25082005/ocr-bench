"""Giao diện web: kéo thả PDF / ảnh → tải về DOCX.

    ocrbench serve --config /kaggle/working/config.yaml --model <tên model>

Mặc định chỉ nghe ở 127.0.0.1 (an toàn cho tài liệu nội bộ). Trên Kaggle, mở từ máy mình bằng SSH:
    ssh -L 7860:localhost:7860 kaggle-ngrok      rồi vào http://localhost:7860
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .convert import Converter

CSS = """
.gradio-container {max-width: 1100px !important; margin: auto}
#title h1 {margin-bottom: 0}
"""


def build_app(converter: Converter):
    import gradio as gr

    modes = list(converter.modes)

    def run(files, use_text_layer, doc_type, pages, mode=None, progress=gr.Progress()):
        if not files:
            raise gr.Error("Chưa có file nào. Kéo thả PDF hoặc ảnh vào ô phía trên.")
        out_dir = Path(tempfile.mkdtemp(prefix="ocr_docx_"))
        docx_files, rows, previews, layouts = [], [], [], []
        t0 = time.perf_counter()
        for f in files:
            path = Path(f if isinstance(f, str) else f.name)

            def step(k, n, msg):
                progress((k, n), desc=msg)

            try:
                res = converter.convert_file(path, out_dir, force_ocr=not use_text_layer, pages=pages or None,
                                             doc_type="table" if doc_type == "Có bảng" else "text", progress=step,
                                             mode=mode or None)
            except Exception as e:
                rows.append([path.name, "—", "lỗi", 0.0, f"{type(e).__name__}: {e}"])
                continue
            docx_files.append(str(res.docx))
            for p in res.pages:
                rows.append([path.name, p.index, p.source, round(p.seconds, 1), p.error or p.note or ""])
                if p.layout_image:
                    layouts.append((str(p.layout_image), f"{path.name} · trang {p.index}"))
            first = next((p for p in res.pages if p.text), None)
            if first:
                previews.append(f"### {path.name} — trang {first.index} ({first.source})\n\n{first.text[:3000]}")
        summary = f"Xong {len(docx_files)}/{len(files)} file trong {time.perf_counter() - t0:.0f} giây."
        return (docx_files, rows, summary + "\n\n" + ("\n\n---\n\n".join(previews) or "_(không có chữ để xem trước)_"),
                layouts)

    with gr.Blocks(title="OCR → DOCX") as demo:
        gr.Markdown(
            f"# Chuyển PDF / ảnh sang DOCX\n"
            f"Model OCR: **`{converter.model}`** · trang PDF đã có lớp chữ thì lấy chữ trực tiếp, trang scan mới qua OCR.",
            elem_id="title",
        )
        files = gr.File(label="Kéo thả PDF hoặc ảnh (nhiều file được)", file_count="multiple",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"])
        with gr.Row():
            use_text_layer = gr.Checkbox(value=True, label="Dùng lớp chữ có sẵn của PDF",
                                         info="Chỉ với trang không có chữ Ả Rập (lớp chữ Ả Rập trong PDF hay bị đảo). "
                                              "Tắt nếu cần dựng lại bảng bằng OCR")
            doc_type = gr.Radio(["Văn bản", "Có bảng"], value="Văn bản", label="Loại tài liệu",
                                info="Chọn prompt cho model (nếu config có prompt riêng cho bảng)")
            pages = gr.Textbox(label="Trang", placeholder="vd. 1-3,5 · để trống = tất cả")
        mode = gr.Radio(modes, value=modes[0] if modes else None, label="Chế độ đọc",
                        info="Bố cục: model trả từng khối (tiêu đề, đoạn, bảng...) kèm vị trí trên trang",
                        visible=bool(modes))
        btn = gr.Button("Chuyển sang DOCX", variant="primary")
        out = gr.File(label="Tải DOCX", file_count="multiple")
        table = gr.Dataframe(headers=["File", "Trang", "Nguồn", "Giây", "Lỗi / cần soát"], label="Chi tiết từng trang",
                             interactive=False, wrap=True)
        preview = gr.Markdown(label="Xem trước")
        layout = gr.Gallery(label="Bố cục từng trang (khung + thứ tự đọc)", columns=3, height="auto")
        btn.click(run, [files, use_text_layer, doc_type, pages, mode], [out, table, preview, layout])
    demo.queue(default_concurrency_limit=1)  # một GPU: xử lý lần lượt từng yêu cầu
    return demo


def serve(config: str, model: str, host: str = "127.0.0.1", port: int = 7860, share: bool = False, dpi: int = 200,
          params: dict | None = None, gpus: str | None = "auto"):
    print(f"Đang nạp model '{model}'...", flush=True)
    converter = Converter(config, model, dpi=dpi, params=params, gpus=gpus)
    where = f"{len(converter.adapters)} GPU" if converter.devices else "thiết bị mặc định"
    print(f"Nạp xong sau {converter.load_s:.0f}s ({where}, batch {converter.batch}). Mở http://{host}:{port}", flush=True)
    import gradio as gr

    build_app(converter).launch(server_name=host, server_port=port, share=share, css=CSS,
                                theme=gr.themes.Soft())
