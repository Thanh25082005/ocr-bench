"""Giao diện web: kéo thả PDF / ảnh → xem SONG SONG trang gốc và kết quả OCR, tải DOCX, xem lại LỊCH SỬ.

    ocrbench serve --config /kaggle/working/config.yaml --model <tên model> [--history-dir DIR]

Mặc định chỉ nghe ở 127.0.0.1 (an toàn cho tài liệu nội bộ). Trên Kaggle, mở từ máy mình bằng SSH:
    ssh -L 7860:localhost:7860 kaggle-ngrok      rồi vào http://localhost:7860

Mỗi lần chuyển được lưu vào thư mục lịch sử (file gốc, ảnh trang, chữ OCR, ảnh bố cục, DOCX) — xem ocrbench/history.py.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import history as H
from .convert import Converter
from .viewer import CSS as VIEW_CSS
from .viewer import ocr_html

CSS = """
.gradio-container {max-width: 1500px !important; margin: auto}
#title h1 {margin-bottom: 0}
.page-img img {object-fit: contain}
.gradio-container table, .gradio-container td, .gradio-container th {font-family: inherit !important}
""" + VIEW_CSS


def default_history_dir() -> Path:
    if os.environ.get("OCR_HISTORY_DIR"):
        return Path(os.environ["OCR_HISTORY_DIR"])
    return Path("/kaggle/working/ocr_history") if Path("/kaggle/working").exists() else Path("ocr_history")


# ---------------------------------------------------------------- khung xem song song (dùng chung 2 tab)

def _page_choices(job, fi):
    pages = job["files"][fi]["pages"] if job and job.get("files") else []
    out = []
    for p in pages:
        flag = " ⚠" if (p.get("error") or p.get("note")) else ""
        out.append(f"Trang {p['index']}{flag}")
    return out


def _render(job, fi, page_label, show_layout):
    """→ (ảnh trang, HTML kết quả, văn bản thô, dòng thông tin)"""
    if not job or not job.get("files"):
        return None, ocr_html(""), "", "_Chưa có kết quả. Chuyển một file hoặc mở một lần chạy trong Lịch sử._"
    f = job["files"][fi]
    labels = _page_choices(job, fi)
    if not f["pages"]:
        return None, ocr_html(""), "", f"**{f['name']}**: không có trang nào (lỗi: {f.get('error') or '—'})"
    pi = labels.index(page_label) if page_label in labels else 0
    p = f["pages"][pi]
    img = H.abs_path(job, p.get("layout_image") if show_layout and p.get("layout_image") else p.get("image"))
    info = (f"**{f['name']}** · trang {p['index']} ({pi + 1}/{len(f['pages'])}) · nguồn: {p['source']}"
            f" · {p.get('seconds', 0)} giây" + (f" · {p['n_blocks']} khối bố cục" if p.get("n_blocks") else ""))
    if p.get("error"):
        info += f"\n\n❌ **Lỗi:** {p['error']}"
    if p.get("note"):
        info += f"\n\n⚠ **Cần soát:** {p['note']}"
    return img, ocr_html(p.get("text", "")), p.get("text", ""), info


def _downloads(job, keys=("docx",)):
    if not job:
        return []
    out = []
    for f in job.get("files", []):
        for key in keys:
            p = H.abs_path(job, f.get(key))
            if p:
                out.append(p)
    return out


def make_viewer(gr, label: str):
    """Tạo khung xem; trả về (state, các component, hàm nạp job → updates, danh sách output của hàm đó)."""
    job_state = gr.State(None)
    with gr.Row():
        file_dd = gr.Dropdown(label="File", choices=[], value=None, scale=3, interactive=True)
        prev_btn = gr.Button("◀ Trang trước", scale=1, size="sm")
        page_dd = gr.Dropdown(label="Trang", choices=[], value=None, scale=2, interactive=True)
        next_btn = gr.Button("Trang sau ▶", scale=1, size="sm")
        show_layout = gr.Checkbox(label="Hiện khung bố cục", value=False, scale=2)
    info = gr.Markdown()
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            img = gr.Image(label=f"Bản gốc ({label})", type="filepath", interactive=False, height=820,
                           elem_classes="page-img")
        with gr.Column(scale=1):
            with gr.Tab("Kết quả OCR"):
                html_out = gr.HTML()
            with gr.Tab("Văn bản thô (sao chép)"):
                raw = gr.Textbox(lines=30, max_lines=60, show_label=False, interactive=False)
    with gr.Row():
        dl_edit = gr.File(label="📄 DOCX sửa được (soạn thảo lại)", file_count="multiple", interactive=False)
        dl_exact = gr.File(label="🧱 DOCX danh sách khối (Title / Text / Table / Picture... từ trên xuống)",
                           file_count="multiple",
                           interactive=False)
        dl_src = gr.File(label="File gốc", file_count="multiple", interactive=False)

    view_outputs = [img, html_out, raw, info]

    def render(job, fname, plabel, lay):
        fi = _file_index(job, fname)
        return _render(job, fi, plabel, lay)

    def on_file(job, fname, lay):
        fi = _file_index(job, fname)
        labels = _page_choices(job, fi)
        first = labels[0] if labels else None
        return (gr.update(choices=labels, value=first), *_render(job, fi, first, lay))

    def step(job, fname, plabel, delta):
        labels = _page_choices(job, _file_index(job, fname))
        if not labels:
            return gr.update()
        i = labels.index(plabel) if plabel in labels else 0
        return gr.update(value=labels[max(0, min(len(labels) - 1, i + delta))])

    file_dd.change(on_file, [job_state, file_dd, show_layout], [page_dd, *view_outputs])
    page_dd.change(render, [job_state, file_dd, page_dd, show_layout], view_outputs)
    show_layout.change(render, [job_state, file_dd, page_dd, show_layout], view_outputs)
    prev_btn.click(lambda j, f, p: step(j, f, p, -1), [job_state, file_dd, page_dd], page_dd)
    next_btn.click(lambda j, f, p: step(j, f, p, +1), [job_state, file_dd, page_dd], page_dd)

    def load(job):
        names = [f["name"] for f in job["files"]] if job else []
        first = names[0] if names else None
        labels = _page_choices(job, 0) if job else []
        page = labels[0] if labels else None
        has_layout = bool(job) and any(p.get("layout_image") for f in job["files"] for p in f["pages"])
        return (job, gr.update(choices=names, value=first), gr.update(choices=labels, value=page),
                gr.update(value=has_layout, visible=has_layout), *_render(job, 0, page, has_layout),
                _downloads(job, ("docx",)), _downloads(job, ("docx_blocks", "docx_exact")) or None, _downloads(job, ("source",)))

    load_outputs = [job_state, file_dd, page_dd, show_layout, *view_outputs, dl_edit, dl_exact, dl_src]
    return load, load_outputs


def _file_index(job, fname):
    if not job or not job.get("files"):
        return 0
    names = [f["name"] for f in job["files"]]
    return names.index(fname) if fname in names else 0


# ---------------------------------------------------------------- lịch sử

def _history_rows(jobs):
    rows = []
    for j in jobs:
        n_pages = sum(len(f["pages"]) for f in j.get("files", []))
        warn = sum(1 for f in j.get("files", []) for p in f["pages"] if p.get("error") or p.get("note"))
        rows.append([j.get("started", j.get("time", ""))[:19].replace("T", " "),
                     ", ".join(f["name"] for f in j.get("files", [])) or "—", n_pages, warn,
                     j.get("mode") or "—", j.get("model", ""), j.get("seconds", ""), j.get("id", "")])
    return rows


def convert_to_job(converter: Converter, hist: Path, paths: list[Path], use_text_layer: bool = True,
                   doc_type: str = "Văn bản", pages: str | None = None, mode: str | None = None, progress=None):
    """Chuyển các file, lưu tất cả vào một lần chạy trong thư mục lịch sử. → (job, các dòng bảng chi tiết)"""
    job_id, job_dir = H.new_job(hist)
    job = {"id": job_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": converter.model,
           "mode": mode or None, "pages_filter": pages or None, "text_layer": bool(use_text_layer), "files": []}
    rows = []
    t0 = time.perf_counter()
    for i, path in enumerate(paths, 1):
        fdir = job_dir / f"f{i}_{path.stem}"[:120]
        fdir.mkdir(parents=True, exist_ok=True)
        src = fdir / path.name
        shutil.copy2(path, src)
        try:
            res = converter.convert_file(src, fdir, force_ocr=not use_text_layer, pages=pages or None,
                                         doc_type="table" if doc_type == "Có bảng" else "text", progress=progress,
                                         mode=mode or None, save_pages=True, exact=False)
        except Exception as e:
            rows.append([path.name, "—", "lỗi", 0.0, f"{type(e).__name__}: {e}"])
            job["files"].append({"name": path.name, "source": H.rel(src, job_dir), "docx": None, "pages": [],
                                 "error": f"{type(e).__name__}: {e}"})
            continue
        job["files"].append(H.file_entry(job_dir, path.name, src, res))
        for p in res.pages:
            rows.append([path.name, p.index, p.source, round(p.seconds, 1), p.error or p.note or ""])
    job["seconds"] = round(time.perf_counter() - t0, 1)
    H.save_job(job_dir, job)
    job["_dir"] = str(job_dir)
    return job, rows


HISTORY_HEADERS = ["Thời gian", "File", "Số trang", "⚠ cần soát", "Chế độ", "Model", "Giây", "Mã"]


def build_app(converter: Converter, text_layer: bool = True, title: str | None = None,
              history_dir: str | Path | None = None):
    import gradio as gr

    hist = Path(history_dir) if history_dir else default_history_dir()
    hist.mkdir(parents=True, exist_ok=True)
    modes = list(converter.modes)

    def run(files, use_text_layer, doc_type, pages, mode=None, progress=gr.Progress()):
        if not files:
            raise gr.Error("Chưa có file nào. Kéo thả PDF hoặc ảnh vào ô phía trên.")
        paths = [Path(f if isinstance(f, str) else f.name) for f in files]
        job, rows = convert_to_job(converter, hist, paths, use_text_layer, doc_type, pages, mode,
                                   progress=lambda k, n, msg: progress((k, n), desc=msg))
        ok = sum(1 for f in job["files"] if f.get("docx"))
        summary = (f"✔ Xong {ok}/{len(paths)} file, {sum(len(f['pages']) for f in job['files'])} trang, "
                   f"{job['seconds']:.0f} giây · đã lưu vào Lịch sử (mã `{job['id']}`).")
        return job, summary, rows

    def refresh_history():
        jobs = H.list_jobs(hist)
        return jobs, _history_rows(jobs)

    with gr.Blocks(title="OCR → DOCX") as demo:
        gr.Markdown(
            f"# {title or 'Chuyển PDF / ảnh sang DOCX'}\n"
            f"Model OCR: **`{converter.model}`** · bên trái là trang gốc, bên phải là kết quả OCR của đúng trang đó.",
            elem_id="title",
        )
        with gr.Tabs():
            with gr.Tab("Chuyển đổi"):
                with gr.Row():
                    with gr.Column(scale=2):
                        files = gr.File(label="Kéo thả PDF hoặc ảnh (nhiều file được)", file_count="multiple",
                                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"])
                    with gr.Column(scale=3):
                        mode = gr.Radio(modes, value=modes[0] if modes else None, label="Chế độ đọc",
                                        info="Bố cục: model trả từng khối (tiêu đề, đoạn, bảng...) kèm vị trí",
                                        visible=bool(modes))
                        with gr.Row():
                            use_text_layer = gr.Checkbox(
                                value=text_layer, label="Dùng lớp chữ có sẵn của PDF",
                                info="Chỉ trang không có chữ Ả Rập. Tắt để mọi trang qua model")
                            doc_type = gr.Radio(["Văn bản", "Có bảng"], value="Văn bản", label="Loại tài liệu",
                                                visible=not modes)
                            pages = gr.Textbox(label="Trang", placeholder="vd. 1-3,5 · trống = tất cả")
                        btn = gr.Button("Chuyển", variant="primary")
                summary = gr.Markdown()
                with gr.Accordion("Chi tiết từng trang", open=False):
                    table = gr.Dataframe(headers=["File", "Trang", "Nguồn", "Giây", "Lỗi / cần soát"],
                                         interactive=False, wrap=True)
                load_conv, conv_outputs = make_viewer(gr, "trang vừa chuyển")
                new_job = gr.State(None)

            with gr.Tab("Lịch sử") as hist_tab:
                jobs_state = gr.State([])
                with gr.Row():
                    refresh = gr.Button("↻ Làm mới", size="sm", scale=1)
                    selected = gr.Textbox(label="Lần chạy đang chọn (mã)", interactive=False, scale=3)
                    open_btn = gr.Button("Mở", variant="primary", size="sm", scale=1)
                    del_btn = gr.Button("🗑 Xoá lần chạy này", variant="stop", size="sm", scale=1)
                jobs_table = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, wrap=True,
                                          label="Bấm một dòng để chọn, rồi bấm Mở")
                load_hist, hist_outputs = make_viewer(gr, "lịch sử")

        btn.click(run, [files, use_text_layer, doc_type, pages, mode], [new_job, summary, table]).then(
            load_conv, new_job, conv_outputs).then(refresh_history, None, [jobs_state, jobs_table])

        def pick(jobs, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return jobs[row]["id"] if 0 <= row < len(jobs) else ""

        def open_job(job_id):
            if not job_id:
                raise gr.Error("Chọn một dòng trong bảng trước.")
            return load_hist(H.load_job(hist, job_id))

        def delete(job_id):
            if not job_id:
                raise gr.Error("Chọn một dòng trong bảng trước.")
            H.delete_job(hist, job_id)
            jobs, rows = refresh_history()
            return jobs, rows, "", *load_hist(None)

        pick.__annotations__["evt"] = gr.SelectData  # file dùng `from __future__ import annotations`: Gradio cần class thật
        jobs_table.select(pick, [jobs_state], selected)
        open_btn.click(open_job, selected, hist_outputs)
        del_btn.click(delete, selected, [jobs_state, jobs_table, selected, *hist_outputs])
        refresh.click(refresh_history, None, [jobs_state, jobs_table])
        hist_tab.select(refresh_history, None, [jobs_state, jobs_table])
        demo.load(refresh_history, None, [jobs_state, jobs_table])
    demo.queue(default_concurrency_limit=1)  # GPU dùng chung: xử lý lần lượt từng yêu cầu
    demo.history_dir = hist
    return demo


def serve(config: str, model: str, host: str = "127.0.0.1", port: int = 7860, share: bool = False, dpi: int = 200,
          params: dict | None = None, gpus: str | None = "auto", text_layer: bool = True, title: str | None = None,
          history_dir: str | None = None, auth: str | None = None):
    """auth: "tên:mật_khẩu" (nhiều người: "a:1,b:2") — BẮT BUỘC khi mở web ra ngoài (ngrok, --share)."""
    users = None
    if auth:
        users = [tuple(x.split(":", 1)) for x in auth.split(",") if ":" in x]
        if not users or any(not u or not pw for u, pw in users):
            raise ValueError("auth phải có dạng ten:matkhau (nhiều người: a:1,b:2)")
    print(f"Đang nạp model '{model}'...", flush=True)
    converter = Converter(config, model, dpi=dpi, params=params, gpus=gpus)
    where = f"{len(converter.adapters)} GPU" if converter.devices else "thiết bị mặc định"
    print(f"Nạp xong sau {converter.load_s:.0f}s ({where}, batch {converter.batch}). Mở http://{host}:{port}", flush=True)
    import gradio as gr

    demo = build_app(converter, text_layer=text_layer, title=title, history_dir=history_dir)
    print(f"Lịch sử lưu ở: {demo.history_dir}", flush=True)
    print("Đăng nhập: BẬT (" + ", ".join(u for u, _ in users) + ")" if users else
          "Đăng nhập: TẮT — chỉ dùng khi web chỉ mở qua SSH tunnel", flush=True)
    demo.launch(server_name=host, server_port=port, share=share, css=CSS, theme=gr.themes.Soft(),
                allowed_paths=[str(Path(demo.history_dir).resolve())], auth=users,
                auth_message="Demo OCR — đăng nhập bằng tài khoản được cấp")
