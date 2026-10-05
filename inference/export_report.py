"""Gom vài lần chuyển trong Lịch sử web demo thành một thư mục báo cáo: input / output / so sánh song song.

    python inference/export_report.py --files 01_hoa_don_tieng_anh.png,08_anh_chup_xau_hoa_don_a_rap.jpg \\
        [--history /kaggle/working/ocr_history] [--out /kaggle/working/bao_cao] [--zip]

Mỗi tên file lấy LẦN CHẠY GẦN NHẤT có file đó. Kết quả:

    <out>/
      README.md                       # bảng tổng hợp: file, số trang, thời gian, cần soát, CER (nếu có đáp án mẫu)
      01_<tên>/
        input/<file gốc>
        output/<tên>.docx             # DOCX như người dùng tải về
        output/<tên>_trangN.md        # chữ OCR thô từng trang
        output/bocuc_trangN.jpg       # ảnh bố cục (khung + thứ tự đọc), nếu có
        so_sanh.html                  # mở bằng trình duyệt: ảnh trang gốc | kết quả OCR, từng trang
        trang/trangN.jpg              # ảnh trang (dùng cho so_sanh.html)

Chỉ ĐỌC thư mục lịch sử, không sửa / xoá gì trong đó.
"""

from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ocrbench import history as H  # noqa: E402
from ocrbench.app import default_history_dir  # noqa: E402
from ocrbench.viewer import CSS, ocr_html  # noqa: E402

PAGE_CSS = """
body{font-family:Arial,Helvetica,sans-serif;margin:16px;color:#111;background:#fff}
h1{font-size:20px;margin:0 0 4px} .meta{color:#555;font-size:13px;margin-bottom:14px}
.page{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:0 0 28px;border-top:1px solid #ddd;padding-top:10px}
.page h2{grid-column:1/3;font-size:15px;margin:0} .page img{width:100%;border:1px solid #ccc}
.note{grid-column:1/3;color:#b45309;font-size:13px}
@media (max-width:900px){.page{grid-template-columns:1fr}.page h2,.note{grid-column:1}}
""" + CSS.replace("max-height: 82vh; overflow: auto;", "")


def find_runs(hist: Path, names: list[str]) -> list[tuple[str, dict, dict]]:
    """→ [(tên file, job, mục file trong job)] — lần chạy gần nhất có file đó."""
    jobs = H.list_jobs(hist)  # mới nhất trước
    out = []
    for name in names:
        hit = next(((j, f) for j in jobs for f in j.get("files", []) if f["name"] == name), None)
        if hit is None:
            raise SystemExit(f"✘ Không thấy '{name}' trong lịch sử {hist}. Các file có: "
                             + ", ".join(sorted({f['name'] for j in jobs for f in j.get('files', [])})))
        out.append((name, *hit))
    return out


def sample_cer(name: str, pages: list[dict]) -> float | None:
    """CER so với đáp án nếu là tài liệu mẫu trong inference/samples (có dap_an)."""
    stem = Path(name).stem
    gt = next(iter((ROOT / "inference/samples/dap_an").glob(stem + ".*")), None)
    if gt is None:
        return None
    from ocrbench.metrics import edit_stats, error_rate
    from ocrbench.normalize import NormConfig, normalize

    text = "\n\n".join(p.get("text", "") for p in pages)
    s = edit_stats(normalize(gt.read_text(encoding="utf-8"), NormConfig()), normalize(text, NormConfig()))
    return round(error_rate(s["char_edits"], s["ref_chars"]) * 100, 2)


def export_one(i: int, name: str, job: dict, f: dict, out: Path) -> dict:
    stem = Path(name).stem
    d = out / f"{i:02d}_{stem}"
    (d / "input").mkdir(parents=True, exist_ok=True)
    (d / "output").mkdir(parents=True, exist_ok=True)
    (d / "trang").mkdir(exist_ok=True)
    src = H.abs_path(job, f.get("source"))
    if src:
        shutil.copy2(src, d / "input" / name)
    docx = H.abs_path(job, f.get("docx"))
    if docx:
        shutil.copy2(docx, d / "output" / Path(docx).name)
    blocks = []
    for p in f["pages"]:
        n = p["index"]
        (d / "output" / f"{stem}_trang{n}.md").write_text(p.get("text", ""), encoding="utf-8")
        img = H.abs_path(job, p.get("image"))
        rel_img = None
        if img:
            shutil.copy2(img, d / "trang" / f"trang{n}.jpg")
            rel_img = f"trang/trang{n}.jpg"
        lay = H.abs_path(job, p.get("layout_image"))
        if lay:
            shutil.copy2(lay, d / "output" / f"bocuc_trang{n}.jpg")
        note = " · ".join(x for x in (p.get("error"), p.get("note")) if x)
        blocks.append(
            f'<section class="page"><h2>Trang {n} · nguồn: {html.escape(p["source"])} · {p.get("seconds", 0)} giây'
            f'</h2>{f"<div class=note>⚠ {html.escape(note)}</div>" if note else ""}'
            f'<div>{f"<img src={chr(34)}{rel_img}{chr(34)} alt={chr(34)}trang {n}{chr(34)}>" if rel_img else "(không có ảnh)"}</div>'
            f'<div>{ocr_html(p.get("text", ""))}</div></section>')
    secs = round(sum(p.get("seconds", 0) for p in f["pages"]), 1)
    warn = sum(1 for p in f["pages"] if p.get("error") or p.get("note"))
    cer = sample_cer(name, f["pages"])
    page = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{html.escape(name)}</title>'
            f'<style>{PAGE_CSS}</style></head><body><h1>{html.escape(name)}</h1>'
            f'<div class="meta">Model {html.escape(job.get("model", ""))} · chế độ {html.escape(job.get("mode") or "—")} · '
            f'{len(f["pages"])} trang · {secs} giây · lần chạy {html.escape(job.get("id", ""))}'
            f'{f" · CER so với đáp án mẫu: {cer}%" if cer is not None else ""}</div>{"".join(blocks)}</body></html>')
    (d / "so_sanh.html").write_text(page, encoding="utf-8")
    return {"dir": d.name, "name": name, "pages": len(f["pages"]), "seconds": secs, "warn": warn, "cer": cer,
            "job": job.get("id", ""), "time": job.get("started", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="tên file như trong Lịch sử, phân cách bằng dấu phẩy")
    ap.add_argument("--history", default=None)
    ap.add_argument("--out", default="/kaggle/working/bao_cao")
    ap.add_argument("--zip", action="store_true", help="nén thêm <out>.zip để tải về")
    a = ap.parse_args()
    hist = Path(a.history) if a.history else default_history_dir()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"✘ {out} đã có — chọn --out khác (script không ghi đè).")
    names = [x.strip() for x in a.files.split(",") if x.strip()]
    rows = [export_one(i, name, job, f, out) for i, (name, job, f) in enumerate(find_runs(hist, names), 1)]
    lines = ["# Báo cáo OCR — dots.mocr (web demo)", "",
             "Mỗi thư mục: `input/` (file gốc) · `output/` (DOCX, chữ OCR từng trang `.md`, ảnh bố cục) · "
             "`so_sanh.html` (mở bằng trình duyệt: trang gốc | kết quả OCR).", "",
             "| # | File | Trang | Thời gian xử lý | Trang cần soát | CER so với đáp án | Lần chạy |",
             "|---|---|---:|---:|---:|---:|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | [{r['name']}]({r['dir']}/so_sanh.html) | {r['pages']} | {r['seconds']} giây | {r['warn']} | "
                     f"{'—' if r['cer'] is None else str(r['cer']) + '%'} | {r['time'][:16].replace('T', ' ')} |")
    lines += ["", "CER = tỉ lệ lỗi ký tự (thấp = tốt), chỉ có với tài liệu mẫu tự tạo (`inference/samples`, có đáp án)."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if a.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"\n✔ Đã nén: {z} ({Path(z).stat().st_size / 1e6:.1f} MB)")
    print(f"✔ Xong: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
