"""Lịch sử chuyển đổi của giao diện web: mỗi lần bấm "Chuyển" là một thư mục.

    <gốc>/<YYYYmmdd-HHMMSS>-<6 ký tự>/
        job.json                  # thông tin lần chạy + chữ OCR từng trang
        f1_<tên file>/            # mỗi file tải lên
            <tên file gốc>        # bản sao file gốc (tải lại được)
            <tên>.docx
            trang/<tên>_trangN.jpg        # ảnh trang (để xem song song)
            <tên>_trangN_bocuc.jpg        # ảnh bố cục (nếu model đọc bố cục)

Đường dẫn trong job.json là tương đối với thư mục lần chạy → chuyển cả thư mục đi nơi khác vẫn mở được.
"""

from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path


def new_job(root: str | Path) -> tuple[str, Path]:
    root = Path(root)
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    d = root / job_id
    d.mkdir(parents=True, exist_ok=False)
    return job_id, d


def rel(p, base: Path) -> str | None:
    if not p:
        return None
    try:
        return str(Path(p).resolve().relative_to(base.resolve()))
    except ValueError:
        return None


def file_entry(job_dir: Path, name: str, source: Path | None, result) -> dict:
    """Từ FileResult của Converter → mục lưu trong job.json."""
    return {
        "name": name,
        "source": rel(source, job_dir),
        "docx": rel(result.docx, job_dir),
        "docx_exact": rel(getattr(result, "docx_exact", None), job_dir),
        "pages": [{
            "index": p.index, "source": p.source, "seconds": round(p.seconds, 1), "error": p.error, "note": p.note,
            "text": p.text, "image": rel(p.page_image, job_dir), "layout_image": rel(p.layout_image, job_dir),
            "n_blocks": len(p.blocks or []),
        } for p in result.pages],
    }


def save_job(job_dir: Path, job: dict) -> None:
    job.setdefault("time", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")


def load_job(root: str | Path, job_id: str) -> dict:
    d = Path(root) / job_id
    job = json.loads((d / "job.json").read_text(encoding="utf-8"))
    job["_dir"] = str(d)
    return job


def list_jobs(root: str | Path) -> list[dict]:
    root = Path(root)
    jobs = []
    if root.exists():
        for f in root.glob("*/job.json"):
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            job["_dir"] = str(f.parent)
            jobs.append(job)
    return sorted(jobs, key=lambda j: j.get("id", ""), reverse=True)


def delete_job(root: str | Path, job_id: str) -> None:
    d = (Path(root) / job_id).resolve()
    if d.parent != Path(root).resolve() or not (d / "job.json").exists():
        raise ValueError(f"không có lần chạy '{job_id}'")
    shutil.rmtree(d)


def abs_path(job: dict, rel: str | None) -> str | None:
    if not rel:
        return None
    p = Path(job["_dir"]) / rel
    return str(p) if p.exists() else None
