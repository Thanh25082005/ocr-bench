"""Chạy MỘT model trên bộ dữ liệu, ghi kết quả từng trang vào predictions.jsonl.

Thường được `ocrbench run` gọi trong một tiến trình con riêng cho mỗi model, để khi model
chạy xong thì toàn bộ VRAM được trả lại trước khi model tiếp theo nạp vào.

Chạy lại sẽ bỏ qua những trang đã có kết quả (tiếp tục được khi phiên Kaggle bị ngắt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .adapters import create_adapter
from .config import ModelSpec, load_config
from .dataset import fingerprint, load_manifest

MAX_CONSECUTIVE_ERRORS = 10
EXIT_SPEC_CHANGED = 3


def spec_fingerprint(adapter: str, params: dict, env: dict | None = None) -> str:
    """Những gì ảnh hưởng tới kết quả của model. Không gồm `gpus` (chỉ là cách xếp lịch)."""
    blob = json.dumps({"adapter": adapter, "params": params, "env": env or {}}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_dir(output_dir: Path, split: str, model: str) -> Path:
    return output_dir / split / model


def load_predictions(path: Path) -> dict[str, dict]:
    """id -> bản ghi; dòng sau ghi đè dòng trước (khi chạy lại các trang lỗi)."""
    preds: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # dòng cuối bị cắt dở khi tiến trình bị dừng đột ngột
                    preds[rec["id"]] = rec
    return preds


def load_image(path: Path, max_side: int | None) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


class VramMonitor:
    """Đo VRAM cao nhất bằng nvidia-smi (dùng được cho mọi framework: torch, paddle...)."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._ids = vis if vis and all(x.strip().isdigit() for x in vis.split(",")) else None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _query(self) -> int | None:
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self._ids:
            cmd += ["-i", self._ids]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
            return sum(int(x) for x in out.split())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _loop(self):
        while not self._stop.is_set():
            used = self._query()
            if used is None:
                return
            self.peak_mib = max(self.peak_mib, used)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def _gpu_names() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10
        ).stdout
        return [x.strip() for x in out.splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def run_model(
    config_path: str | Path,
    spec: ModelSpec,
    split: str,
    categories: list[str] | None = None,
    limit: int | None = None,
    retry_errors: bool = False,
    per_category: int | None = None,
) -> int:
    cfg = load_config(config_path)
    items = load_manifest(cfg.dataset, split=split, categories=categories, limit=limit, per_category=per_category)
    out = run_dir(cfg.output_dir, split, spec.name)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"

    done = load_predictions(pred_path)

    # Chặn trộn kết quả: đã có kết quả của cấu hình cũ thì không được chạy tiếp với cấu hình khác dưới cùng tên
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"sessions": []}
    spec_hash = spec_fingerprint(spec.adapter, spec.params, spec.env)
    old_hash = meta.get("spec_hash") or (
        spec_fingerprint(meta["sessions"][-1]["adapter"], meta["sessions"][-1]["params"], meta["sessions"][-1].get("env"))
        if meta.get("sessions") else None
    )
    if done and old_hash and old_hash != spec_hash:
        print(
            f"[{spec.name}] TỪ CHỐI CHẠY: cấu hình (adapter/params/env) đã khác với lúc tạo {len(done)} kết quả có sẵn "
            f"trong {out}.\n  Chạy tiếp sẽ trộn kết quả của hai cấu hình khác nhau. Hãy ĐẶT TÊN MODEL MỚI cho cấu hình mới "
            f"(vd. {spec.name}__v2), hoặc đưa cấu hình về như cũ.",
            flush=True,
        )
        return EXIT_SPEC_CHANGED
    if retry_errors:
        done = {k: v for k, v in done.items() if not v.get("error")}
    todo = [it for it in items if it.id not in done]
    print(f"[{spec.name}] {len(items)} mẫu, đã có {len(items) - len(todo)}, cần chạy {len(todo)}", flush=True)
    if not todo:
        return 0

    for k, v in spec.env.items():
        os.environ[k] = v
    data_fp = fingerprint(load_manifest(cfg.dataset))
    meta["spec_hash"] = spec_hash
    meta["dataset_fingerprint"] = data_fp
    session = {
        "dataset_fingerprint": data_fp,
        "env": spec.env,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "adapter": spec.adapter,
        "params": spec.params,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "gpus": _gpu_names(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    adapter = create_adapter(spec.adapter, spec.params)
    max_side = spec.params.get("max_image_side")
    exit_code = 0
    with VramMonitor() as vram:
        t0 = time.perf_counter()
        adapter.load()
        session["load_s"] = round(time.perf_counter() - t0, 1)
        print(f"[{spec.name}] nạp model xong sau {session['load_s']}s", flush=True)

        consecutive_errors = 0
        n_errors = 0
        with pred_path.open("a", encoding="utf-8") as f:
            for i, it in enumerate(todo, 1):
                rec = {"id": it.id, "text": "", "confidence": None, "latency_s": None, "extra": {}, "error": None}
                try:
                    pages = [load_image(p, max_side) for p in it.images]
                    t = time.perf_counter()
                    pred = adapter.predict(pages[0], it) if len(pages) == 1 else adapter.predict_pages(pages, it)
                    rec.update(
                        text=pred.text,
                        confidence=pred.confidence,
                        latency_s=round(time.perf_counter() - t, 3),
                        extra=pred.extra,
                    )
                    consecutive_errors = 0
                except Exception as e:  # một trang lỗi không được làm dừng cả lượt chạy
                    rec["error"] = f"{type(e).__name__}: {e}"[:2000]
                    consecutive_errors += 1
                    n_errors += 1
                    traceback.print_exc()
                    _free_cuda_cache()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                status = "LỖI " + rec["error"][:120] if rec["error"] else f"{rec['latency_s']:.2f}s"
                print(f"[{spec.name}] {i}/{len(todo)} {it.id} {status}", flush=True)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(
                        f"[{spec.name}] DỪNG: {MAX_CONSECUTIVE_ERRORS} trang liên tiếp bị lỗi, "
                        "nhiều khả năng cấu hình model sai. Xem log ở trên.",
                        flush=True,
                    )
                    exit_code = 2
                    break
        adapter.close()
    if n_errors:
        print(f"[{spec.name}] {n_errors} mẫu bị lỗi; sửa xong chạy lại với --retry-errors", flush=True)
        exit_code = exit_code or 1

    session["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session["peak_vram_mib"] = vram.peak_mib or None
    meta["sessions"].append(session)
    meta["peak_vram_mib"] = max((s.get("peak_vram_mib") or 0) for s in meta["sessions"]) or None
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return exit_code


def _free_cuda_cache():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ocrbench.worker")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--categories")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--per-category", type=int)
    a = ap.parse_args(argv)
    spec = load_config(a.config).model(a.model)
    cats = a.categories.split(",") if a.categories else None
    sys.exit(run_model(a.config, spec, a.split, cats, a.limit, a.retry_errors, a.per_category))


if __name__ == "__main__":
    main()
