"""Điều phối: chạy lần lượt (hoặc song song trên nhiều GPU) các model trong config.

Mỗi model chạy trong một tiến trình con riêng:
- model xong là VRAM được trả lại hết, model sau nạp vào sạch sẽ;
- model này lỗi / hết bộ nhớ không kéo model khác chết theo;
- mỗi model có thể dùng một python (venv) riêng nếu thư viện xung đột nhau.

Với `--gpus 0,1` (Kaggle 2×T4), hai model 1-GPU chạy cùng lúc, mỗi model một card;
model khai báo `gpus: 2` sẽ đợi đến khi cả hai card rảnh.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from .config import Config, ModelSpec
from .dataset import load_manifest
from .worker import load_predictions, run_dir, run_model

STATUS_EVERY_S = 60


def _count_done(path: Path) -> int:
    return len(load_predictions(path))


def run_all(
    cfg: Config,
    specs: list[ModelSpec],
    split: str,
    categories: list[str] | None = None,
    limit: int | None = None,
    retry_errors: bool = False,
    gpus: list[str] | None = None,
    inline: bool = False,
    per_category: int | None = None,
) -> dict[str, int]:
    """Trả về {tên model: mã thoát}."""
    if inline:
        return {s.name: run_model(cfg.path, s, split, categories, limit, retry_errors, per_category) for s in specs}

    total = len(load_manifest(cfg.dataset, split=split, categories=categories, limit=limit, per_category=per_category))
    slots: list[str | None] = list(gpus) if gpus else [None]
    free = list(slots)
    queue = deque(specs)
    running: dict[subprocess.Popen, tuple[ModelSpec, list, object, float]] = {}
    results: dict[str, int] = {}
    last_status = time.monotonic()

    def start(spec: ModelSpec, taken: list):
        out = run_dir(cfg.output_dir, split, spec.name)
        out.mkdir(parents=True, exist_ok=True)
        log = (out / "run.log").open("a", encoding="utf-8")
        env = {**os.environ, **spec.env, "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        if taken[0] is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(taken)
        cmd = [spec.python or sys.executable, "-m", "ocrbench.worker", "--config", str(cfg.path),
               "--model", spec.name, "--split", split]
        if categories:
            cmd += ["--categories", ",".join(categories)]
        if limit:
            cmd += ["--limit", str(limit)]
        if per_category:
            cmd += ["--per-category", str(per_category)]
        if retry_errors:
            cmd.append("--retry-errors")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        running[proc] = (spec, taken, log, time.monotonic())
        where = f"GPU {','.join(taken)}" if taken[0] is not None else "thiết bị mặc định"
        print(f"▶ {spec.name}: bắt đầu trên {where} (log: {out / 'run.log'})", flush=True)

    try:
        while queue or running:
            # Khởi động model tiếp theo nếu đủ GPU rảnh (giữ đúng thứ tự trong config)
            while queue:
                need = min(max(queue[0].gpus, 1), len(slots))
                if queue[0].gpus > len(slots):
                    print(f"⚠ {queue[0].name} cần {queue[0].gpus} GPU nhưng chỉ có {len(slots)}; chạy với {len(slots)}")
                if len(free) < need:
                    break
                taken, free[:] = free[:need], free[need:]
                start(queue.popleft(), taken)

            time.sleep(2)
            for proc in list(running):
                code = proc.poll()
                if code is None:
                    continue
                spec, taken, log, t0 = running.pop(proc)
                log.close()
                free.extend(taken)
                results[spec.name] = code
                mins = (time.monotonic() - t0) / 60
                done = _count_done(run_dir(cfg.output_dir, split, spec.name) / "predictions.jsonl")
                mark = "✔" if code == 0 else "✘"
                print(f"{mark} {spec.name}: kết thúc (mã {code}) sau {mins:.1f} phút, {done}/{total} mẫu", flush=True)
                if code != 0:
                    _print_tail(run_dir(cfg.output_dir, split, spec.name) / "run.log")

            if running and time.monotonic() - last_status >= STATUS_EVERY_S:
                last_status = time.monotonic()
                parts = [
                    f"{s.name} {_count_done(run_dir(cfg.output_dir, split, s.name) / 'predictions.jsonl')}/{total}"
                    for s, *_ in running.values()
                ]
                print("… đang chạy: " + " | ".join(parts) + (f" | chờ: {len(queue)} model" if queue else ""), flush=True)
    except KeyboardInterrupt:
        print("Đang dừng các tiến trình con... (chạy lại lệnh để tiếp tục từ chỗ dừng)", flush=True)
        for proc in running:
            proc.terminate()
        for proc in running:
            proc.wait(timeout=30)
        raise
    return results


def _print_tail(path: Path, n: int = 25):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return
    print("   ── cuối log ──")
    for line in lines:
        print("   " + line)
