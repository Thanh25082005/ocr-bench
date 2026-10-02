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


def _count_done(path: Path, ids: set[str] | None = None) -> int:
    preds = load_predictions(path)
    return len(preds) if ids is None else sum(1 for k in preds if k in ids)


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
    shard: bool = True,
) -> dict[str, int]:
    """Trả về {tên model: mã thoát}.

    Chỉ chạy MỘT model, model đó vừa 1 GPU, và có từ 2 GPU trở lên (`shard=True`): tool nạp model lên mọi GPU
    và chia đều số mẫu (mỗi tiến trình một phần cố định), nhanh hơn ~số GPU lần. Nhiều model thì mỗi model một GPU.

    Mỗi khi một model kết thúc (và định kỳ `status.every_min` phút trong lúc chạy), ghi OUT/status.md và,
    nếu config bật `status.push`, đẩy kết quả lên GitHub, để server sập cũng không mất tiến độ.
    """
    from .status import safe_push

    if inline:
        results = {}
        for s in specs:
            safe_push(cfg, [s.name], f"bắt đầu {s.name}")
            results[s.name] = run_model(cfg.path, s, split, categories, limit, retry_errors, per_category)
            safe_push(cfg, [], f"{s.name} kết thúc (mã {results[s.name]})")
        return results

    run_ids = {it.id for it in load_manifest(cfg.dataset, split=split, categories=categories, limit=limit,
                                             per_category=per_category)}
    total = len(run_ids)
    slots: list[str | None] = list(gpus) if gpus else [None]
    free = list(slots)
    n_gpu_slots = len([s for s in slots if s is not None])
    if shard and len(specs) == 1 and specs[0].gpus == 1 and n_gpu_slots >= 2:
        queue = deque((specs[0], k, n_gpu_slots) for k in range(n_gpu_slots))
        print(f"⇉ {specs[0].name}: vừa 1 GPU, chia mẫu cho {n_gpu_slots} GPU (mỗi GPU một bản model)", flush=True)
    else:
        queue = deque((s, 0, 1) for s in specs)
    running: dict[subprocess.Popen, tuple] = {}
    results: dict[str, int] = {}
    shards_left: dict[str, int] = {}
    t_start: dict[str, float] = {}
    last_status = time.monotonic()
    last_push = time.monotonic()

    def running_names():
        return sorted({s.name for s, *_ in running.values()})

    def start(spec: ModelSpec, taken: list, k: int = 0, n: int = 1):
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
        if n > 1:
            cmd += ["--shard", str(k), "--num-shards", str(n)]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        first = spec.name not in shards_left
        shards_left[spec.name] = shards_left.get(spec.name, 0) + 1
        t_start.setdefault(spec.name, time.monotonic())
        running[proc] = (spec, taken, log, k, n)
        where = f"GPU {','.join(taken)}" if taken[0] is not None else "thiết bị mặc định"
        part = f" (phần {k + 1}/{n})" if n > 1 else ""
        print(f"▶ {spec.name}{part}: bắt đầu trên {where} (log: {out / 'run.log'})", flush=True)
        if first:
            safe_push(cfg, running_names(), f"bắt đầu {spec.name}")

    try:
        while queue or running:
            # Khởi động model tiếp theo nếu đủ GPU rảnh (giữ đúng thứ tự trong config)
            while queue:
                head = queue[0][0]
                need = min(max(head.gpus, 1), len(slots))
                if head.gpus > len(slots):
                    print(f"⚠ {head.name} cần {head.gpus} GPU nhưng chỉ có {len(slots)}; chạy với {len(slots)}")
                if len(free) < need:
                    break
                taken, free[:] = free[:need], free[need:]
                spec, k, n = queue.popleft()
                start(spec, taken, k, n)

            time.sleep(2)
            for proc in list(running):
                code = proc.poll()
                if code is None:
                    continue
                spec, taken, log, k, n = running.pop(proc)
                log.close()
                free.extend(taken)
                results[spec.name] = max(results.get(spec.name, 0), code)
                shards_left[spec.name] -= 1
                if shards_left[spec.name] > 0:  # còn phần khác của cùng model đang chạy
                    print(f"  · {spec.name} phần {k + 1}/{n} xong (mã {code}), chờ các phần còn lại", flush=True)
                    continue
                code = results[spec.name]
                mins = (time.monotonic() - t_start[spec.name]) / 60
                done = _count_done(run_dir(cfg.output_dir, split, spec.name) / "predictions.jsonl", run_ids)
                mark = "✔" if code == 0 else "✘"
                print(f"{mark} {spec.name}: kết thúc (mã {code}) sau {mins:.1f} phút, {done}/{total} mẫu", flush=True)
                if code != 0:
                    _print_tail(run_dir(cfg.output_dir, split, spec.name) / "run.log")
                safe_push(cfg, running_names(), f"{spec.name} kết thúc (mã {code}, {done}/{total} mẫu)")
                last_push = time.monotonic()

            if running and time.monotonic() - last_push >= cfg.status.every_min * 60:
                safe_push(cfg, running_names(), "cập nhật định kỳ")
                last_push = time.monotonic()

            if running and time.monotonic() - last_status >= STATUS_EVERY_S:
                last_status = time.monotonic()
                parts = [
                    f"{name} {_count_done(run_dir(cfg.output_dir, split, name) / 'predictions.jsonl', run_ids)}/{total}"
                    for name in running_names()
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
