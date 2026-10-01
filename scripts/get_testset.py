#!/usr/bin/env python
"""Tải bộ test từ Google Drive về máy (vd. server Kaggle).

    pip install gdown
    python scripts/get_testset.py --out /kaggle/working/testset

- ID Drive có thể là một THƯ MỤC (tải từng file, song song, thử lại khi lỗi) hoặc một FILE nén
  .zip / .tar / .tar.gz (tải rồi giải nén).
- Chạy lại sẽ bỏ qua file đã tải xong, nên bị ngắt giữa chừng thì cứ chạy lại.
- Cuối cùng kiểm tra đủ số file và so dấu vân tay dữ liệu với giá trị mong đợi.

Thư mục/file trên Drive phải để chế độ "Bất kỳ ai có đường liên kết đều xem được".
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_ID = "1Hv5xn8vKAMnje-pf-hAZSRLd1FIUYGeD"
EXPECTED_FINGERPRINT = "aa8fdd44e1715845"  # dấu vân tay bộ test gốc (ocrbench validate)


def _download_one(gdown, file_id: str, dest: Path, retries: int) -> str | None:
    """Tải một file; trả về None nếu thành công, ngược lại là thông báo lỗi."""
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            out = gdown.download(id=file_id, output=str(tmp), quiet=True)
            if out and tmp.exists() and tmp.stat().st_size > 0:
                tmp.rename(dest)
                return None
            err = "tải về rỗng"
        except Exception as e:  # mạng chập chờn / Drive tạm chặn
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 3 * 2 ** attempt))
    tmp.unlink(missing_ok=True)
    return f"{dest}: {err}"


def fetch_folder(gdown, folder_id: str, out: Path, workers: int, retries: int, max_files: int | None) -> int:
    print("Đang đọc danh sách file trên Drive (khoảng 30 giây)...", flush=True)
    files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
    if max_files:
        files = files[:max_files]
    todo = []
    for f in files:
        dest = out / f.path
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        todo.append((f.id, dest))
    print(f"Trên Drive: {len(files)} file · đã có: {len(files) - len(todo)} · cần tải: {len(todo)}", flush=True)

    errors, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, gdown, fid, dest, retries) for fid, dest in todo]
        for fut in as_completed(futures):
            done += 1
            if (err := fut.result()):
                errors.append(err)
            if done % 200 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-6)
                left = (len(todo) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(todo)} file · {rate:.1f} file/s · còn khoảng {left / 60:.0f} phút"
                      + (f" · lỗi: {len(errors)}" if errors else ""), flush=True)
    for e in errors[:10]:
        print("  LỖI", e)
    if errors:
        print(f"\n{len(errors)} file chưa tải được. Chạy lại đúng lệnh này để tải tiếp phần còn thiếu.")
    return len(files)


def fetch_archive(gdown, file_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(gdown.download(id=file_id, output=str(out.parent) + "/", quiet=False))
    print(f"Giải nén {archive.name} ...", flush=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
    else:
        sys.exit(f"{archive} không phải .zip hay .tar — không biết giải nén thế nào.")
    archive.unlink()
    # nén kèm thư mục gốc (testset/manifest.jsonl) thì đưa nội dung lên một cấp
    inner = [p for p in out.iterdir() if p.is_dir()]
    if not (out / "manifest.jsonl").exists() and len(inner) == 1 and (inner[0] / "manifest.jsonl").exists():
        for p in inner[0].iterdir():
            shutil.move(str(p), out / p.name)
        inner[0].rmdir()


def verify(out: Path, expected_files: int | None, expected_fp: str | None) -> bool:
    manifest = out / "manifest.jsonl"
    if not manifest.exists():
        print(f"✘ Không thấy {manifest}")
        return False
    n_files = sum(1 for p in out.rglob("*") if p.is_file() and not p.name.endswith(".part"))
    ok = True
    if expected_files is not None:
        mark = "✔" if n_files >= expected_files else "✘"
        ok &= n_files >= expected_files
        print(f"{mark} Số file: {n_files} / {expected_files}")
    try:
        from ocrbench.dataset import fingerprint, load_manifest, summarize
    except ImportError:
        print("(chưa cài ocrbench nên bỏ qua kiểm tra dấu vân tay)")
        return ok
    items = load_manifest(manifest)
    missing = [it.id for it in items if not all(p.exists() for p in it.images)]
    if missing:
        ok = False
        print(f"✘ {len(missing)} mẫu thiếu ảnh, ví dụ: {missing[:3]}")
    fp = fingerprint(items)
    if expected_fp:
        same = fp == expected_fp
        ok &= same
        print(f"{'✔' if same else '✘'} Dấu vân tay: {fp} (mong đợi {expected_fp})")
    else:
        print(f"Dấu vân tay: {fp}")
    print()
    print(summarize(items))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=DEFAULT_ID, help="ID thư mục hoặc file trên Google Drive")
    ap.add_argument("--out", default="testset", help="thư mục đích (sẽ chứa manifest.jsonl)")
    ap.add_argument("--workers", type=int, default=16,
                    help="số luồng tải song song; 16 luồng ≈ 3 file/s (đã đo). Bị Drive chặn thì giảm xuống 4")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--expect-fingerprint", default=EXPECTED_FINGERPRINT, help="để trống ('') nếu bộ test đã đổi")
    ap.add_argument("--max-files", type=int, help="chỉ tải N file đầu (để thử)")
    a = ap.parse_args()

    try:
        import gdown
    except ImportError:
        sys.exit("Thiếu gdown: pip install gdown")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        n = fetch_folder(gdown, a.id, out, a.workers, a.retries, a.max_files)
    except Exception as e:
        # không phải thư mục -> thử như một file nén
        print(f"Không đọc được như thư mục ({type(e).__name__}); thử tải như một file nén.", flush=True)
        fetch_archive(gdown, a.id, out)
        n = None
    if a.max_files:
        print(f"Đã thử tải {a.max_files} file đầu; bỏ qua bước kiểm tra.")
        return
    ok = verify(out, n, a.expect_fingerprint or None)
    print("\nXONG: bộ test đầy đủ và khớp." if ok else "\nCHƯA ĐỦ: chạy lại lệnh để tải tiếp.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
