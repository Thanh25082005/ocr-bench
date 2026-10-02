"""`ocrbench speedtest`: đo TỐC ĐỘ và ĐỘ CHÍNH XÁC của một model với nhiều cấu hình tối ưu, trên cùng một tập trang
lấy từ bộ dev (mỗi nhóm vài trang, chỉ tài liệu 1 trang). Dùng để chọn cấu hình triển khai: nhanh nhất mà CER không tăng.

    ocrbench speedtest --config C --model sherif_handwriting --per-category 3 \\
        --variant "max_new_tokens=4096" --variant "max_new_tokens=4096,batch_size=4" ...

Mỗi biến thể là danh sách khóa=giá trị ghi đè tham số model; `max_pixels=N` là viết tắt của processor_kwargs.max_pixels.
"""

from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path

import yaml

from .convert import Converter
from .dataset import load_manifest
from .metrics import find_repetition
from .normalize import normalize
from .worker import load_image

DEFAULT_VARIANTS = [
    "max_new_tokens=4096",
    "max_new_tokens=4096,batch_size=4",
    "max_new_tokens=4096,batch_size=4,stop_on_loop=true",
]
# Không có biến thể giảm độ phân giải (max_pixels) trong danh sách mặc định: người dùng không chấp nhận đánh đổi
# độ chính xác. Muốn thử thì thêm bằng --variant.


def parse_variant(v: str) -> dict:
    params: dict = {}
    for part in filter(None, (x.strip() for x in v.split(","))):
        key, _, val = part.partition("=")
        val = yaml.safe_load(val)
        if key == "max_pixels":
            params.setdefault("processor_kwargs", {})["max_pixels"] = val
        else:
            params[key] = val
    return params


def run_speedtest(config_path, model: str, variants: list[str], per_category: int = 3, gpus: str = "auto",
                  out_path: Path | None = None) -> str:
    from rapidfuzz.distance import Levenshtein

    from .config import load_config

    cfg = load_config(config_path)
    items = [it for it in load_manifest(cfg.dataset, split="dev", per_category=per_category) if it.n_pages == 1]
    images = [load_image(it.image, None) for it in items]
    print(f"{len(items)} trang (dev, {per_category} trang mỗi nhóm, chỉ tài liệu 1 trang)", flush=True)
    rows = []
    for v in variants:
        params = parse_variant(v)
        print(f"\n▶ {v}: nạp model...", flush=True)
        conv = Converter(config_path, model, params=params, gpus=gpus)
        ready = []
        for k, (it, img) in enumerate(zip(items, images)):
            im, item = conv._prepare(img.copy(), it.image, 1, "table" if it.gt_type == "table_html" else "text")
            item.category, item.id = it.category, it.id  # để prompt theo nhóm (nếu config có) giống benchmark
            ready.append((k, 1, im, item))
        batches = [ready[i:i + conv.batch] for i in range(0, len(ready), conv.batch)]
        t = time.perf_counter()
        results = dict(conv._run_batches(batches))
        wall = time.perf_counter() - t
        cers, n_cut, n_loop, n_err = [], 0, 0, 0
        for k, it in enumerate(items):
            r = results[k]
            if r.error:
                n_err += 1
            ref, hyp = normalize(it.gt, cfg.normalization), normalize(r.text, cfg.normalization)
            cers.append(Levenshtein.distance(ref, hyp) / max(len(ref), 1))
            n_cut += bool(r.note and "bị cắt" in r.note)
            n_loop += bool((r.note and "lặp" in r.note) or find_repetition(r.text))
        rows.append(dict(variant=v, gpus=len(conv.adapters), batch=conv.batch, wall=wall,
                         ppm=len(items) / wall * 60, cer=statistics.fmean(cers), cer_med=statistics.median(cers),
                         cut=n_cut, loop=n_loop, err=n_err, fallbacks=conv.fallbacks))
        print(f"  {len(items)} trang trong {wall:.0f}s = {rows[-1]['ppm']:.1f} trang/phút · CER {rows[-1]['cer']:.1%}",
              flush=True)
        conv.close()
        del conv
        gc.collect()
    base = rows[0]
    L = [f"# Speedtest `{model}` — {len(items)} trang dev", "",
         "| Cấu hình | GPU | Batch | Trang/phút | Nhanh gấp | CER TB | CER trung vị | Bị cắt | Lặp | Lỗi | Batch phải chạy lại |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| `{r['variant']}` | {r['gpus']} | {r['batch']} | {r['ppm']:.1f} | {r['ppm'] / base['ppm']:.2f}× "
                 f"| {r['cer']:.1%} | {r['cer_med']:.1%} | {r['cut']} | {r['loop']} | {r['err']} | {r['fallbacks']} |")
    L += ["", "Chọn cấu hình nhanh nhất mà CER TB không cao hơn dòng đầu quá ~0,5 điểm % và cột Lỗi = 0. "
          "'Batch phải chạy lại' > 0 thường là hết VRAM: giảm batch_size."]
    text = "\n".join(L)
    if out_path:
        Path(out_path).write_text(text + "\n", encoding="utf-8")
    return text
