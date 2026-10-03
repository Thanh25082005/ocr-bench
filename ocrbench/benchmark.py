"""`ocrbench benchmark`: sinh WORK/BENCHMARK.md — bảng xếp hạng mọi model trong hàng đợi /goal đã có kết quả.

Mỗi model trong hàng đợi là MỘT dòng, ghép từ các biến thể của nó (vd. `amad_vlm6__2gpu` cho nhóm thường,
`amad_vlm6__long` cho tài liệu dài). Số liệu chỉ lấy từ chấm điểm của tool, không ai gõ tay.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .dataset import fingerprint, load_manifest
from .goal import LONG_CATEGORIES, SKIP_REASONS, evaluate, goal_items, load_queue, load_skips
from .score import aggregate, score_model


def _pct(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def build_benchmark(cfg: Config) -> str:
    q = load_queue()
    split = q["split"]
    items = goal_items(cfg, q)  # nhóm tài liệu dài: tập con cố định (nếu goal/queue.yaml đặt long_per_category)
    cats = sorted({it.category for it in items})
    # nhóm bảng = MỌI mẫu đều là bảng; nhóm trộn (vd. syn_degraded có lẫn hóa đơn) chấm bằng CER
    table_cats = {c for c in cats if all(it.gt_type == "table_html" for it in items if it.category == c)}
    states, _ = evaluate(cfg)
    skips = load_skips(cfg)

    rows = []  # (queue name, variants used, per-category agg, overall dict)
    for st in states:
        if st.normal is None or st.normal[1] == 0:
            continue
        v_norm = st.normal[0]
        v_long = st.long[0] if st.long and st.long[1] else None
        # nhóm tài liệu dài lấy từ biến thể dành cho tài liệu dài (nếu có), các nhóm khác từ biến thể thường
        used = {c: (v_long if c in LONG_CATEGORIES and v_long else v_norm) for c in cats}
        scored = {v: score_model(cfg, split, v, items) for v in set(used.values())}
        per_cat, all_rows = {}, []
        for c in cats:
            rows_c = [x for x in scored[used[c]][0] if x["category"] == c]
            per_cat[c] = aggregate(rows_c)
            all_rows += rows_c
        metas = {v: meta for v, (_, meta) in scored.items()}
        text_cers = [per_cat[c]["cer_mean"] for c in cats if c in per_cat and c not in table_cats
                     and per_cat[c]["complete"] and per_cat[c]["cer_mean"] is not None]
        cells = [per_cat[c]["cell_exact_mean"] for c in cats if c in table_cats and c in per_cat
                 and per_cat[c]["complete"] and per_cat[c]["cell_exact_mean"] is not None]
        overall = aggregate(all_rows) if all_rows else None
        vram = max((m.get("peak_vram_mib") or 0) for m in metas.values()) if metas else 0
        n_done = sum(per_cat[c]["n_scored"] for c in cats)
        coverage = "đủ" if n_done == len(items) else f"{n_done}/{len(items)}"
        rows.append(dict(name=st.name, state=coverage, used=used, per_cat=per_cat, overall=overall,
                         text=statistics.fmean(text_cers) if text_cers else None,
                         cells=statistics.fmean(cells) if cells else None, vram=vram,
                         variants=sorted(set(used.values()))))

    rows.sort(key=lambda r: (r["text"] is None, r["text"] if r["text"] is not None else 9))
    now = datetime.now(timezone.utc)
    L = ["# Benchmark OCR (tiếng Anh + tiếng Ả Rập)", "",
         f"Cập nhật **{now:%Y-%m-%d %H:%M} UTC** · split `{split}` · {len(items)} mẫu"
         + (f" (tài liệu dài: {q['long_per_category']} mẫu cố định mỗi nhóm)" if q.get("long_per_category") else "")
         + " · "
         f"dấu vân tay bộ test `{fingerprint(load_manifest(cfg.dataset))}` · sinh tự động bởi `ocrbench benchmark`.", "",
         "Chỉ số: **CER** = tỉ lệ lỗi ký tự (thấp = tốt) cho văn bản; **Ô đúng** = ô bảng đúng giá trị và đúng vị trí "
         "(cao = tốt). Giải thích đầy đủ: `docs/METRICS.md` (nhánh `main`).", "",
         "## Bảng tổng hợp", "",
         "Xếp theo CER văn bản trung bình (trung bình các nhóm văn bản, mỗi nhóm nặng như nhau). "
         "Chỉ tính nhóm model đã chạy đủ mẫu.", "",
         "| # | Model | Đã chạy | CER văn bản | Ô đúng (bảng) | Lặp/thừa | Lỗi/rỗng | s/mẫu | VRAM đỉnh | Biến thể đã dùng |",
         "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(rows, 1):
        o = r["overall"] or {}
        vram_s = f"{r['vram'] / 1024:.1f} GB" if r["vram"] else "—"
        L.append(f"| {i} | {r['name']} | {r['state']} | {_pct(r['text'])} | {_pct(r['cells'])} "
                 f"| {o.get('n_hallucination', '—')} | {o.get('n_fail', '—')} "
                 f"| {o.get('latency_mean_s') or 0:.2f} | {vram_s} "
                 f"| {', '.join(f'`{v}`' for v in r['variants'])} |")
    if not rows:
        L.append("| — | (chưa có model nào chạy xong) | | | | | | | | |")

    if rows:
        L += ["", "## Theo nhóm", "",
              "Nhóm văn bản: CER ±95% (thấp = tốt). Nhóm bảng (`*`): ô đúng vị trí ±95% (cao = tốt), kèm CER của chữ "
              "trong bảng (đọc đúng chữ nhưng không dựng lại được bảng thì ô đúng = 0% mà CER vẫn thấp). "
              "`—` = chưa chạy đủ.", "",
              "| Nhóm | " + " | ".join(r["name"] for r in rows) + " |", "|---|" + "---:|" * len(rows)]
        for c in cats:
            cells = []
            for r in rows:
                a = r["per_cat"].get(c)
                if not a or not a["complete"]:
                    cells.append("—")
                elif c in table_cats:
                    ci = a["cell_exact_ci95"]
                    cells.append(f"{_pct(a['cell_exact_mean'])}" + (f" ±{ci * 100:.1f}" if ci else "")
                                 + f" · CER {_pct(a['cer_mean'])}")
                else:
                    ci = a["cer_ci95"]
                    cells.append(f"{_pct(a['cer_mean'])}" + (f" ±{ci * 100:.1f}" if ci else ""))
            L.append(f"| {c}{' *' if c in table_cats else ''} | " + " | ".join(cells) + " |")

        long_cats = [c for c in cats if c in LONG_CATEGORIES]
        if long_cats:
            L += ["", "## Tài liệu dài: độ chính xác theo vị trí (đầu Q1 → cuối Q4)", "",
                  "| Model | Nhóm | Q1 | Q2 | Q3 | Q4 | Bị cắt | Hàng sai số cột |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                for c in long_cats:
                    a = r["per_cat"].get(c)
                    if a and a["n_scored"]:
                        L.append(f"| {r['name']} | {c} | " + " | ".join(_pct(x, 0) for x in a["pos_q"])
                                 + f" | {a['n_hit_max_tokens']} | {a['rows_wrong_ncols']} |")

    if skips:
        L += ["", "## Model bị bỏ qua", "", "| Model | Lý do | Chi tiết |", "|---|---|---|"]
        for name, s in skips.items():
            L.append(f"| {name} | {SKIP_REASONS.get(s.get('reason_code'), s.get('reason_code'))} "
                     f"| {str(s.get('detail', '')).replace('|', '/')} |")

    not_started = [st.name for st in states if (st.normal is None or st.normal[1] == 0) and st.name not in skips]
    if not_started:
        L += ["", f"Chưa chạy: {', '.join(not_started)}."]
    L.append("")
    return "\n".join(L)


def write_benchmark(cfg: Config) -> Path:
    path = cfg.work_dir / "BENCHMARK.md"
    path.write_text(build_benchmark(cfg), encoding="utf-8")
    return path
