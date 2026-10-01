"""Tạo báo cáo so sánh các model: report.md (đọc) + summary.csv (phân tích tiếp)."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .config import Config
from .dataset import Item
from .score import aggregate, score_model

WORST_K = 5


def _pct(x, digits=1):
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def _ci(m, ci):
    if m is None:
        return "—"
    return f"{m * 100:.1f}% ±{ci * 100:.1f}" if ci is not None else f"{m * 100:.1f}%"


def _num(x, fmt="{:.3f}"):
    return "—" if x is None else fmt.format(x)


def _clip(s: str, n: int = 80) -> str:
    s = " ".join((s or "").split())
    s = s if len(s) <= n else s[: n - 1] + "…"
    return s.replace("|", "\\|")


def build_report(cfg: Config, split: str, models: list[str], items: list[Item]) -> Path:
    by_id = {it.id: it for it in items}
    categories = sorted({it.category for it in items})
    table_cats = sorted({it.category for it in items if it.gt_type == "table_html"})

    results = {}  # model -> (rows, meta, overall, per_cat)
    for m in models:
        rows, meta = score_model(cfg, split, m, items)
        per_cat = {c: aggregate([r for r in rows if r["category"] == c]) for c in categories}
        results[m] = (rows, meta, aggregate(rows), per_cat)

    def sort_key(m):
        ov = results[m][2]
        return (not ov["complete"], ov["cer_mean"] if ov["cer_mean"] is not None else 9e9)

    order = sorted(models, key=sort_key)
    out_dir = Path(cfg.output_dir) / split / "_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV ---------------------------------------------------------------
    fields = ["model", "category", "n", "n_scored", "complete", "cer_mean", "cer_ci95", "cer_median", "cer_p90",
              "cer_micro", "cer_raw_mean", "wer_mean", "teds_n", "teds_mean", "teds_ci95", "teds_struct_mean",
              "n_fail", "n_hallucination", "hallucination_upper95", "latency_mean_s", "peak_vram_mib",
              "cell_exact_mean", "cell_exact_ci95", "cell_aligned_mean", "rows_wrong_ncols", "row_count_off", "n_hit_max_tokens",
              "pos_q1", "pos_q2", "pos_q3", "pos_q4"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in order:
            rows, meta, overall, per_cat = results[m]
            for cat, agg in [("ALL", overall), *per_cat.items()]:
                w.writerow({"model": m, "category": cat, "peak_vram_mib": meta.get("peak_vram_mib"),
                            **{k: agg[k] for k in fields if k in agg},
                            **{f"pos_q{k + 1}": agg["pos_q"][k] for k in range(4)}})

    # --- Markdown ----------------------------------------------------------
    L = []
    L.append(f"# Kết quả OCR benchmark — split `{split}`")
    L.append("")
    L.append(f"Tạo lúc {datetime.now():%Y-%m-%d %H:%M} · {len(items)} mẫu · {len(categories)} nhóm · "
             f"{len(models)} model · manifest `{cfg.dataset}`")
    L.append("")
    from .dataset import fingerprint, load_manifest

    current_fp = fingerprint(load_manifest(cfg.dataset))
    stale = [m for m in order
             if any(s.get("dataset_fingerprint") not in (None, current_fp) for s in results[m][1].get("sessions", []))]
    if stale:
        L.append(f"> ⚠ **CẢNH BÁO: dữ liệu đã thay đổi.** Các model sau đã chạy (toàn bộ hoặc một phần) trên bộ dữ liệu "
                 f"khác với bộ hiện tại (dấu vân tay `{current_fp}`): {', '.join(stale)}. Kết quả của chúng không so sánh "
                 "được với các model khác. Phải xóa thư mục kết quả của các model này và chạy lại.")
        L.append("")
    L.append("## Tổng quan")
    L.append("")
    L.append("| Model | Mẫu | CER (norm) ±95% | CER micro | CER raw | WER | TEDS bảng | Lỗi/rỗng | Lặp/thừa (trần 95%) | s/mẫu | VRAM đỉnh |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in order:
        rows, meta, ov, _ = results[m]
        name = m if ov["complete"] else f"{m} ⚠ chưa đủ"
        vram = meta.get("peak_vram_mib")
        L.append(
            f"| {name} | {ov['n_scored']}/{ov['n']} | {_ci(ov['cer_mean'], ov['cer_ci95'])} | {_pct(ov['cer_micro'])} "
            f"| {_pct(ov['cer_raw_mean'])} | {_pct(ov['wer_mean'])} | {_num(ov['teds_mean'])} "
            f"| {ov['n_fail']} | {ov['n_hallucination']} (≤{_pct(ov['hallucination_upper95'])}) "
            f"| {_num(ov['latency_mean_s'], '{:.2f}')} | {f'{vram / 1024:.1f} GB' if vram else '—'} |"
        )
    L.append("")

    L.append("## CER (norm) theo nhóm — thấp hơn là tốt hơn")
    L.append("")
    L.append("| Nhóm (số mẫu) | " + " | ".join(order) + " |")
    L.append("|---|" + "---:|" * len(order))
    for c in categories:
        n = sum(1 for it in items if it.category == c)
        cells = []
        for m in order:
            a = results[m][3][c]
            cell = _ci(a["cer_mean"], a["cer_ci95"])
            if not a["complete"]:
                cell += f" ({a['n_scored']}/{a['n']})"
            cells.append(cell)
        L.append(f"| {c} ({n}) | " + " | ".join(cells) + " |")
    L.append("")

    if table_cats:
        L.append("## TEDS theo nhóm có bảng — cao hơn là tốt hơn (1.0 = khớp hoàn toàn)")
        L.append("")
        L.append("Ô ghi `nội dung / cấu trúc`: TEDS đầy đủ và TEDS chỉ xét cấu trúc hàng-cột.")
        L.append("")
        L.append("| Nhóm | " + " | ".join(order) + " |")
        L.append("|---|" + "---:|" * len(order))
        for c in table_cats:
            cells = []
            for m in order:
                a = results[m][3][c]
                cells.append(f"{_num(a['teds_mean'])} / {_num(a['teds_struct_mean'])}")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")

    long_cats = sorted({it.category for it in items if it.meta.get("length_bin")})
    for c in long_cats:
        is_table = any(it.gt_type == "table_html" for it in items if it.category == c)
        bins = sorted({it.meta["length_bin"] for it in items if it.category == c})
        L.append(f"## Tài liệu dài: `{c}`")
        L.append("")
        if is_table:
            L.append("Ô ghi `ô đúng vị trí / ô đúng sau căn hàng · TEDS`. Hai tỉ lệ đầu chênh nhau nhiều nghĩa là model "
                     "bỏ sót hoặc thêm hàng, làm mọi giá trị phía sau bị đẩy lệch hàng.")
        else:
            L.append("Ô ghi `CER · độ phủ phần cuối (Q4)`: độ phủ Q4 là tỉ lệ đoạn ở 1/4 cuối tài liệu có mặt trong kết quả.")
        L.append("")
        L.append("| Độ dài (số mẫu) | " + " | ".join(order) + " |")
        L.append("|---|" + "---:|" * len(order))
        for b in bins:
            n = sum(1 for it in items if it.category == c and it.meta.get("length_bin") == b)
            cells = []
            for m in order:
                a = aggregate([r for r in results[m][0] if r["category"] == c and r.get("length_bin") == b])
                if is_table:
                    cells.append(f"{_pct(a['cell_exact_mean'], 0)} / {_pct(a['cell_aligned_mean'], 0)} · {_num(a['teds_mean'], '{:.2f}')}")
                else:
                    cells.append(f"{_pct(a['cer_mean'])} · {_pct(a['pos_q'][3], 0)}")
            L.append(f"| {b} ({n}) | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Độ chính xác theo vị trí trong tài liệu (đầu → cuối), cột cuối là số mẫu bị cắt vì hết `max_new_tokens`"
                 + (", số hàng sai số cột (dấu hiệu dồn cột) và số mẫu đọc sai số hàng:" if is_table else ":"))
        L.append("")
        extra_head = " | Hàng sai số cột | Sai số hàng" if is_table else ""
        L.append("| Model | Độ dài | Q1 | Q2 | Q3 | Q4 | Bị cắt" + extra_head + " |")
        L.append("|---|---|---:|---:|---:|---:|---:|" + ("---:|---:|" if is_table else ""))
        for m in order:
            for b in bins:
                a = aggregate([r for r in results[m][0] if r["category"] == c and r.get("length_bin") == b])
                q = " | ".join(_pct(x, 0) for x in a["pos_q"])
                tail = f" | {a['rows_wrong_ncols']} | {a['row_count_off']}" if is_table else ""
                L.append(f"| {m} | {b} | {q} | {a['n_hit_max_tokens']}{tail} |")
        L.append("")

    L.append(f"## {WORST_K} mẫu tệ nhất của mỗi model")
    L.append("")
    for m in order:
        rows = [r for r in results[m][0] if not r["missing"]]
        worst = sorted(rows, key=lambda r: r["cer"], reverse=True)[:WORST_K]
        if not worst:
            continue
        L.append(f"### {m}")
        L.append("")
        L.append("| id | nhóm | CER | cờ | đáp án (đầu) | model đọc (đầu) |")
        L.append("|---|---|---:|---|---|---|")
        preds = _load_texts(cfg, split, m)
        for r in worst:
            hyp = r["error"] or preds.get(r["id"], "")
            L.append(f"| `{r['id']}` | {r['category']} | {_pct(r['cer'])} | {', '.join(r['flags']) or '—'} "
                     f"| {_clip(by_id[r['id']].gt)} | {_clip(hyp)} |")
        L.append("")

    L.append("## Cách đọc")
    L.append("")
    L.append("Giải thích đầy đủ, kèm ví dụ: `docs/METRICS.md` trong thư mục code của tool.")
    L.append("")
    L.append("- **CER (norm)**: tỉ lệ lỗi ký tự sau khi chuẩn hóa (mục `normalization` trong config), trung bình theo từng mẫu, "
             "kèm khoảng tin cậy 95%. Có thể > 100% khi model sinh thừa nhiều chữ.")
    L.append("- **CER micro**: tổng số lỗi / tổng số ký tự của cả bộ (trang dài có trọng số lớn hơn).")
    L.append("- **CER raw**: không chuẩn hóa; chênh lệch lớn so với CER norm thường do định dạng (Markdown, tashkeel...).")
    L.append("- **Lỗi/rỗng**: model báo lỗi hoặc trả về chuỗi rỗng. **Lặp/thừa**: kẹt vòng lặp hoặc dài hơn đáp án >1,5 lần "
             "(dấu hiệu bịa chữ); số trong ngoặc là cận trên 95% của tỉ lệ thật.")
    L.append("- **s/mẫu** và **VRAM đỉnh** chỉ để tham khảo; hai model chạy song song trên hai GPU không ảnh hưởng nhau, "
             "nhưng dùng chung CPU và ổ đĩa.")
    L.append("- Model có ⚠ chưa chạy hết bộ dữ liệu nên không so sánh trực tiếp được với các model khác.")
    L.append("- **Tài liệu dài**: *ô đúng vị trí* yêu cầu đúng cả giá trị lẫn hàng/cột; *ô đúng sau căn hàng* bỏ qua việc "
             "thiếu/thừa hàng. Q1→Q4 là độ chính xác theo vị trí trong tài liệu; tụt dần về Q4 nghĩa là model mất ngữ cảnh "
             "hoặc bị cắt khi tài liệu dài. Mẫu *bị cắt* cần tăng `max_new_tokens`.")
    L.append("")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    return out_dir


def _load_texts(cfg: Config, split: str, model: str) -> dict[str, str]:
    from .worker import load_predictions, run_dir

    return {k: v.get("text", "") for k, v in load_predictions(run_dir(cfg.output_dir, split, model) / "predictions.jsonl").items()}
