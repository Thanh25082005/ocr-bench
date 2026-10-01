"""Chấm điểm kết quả đã lưu (không cần GPU, chạy lại bao nhiêu lần cũng được)."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from .config import Config
from .dataset import Item
from .metrics import (
    edit_stats,
    error_rate,
    flags_for,
    merge_tables_html,
    position_recall,
    table_cell_metrics,
    teds,
)
from .normalize import normalize, normalize_raw
from .worker import load_predictions, run_dir

FAIL_FLAGS = {"error", "empty"}
HALLUCINATION_FLAGS = {"repetition", "too_long"}


def score_items(cfg: Config, items: list[Item], preds: dict[str, dict]) -> list[dict]:
    norm = cfg.normalization
    rows = []
    for it in items:
        base = {"id": it.id, "category": it.category, "gt_type": it.gt_type, "doc_id": it.doc_id,
                "pages": it.n_pages, "length_bin": it.meta.get("length_bin")}
        p = preds.get(it.id)
        if p is None:
            rows.append({**base, "missing": True})
            continue
        hyp = p.get("text") or ""
        ref_n, hyp_n = normalize(it.gt, norm), normalize(hyp, norm)
        sn = edit_stats(ref_n, hyp_n)
        sr = edit_stats(normalize_raw(it.gt), normalize_raw(hyp))
        flags = flags_for(ref_n, hyp_n, hyp, p.get("error"))
        if (p.get("extra") or {}).get("hit_max_tokens"):
            flags.append("hit_max_tokens")
        row = {
            **base,
            "missing": False,
            "cer": error_rate(sn["char_edits"], sn["ref_chars"]),
            "wer": error_rate(sn["word_edits"], sn["ref_words"]),
            "cer_raw": error_rate(sr["char_edits"], sr["ref_chars"]),
            "wer_raw": error_rate(sr["word_edits"], sr["ref_words"]),
            **sn,
            "teds": None,
            "teds_struct": None,
            "flags": flags,
            "latency_s": p.get("latency_s"),
            "confidence": p.get("confidence"),
            "error": p.get("error"),
        }
        if it.gt_type == "table_html":
            # bảng ngắt qua nhiều trang: gộp các bảng model đọc được thành một trước khi so
            pred_tables = merge_tables_html(hyp, norm) if it.n_pages > 1 else hyp
            row["teds"] = teds(it.gt, pred_tables, norm)
            row["teds_struct"] = teds(it.gt, pred_tables, norm, structure_only=True)
            row.update(table_cell_metrics(it.gt, hyp, norm) or {})
        else:
            row["pos_q"] = position_recall(ref_n, hyp_n)
        rows.append(row)
    return rows


def _mean_ci(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, None
    return m, 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def wilson_upper(k: int, n: int, z: float = 1.96) -> float | None:
    """Cận trên 95% của tỉ lệ thật khi quan sát k/n (k=0 thì xấp xỉ 3/n)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / denom)


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if not r["missing"]]
    cers = [r["cer"] for r in scored]
    cer_mean, cer_ci = _mean_ci(cers)
    teds_vals = [r["teds"] for r in scored if r["teds"] is not None]
    teds_mean, teds_ci = _mean_ci(teds_vals)
    teds_struct = [r["teds_struct"] for r in scored if r["teds_struct"] is not None]
    lat = [r["latency_s"] for r in scored if r["latency_s"] is not None]
    cell_exact = [r["cell_exact"] for r in scored if r.get("cell_exact") is not None]
    cell_exact_mean, cell_exact_ci = _mean_ci(cell_exact)
    cell_aligned = [r["cell_aligned"] for r in scored if r.get("cell_aligned") is not None]
    pos_q = [[r["pos_q"][k] for r in scored if r.get("pos_q") and r["pos_q"][k] is not None] for k in range(4)]
    n_fail = sum(1 for r in scored if FAIL_FLAGS & set(r["flags"]))
    n_hall = sum(1 for r in scored if HALLUCINATION_FLAGS & set(r["flags"]))
    ref_chars = sum(r["ref_chars"] for r in scored)
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "complete": len(scored) == len(rows),
        "cer_mean": cer_mean,
        "cer_ci95": cer_ci,
        "cer_median": statistics.median(cers) if cers else None,
        "cer_p90": _quantile(cers, 0.9),
        "cer_micro": (sum(r["char_edits"] for r in scored) / ref_chars) if ref_chars else None,
        "cer_raw_mean": statistics.fmean([r["cer_raw"] for r in scored]) if scored else None,
        "wer_mean": statistics.fmean([r["wer"] for r in scored]) if scored else None,
        "teds_n": len(teds_vals),
        "teds_mean": teds_mean,
        "teds_ci95": teds_ci,
        "teds_struct_mean": statistics.fmean(teds_struct) if teds_struct else None,
        "n_fail": n_fail,
        "n_hallucination": n_hall,
        "hallucination_upper95": wilson_upper(n_hall, len(scored)),
        "latency_mean_s": statistics.fmean(lat) if lat else None,
        "cell_exact_mean": cell_exact_mean,
        "cell_exact_ci95": cell_exact_ci,
        "cell_aligned_mean": statistics.fmean(cell_aligned) if cell_aligned else None,
        "rows_wrong_ncols": sum(r.get("rows_wrong_ncols", 0) for r in scored),
        "row_count_off": sum(1 for r in scored if r.get("gt_rows") is not None and r["gt_rows"] != r["pred_rows"]),
        "pos_q": [statistics.fmean(v) if v else None for v in pos_q],
        "n_hit_max_tokens": sum(1 for r in scored if "hit_max_tokens" in r["flags"]),
    }


def score_model(cfg: Config, split: str, model: str, items: list[Item]) -> tuple[list[dict], dict]:
    out = run_dir(cfg.output_dir, split, model)
    rows = score_items(cfg, items, load_predictions(out / "predictions.jsonl"))
    with (out / "scores.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return rows, meta


def list_scored_models(cfg: Config, split: str) -> list[str]:
    d = Path(cfg.output_dir) / split
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "predictions.jsonl").exists())
