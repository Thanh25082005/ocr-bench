"""`ocrbench decide`: áp luật loại / chọn model một cách máy móc, để không ai (người hay agent) phải tự
tính khoảng tin cậy bằng tay.

Chỉ số chính của mỗi nhóm:
- nhóm có đáp án dạng bảng (gt_type = table_html): `cell_exact` — ô đúng vị trí, CAO hơn là tốt hơn;
- các nhóm còn lại: `cer` — tỉ lệ lỗi ký tự (norm), THẤP hơn là tốt hơn.

Luật (áp dụng riêng từng nhóm, chỉ xét model đã chạy ĐỦ mọi mẫu của nhóm):
1. KHÔNG ỔN ĐỊNH: (số mẫu lỗi/rỗng + số mẫu lặp/thừa) / số mẫu >= --max-unstable (mặc định 15%) → loại.
2. KÉM CHẮC CHẮN: khoảng tin cậy 95% của model không chồng với khoảng của model tốt nhất nhóm,
   về phía xấu hơn → loại.
3. Còn lại xếp theo chỉ số chính. --stage screening: lấy tối đa --max-finalists model vào chung kết, cộng thêm
   model OCR truyền thống tốt nhất làm mốc. --stage final: model tốt nhất thắng; model nào có khoảng tin cậy
   chồng với nó thì HÒA, phân định bằng: ít mẫu lặp/thừa hơn → nhanh hơn (s/mẫu) → ít VRAM hơn.
"""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import load_manifest
from .score import aggregate, list_scored_models, score_model

TRADITIONAL = {"tesseract", "easyocr", "paddleocr"}


def _primary(agg: dict, is_table: bool):
    if is_table:
        m, ci = agg["cell_exact_mean"], agg["cell_exact_ci95"] or 0.0
        return m, (m - ci, m + ci) if m is not None else None, True
    m, ci = agg["cer_mean"], agg["cer_ci95"] or 0.0
    return m, (m - ci, m + ci) if m is not None else None, False


def decide(cfg, split: str, stage: str, models: list[str] | None = None, categories=None, per_category=None,
           max_unstable: float = 0.15, max_finalists: int = 4) -> dict:
    items = load_manifest(cfg.dataset, split=split, categories=categories, per_category=per_category)
    models = models or list_scored_models(cfg, split)
    adapters = {m.name: m.adapter for m in cfg.models}
    rows = {m: score_model(cfg, split, m, items) for m in models}
    out = {"stage": stage, "split": split, "max_unstable": max_unstable, "categories": {}}
    for cat in sorted({it.category for it in items}):
        # nhóm bảng = MỌI mẫu đều là bảng; nhóm trộn (vd. syn_degraded có lẫn hóa đơn) chấm bằng CER
        is_table = all(it.gt_type == "table_html" for it in items if it.category == cat)
        entries, skipped = [], []
        for m in models:
            agg = aggregate([r for r in rows[m][0] if r["category"] == cat])
            if not agg["complete"]:
                skipped.append({"model": m, "reason": f"chưa chạy đủ ({agg['n_scored']}/{agg['n']} mẫu)"})
                continue
            value, ci, higher = _primary(agg, is_table)
            if value is None:
                skipped.append({"model": m, "reason": "không tính được chỉ số chính"})
                continue
            unstable = (agg["n_fail"] + agg["n_hallucination"]) / max(agg["n_scored"], 1)
            entries.append({"model": m, "adapter": adapters.get(m), "value": value, "ci": ci, "unstable": unstable,
                            "n": agg["n_scored"], "n_hallucination": agg["n_hallucination"],
                            "latency": agg["latency_mean_s"] or 0.0,
                            "vram": rows[m][1].get("peak_vram_mib") or 0, "hit_max_tokens": agg["n_hit_max_tokens"]})
        metric = "cell_exact (ô đúng vị trí, cao = tốt)" if is_table else "cer (lỗi ký tự, thấp = tốt)"
        better = (lambda a, b: a > b) if is_table else (lambda a, b: a < b)
        for e in entries:
            e["status"], e["reason"] = "giữ", ""
            if e["unstable"] >= max_unstable:
                e["status"], e["reason"] = "loại", f"không ổn định: {e['unstable']:.0%} mẫu lỗi/rỗng/lặp/thừa (≥ {max_unstable:.0%})"
        alive = [e for e in entries if e["status"] == "giữ"]
        alive.sort(key=lambda e: -e["value"] if is_table else e["value"])
        best = alive[0] if alive else None
        if best:
            for e in alive[1:]:
                worse_than_best = e["ci"][1] < best["ci"][0] if is_table else e["ci"][0] > best["ci"][1]
                if worse_than_best:
                    e["status"] = "loại"
                    e["reason"] = (f"kém chắc chắn hơn {best['model']}: khoảng [{e['ci'][0]:.1%}, {e['ci'][1]:.1%}] "
                                   f"không chồng với [{best['ci'][0]:.1%}, {best['ci'][1]:.1%}]")
        alive = [e for e in alive if e["status"] == "giữ"]
        result = {"metric": metric, "best": best["model"] if best else None, "skipped": skipped}
        if stage == "screening":
            finalists = [e["model"] for e in alive[:max_finalists]]
            for e in alive[max_finalists:]:
                e["status"], e["reason"] = "loại", f"ngoài top {max_finalists}"
            # mốc so sánh: OCR truyền thống tốt nhất, giữ lại kể cả khi kém hơn (trừ khi không ổn định)
            trad = [e for e in entries if e["adapter"] in TRADITIONAL and e["unstable"] < max_unstable]
            trad.sort(key=lambda e: -e["value"] if is_table else e["value"])
            if trad and trad[0]["model"] not in finalists:
                finalists.append(trad[0]["model"])
                trad[0]["status"], trad[0]["reason"] = "giữ", "mốc so sánh (OCR truyền thống tốt nhất)"
            result["finalists"] = finalists
        else:
            ties = [e for e in alive if e is not best and not (
                e["ci"][1] < best["ci"][0] if is_table else e["ci"][0] > best["ci"][1])] if best else []
            group = ([best] + ties) if best else []
            group.sort(key=lambda e: (e["n_hallucination"], e["latency"], e["vram"]))
            result["winner"] = group[0]["model"] if group else None
            result["tied_with"] = [e["model"] for e in group[1:]]
            if ties:
                result["tie_break"] = "hòa về độ chính xác; chọn theo: ít lặp/thừa hơn → nhanh hơn → ít VRAM hơn"
        for e in entries:
            if e["hit_max_tokens"]:
                e["reason"] = (e["reason"] + "; " if e["reason"] else "") + \
                              f"⚠ {e['hit_max_tokens']} mẫu bị cắt (hết max_new_tokens) — kết quả chưa đáng tin"
        result["models"] = entries
        out["categories"][cat] = result
    return out


def render(dec: dict) -> str:
    L = [f"# Quyết định — giai đoạn `{dec['stage']}` (split `{dec['split']}`)", "",
         "Sinh tự động bởi `ocrbench decide`. Luật: xem đầu file `ocrbench/decide.py`.", ""]
    for cat, r in dec["categories"].items():
        L.append(f"## {cat} — chỉ số chính: {r['metric']}")
        L.append("")
        if dec["stage"] == "screening":
            L.append(f"**Vào chung kết:** {', '.join(r.get('finalists') or []) or '(không có)'}")
        else:
            tied = f" (hòa với: {', '.join(r['tied_with'])}; {r.get('tie_break', '')})" if r.get("tied_with") else ""
            L.append(f"**Thắng:** {r.get('winner') or '(không có)'}{tied}")
        L.append("")
        L.append("| Model | Chỉ số chính | Khoảng 95% | Lỗi+lặp | s/mẫu | Kết luận | Lý do |")
        L.append("|---|---:|---|---:|---:|---|---|")
        for e in sorted(r["models"], key=lambda e: (e["status"] != "giữ", e["value"] if "cer" in r["metric"] else -e["value"])):
            L.append(f"| {e['model']} | {e['value']:.1%} | [{e['ci'][0]:.1%}, {e['ci'][1]:.1%}] | {e['unstable']:.0%} "
                     f"| {e['latency']:.2f} | {e['status']} | {e['reason']} |")
        for s in r["skipped"]:
            L.append(f"| {s['model']} | — | — | — | — | chưa xét | {s['reason']} |")
        L.append("")
    return "\n".join(L)


def write_decision(cfg, dec: dict) -> Path:
    out = Path(cfg.output_dir) / dec["split"] / "_report"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"decision_{dec['stage']}.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2))
    (out / f"decision_{dec['stage']}.md").write_text(render(dec), encoding="utf-8")
    return out / f"decision_{dec['stage']}.md"
