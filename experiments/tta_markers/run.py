"""Thí nghiệm TTA + ghép ký hiệu nhỏ cho dots.mocr (chạy trên GPU Kaggle, trong venv dots).

    CUDA_VISIBLE_DEVICES=0 /kaggle/working/venvs/dots/bin/python experiments/tta_markers/run.py

Mỗi ảnh:
  1. lượt GỐC: đúng như web demo (inference/config.yaml: phóng ảnh nhỏ ≤ ×3, prompt_layout_all_en);
  2. các lượt PHỤ (--passes): ảnh gốc phóng ×k (+ lề trắng nếu có 'pad'), chế độ 'ocr' (prompt_ocr) hoặc 'layout';
  3. ghép ký hiệu (fuse.py) với min_votes = 1 và 2;
  4. chấm: CER lượt gốc / sau ghép, số ký hiệu trước số giữ được, số ký hiệu thêm SAI.
→ <out>/report.md (bảng tổng hợp) + <out>/<ảnh>.json (kết quả từng lượt).

KHÔNG sửa gì ngoài thư mục --out. Không dùng holdout (trừ mẫu trong goal/holdout_excluded.yaml).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from fuse import fuse_markers, gt_markers, markers_in  # noqa: E402

DEFAULT_IDS = [
    "pub_tables_en/pubtabnet__552595",  # holdout ĐÃ LOẠI (goal/holdout_excluded.yaml): †105 ‡88 §46
    "pub_tables_en/pubtabnet__699374", "pub_tables_en/pubtabnet__684148", "pub_tables_en/pubtabnet__707833",
    "pub_tables_en/pubtabnet__644357", "pub_tables_en/pubtabnet__590407",  # dev: < ≥ ± trước số
    "pub_tables_en/pubtabnet__550360", "pub_tables_en/pubtabnet__729650",  # dev: chữ mũ a/b, danh sách lồng
]
DEFAULT_SAMPLES = ["01_hoa_don_tieng_anh.png", "08_anh_chup_xau_hoa_don_a_rap.jpg"]  # đối chứng: không được tệ đi
DEFAULT_PASSES = "x4_ocr,x5_ocr,x5_layout"
_DATA_IMG = re.compile(r"!\[[^\]]*\]\(data:image/[^)]*\)")


def load_synth(path):
    """Bộ bảng tự sinh experiments/marker_bench (make.py) → [(tên, ảnh, đáp án HTML)]."""
    d = Path(path)
    recs = [json.loads(line) for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(r["name"], d / r["image"], r["gt"]) for r in recs]


def load_items(ids, samples):
    from ocrbench.dataset import holdout_excluded, load_manifest

    manifest = ROOT / "testset" / "manifest.jsonl"
    if not manifest.exists():
        manifest = Path("/kaggle/working/testset/manifest.jsonl")
    by_id = {it.id: it for it in load_manifest(manifest)}
    excluded = holdout_excluded()
    out = []
    for i in ids:
        it = by_id[i]
        if it.split == "holdout" and it.id not in excluded:
            raise SystemExit(f"TỪ CHỐI: {i} thuộc holdout và KHÔNG nằm trong goal/holdout_excluded.yaml")
        out.append((i.split("/")[-1], it.image, it.gt))
    for s in samples:
        p = ROOT / "inference" / "samples" / s
        stem = p.stem
        gt = next((g.read_text(encoding="utf-8") for g in (ROOT / "inference/samples/dap_an").glob(stem + ".*")), "")
        out.append((stem, p, gt))
    return out


def make_adapter():
    import yaml

    from ocrbench.adapters.dots import DotsAdapter

    cfg = yaml.safe_load((ROOT / "inference" / "config.yaml").read_text(encoding="utf-8"))
    params = dict(cfg["models"][0]["params"])
    pre = params.pop("preprocess", None)
    params.pop("modes", None)
    a = DotsAdapter(**params)
    a.load()
    return a, pre


def read(adapter, image, mode):
    t = time.perf_counter()
    r = adapter.parse_image(image, "prompt_ocr" if mode == "ocr" else "prompt_layout_all_en")
    md = _DATA_IMG.sub("", r["md"] or "").strip()
    return {"md": md, "seconds": round(time.perf_counter() - t, 1), "extra": r["extra"]}


MAX_SIDE = 2600  # ảnh phóng to hơn thế dễ hết VRAM T4 (thử: 4.024 px → lỗi) — ảnh cỡ trang gần như không phóng


def variant(img, spec, max_side: int = MAX_SIDE):
    """'x5_ocr' / 'x4pad_layout' → (ảnh, chế độ). Hệ số thực = min(k, max_side / cạnh dài)."""
    m = re.match(r"x(\d+(?:\.\d+)?)(pad)?_(ocr|layout)$", spec)
    if not m:
        raise SystemExit(f"--passes sai: {spec} (dạng x5_ocr, x4pad_layout)")
    k, pad, mode = float(m.group(1)), bool(m.group(2)), m.group(3)
    k = max(1.0, min(k, max_side / max(img.size)))
    out = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    if pad:
        p = round(16 * k)
        canvas = Image.new("RGB", (out.width + 2 * p, out.height + 2 * p), "white")
        canvas.paste(out, (p, p))
        out = canvas
    return out, mode


def cer(gt, md):
    from ocrbench.metrics import edit_stats, error_rate
    from ocrbench.normalize import NormConfig, normalize

    if not gt:
        return None
    s = edit_stats(normalize(gt, NormConfig()), normalize(md, NormConfig()))
    return round(error_rate(s["char_edits"], s["ref_chars"]) * 100, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    ap.add_argument("--passes", default=DEFAULT_PASSES)
    ap.add_argument("--out", default="/kaggle/working/exp_tta")
    ap.add_argument("--synth", default="", help="thư mục bộ tự sinh (experiments/marker_bench/data); thêm vào danh sách")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items([x for x in a.ids.split(",") if x], [x for x in a.samples.split(",") if x])
    if a.synth:
        items += load_synth(a.synth)
    passes = [p for p in a.passes.split(",") if p]
    import torch

    from ocrbench.preprocess import preprocess

    adapter, pre = make_adapter()
    rows = []
    for name, path, gt in items:
        print(f"== {name}", flush=True)
        img = Image.open(path).convert("RGB")
        rec = {"name": name, "size": img.size, "passes": {}}
        base_img, _ = preprocess(img, pre) if pre else (img, {})
        rec["base"] = read(adapter, base_img, "layout")
        for spec in passes:
            try:
                v, mode = variant(img, spec)
                rec["passes"][spec] = read(adapter, v, mode)
            except Exception as e:  # vd. hết VRAM ở ảnh phóng to — ghi lại, chạy tiếp
                rec["passes"][spec] = {"error": f"{type(e).__name__}: {e}"[:300], "md": ""}
                traceback.print_exc()
                torch.cuda.empty_cache()
            print(f"   {spec}: {rec['passes'][spec].get('seconds', '—')} s {rec['passes'][spec].get('error', '')}",
                  flush=True)
        details = [p["md"] for p in rec["passes"].values() if p.get("md")]
        g = set(gt_markers(gt)) if gt else set()
        b = markers_in(rec["base"]["md"])
        for mv in (1, 2):
            fused, changes = fuse_markers(rec["base"]["md"], details, min_votes=mv)
            f = markers_in(fused)
            # "thêm sai" = ký hiệu do bước ghép thêm vào mà đáp án không có (không tính cái lượt gốc vốn có)
            rec[f"fused_v{mv}"] = {"md": fused, "changes": changes, "cer": cer(gt, fused),
                                   "markers_ok": len(g & f), "markers_wrong": len((f - b) - g) if gt else None}
        rec["gt_markers"] = len(g)
        rec["per_pass"] = {}  # ghép với TỪNG lượt phụ riêng lẻ → biết lượt nào thực sự có ích
        for spec, p in rec["passes"].items():
            fused1, _ = fuse_markers(rec["base"]["md"], [p.get("md", "")], min_votes=1)
            f1 = markers_in(fused1)
            rec["per_pass"][spec] = {"ok": len(g & f1), "wrong": len((f1 - b) - g) if gt else None,
                                     "seconds": p.get("seconds")}
        rec["base_cer"], rec["base_markers_ok"] = cer(gt, rec["base"]["md"]), len(g & b)
        (out / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append(rec)
        print(f"   ký hiệu đáp án {len(g)} · gốc {len(g & b)} · ghép(v1) {rec['fused_v1']['markers_ok']} "
              f"(sai {rec['fused_v1']['markers_wrong']}) · ghép(v2) {rec['fused_v2']['markers_ok']} "
              f"(sai {rec['fused_v2']['markers_wrong']}) · CER {rec['base_cer']} → {rec['fused_v1']['cer']}", flush=True)

    lines = ["# Thí nghiệm TTA + ghép ký hiệu nhỏ (dots.mocr)", "",
             f"Lượt phụ: `{a.passes}` · min_votes 1 (v1) và 2 (v2)", "",
             "| Ảnh | Kích thước | Ký hiệu đáp án | Gốc giữ | Ghép v1 (sai) | Ghép v2 (sai) | CER gốc | CER v1 | CER v2 | "
             "Giây gốc | Giây phụ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tot = {"g": 0, "b": 0, "v1": 0, "w1": 0, "v2": 0, "w2": 0}
    for r in rows:
        sec = sum(p.get("seconds", 0) for p in r["passes"].values())
        lines.append(f"| {r['name']} | {r['size'][0]}×{r['size'][1]} | {r['gt_markers']} | {r['base_markers_ok']} | "
                     f"{r['fused_v1']['markers_ok']} ({r['fused_v1']['markers_wrong']}) | "
                     f"{r['fused_v2']['markers_ok']} ({r['fused_v2']['markers_wrong']}) | {r['base_cer']} | "
                     f"{r['fused_v1']['cer']} | {r['fused_v2']['cer']} | {r['base']['seconds']} | {round(sec, 1)} |")
        tot["g"] += r["gt_markers"]
        tot["b"] += r["base_markers_ok"]
        tot["v1"] += r["fused_v1"]["markers_ok"]
        tot["w1"] += r["fused_v1"]["markers_wrong"] or 0
        tot["v2"] += r["fused_v2"]["markers_ok"]
        tot["w2"] += r["fused_v2"]["markers_wrong"] or 0
    lines += ["", f"**Tổng ký hiệu trước số:** đáp án {tot['g']} · gốc giữ {tot['b']} · ghép v1 {tot['v1']} "
                  f"(thêm sai {tot['w1']}) · ghép v2 {tot['v2']} (thêm sai {tot['w2']})"]
    lines += ["", "**Từng lượt phụ riêng lẻ** (ghép chỉ với lượt đó):", "",
              "| Lượt phụ | Ký hiệu thêm đúng | Thêm sai | Tổng giây |", "|---|---:|---:|---:|"]
    for spec in passes:
        ok = sum(r["per_pass"].get(spec, {}).get("ok", 0) - r["base_markers_ok"] for r in rows)
        wrong = sum(r["per_pass"].get(spec, {}).get("wrong") or 0 for r in rows)
        sec = sum(r["per_pass"].get(spec, {}).get("seconds") or 0 for r in rows)
        lines.append(f"| {spec} | {ok} | {wrong} | {round(sec)} |")
    base_sec = sum(r["base"]["seconds"] for r in rows)
    lines += ["", f"Tổng giây lượt gốc: {round(base_sec)}"]
    errs = [(r["name"], s, p["error"]) for r in rows for s, p in r["passes"].items() if p.get("error")]
    if errs:
        lines += ["", "**Lượt phụ lỗi:**"] + [f"- {n} · {s}: {e}" for n, s, e in errs]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
