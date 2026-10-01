"""Dựng bộ test: lấy mẫu từ bộ công khai + sinh tài liệu tổng hợp, rồi ghi manifest.jsonl.

Cấu trúc đầu ra:

    OUT/images/<nhóm>/<tên>.png|jpg
    OUT/gt/<nhóm>/<tên>.txt|html
    OUT/manifest.jsonl
    OUT/DATASET_CARD.md

Chạy lại sẽ bỏ qua các mẫu đã có (tiếp tục được khi bị ngắt). Mọi thứ đều cố định theo
seed, nên cùng seed thì ra cùng bộ test.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..dataset import _split_for_doc
from .templates import KINDS

SYNTH_PLAN = {
    "syn_text_ar": dict(kinds=["contract", "letter"], langs=["ar"]),
    "syn_text_en": dict(kinds=["contract", "letter"], langs=["en"]),
    "syn_text_mixed": dict(kinds=["contract", "letter"], langs=["mixed"]),
    "syn_invoice_ar": dict(kinds=["invoice"], langs=["ar"]),
    "syn_invoice_en": dict(kinds=["invoice"], langs=["en"]),
    "syn_invoice_mixed": dict(kinds=["invoice"], langs=["mixed"]),
    "syn_form_ar": dict(kinds=["form"], langs=["ar", "mixed"]),
    "syn_form_en": dict(kinds=["form"], langs=["en"]),
    "syn_degraded": dict(kinds=list(KINDS), langs=["ar", "en", "mixed"], degrade=["medium", "hard"]),
}
LONG_CATEGORIES = ("syn_longtable", "syn_longtext")


def _save_gt(path: Path, gt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gt, encoding="utf-8")


def _record(out: Path, cat: str, name: str, img_path: Path, gt_path: Path, gt_type: str, doc_id: str,
            holdout_ratio: float, seed: int, meta: dict) -> dict:
    return {
        "id": f"{cat}/{name}",
        "image": str(img_path.relative_to(out)),
        "gt_file": str(gt_path.relative_to(out)),
        "gt_type": gt_type,
        "category": cat,
        "doc_id": doc_id,
        "split": _split_for_doc(doc_id, holdout_ratio, str(seed)),
        **meta,
    }


def build_public(out: Path, n: int, seed: int, holdout_ratio: float, keys: list[str] | None, log=print) -> list[dict]:
    from .public import SOURCES, iter_samples

    records = []
    for src in SOURCES:
        if keys and src.key not in keys:
            continue
        log(f"[công khai] {src.key}: lấy {n} mẫu từ {src.hf_id} ({src.split})")
        cached = out / f".cache_{src.key}.jsonl"
        if cached.exists() and len(cached.read_text().splitlines()) >= n:
            records += [json.loads(line) for line in cached.read_text().splitlines()[:n]]
            continue
        recs = []
        for uid, img, gt, gt_type in iter_samples(src, n, seed):
            name = f"{src.key}__{uid}".replace("/", "_")
            img_path = out / "images" / src.category / f"{name}.png"
            gt_path = out / "gt" / src.category / f"{name}.{'html' if gt_type == 'table_html' else 'txt'}"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            if not img_path.exists():
                img.convert("RGB").save(img_path, optimize=True)
            _save_gt(gt_path, gt)
            recs.append(_record(out, src.category, name, img_path, gt_path, gt_type, f"{src.key}/{uid}",
                                holdout_ratio, seed,
                                {"lang": src.lang, "source": src.key, "license": src.license,
                                 "granularity": src.granularity}))
        cached.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs))
        records += recs
    return records


def build_synth(out: Path, n: int, seed: int, holdout_ratio: float, categories: list[str] | None,
                scale: float = 1.5, log=print) -> list[dict]:
    from .degrade import degrade
    from .fonts import ensure_fonts, font_face_css
    from .render import Renderer
    from .templates import make_doc

    faces = font_face_css(ensure_fonts())
    records = []
    with Renderer(scale=scale) as renderer:
        for cat, plan in SYNTH_PLAN.items():
            if categories and cat not in categories:
                continue
            log(f"[tổng hợp] {cat}: {n} tài liệu")
            for i in range(n):
                rng = random.Random(f"{seed}:{cat}:{i}")
                kind, lang = rng.choice(plan["kinds"]), rng.choice(plan["langs"])
                doc = make_doc(kind, lang, rng, faces)
                level = rng.choice(plan["degrade"]) if plan.get("degrade") else None
                name = f"{cat}_{i:04d}"
                ext = "jpg" if level else "png"
                img_path = out / "images" / cat / f"{name}.{ext}"
                gt_path = out / "gt" / cat / f"{name}.{'html' if doc.gt_type == 'table_html' else 'txt'}"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                meta = {"lang": lang, "source": "synthetic", "license": "tự tạo", **doc.meta}
                if not img_path.exists():
                    if level:
                        tmp = img_path.with_suffix(".clean.png")
                        renderer.render(doc.page_html, tmp)
                        bad, info = degrade(Image.open(tmp), rng, level)
                        bad.save(img_path, quality=92)
                        tmp.unlink()
                        meta["degradation"] = info
                    else:
                        renderer.render(doc.page_html, img_path)
                elif level:
                    # rng phải đi qua cùng các bước để mẫu sau không đổi; ghi lại mức độ
                    meta["degradation"] = {"level": level}
                _save_gt(gt_path, doc.gt)
                records.append(_record(out, cat, name, img_path, gt_path, doc.gt_type, f"{cat}/{name}",
                                       holdout_ratio, seed, meta))
        records += build_long(out, n, seed, holdout_ratio, categories, renderer, faces, log)
    return records


def build_long(out: Path, n: int, seed: int, holdout_ratio: float, categories, renderer, faces, log=print) -> list[dict]:
    """Bảng dài và văn bản dài, nhiều trang. Mỗi mẫu là một danh sách ảnh trang."""
    from .longdocs import TABLE_BINS, TEXT_BINS, long_table, long_text, wiki_articles

    records = []
    jobs = []
    if not categories or "syn_longtable" in categories:
        for i in range(n):
            rng = random.Random(f"{seed}:syn_longtable:{i}")
            jobs.append(("syn_longtable", i, rng, lambda rng=rng, i=i: long_table(
                rng.choice(["ar", "en", "mixed"]), rng, faces, i % len(TABLE_BINS))))
    if not categories or "syn_longtext" in categories:
        n_ar = (2 * n + 2) // 3  # 2/3 tiếng Ả Rập, 1/3 tiếng Anh
        log(f"[dài] tải {n} bài Wikipedia ({n_ar} ar, {n - n_ar} en)")
        articles = [("ar", a) for a in wiki_articles("ar", n_ar)] + [("en", a) for a in wiki_articles("en", n - n_ar)]
        for i, (lang, art) in enumerate(articles):
            rng = random.Random(f"{seed}:syn_longtext:{i}")
            jobs.append(("syn_longtext", i, rng, lambda rng=rng, i=i, lang=lang, art=art: long_text(
                lang, art, rng, faces, i % len(TEXT_BINS))))
    for cat, i, rng, make in jobs:
        if i == 0:
            log(f"[dài] {cat}")
        pages, gt, gt_type, meta = make()
        name = f"{cat}_{i:04d}"
        img_dir = out / "images" / cat
        img_dir.mkdir(parents=True, exist_ok=True)
        paths = [img_dir / f"{name}_p{k + 1}.png" for k in range(len(pages))]
        for html_page, path in zip(pages, paths):
            if not path.exists():
                renderer.render(html_page, path)
        gt_path = out / "gt" / cat / f"{name}.{'html' if gt_type == 'table_html' else 'txt'}"
        _save_gt(gt_path, gt)
        rec = _record(out, cat, name, paths[0], gt_path, gt_type, f"{cat}/{name}", holdout_ratio, seed,
                      {"source": "synthetic" if cat == "syn_longtable" else "wikipedia+synthetic",
                       "license": "tự tạo" if cat == "syn_longtable" else "CC BY-SA 4.0 (Wikipedia)", **meta})
        rec["image"] = [str(p.relative_to(out)) for p in paths]
        records.append(rec)
    return records


def write_card(out: Path, records: list[dict], args: dict) -> None:
    from .public import SOURCES

    counts = Counter((r["category"], r["split"]) for r in records)
    cats = sorted({r["category"] for r in records})
    L = ["# Bộ test OCR (tiếng Anh + tiếng Ả Rập)", "",
         f"Dựng lúc {datetime.now():%Y-%m-%d %H:%M} bằng `ocrbench build-testset` với tham số: `{json.dumps(args, ensure_ascii=False)}`.", "",
         "## Số mẫu", "", "| Nhóm | dev | holdout | Loại đáp án | Nguồn |", "|---|---:|---:|---|---|"]
    for c in cats:
        rs = [r for r in records if r["category"] == c]
        L.append(f"| {c} | {counts[(c, 'dev')]} | {counts[(c, 'holdout')]} | "
                 f"{', '.join(sorted({r['gt_type'] for r in rs}))} | {', '.join(sorted({r['source'] for r in rs}))} |")
    L.append(f"| **Tổng** | {sum(v for (c, s), v in counts.items() if s == 'dev')} | "
             f"{sum(v for (c, s), v in counts.items() if s == 'holdout')} | | |")
    used = {r["source"] for r in records}
    L += ["", "## Nguồn công khai", "", "| Nguồn | Dataset | Giấy phép | Ghi chú |", "|---|---|---|---|"]
    for s in SOURCES:
        if s.key in used:
            L.append(f"| {s.key} | [{s.hf_id}](https://huggingface.co/datasets/{s.hf_id}) ({s.split}) | {s.license} | {s.note} |")
    L += ["", "## Dữ liệu tổng hợp (`syn_*`)", "",
          "- Tài liệu sinh từ mẫu HTML (hợp đồng, công văn, hóa đơn, biểu mẫu), dựng ảnh bằng Chromium. Nội dung "
          "(tên, công ty, số tiền, điều khoản) là giả nhưng có nghĩa, không phải chữ ngẫu nhiên.",
          "- Đáp án sinh cùng lúc với ảnh nên chính xác tuyệt đối. Logo, chữ ký, con dấu là hình không chứa chữ.",
          "- Khoảng một nửa tài liệu tiếng Ả Rập dùng chữ số Ả Rập-Ấn (٠١٢).",
          "- `syn_form_*`: phần điền tay dùng **font giả chữ viết tay** (Aref Ruqaa, Caveat...). Đây chỉ là xấp xỉ, "
          "dễ hơn chữ viết tay thật nhiều; đánh giá chữ viết tay thật phải dựa vào `pub_handwriting_*`.",
          "- `syn_degraded`: tài liệu tổng hợp đã làm xấu như ảnh chụp điện thoại (nghiêng, méo, mờ, nhiễu, JPEG), "
          "gồm mức medium và hard.",
          "- `syn_longtable`: bảng dài 1–3 trang (sao kê tài khoản có ô nợ/có để trống xen kẽ, hoặc bảng kê hàng 8 cột). "
          "Một nửa số bảng nhiều trang KHÔNG in lại hàng tiêu đề ở trang sau. Đáp án là một bảng logic duy nhất. "
          "Dùng để đo giá trị có bị lệch hàng/lệch cột không (`length_bin` ghi độ dài).",
          "- `syn_longtext`: văn bản dài 1–2 trang lấy từ bài Wikipedia (CC BY-SA 4.0); khoảng một nửa con số đã bị thay "
          "ngẫu nhiên, nên model 'nhớ' bài gốc thay vì đọc sẽ bị tính sai. Mẫu nhiều trang có trường `image` là danh sách ảnh.", "",
          "## Hạn chế cần nhớ", "",
          "- **Đây là bộ thay thế khi chưa có dữ liệu thật.** Kết quả trên bộ này chỉ cho biết thứ hạng tương đối giữa "
          "các model; độ chính xác trên tài liệu thật của dự án có thể khác.",
          "- **Rủi ro model đã gặp dữ liệu công khai lúc huấn luyện** (vd. Baseer với Misraj, model của sherif1313 với "
          "KHATT/IAM). Nếu một model chỉ tốt ở nhóm `pub_*` mà kém ở nhóm `syn_*` tương ứng, cần nghi ngờ điều này.",
          "- Dữ liệu dòng chữ (KHATT, IAM) không có mã người viết, nên dev/holdout chia theo từng dòng.",
          "- Giấy phép: một số nguồn chỉ cho phép nghiên cứu/phi thương mại hoặc chưa ghi rõ; chỉ dùng nội bộ để đánh giá, "
          "không phát hành lại."]
    (out / "DATASET_CARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_testset(out: str | Path, public_n: int = 80, synth_n: int = 80, holdout_ratio: float = 0.5, seed: int = 0,
                  public: bool = True, synth: bool = True, sources: list[str] | None = None,
                  categories: list[str] | None = None, scale: float = 1.5, log=print) -> list[dict]:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    if public:
        records += build_public(out, public_n, seed, holdout_ratio, sources, log)
    if synth:
        records += build_synth(out, synth_n, seed, holdout_ratio, categories, scale, log)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_card(out, records, dict(public_n=public_n, synth_n=synth_n, holdout_ratio=holdout_ratio, seed=seed,
                                  public=public, synth=synth, sources=sources, categories=categories, scale=scale))
    return records
