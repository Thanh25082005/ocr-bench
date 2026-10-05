"""Sinh bộ bảng TỰ TẠO có ký hiệu chú thích nhỏ trước số (†, ‡, §, ¶, *, #, chữ mũ a–d), độ phân giải thấp
kiểu PubTabNet (ký hiệu chỉ còn ~3–4 px) — thước đo chung cho các thí nghiệm "ký hiệu nhỏ" (hướng 2, hướng 3).

    python experiments/marker_bench/make.py            # → experiments/marker_bench/data/{*.png, manifest.jsonl}

Cô lập: không đụng bộ test, không sửa ocrbench. Xoá: rm -rf experiments/marker_bench
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ocrbench.testset.render import Renderer  # noqa: E402

OUT = HERE / "data"
N = 20
SEED = "marker-bench-2026"
MARKS = ["†", "‡", "§", "¶", "*", "#", "a", "b", "c", "d"]
LABELS = ["Neurological", "Cardiovascular", "Respiratory", "Renal", "Hepatic", "Endocrine", "Infection", "Cancer",
          "Diabetes mellitus", "Hypertension", "Obesity", "Depression", "Anxiety", "Asthma", "Arthritis", "Stroke",
          "Anemia", "Migraine", "Epilepsy", "Thyroid disease", "Allergy", "Osteoporosis", "Dementia", "Insomnia"]
GROUPS = [("Telephone interview", "Clinic evaluation"), ("Baseline", "Follow-up"), ("Male", "Female"),
          ("Cases", "Controls"), ("Group A", "Group B"), ("2019", "2023")]


def _num(rng):
    k = rng.random()
    if k < 0.5:
        return str(rng.randint(1, 999))
    if k < 0.8:
        return f"{rng.uniform(0, 99):.1f}"
    return f"{rng.uniform(0, 9):.2f}"


def make_table(rng: random.Random):
    g1, g2 = rng.choice(GROUPS)
    n_rows = rng.randint(7, 12)
    labels = rng.sample(LABELS, n_rows)
    head = ["", "n", "(%)", "n", "(%)"]
    rows, gt_rows = [], []
    n_marks = 0
    for lab in labels:
        cells_html, cells_gt = [lab], [lab]
        for j in range(4):
            v = _num(rng)
            if j in (0, 2) and rng.random() < 0.28:  # ký hiệu trước số ở cột "n"
                mk = rng.choice(MARKS)
                cells_html.append(f"<sup>{mk}</sup>{v}")
                cells_gt.append(f"<sup>{mk}</sup>{v}")
                n_marks += 1
            else:
                cells_html.append(v)
                cells_gt.append(v)
        rows.append(cells_html)
        gt_rows.append(cells_gt)
    font = rng.choice(["Arial", "Helvetica", "Times New Roman", "Georgia", "Verdana"])
    size = rng.choice([11, 12, 13])
    css = (f"body{{margin:8px;font-family:'{font}';font-size:{size}px}} table{{border-collapse:collapse}}"
           "td,th{padding:2px 14px;text-align:right} td:first-child{text-align:left;padding-left:14px}"
           "thead tr:last-child th{border-bottom:1px solid #000} table{border-top:1px solid #000;"
           "border-bottom:1px solid #000} sup{font-size:68%}")
    top = (f"<tr><th></th><th colspan=2>{g1}</th><th colspan=2>{g2}</th></tr>"
           "<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>")
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    html = f"<html><head><style>{css}</style></head><body><table><thead>{top}</thead><tbody>{body}</tbody></table></body></html>"
    gt = (f"<table><tr><td></td><td colspan=\"2\">{g1}</td><td colspan=\"2\">{g2}</td></tr>"
          "<tr>" + "".join(f"<td>{h}</td>" for h in head) + "</tr>"
          + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in gt_rows) + "</table>")
    return html, gt, n_marks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    with Renderer(scale=1.0, width=760) as r:
        for i in range(N):
            rng = random.Random(f"{SEED}:{i}")
            html, gt, n_marks = make_table(rng)
            full = OUT / f"_full{i}.png"
            r.render(html, full)
            im = Image.open(full).convert("RGB")
            box = Image.eval(im.convert("L"), lambda v: 255 - v).getbbox()
            im = im.crop((max(0, box[0] - 6), max(0, box[1] - 6), box[2] + 6, box[3] + 6))
            # thu nhỏ theo CHIỀU CAO DÒNG (không theo bề rộng): mỗi dòng ~10–12 px như pubtabnet__552595
            # → chữ ~7–8 px, ký hiệu mũ (68%) chỉ còn ~3–4 px
            n_lines = gt.count("<tr>")
            k = rng.uniform(10, 12) * n_lines / im.height
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
            name = f"mb_{i:02d}"
            im.save(OUT / f"{name}.png")
            full.unlink()
            recs.append({"name": name, "image": f"{name}.png", "gt": gt, "n_markers": n_marks, "size": im.size})
            print(name, im.size, "ký hiệu:", n_marks)
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Tổng ký hiệu:", sum(r["n_markers"] for r in recs))


if __name__ == "__main__":
    main()
