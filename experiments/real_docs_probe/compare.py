"""Gộp kết quả các hệ thống → <out>/so_sanh.html (mở bằng trình duyệt) + <out>/tom_tat.md (số liệu, KHÔNG chứa chữ).

    python experiments/real_docs_probe/compare.py /kaggle/working/real_docs_out

Chưa có đáp án → không tính được độ chính xác. Báo cáo các chỉ số thay thế:
- tashkeel/chữ: tỉ lệ dấu nguyên âm trên chữ cái — giấy tờ đánh máy gần như KHÔNG có dấu, nên tỉ lệ cao = model bịa
- đồng thuận: trung bình (1 − CER) giữa hệ thống này với các hệ thống còn lại (đã bỏ dấu, thống nhất alef/yeh/số).
  Các hệ thống ĐỘC LẬP cùng ra một chữ thì chữ đó nhiều khả năng đúng; hệ thống "lạc loài" thường là bịa / bỏ sót
- lặp: có cụm ≥ 5 từ lặp liên tiếp (vòng lặp sinh chữ)
tom_tat.md chỉ có số liệu (được phép ghi vào nhật ký); so_sanh.html có chữ OCR → CHỈ xem trên server.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ocrbench.metrics import edit_stats, error_rate  # noqa: E402
from ocrbench.normalize import NormConfig, normalize  # noqa: E402

HARAKAT = re.compile(r"[ً-ْٰ]")
ARABIC_LETTER = re.compile(r"[ء-غف-يٱ-ۓ]")
HEBREW = re.compile(r"[א-ת]")
NORM = NormConfig(strip_diacritics=True, strip_tatweel=True, unify_alef=True, unify_yeh=True, unify_digits=True,
                  strip_markdown=True)


def stats(text: str) -> dict:
    letters = len(ARABIC_LETTER.findall(text))
    words = text.split()
    rep = any(len(set(words[i:i + 5])) == 1 for i in range(max(0, len(words) - 4))) if len(words) >= 5 else False
    return {"ky_tu": len(text), "chu_ar": letters, "tashkeel": round(len(HARAKAT.findall(text)) / max(1, letters), 3),
            "hebrew": len(HEBREW.findall(text)), "lap": rep}


def agree(a: str, b: str) -> float:
    na, nb = normalize(a, NORM), normalize(b, NORM)
    if not na and not nb:
        return 1.0
    s = edit_stats(na, nb)
    return round(max(0.0, 1 - error_rate(s["char_edits"], max(1, len(na)))), 3)


def thumb(p: Path, side: int = 900) -> str:
    from PIL import Image

    im = Image.open(p).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def main(out: str) -> None:
    out = Path(out)
    imgs = sorted((out / "img").glob("*.png"))
    systems = sorted(d for d in out.iterdir() if d.is_dir() and d.name not in ("img", "_runs")
                     and any(d.glob("*.txt")))
    texts = {s.name: {p.stem: (s / f"{p.stem}.txt").read_text(encoding="utf-8") if (s / f"{p.stem}.txt").exists()
                      else None for p in imgs} for s in systems}
    metas = {s.name: json.loads((s / "meta.json").read_text()) if (s / "meta.json").exists() else {} for s in systems}
    rows, sections = [], []
    agg: dict[str, dict] = {s.name: {"tashkeel": [], "dong_thuan": [], "giay": [], "lap": 0, "rong": 0} for s in systems}
    for p in imgs:
        cells = []
        for s in systems:
            t = texts[s.name][p.stem]
            if t is None:
                continue
            st = stats(t)
            others = [texts[o.name][p.stem] for o in systems if o.name != s.name and texts[o.name][p.stem]]
            cons = round(sum(agree(t, o) for o in others) / len(others), 3) if others else None
            sec = metas[s.name].get(p.stem, {}).get("giay")
            a = agg[s.name]
            a["tashkeel"].append(st["tashkeel"])
            a["lap"] += st["lap"]
            a["rong"] += st["ky_tu"] < 20
            if cons is not None:
                a["dong_thuan"].append(cons)
            if sec is not None:
                a["giay"].append(sec)
            rows.append(f"| {p.stem} | {s.name} | {st['ky_tu']} | {st['tashkeel']} | {cons} | {st['hebrew']} | "
                        f"{'⚠' if st['lap'] else ''} | {sec} |")
            cells.append(f"<div class=col><h3>{html.escape(s.name)}</h3><div class=m>{st['ky_tu']} ký tự · "
                         f"tashkeel/chữ {st['tashkeel']} · đồng thuận {cons}{' · ⚠ LẶP' if st['lap'] else ''}"
                         f"{f' · {sec}s' if sec else ''}</div><pre dir=rtl>{html.escape(t)}</pre></div>")
        sections.append(f"<section><h2>{p.stem}</h2><div class=row><div class=col><img src='data:image/jpeg;base64,"
                        f"{thumb(p)}'></div>{''.join(cells)}</div></section>")
    css = ("body{font-family:sans-serif;margin:12px} .row{display:flex;gap:10px;overflow-x:auto} "
           ".col{min-width:380px;max-width:420px} img{width:420px;border:1px solid #ccc} "
           "pre{white-space:pre-wrap;font-family:'Noto Naskh Arabic','Amiri',serif;font-size:15px;line-height:1.7;"
           "background:#fafafa;border:1px solid #ddd;padding:6px;max-height:900px;overflow:auto} "
           ".m{color:#555;font-size:12px} h3{margin:4px 0;font-size:14px}")
    (out / "so_sanh.html").write_text(f"<!doctype html><meta charset=utf-8><title>So sánh OCR giấy tờ cũ</title>"
                                      f"<style>{css}</style><h1>So sánh OCR — giấy tờ cũ (CHỈ xem trên server)</h1>"
                                      f"{''.join(sections)}", encoding="utf-8")
    mean = lambda v: round(sum(v) / len(v), 3) if v else None  # noqa: E731
    lines = ["# Tóm tắt thí nghiệm giấy tờ cũ (chỉ số liệu, không có chữ OCR)", "",
             "| Hệ thống | tashkeel/chữ TB (thấp = ít bịa dấu) | đồng thuận TB (cao = khớp các hệ khác) | "
             "ảnh lặp | ảnh rỗng | giây/ảnh TB |", "|---|---:|---:|---:|---:|---:|"]
    for s in systems:
        a = agg[s.name]
        lines.append(f"| {s.name} | {mean(a['tashkeel'])} | {mean(a['dong_thuan'])} | {a['lap']} | {a['rong']} | "
                     f"{mean(a['giay'])} |")
    lines += ["", "Chi tiết từng ảnh:", "", "| Ảnh | Hệ thống | ký tự | tashkeel/chữ | đồng thuận | chữ Hebrew | lặp | giây |",
              "|---|---|---:|---:|---:|---:|---|---:|", *rows]
    (out / "tom_tat.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len(systems)]))
    print(f"\n✔ {out / 'so_sanh.html'} (chỉ xem trên server)  ·  ✔ {out / 'tom_tat.md'}")


if __name__ == "__main__":
    main(sys.argv[1])
