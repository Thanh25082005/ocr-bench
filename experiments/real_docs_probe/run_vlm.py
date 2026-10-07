"""Chạy MỘT model VLM trên mọi ảnh trong <out>/img → <out>/<model>[__<chế độ>]/<ảnh>.txt (+ DOCX nếu là dots).

    <python phù hợp> experiments/real_docs_probe/run_vlm.py <out> <model> [--mode "Chỉ chữ"] [--gpus 0]

- dots_mocr: chạy bằng /kaggle/working/venvs/dots/bin/python (transformers 4.56.1); ghi thêm DOCX A + B.
- qari_0_4: python hệ thống (transformers >= 4.57 cho Qwen3-VL).
- churro_3b, sherif_handwriting: python nào cũng được (Qwen2.5-VL).
Ghi meta.json: thời gian từng ảnh, cờ lặp vòng / rỗng. Chạy lại: bỏ qua ảnh đã có .txt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

_XML_TAG = re.compile(r"</?[A-Za-z][^<>]{0,200}>")


def plain(text: str) -> str:
    """CHURRO trả XML: bỏ thẻ, giữ chữ + xuống dòng."""
    if "<" in text and ">" in text and not text.lstrip().startswith(("|", "#")):
        text = re.sub(r"<(br|/p|/line|/l|/lb|lb|/div|/head)\b[^>]*>", "\n", text)
        text = _XML_TAG.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("model")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--gpus", default="0")
    a = ap.parse_args()
    from ocrbench.convert import Converter

    out = Path(a.out)
    name = a.model + (("__" + re.sub(r"\W+", "_", a.mode).strip("_")) if a.mode else "")
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    imgs = sorted((out / "img").glob("*.png"))
    todo = [p for p in imgs if not (d / f"{p.stem}.txt").exists()]
    meta_p = d / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if not todo:
        print(f"{name}: đã xong hết ({len(imgs)} ảnh)")
        return 0
    t0 = time.time()
    conv = Converter(HERE / "config.yaml", a.model, gpus=a.gpus)
    print(f"{name}: nạp model {time.time() - t0:.0f}s, {len(todo)} ảnh", flush=True)
    for p in todo:
        t = time.time()
        res = conv.convert_file(p, d / "_docx", force_ocr=True, mode=a.mode, exact=(a.model == "dots_mocr"))
        text = "\n\n".join(pg.text or "" for pg in res.pages)
        notes = "; ".join(x for pg in res.pages for x in (pg.error, pg.note) if x)
        if a.model == "churro_3b":
            (d / f"{p.stem}.raw.xml").write_text(text, encoding="utf-8")
            text = plain(text)
        (d / f"{p.stem}.txt").write_text(text, encoding="utf-8")
        meta[p.stem] = {"giay": round(time.time() - t, 1), "ky_tu": len(text), "ghi_chu": notes or None}
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {p.stem}: {meta[p.stem]['giay']}s, {len(text)} ký tự {('⚠ ' + notes) if notes else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
