"""Chuẩn bị ảnh thí nghiệm giấy tờ cũ: mỗi ảnh gốc → 2 biến thể (không nhị phân hoá — bài báo cho thấy hại VLM).

    python experiments/real_docs_probe/prep.py /kaggle/working/real_docs /kaggle/working/real_docs_out

    <out>/img/<tên>__goc.png        ảnh gốc (đổi sang PNG, không sửa gì)
    <out>/img/<tên>__cat.png        cắt tờ giấy khỏi nền tối + làm nét nhẹ (unsharp mask: biến đổi DUY NHẤT giúp
                                     VLM trong "When Do VLMs Help Arabic Manuscript OCR?", arXiv 2608.22366)

Cô lập: không sửa ocrbench. Xoá: rm -rf experiments/real_docs_probe
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def crop_paper(im: Image.Image) -> Image.Image:
    """Cắt tờ giấy khỏi nền tối (ảnh chụp đặt trên nền đen): giữ các hàng / cột mà phần lớn là giấy (sáng)."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    paper = g > max(60.0, 0.5 * float(np.percentile(g, 90)))
    rows = np.nonzero(paper.mean(axis=1) > 0.5)[0]
    cols = np.nonzero(paper.mean(axis=0) > 0.5)[0]
    if len(rows) < 0.3 * g.shape[0] or len(cols) < 0.3 * g.shape[1]:
        return im  # không thấy nền tối rõ ràng → giữ nguyên
    pad = 4
    box = (max(0, cols.min() - pad), max(0, rows.min() - pad), min(g.shape[1], cols.max() + pad + 1),
           min(g.shape[0], rows.max() + pad + 1))
    return im.crop(box)


def main(src: str, out: str) -> None:
    d = Path(out) / "img"
    d.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"Không có ảnh trong {src}")
    for i, p in enumerate(files, 1):
        stem = f"anh{i:02d}"  # tên ngắn, không chứa thông tin cá nhân
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.save(d / f"{stem}__goc.png")
        c = crop_paper(im).filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=3))
        c.save(d / f"{stem}__cat.png")
        (d / f"{stem}.nguon.txt").write_text(p.name, encoding="utf-8")
        print(f"{stem}: {p.name} {im.size} → cắt {c.size}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
