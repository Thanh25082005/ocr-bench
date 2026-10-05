"""Tự kiểm tra trước khi mở web: dựng một trang hoá đơn mẫu (tiếng Anh + Ả Rập + bảng), cho dots đọc,
in ra trạng thái JSON bố cục, số khối, có bảng HTML không, thời gian. Lỗi → in lỗi và thoát mã 1."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def sample_page() -> Image.Image:
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    try:
        big, font = ImageFont.truetype("DejaVuSans-Bold.ttf", 44), ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        big = font = ImageFont.load_default(size=30)
    d.text((90, 90), "INVOICE No. 2041", fill="black", font=big)
    d.text((90, 170), "Customer: Al Noor Trading LLC   Date: 05/10/2026", fill="black", font=font)
    rows = [["Item", "Qty", "Price"], ["Laptop", "2", "1,250.00"], ["Printer", "1", "430.50"], ["Total", "", "2,930.50"]]
    x0, y0, cw, rh = 90, 260, [500, 200, 300], 70
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            d.rectangle([x, y0 + r * rh, x + cw[c], y0 + (r + 1) * rh], outline="black", width=2)
            d.text((x + 15, y0 + r * rh + 20), cell, fill="black", font=font)
            x += cw[c]
    d.text((90, 600), "Thank you for your business. Payment due within 30 days.", fill="black", font=font)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="auto")
    a = ap.parse_args()
    from ocrbench.convert import Converter

    t = time.perf_counter()
    conv = Converter(ROOT / "inference/config.yaml", "dots_mocr", gpus=a.gpus)
    print(f"nạp model: {time.perf_counter() - t:.0f}s trên {len(conv.adapters)} bản model", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mau.png"
        sample_page().save(p)
        t = time.perf_counter()
        res = conv.convert_file(p, tmp, force_ocr=True)
        page = res.pages[0]
        dt = time.perf_counter() - t
    conv.close()
    print(f"thời gian 1 trang: {dt:.1f}s · nguồn: {page.source} · lỗi: {page.error or '—'} · ghi chú: {page.note or '—'}")
    print(f"số khối bố cục: {len(page.blocks or [])} · có bảng HTML: {'<table' in page.text.lower()}")
    print("----- kết quả -----\n" + page.text[:1500] + "\n-------------------")
    ok = not page.error and page.text.strip() and page.blocks and "2041" in page.text
    print("✔ KIỂM TRA ĐẠT — mở web được" if ok else "✘ KIỂM TRA KHÔNG ĐẠT — xem kết quả ở trên (gửi cho người sửa code)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
