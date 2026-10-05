"""Sinh bộ tài liệu mẫu cho web demo (inference/samples/): TỰ TẠO hoàn toàn, seed khác bộ test.

    python inference/make_samples.py            # cần Playwright + Chromium (như khi dựng bộ test)

Gồm ảnh và PDF đủ loại: hoá đơn có bảng (Anh / Ả Rập / trộn), hợp đồng, thư, biểu mẫu chữ viết tay,
ảnh chụp xấu, PDF scan nhiều trang (bảng dài 3 trang), PDF có lớp chữ. Đáp án đúng ở samples/dap_an/
để đối chiếu khi demo. Không dùng dữ liệu công khai của bộ test (có nguồn giấy phép phi thương mại).
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ocrbench.testset.degrade import degrade  # noqa: E402
from ocrbench.testset.fonts import ensure_fonts, font_face_css  # noqa: E402
from ocrbench.testset.longdocs import long_table  # noqa: E402
from ocrbench.testset.render import Renderer  # noqa: E402
from ocrbench.testset.templates import make_doc  # noqa: E402

OUT = ROOT / "inference" / "samples"
SEED = "inference-demo-2026"

# tên file, loại, ngôn ngữ, mức làm xấu (None = sạch)
SINGLE = [
    ("01_hoa_don_tieng_anh", "invoice", "en", None),
    ("02_hoa_don_tieng_a_rap", "invoice", "ar", None),
    ("03_hoa_don_tron_anh_a_rap", "invoice", "mixed", None),
    ("04_hop_dong_tieng_a_rap", "contract", "ar", None),
    ("05_thu_tieng_anh", "letter", "en", None),
    ("06_bieu_mau_viet_tay_a_rap", "form", "ar", None),
    ("07_bieu_mau_viet_tay_anh", "form", "en", None),
    ("08_anh_chup_xau_hoa_don_a_rap", "invoice", "ar", "medium"),
    ("09_anh_chup_rat_xau_hop_dong_anh", "contract", "en", "hard"),
]


def _gt_ext(gt_type: str) -> str:
    return "html" if gt_type == "table_html" else "txt"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "dap_an").mkdir(parents=True)
    faces = font_face_css(ensure_fonts())
    with Renderer(scale=1.5) as r:
        for name, kind, lang, level in SINGLE:
            rng = random.Random(f"{SEED}:{name}")
            doc = make_doc(kind, lang, rng, faces)
            png = OUT / f"{name}.png"
            r.render(doc.page_html, png)
            if level:
                bad, _ = degrade(Image.open(png), rng, level)
                bad.save(OUT / f"{name}.jpg", quality=90)
                png.unlink()
            (OUT / "dap_an" / f"{name}.{_gt_ext(doc.gt_type)}").write_text(doc.gt, encoding="utf-8")
            print("✔", name)

        # PDF scan 3 trang: bảng sao kê dài (kiểm tra bảng qua nhiều trang có lệch giá trị không)
        rng = random.Random(f"{SEED}:longtable")
        pages, gt, gt_type, _ = long_table("mixed", rng, faces, 2)
        imgs = []
        for k, html in enumerate(pages):
            p = OUT / f"_p{k}.png"
            r.render(html, p)
            imgs.append(Image.open(p).convert("RGB"))
        name = f"10_pdf_scan_bang_dai_{len(imgs)}_trang"
        imgs[0].save(OUT / f"{name}.pdf", save_all=True, append_images=imgs[1:], resolution=180)
        for k in range(len(pages)):
            (OUT / f"_p{k}.png").unlink()
        (OUT / "dap_an" / f"{name}.{_gt_ext(gt_type)}").write_text(gt, encoding="utf-8")
        print("✔", name)

        # PDF scan 2 trang ghép: hợp đồng tiếng Ả Rập + hoá đơn tiếng Anh (tài liệu nhiều trang, nhiều loại)
        docs = [make_doc("contract", "ar", random.Random(f"{SEED}:mix1"), faces),
                make_doc("invoice", "en", random.Random(f"{SEED}:mix2"), faces)]
        imgs = []
        for k, d in enumerate(docs):
            p = OUT / f"_m{k}.png"
            r.render(d.page_html, p)
            imgs.append(Image.open(p).convert("RGB"))
            p.unlink()
        name = "11_pdf_scan_2_trang_hop_dong_va_hoa_don"
        imgs[0].save(OUT / f"{name}.pdf", save_all=True, append_images=imgs[1:], resolution=180)
        (OUT / "dap_an" / f"{name}.txt").write_text("\n\n=== trang 2 ===\n\n".join(d.gt for d in docs),
                                                   encoding="utf-8")
        print("✔", name)

        # PDF có lớp chữ (xuất thẳng từ HTML): demo bật/tắt "Dùng lớp chữ có sẵn của PDF"
        d = make_doc("letter", "en", random.Random(f"{SEED}:digital"), faces)
        src = Path(r._tmp.name) / "digital.html"
        src.write_text(d.page_html, encoding="utf-8")
        r._page.goto(src.as_uri())
        r._page.evaluate("document.fonts.ready")
        name = "12_pdf_co_lop_chu_thu_tieng_anh"
        r._page.pdf(path=str(OUT / f"{name}.pdf"), format="A4", print_background=True)
        (OUT / "dap_an" / f"{name}.txt").write_text(d.gt, encoding="utf-8")
        print("✔", name)

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Xong: {len(list(OUT.glob('[0-9]*')))} file mẫu, {total / 1e6:.1f} MB → {OUT}")


if __name__ == "__main__":
    main()
