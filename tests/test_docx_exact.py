"""DOCX giữ nguyên bố cục: mỗi khối một khung neo theo trang, đúng toạ độ; không mất chữ; ảnh cắt từ trang gốc."""

import re
import zipfile

import pytest
import yaml
from PIL import Image, ImageDraw, ImageFont

from ocrbench.docx_exact import build_exact_docx, column_bounds, fit_text, text_bands, wrap

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _page():
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    d.text((100, 100), "INVOICE No. 2041", fill="black", font=_font(44))
    for i, line in enumerate(["Customer: Al Noor Trading LLC", "Date: 05/10/2026, due in 30 days"]):
        d.text((100, 200 + i * 40), line, fill="black", font=_font(28))
    d.rectangle((900, 80, 1100, 180), fill="navy")  # "logo"
    for x, col in ((110, "Item"), (610, "Qty"), (900, "Price")):  # bảng 3 cột, khe trắng rõ
        d.text((x, 320), col, fill="black", font=_font(26))
        d.text((x, 380), {"Item": "Laptop", "Qty": "2", "Price": "1,250.00"}[col], fill="black", font=_font(26))
    blocks = [
        {"category": "Title", "bbox": [100, 100, 520, 150], "text": "# INVOICE No. 2041"},
        {"category": "Text", "bbox": [100, 200, 620, 270], "text": "Customer: **Al Noor Trading LLC** Date: 05/10/2026, due in 30 days"},
        {"category": "Picture", "bbox": [900, 80, 1100, 180], "text": ""},
        {"category": "Table", "bbox": [100, 310, 1100, 420],
         "text": "<table><tr><th>Item</th><th>Qty</th><th>Price</th></tr><tr><td>Laptop</td><td>2</td><td>1,250.00</td></tr></table>"},
    ]
    return img, blocks


def test_build_positions_text_and_images(tmp_path):
    img, blocks = _page()
    out = tmp_path / "x.docx"
    st = build_exact_docx([(img, blocks), (img, [])], out, title="t", calibrate=None)
    fonts = st.pop("fonts")
    assert st.pop("arabic_font")  # không có chữ Ả Rập → phông mặc định
    assert st == {"pages": 2, "text": 2, "table": 1, "image": 2, "whole_page_image": 1}
    names = zipfile.ZipFile(out).namelist()
    if fonts:  # phông được nhúng (đo = hiển thị trên mọi máy)
        assert any(n.startswith("word/fonts/") and n.endswith(".odttf") for n in names)
        assert "embedTrueTypeFonts" in zipfile.ZipFile(out).read("word/settings.xml").decode()
    xml = zipfile.ZipFile(out).read("word/document.xml").decode()
    # ảnh (logo + ảnh cả trang) neo theo trang; khối chữ + bảng = BẢNG NỔI theo trang (Google Docs giữ được,
    # khác khung chữ text box mà Google Docs bỏ mất) — không còn khung chữ nào
    assert xml.count("<wp:anchor") == 2 and xml.count('relativeFrom="page"') == 4
    assert xml.count('w:vertAnchor="page" w:horzAnchor="page"') == 3 and "wps:wsp" not in xml
    assert 'w:ascii="Arial"' in xml  # tên phông Latin có sẵn ở Word / Google Docs
    words = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml))
    for w in ["INVOICE", "2041", "Customer:", "Al", "Noor", "Trading", "Laptop", "1,250.00", "Price"]:
        assert w in words  # không mất chữ
    assert "<w:b/>" in xml  # **đậm** và tiêu đề giữ in đậm
    # ô "Qty" (chữ ở x=610 px) phải thụt lề để không bị kéo về mép trái ô
    assert re.search(r'<w:ind w:left="[1-9]\d*" w:right="0"/>', xml)
    # toạ độ: bảng ở x=100 px trên ảnh rộng 1240 px ↔ 210 mm → 16.94 mm = 960 twip
    xs = [int(v) for v in re.findall(r'w:tblpX="(-?\d+)"', xml)]
    assert any(abs(x - 100 / 1240 * 210 / 25.4 * 1440) <= 1 for x in xs)
    # logo ở x=900 px → EMU
    ps = [int(v) for v in re.findall(r'<wp:positionH relativeFrom="page"><wp:posOffset>(\d+)', xml)]
    assert any(abs(x - 900 / 1240 * 210 * 36000) < 2 for x in ps)
    from docx import Document

    d = Document(out)
    assert len(d.sections) == 2 and abs(d.sections[0].page_width.mm - 210) < 0.1
    assert abs(d.sections[0].page_height.mm - 1754 / 1240 * 210) < 0.1
    assert len(zipfile.ZipFile(out).namelist()) > 0 and any(n.startswith("word/media/") for n in zipfile.ZipFile(out).namelist())


def test_wrap_and_fit_respect_line_count():
    text = "The quick brown fox jumps over the lazy dog " * 6
    size, spacing, lines = fit_text(text, 600, 120, 3, 40, False)
    assert len(lines) <= 3 and all(len(line) > 0 for line in lines)
    assert wrap("a b c", 10_000, 12) == ["a b c"]


def test_text_bands_and_columns():
    img, blocks = _page()
    assert len(text_bands(img, blocks[1]["bbox"])) == 2  # khối 2 dòng
    b = column_bounds(img, blocks[3]["bbox"], 3)
    # ranh giới nằm trong khe trắng giữa các cột (cột 2 bắt đầu ở x=510, cột 3 ở x=800 trong khối)
    assert b is not None and len(b) == 4 and 120 < b[1] < 505 and 560 < b[2] < 795


def test_converter_builds_both_docx(tmp_path):
    """Model trả khối có chữ (như dots với blocks_with_text) → converter ghi thêm <tên>_bo_cuc.docx."""
    (tmp_path / "m.jsonl").write_text("")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"dataset": "m.jsonl", "models": [
        {"name": "lay", "adapter": "tests.test_docx_exact:FakeBlocks"}]}))
    from ocrbench.convert import Converter

    img, _ = _page()
    img.save(tmp_path / "a.png")
    res = Converter(tmp_path / "c.yaml", "lay").convert_file(tmp_path / "a.png", tmp_path / "o", force_ocr=True)
    assert res.docx.exists() and res.docx_exact and res.docx_exact.name == "a_bo_cuc.docx" and res.docx_exact.exists()
    assert all(p.image is None for p in res.pages)  # không giữ ảnh trong bộ nhớ sau khi dựng


from ocrbench.adapters.base import Adapter, Prediction  # noqa: E402


class FakeBlocks(Adapter):
    def predict(self, image, item):
        _, blocks = _page()
        return Prediction("# INVOICE No. 2041", extra={"layout": "ok", "blocks": blocks})


def test_exact_skipped_without_block_text(tmp_path):
    pytest.importorskip("docx")
    from ocrbench.convert import Converter

    (tmp_path / "m.jsonl").write_text("")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"dataset": "m.jsonl", "models": [
        {"name": "fake", "adapter": "tests.test_app:FakeOCR"}]}))
    Image.new("RGB", (300, 300), "white").save(tmp_path / "a.png")
    res = Converter(tmp_path / "c.yaml", "fake").convert_file(tmp_path / "a.png", tmp_path / "o")
    assert res.docx.exists() and res.docx_exact is None


def test_measure_by_color_ignores_neighbours():
    """Hiệu chỉnh đo từng khối theo màu riêng: nét đen / khối màu khác nằm sát bên không bị tính vào."""
    import numpy as np

    from ocrbench.docx_exact import _measure, _palette

    pal = _palette(2)
    rgb = np.full((200, 400, 3), 255, np.uint8)
    c0 = [int(pal[0][0][i:i + 2], 16) for i in (0, 2, 4)]
    c1 = [int(pal[1][0][i:i + 2], 16) for i in (0, 2, 4)]
    rgb[50:70, 100:300] = c0
    rgb[72:90, 60:350] = c1  # khối bên dưới, sát và rộng hơn
    rgb[40:48, 20:380] = 0  # nét đen (bảng / ảnh)
    s0, s1 = {"box": (100, 50, 300, 70)}, {"box": (60, 72, 350, 90)}
    got = _measure(rgb, [s0, s1], {id(s0): pal[0], id(s1): pal[1]})
    assert got[id(s0)] == (100, 50, 300, 70) and got[id(s1)] == (60, 72, 350, 90)


def test_calibration_with_libreoffice(tmp_path):
    """Có LibreOffice: dựng thử → đo → sửa; khối chữ không lệch quá 2 mm so với ảnh gốc."""
    from ocrbench.docx_exact import find_soffice

    soffice = find_soffice()
    if not soffice:
        pytest.skip("không có LibreOffice")
    img, blocks = _page()
    st = build_exact_docx([(img, blocks)], tmp_path / "c.docx", calibrate=soffice)
    cal = st["calibration"]
    assert cal["soffice"] and cal["max_dev_mm_final"] <= 2.0, cal
    assert not list(tmp_path.glob("*.cal.docx"))  # bản dựng thử đã xoá


def test_ink_mask_adaptive_on_faded_scan():
    """Nét mờ (xám ~120) vẫn là mực; vùng trắng / không tương phản thì không có mực."""
    import numpy as np

    from ocrbench.docx_exact import ink_mask

    g = np.full((20, 100), 235, np.uint8)
    g[8:12, 10:90] = 120
    m = ink_mask(g)
    assert m[8:12, 10:90].all() and not m[:8].any()
    assert not ink_mask(np.full((20, 100), 240, np.uint8)).any()


def test_identify_arabic_font_from_rendered_lines():
    """So mẫu: dòng chữ dựng bằng Amiri / Tajawal được nhận đúng phông (cần thư viện phông đã tải)."""
    import numpy as np
    from PIL import ImageFilter

    from ocrbench import arabic_fonts as AF

    lib = AF.available()
    if not {"Amiri", "Tajawal", "Noto Sans Arabic"} <= set(lib):
        pytest.skip("chưa tải thư viện phông Ả Rập (python -m ocrbench.arabic_fonts)")
    texts = ["البند الأول: تحرر هذا العقد من نسختين أصليتين", "نرجو التكرم بتزويدنا بعرض أسعار محدث"]
    for fam in ("Amiri", "Tajawal"):
        lines = []
        for t in texts:
            f = ImageFont.truetype(str(lib[fam][0]), 30, layout_engine=ImageFont.Layout.RAQM)
            im = Image.new("L", (900, 80), 255)
            ImageDraw.Draw(im).text((20, 15), t, font=f, fill=0, direction="rtl", language="ar")
            im = im.filter(ImageFilter.GaussianBlur(0.7))  # hơi mờ như ảnh scan
            lines.append((np.asarray(im) < 160, t))
        best, scores = AF.identify(lines, families=["Amiri", "Tajawal", "Noto Sans Arabic"])
        assert best == fam, scores


# ------------------------------------------------------------------ tiếng Ả Rập: hướng, thứ tự, bảng, nét ngoài khối

def test_paragraph_direction_follows_first_strong_char():
    from ocrbench.docx_exact import para_rtl

    assert para_rtl("الهاتف: +971 56 512 3883 contact@firm.co.uk")  # nhiều chữ Latin hơn nhưng bắt đầu bằng Ả Rập
    assert not para_rtl("IBAN: SA39 7931 7540")
    assert not para_rtl("12345")


def test_bidi_candidates_and_directional_marks():
    from ocrbench.arabic_bidi import LRE, PDF, apply, candidates

    t = "على الرقم +٩٧٤ ٨١٨ ٦٧٤٣ أو البريد contact@firm.co.uk."
    segs = [t[a:b] for a, b in candidates(t)]
    assert segs == ["+٩٧٤ ٨١٨ ٦٧٤٣", "contact@firm.co.uk"]
    assert candidates("الرقم: 24725") == []  # một cụm số liền: hướng không đổi được gì
    out = apply(t, candidates(t)[:1])
    assert out.count(LRE) == 1 and out.replace(LRE, "").replace(PDF, "") == t  # chỉ thêm dấu vô hình


def test_rtl_indents_are_physical_and_run_order_follows_schema():
    """Đoạn bidi: thụt lề mép phải ghi vào w:left (ECMA-376); rPr theo đúng thứ tự lược đồ (Word khắt khe)."""
    from ocrbench.docx_exact import _para, _run

    x = _para([[("مرحبا", False)]], 12, 14, True, "right", ind_left=0, ind_right=300)
    assert '<w:ind w:left="300" w:right="0"/>' in x and '<w:jc w:val="start"/>' in x
    r = _run("مرحبا", 11.3, bold=True, rtl=True, color="FF0000", sx=1.05)
    order = [r.index(t) for t in ("<w:b/>", "<w:color", "<w:w ", "<w:sz ", "<w:szCs", "<w:rtl/>")]
    assert order == sorted(order)
    assert 'w:sz w:val="23"' in r  # cỡ chữ làm tròn 0,5 pt; phần chênh bù bằng w:w


def test_ink_on_colored_band_is_the_text_not_the_band():
    import numpy as np

    from ocrbench.docx_exact import ink_colors, ink_rgb

    a = np.full((60, 200, 3), 255, np.uint8)
    a[10:50] = (68, 68, 68)  # hàng tiêu đề nền tối
    a[25:35, 40:160] = (255, 255, 255)  # chữ trắng
    m = ink_rgb(a)
    assert m[25:35, 40:160].all() and not m[12:20, :].any()
    from PIL import Image

    color, fill = ink_colors(Image.fromarray(a[10:50]), (0, 0, 200, 40))
    assert fill == "444444" and color == "FFFFFF"


def test_rtl_table_first_logical_cell_is_rightmost():
    """Bảng phải → trái: ô đầu tiên của dòng trong HTML nằm ở cột NGOÀI CÙNG BÊN PHẢI (bidiVisual)."""
    from PIL import features

    if not features.check("raqm"):
        pytest.skip("cần Raqm")
    from ocrbench.docx_exact import _font, table_spec, table_xml_spec

    img = Image.new("RGB", (900, 200), "white")
    d = ImageDraw.Draw(img)
    f = _font(False, 28, True)
    cells = [["الرقم", "الوصف الكامل للصنف", "الكمية"], ["1", "جهاز حاسوب محمول", "47"]]
    xs_right = [880, 600, 180]  # cột logic 0 → bên phải
    for r, row in enumerate(cells):
        for c, t in enumerate(row):
            d.text((xs_right[c], 40 + r * 80), t, font=f, fill="black", anchor="ra", direction="rtl", language="ar")
    html = "<table>" + "".join("<tr>" + "".join(f"<td>{t}</td>" for t in row) + "</tr>" for row in cells) + "</table>"
    sp = table_spec(img, (20, 20, 890, 180), html)
    assert sp["rtl"]
    first = next(c for c in sp["cells"] if c["r"] == 0 and c["c"] == 0)
    assert first["box"][0] > 600 and first["ink"][2] > 860
    assert "<w:bidiVisual/><w:tblW" in table_xml_spec(sp, 0.5)


def test_residual_layer_keeps_rules_drops_cut_glyph_parts():
    import numpy as np

    from ocrbench.docx_exact import residual_layer

    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    for x in range(20, 580, 4):  # đường kẻ chấm dài, đi qua dưới một khối chữ
        d.rectangle((x, 200, x + 1, 201), fill=(150, 150, 150))
    d.rectangle((100, 80, 300, 120), fill="black")  # "chữ" trong khối
    d.rectangle((301, 110, 306, 124), fill="black")  # đuôi chữ thò ra ngoài khung khối
    box = (100, 80, 300, 205)  # khung khối phủ cả lên đường kẻ
    rim, (rx, ry) = residual_layer(img, [box], [box])
    alpha = np.zeros((300, 600), bool)
    alpha[ry:ry + rim.height, rx:rx + rim.width] = np.asarray(rim)[..., 3] > 0
    assert alpha[200:202, 150:250].any()  # đoạn kẻ dưới khung khối vẫn còn
    assert alpha[200:202, 20:90].any() and alpha[200:202, 400:570].any()
    assert not alpha[105:125, 300:310].any()  # đuôi chữ bị cắt không thành vệt thừa


def test_embed_fonts_setting_in_schema_order():
    """<w:embedTrueTypeFonts/> phải đứng SAU w:zoom (CT_Settings) — đứng trước thì Word báo file hỏng."""
    from ocrbench.docx_exact import _settings_embed

    W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    xml = f'<w:settings {W}><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/></w:settings>'.encode()
    out = _settings_embed(xml).decode()
    assert out.index("zoom") < out.index("embedTrueTypeFonts") < out.index("defaultTabStop")
    assert _settings_embed(out.encode()).decode().count("embedTrueTypeFonts") == 1


def test_residual_layer_ignores_aligned_glyph_slivers():
    """Đầu chữ thẳng hàng ở mép phải các dòng Ả Rập (thò ra ngoài khung) không phải đường kẻ dọc."""
    import numpy as np

    from ocrbench.docx_exact import residual_layer

    img = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(img)
    boxes = []
    for k in range(15):
        y = 30 + k * 50
        for x in range(100, 500, 7):  # dòng chữ (nét dọc cách nhau, như chữ thật — không phải khối đặc)
            d.rectangle((x, y, x + 2, y + 24), fill="black")
        d.rectangle((501, y + 2, 503, y + 22), fill="black")  # đầu chữ thò ra ngoài khung 2–3 px
        boxes.append((100, y, 500, y + 24))
    res = residual_layer(img, boxes, boxes)
    if res is not None:
        rim, (rx, ry) = res
        alpha = np.zeros((800, 600), bool)
        alpha[ry:ry + rim.height, rx:rx + rim.width] = np.asarray(rim)[..., 3] > 0
        assert not alpha[:, 500:506].any()


def test_editable_docx_inserts_pictures_in_reading_order(tmp_path):
    """Bản A (sửa được): khối Picture (logo / chữ ký / con dấu) thành ảnh cắt từ trang gốc, đúng thứ tự đọc;
    ảnh cùng hàng chung một dòng; đoạn Ả Rập căn đầu dòng (jc=start), bidi đúng vị trí trong pPr."""
    from ocrbench.docx_export import add_page, new_document

    img, blocks = _page()
    blocks = blocks + [{"category": "Picture", "bbox": [100, 500, 300, 560], "text": ""},
                       {"category": "Picture", "bbox": [700, 505, 900, 565], "text": ""},
                       {"category": "Text", "bbox": [100, 600, 600, 640], "text": "## الطرف الأول"}]
    doc = new_document("t")
    add_page(doc, "", header="h", first=True, blocks=blocks, image=img)
    out = tmp_path / "a.docx"
    doc.save(out)
    xml = zipfile.ZipFile(out).read("word/document.xml").decode()
    assert xml.count("<pic:pic") == 3  # logo + 2 ảnh cùng hàng
    body = re.sub(r"<w:drawing>.*?</w:drawing>", "[ẢNH]", xml, flags=re.S)
    words = re.findall(r"<w:t[^>]*>([^<]*)</w:t>|(\[ẢNH\])", body)
    seq = [a or b for a, b in words]
    assert seq.index("[ẢNH]") > seq.index("INVOICE No. 2041")  # logo sau tiêu đề như thứ tự khối
    p = re.search(r"<w:p>(?:(?!</w:p>).)*\[ẢNH\](?:(?!</w:p>).)*\[ẢNH\](?:(?!</w:p>).)*</w:p>", body, flags=re.S)
    assert p  # hai ảnh cùng hàng → cùng một đoạn
    rtl_p = re.search(r"<w:pPr>((?:(?!</w:pPr>).)*)</w:pPr>(?:(?!</w:p>).)*الطرف", xml, flags=re.S).group(1)
    assert 'w:jc w:val="start"' in rtl_p and rtl_p.index("pStyle") < rtl_p.index("bidi") < rtl_p.index("jc")
