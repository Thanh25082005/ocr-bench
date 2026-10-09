"""Tìm vùng hình (con dấu, vân tay, chữ ký) cho model chỉ đọc chữ: ocrbench/pictures.py."""

from PIL import Image, ImageDraw

from ocrbench.pictures import add_pictures, combine, find_pictures

LINE = 20  # cỡ chữ giả lập (chiều cao "từ" OCR đã đọc)


def _page():
    """Trang trắng 1000×1000 trên nền tối (như ảnh chụp điện thoại): 3 dòng chữ, dấu đỏ, chữ ký, khung bảng, vết ố."""
    img = Image.new("RGB", (1100, 1100), (25, 25, 25))
    d = ImageDraw.Draw(img)
    d.rectangle((50, 50, 1050, 1050), fill=(245, 240, 230))
    words = []
    for row in range(3):  # "chữ": vạch đen nhỏ, OCR đọc chắc
        y = 100 + row * 40
        for col in range(8):
            x = 100 + col * 100
            d.rectangle((x, y, x + 70, y + LINE), fill=(30, 30, 30))
            words.append(([x, y, x + 70, y + LINE], 90.0, (1, 1, row), "word"))
    d.ellipse((150, 400, 350, 600), fill=(220, 30, 30))  # dấu đỏ
    d.line([(600, 450), (650, 400), (700, 500), (750, 420), (800, 480)], fill=(20, 20, 60), width=4)  # chữ ký
    for k in range(4):  # khung bảng thưa
        d.line([(500, 700 + k * 60), (950, 700 + k * 60)], fill=(0, 0, 0), width=2)
        d.line([(500 + k * 150, 700), (500 + k * 150, 880)], fill=(0, 0, 0), width=2)
    d.ellipse((150, 750, 230, 800), fill=(200, 160, 120))  # vết ố nhạt
    return img, words


def test_finds_seal_and_signature_only():
    img, words = _page()
    pics = find_pictures(img, words)
    assert len(pics) == 2
    seal, sig = sorted(p["bbox"] for p in pics)
    assert abs(seal[0] - 150) < 10 and abs(seal[3] - 600) < 10
    assert abs(sig[0] - 600) < 10 and abs(sig[2] - 800) < 10


def test_add_pictures_reading_order_and_drops_garbage_text():
    blocks = [{"category": "Text", "bbox": [0, 0, 100, 20], "text": "a"},
              {"category": "Text", "bbox": [160, 450, 200, 470], "text": "rác trên con dấu"},
              {"category": "Text", "bbox": [0, 700, 100, 720], "text": "b"}]
    out = add_pictures(blocks, [{"category": "Picture", "bbox": [150, 400, 350, 600]}])
    assert [b["category"] for b in out] == ["Text", "Picture", "Text"]
    assert out[1]["bbox"] == [150, 400, 350, 600]


def test_faint_fingerprint_found_but_rust_stain_and_punch_hole_are_not():
    img, words = _page()
    d = ImageDraw.Draw(img)
    d.ellipse((150, 650, 230, 730), fill=(200, 205, 215))  # vân tay mực xanh nhạt (giấy vàng → xanh hơn giấy)
    d.ellipse((985, 400, 1025, 440), fill=(5, 5, 5))  # lỗ đục hồ sơ sát mép phải tờ giấy (mép ở x = 1050)
    pics = find_pictures(img, words)
    assert any(p["kind"] == "tone" and abs(p["bbox"][0] - 150) < 15 for p in pics)
    assert not any(p["bbox"][0] > 960 for p in pics)  # lỗ đục
    assert not any(140 < p["bbox"][1] < 260 and p["bbox"][1] > 740 for p in pics)  # vết ố (y 750–800)


def _pic(bbox, kind="dark", fill=0.5):
    return {"bbox": bbox, "kind": kind, "fill": fill}


def test_combine_signatures_with_pictures():
    pics = [_pic([0, 0, 300, 400], fill=0.45),            # cụm tem + dấu
            _pic([500, 500, 560, 580], fill=0.1),         # mảnh chữ ký luật tìm được
            _pic([700, 100, 900, 140], "blue", 0.13),     # chữ viết tay dài, thưa
            _pic([700, 300, 780, 380], "blue", 0.3)]      # logo tròn
    sigs = [[50, 100, 120, 200],                          # nằm trong cụm tem → bỏ
            [480, 490, 640, 600]]                         # trùm mảnh chữ ký → gộp
    out = combine(pics, sigs)
    assert {(r["category"], tuple(r["bbox"])) for r in out} == {
        ("Picture", (0, 0, 300, 400)), ("Picture", (700, 300, 780, 380)), ("Signature", (480, 490, 640, 600))}
    # không có model chữ ký: giữ nguyên mọi hình luật tìm được
    assert len(combine(pics, None)) == 4
