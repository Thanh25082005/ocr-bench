"""Tìm vùng hình (con dấu, vân tay, chữ ký) cho model chỉ đọc chữ: ocrbench/pictures.py."""

from PIL import Image, ImageDraw

from ocrbench.pictures import add_pictures, find_pictures

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
    seal, sig = sorted(pics)
    assert abs(seal[0] - 150) < 10 and abs(seal[3] - 600) < 10
    assert abs(sig[0] - 600) < 10 and abs(sig[2] - 800) < 10


def test_add_pictures_reading_order_and_drops_garbage_text():
    blocks = [{"category": "Text", "bbox": [0, 0, 100, 20], "text": "a"},
              {"category": "Text", "bbox": [160, 450, 200, 470], "text": "rác trên con dấu"},
              {"category": "Text", "bbox": [0, 700, 100, 720], "text": "b"}]
    out = add_pictures(blocks, [[150, 400, 350, 600]])
    assert [b["category"] for b in out] == ["Text", "Picture", "Text"]
    assert out[1]["bbox"] == [150, 400, 350, 600]
