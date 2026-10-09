"""Bố cục kraken (ocrbench/adapters/kraken_layout.py) — tách dòng giả lập, không cần venv kraken."""

import os

import pytest
from PIL import Image, ImageDraw

from ocrbench.adapters.base import create_adapter
from ocrbench.adapters.kraken_layout import (KrakenLayoutAdapter, default_python, drop_faint_lines, line_crop,
                                             parse_lines, text_mask)


def _rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _page():
    """Giấy 1000×800: 3 dòng chữ đậm, 1 dòng watermark nhạt, dấu đỏ (kraken coi một phần dấu là "dòng")."""
    img = Image.new("RGB", (1000, 800), (245, 240, 230))
    d = ImageDraw.Draw(img)
    for k in range(3):
        y = 100 + k * 60
        for x in range(150, 850, 60):
            d.rectangle((x, y + 8, x + 40, y + 30), fill=(30, 30, 30))
    d.text((400, 300), "watermark", fill=(235, 200, 200))
    d.rectangle((380, 290, 600, 320), fill=(238, 215, 212))  # mảng watermark nhạt
    d.ellipse((150, 450, 350, 650), fill=(220, 30, 30))  # dấu đỏ
    seg = {"lines": [{"boundary": _rect(140, 100 + k * 60, 860, 140 + k * 60), "baseline": []} for k in range(3)]
           + [{"boundary": _rect(370, 285, 610, 325)},                      # watermark
              {"boundary": _rect(160, 520, 340, 560)},                      # "dòng" trên dấu đỏ
              {"boundary": [[1, 1], [2, 2], [1, 2]]}],                      # suy biến
           "regions": [], "seconds": 0.1}
    return img, seg


class FakeWorker:
    def __init__(self, seg):
        self.seg = seg

    def segment(self, path, text_direction):
        assert os.path.exists(path) and text_direction == "horizontal-rl"
        return self.seg

    def close(self):
        pass


def _adapter(seg, **params):
    from concurrent.futures import ThreadPoolExecutor

    ad = KrakenLayoutAdapter(recognizer="none", signatures=False, **params)
    ad.worker, ad.pool = FakeWorker(seg), ThreadPoolExecutor(2)
    return ad


def test_registered():
    assert isinstance(create_adapter("kraken_layout", {}), KrakenLayoutAdapter)


def test_parse_lines_drops_degenerate():
    _, seg = _page()
    lines = parse_lines(seg)
    assert len(lines) == 5
    assert lines[0]["bbox"] == [140, 100, 860, 140] and lines[0]["polygon"] == _rect(140, 100, 860, 140)


def test_drop_faint_lines_removes_watermark_only():
    img, seg = _page()
    kept = drop_faint_lines(img, parse_lines(seg))
    assert [ln["bbox"][1] for ln in kept] == [100, 160, 220, 520]  # watermark (285) bị bỏ, dấu đỏ còn (đậm)


def test_text_mask_and_line_height():
    _, seg = _page()
    mask, lh = text_mask((1000, 800), parse_lines(seg)[:3])
    assert mask.shape == (800, 1000) and mask[120, 500] and not mask[400, 500]
    assert lh == pytest.approx(0.58 * 40)


def test_line_crop_whitens_outside_polygon_and_upscales():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    ln = {"bbox": [10, 10, 110, 30], "polygon": [[10, 10], [110, 10], [10, 30]]}  # tam giác
    crop = line_crop(img, ln, pad=0, min_height=40)
    assert crop.height == 40 and crop.width == 200
    assert crop.getpixel((crop.width - 2, crop.height - 2)) == (255, 255, 255)  # ngoài đa giác → trắng
    assert crop.getpixel((2, 2)) == (0, 0, 0)


def test_predict_blocks_lines_and_pictures_in_reading_order():
    img, seg = _page()
    pred = _adapter(seg).predict(img, None)
    cats = [(b["category"], b["bbox"][1]) for b in pred.extra["blocks"]]
    # 3 dòng chữ; watermark bị bỏ; "dòng" trên dấu đỏ bị bỏ, thay bằng khối Picture
    assert [c for c, _ in cats] == ["Text", "Text", "Text", "Picture"]
    assert all("polygon" in b for b in pred.extra["blocks"] if b["category"] == "Text")
    assert pred.extra["lines"] == 3


def test_missing_venv_message(monkeypatch):
    monkeypatch.setattr("ocrbench.adapters.kraken_layout.default_python", lambda: None)
    with pytest.raises(RuntimeError, match="setup_kraken.sh"):
        KrakenLayoutAdapter().load()


@pytest.mark.skipif(not default_python(), reason="chưa cài venv kraken (inference/setup_kraken.sh)")
def test_real_kraken_on_sample():
    ad = KrakenLayoutAdapter(recognizer="none", signatures=False, pictures=False)
    ad.load()
    try:
        pred = ad.predict(Image.open("inference/samples/05_thu_tieng_anh.png"), None)
    finally:
        ad.close()
    assert pred.extra["lines"] >= 10
