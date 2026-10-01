"""Kiểm tra phần dựng bộ test tổng hợp (không cần mạng; phần dựng ảnh bỏ qua nếu không có Chromium)."""

import random
import re

import pytest
from PIL import Image

from ocrbench.metrics import teds
from ocrbench.normalize import NormConfig, normalize, to_plain_text
from ocrbench.testset.degrade import degrade
from ocrbench.testset.templates import KINDS, isolate_ltr_runs, make_doc

LANGS = ("ar", "en", "mixed")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("lang", LANGS)
def test_docs_have_consistent_ground_truth(kind, lang):
    for seed in range(5):
        doc = make_doc(kind, lang, random.Random(seed), "")
        assert doc.gt.strip()
        # hình trang trí không được lọt vào đáp án
        assert "<svg" not in doc.gt and "class=" not in doc.gt
        if kind == "invoice":
            assert doc.gt_type == "table_html" and doc.gt.count("<table") == 2
            assert teds(doc.gt, doc.gt, NormConfig()) == pytest.approx(1.0)
        else:
            assert doc.gt_type == "text" and "<" not in doc.gt
        # mọi dòng đáp án phải xuất hiện trong trang hiển thị (không có chữ "ma")
        page = re.sub(r"<style>.*?</style>", "", doc.page_html, flags=re.S)
        page = re.sub(r'</?span( dir="ltr")?>', "", page)  # thẻ inline không tạo khoảng trắng trên ảnh
        page_text = normalize(to_plain_text(page), NormConfig())
        for line in to_plain_text(doc.gt).splitlines():
            line = normalize(line, NormConfig())
            if line:
                assert line in page_text, line


def test_same_seed_same_document():
    a = make_doc("invoice", "ar", random.Random("x"), "")
    b = make_doc("invoice", "ar", random.Random("x"), "")
    assert a.gt == b.gt and a.page_html == b.page_html


def test_language_content():
    ar = make_doc("contract", "ar", random.Random(1), "")
    en = make_doc("contract", "en", random.Random(1), "")
    assert re.search(r"[؀-ۿ]", ar.gt) and not re.search(r"[؀-ۿ]", en.gt)
    assert 'dir="rtl"' in ar.page_html and 'dir="ltr"' in en.page_html


def test_ltr_isolation_keeps_text():
    html = "<p>الهاتف: +966 58 440 7282 info@example.com</p><td>Dell Latitude 5440</td>"
    out = isolate_ltr_runs(html)
    assert '<span dir="ltr">+966 58 440 7282 info@example.com</span>' in out
    assert re.sub(r'</?span( dir="ltr")?>', "", out) == html


def test_degrade():
    img = Image.new("RGB", (800, 1000), "white")
    for level in ("medium", "hard"):
        out, info = degrade(img, random.Random(0), level)
        assert out.mode == "RGB" and out.width < 800 and info["level"] == level


def test_render_arabic_page(tmp_path):
    pytest.importorskip("playwright")
    from ocrbench.testset.render import Renderer

    doc = make_doc("invoice", "ar", random.Random(0), "")
    try:
        with Renderer(scale=1.0) as r:
            r.render(doc.page_html, tmp_path / "x.png")
    except RuntimeError as e:
        pytest.skip(f"không có Chromium: {e}")
    img = Image.open(tmp_path / "x.png")
    assert img.width == 1000 and img.height > 500
