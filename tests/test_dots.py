"""Adapter dots (giống source gốc dots.ocr): hậu xử lý bằng chính hàm của họ, ghi file như parse_file của họ."""

import json

import pytest
from PIL import Image

pytest.importorskip("fitz")  # dots_ocr/utils cần PyMuPDF

from ocrbench.adapters.dots import DotsAdapter, dots_utils, parse_file, postprocess_response  # noqa: E402
from ocrbench.layout import smart_resize  # noqa: E402

W, H = 1500, 2000
IH, IW = smart_resize(H, W)
CELLS = [
    {"bbox": [10, 10, 600, 60], "category": "Page-header", "text": "ACME Ltd"},
    {"bbox": [100, 100, IW - 100, 180], "category": "Title", "text": "# Invoice"},
    {"bbox": [100, 200, IW - 100, 400], "category": "Table", "text": "<table><tr><td>Pen</td><td>2</td></tr></table>"},
    {"bbox": [100, 450, 400, 700], "category": "Picture"},
]


def test_vendor_is_verbatim():
    """Bản chép phải khớp sha256 ghi trong VENDORED.md (không ai được sửa code của họ)."""
    import hashlib
    import re
    from pathlib import Path

    v = Path(__file__).resolve().parent.parent / "ocrbench/_vendor/dots_ocr"
    sums = re.findall(r"^([0-9a-f]{64})  (\S+\.py)$", (v / "VENDORED.md").read_text(), flags=re.M)
    assert len(sums) == 8
    for h, name in sums:
        assert hashlib.sha256((v / "utils" / name).read_bytes()).hexdigest() == h, name


def test_prompts_are_theirs():
    _, prompts, *_ = dots_utils()
    from ocrbench.layout import DOTS_PROMPTS

    assert prompts.dict_promptmode_to_prompt["prompt_layout_all_en"] == DOTS_PROMPTS["layout"]
    assert prompts.dict_promptmode_to_prompt["prompt_ocr"] == DOTS_PROMPTS["text"]


def test_postprocess_layout_ok():
    origin = Image.new("RGB", (W, H), "white")
    image = origin  # fitz_preprocess tắt: ảnh đưa vào = ảnh gốc
    r = postprocess_response(json.dumps(CELLS), "prompt_layout_all_en", origin, image, None, None)
    assert r["extra"]["layout"] == "ok" and not r["filtered"]
    assert r["extra"]["blocks"][1]["bbox"][2] == int((IW - 100) / (IW / W))  # bbox đổi về ảnh gốc như code của họ
    assert "ACME Ltd" in r["md"] and "ACME Ltd" not in r["md_nohf"]
    assert "<table>" in r["md"] and "data:image" in r["md"]  # layoutjson2md nhúng ảnh khối Picture


def test_postprocess_broken_json_uses_their_cleaner():
    origin = Image.new("RGB", (W, H), "white")
    raw = json.dumps(CELLS[:3])[:-40]  # bị cắt
    r = postprocess_response(raw, "prompt_layout_all_en", origin, origin, None, None)
    assert r["filtered"] and r["extra"]["layout"] == "filtered"
    assert "Invoice" in r["md"]


def test_postprocess_text_mode_passthrough():
    origin = Image.new("RGB", (W, H), "white")
    r = postprocess_response("plain text", "prompt_ocr", origin, origin, None, None)
    assert r["md"] == "plain text" and r["cells"] is None


class FakeDots(DotsAdapter):
    def load(self):
        pass

    def _inference_with_hf(self, image, prompt):
        if "layout" in prompt.lower() and "bbox" in prompt:
            return json.dumps(CELLS), {"new_tokens": 10, "hit_max_tokens": False}
        return "plain text", {"new_tokens": 2, "hit_max_tokens": False}


def test_predict_strips_embedded_images_and_respects_nohf():
    a = FakeDots()
    text = a.predict(Image.new("RGB", (W, H), "white"), None).text
    assert "data:image" not in text and "ACME Ltd" in text
    a = FakeDots(no_page_hf=True)
    assert "ACME Ltd" not in a.predict(Image.new("RGB", (W, H), "white"), None).text


def test_parse_file_writes_same_files_as_their_tool(tmp_path):
    Image.new("RGB", (W, H), "white").save(tmp_path / "doc.png")
    res = parse_file(FakeDots(), tmp_path / "doc.png", tmp_path / "out")
    d = tmp_path / "out" / "doc"
    assert sorted(p.name for p in d.iterdir()) == ["doc.jpg", "doc.json", "doc.md", "doc_nohf.md"]
    assert (tmp_path / "out" / "doc.jsonl").exists() and res[0]["page_no"] == 0
    assert json.loads((d / "doc.json").read_text())[1]["category"] == "Title"
    # PDF 2 trang → doc2_page_0.*, doc2_page_1.*
    Image.new("RGB", (600, 800), "white").save(tmp_path / "doc2.pdf", save_all=True,
                                               append_images=[Image.new("RGB", (600, 800), "white")])
    parse_file(FakeDots(), tmp_path / "doc2.pdf", tmp_path / "out", prompt_mode="prompt_ocr")
    names = sorted(p.name for p in (tmp_path / "out" / "doc2").iterdir())
    assert names == ["doc2_page_0.jpg", "doc2_page_0.md", "doc2_page_1.jpg", "doc2_page_1.md"]
