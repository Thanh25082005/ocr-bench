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


def test_postprocess_inverted_picture_bbox_does_not_crash():
    """Model trả khối Picture có y2 < y1 (gặp thật trên ảnh giấy tờ cũ): bản gốc crop lỗi ValueError → mất trang."""
    origin = Image.new("RGB", (W, H), "white")
    cells = [{"bbox": [100, 900, 400, 700], "category": "Picture"},
             {"bbox": [IW - 50, 10, IW + 300, 60], "category": "Text", "text": "tràn mép phải"},
             {"bbox": [100, 100, 600, 180], "category": "Text", "text": "bình thường"}]
    r = postprocess_response(json.dumps(cells), "prompt_layout_all_en", origin, origin, None, None)
    assert r["extra"]["layout"] == "ok" and r["extra"]["bbox_fixed"] == 2
    for b in r["extra"]["blocks"]:
        x1, y1, x2, y2 = b["bbox"]
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
    assert r["extra"]["blocks"][2]["bbox"] == [int(v / (IW / W)) if i % 2 == 0 else int(v / (IH / H))
                                               for i, v in enumerate([100, 100, 600, 180])]  # khối hợp lệ giữ nguyên
    assert "data:image" in r["md"]


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


def test_layout_loop_stops_only_on_repeated_cell():
    """dots lặp cùng một ô (toạ độ khác nhau) tới hết token — dừng sớm; khối khác nhau / chữ ngắn / ít lần → không."""
    from ocrbench.adapters.dots import layout_loop

    rep = ", ".join('{"bbox": [10, %d, 200, %d], "category": "Text", "text": "Binds to Nucleocapsid"}' % (i, i + 9)
                    for i in range(30))
    assert layout_loop('[{"bbox": [1,2,3,4], "category": "Title", "text": "T"}, ' + rep)
    diff = ", ".join('{"bbox": [1,2,3,4], "category": "Text", "text": "dòng %d"}' % i for i in range(60))
    assert not layout_loop("[" + diff)
    short = ", ".join('{"bbox": [1,%d,3,4], "category": "Text", "text": "1"}' % i for i in range(60))
    assert not layout_loop("[" + short)  # ô số ngắn lặp hợp lệ (bảng)
    few = ", ".join('{"bbox": [1,%d,3,4], "category": "Text", "text": "Binds to Nucleocapsid"}' % i for i in range(10))
    assert not layout_loop("[" + few)


def test_loop_stop_long_period():
    """Dòng ~100 token lặp mãi (ảnh giấy tờ cũ): chỉ bắt khi bật long_max_period, và phải lặp ≥ 8 lần."""
    import torch

    from ocrbench.adapters.hf_vlm import _LoopStop

    line = list(range(1000, 1100))  # chu kỳ 100 token
    prefix = list(range(1, 300))

    def run(gen, **kw):
        stop = _LoopStop(torch, 0, 1, 60, 600, every=1, **kw)
        return bool(stop(torch.tensor([gen]), None)[0])

    assert not run(prefix + line * 10)  # mặc định (benchmark): như cũ, không bắt
    assert run(prefix + line * 8, long_max_period=256)
    assert not run(prefix + line * 7, long_max_period=256)  # 7 dòng trống giống nhau của bảng: không cắt
    assert run(prefix + [5, 6] * 300, long_max_period=256)  # vòng ngắn vẫn bắt như cũ


def test_close_truncated_layout_keeps_last_block():
    """Dừng giữa khối đang lặp: giữ chữ trước chỗ lặp + một bản dòng lặp, đóng JSON; JSON đủ → không đổi."""
    import json

    from ocrbench.adapters.dots import close_truncated_layout

    head = '[{"bbox": [1, 2, 3, 4], "category": "Title", "text": "عنوان"}, {"bbox": [5, 6, 7, 8], "category": "Text", '
    rep = "وإذا الوكيل الخذير بالأقرار والتقرير\\n"
    raw = head + '"text": "سطر أول صحيح\\n' + rep * 20 + rep[:9]
    data = json.loads(close_truncated_layout(raw))
    assert [d["category"] for d in data] == ["Title", "Text"]
    assert data[1]["text"].startswith("سطر أول صحيح\n") and data[1]["text"].count("الخذير") <= 2
    ok = head + '"text": "x....................................."}]'
    assert close_truncated_layout(ok) == ok
    assert close_truncated_layout('[{"bbox": [1, 2') == '[{"bbox": [1, 2'  # không sửa được → giữ nguyên
    cut = '[{"bbox": [1, 2, 3, 4], "category": "Text", "text": "abc\\'  # cắt giữa dấu thoát
    assert json.loads(close_truncated_layout(cut))[0]["text"] == "abc"
