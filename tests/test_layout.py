"""Kết quả dạng bố cục (dots.ocr / dots.mocr) và bản vá chạy dots trên T4."""

import json
import sys
import types

import pytest
from PIL import Image

from ocrbench.layout import Block, blocks_to_markdown, draw_blocks, parse_layout, smart_resize

SAMPLE = [
    {"bbox": [10, 10, 500, 60], "category": "Page-header", "text": "Company Ltd"},
    {"bbox": [100, 100, 900, 160], "category": "Title", "text": "Invoice"},
    {"bbox": [100, 200, 900, 300], "category": "Text", "text": "Customer: **Ali**"},
    {"bbox": [100, 320, 900, 600], "category": "Table",
     "text": "<table><tr><td>Item</td><td>Qty</td></tr><tr><td>Pen</td><td>2</td></tr></table>"},
    {"bbox": [100, 650, 900, 700], "category": "List-item", "text": "Paid in cash"},
    {"bbox": [100, 720, 400, 900], "category": "Picture"},
]


def test_smart_resize_matches_qwen():
    assert smart_resize(1000, 1000) == (1008, 1008)
    h, w = smart_resize(20000, 15000)
    assert h % 28 == 0 and w % 28 == 0 and h * w <= 11289600


def test_parse_ok_and_rescale_to_original():
    w, h = 1500, 2000
    ih, iw = smart_resize(h, w)
    raw = json.dumps([{"bbox": [0, 0, iw, ih], "category": "Text", "text": "x"}])
    blocks, status = parse_layout(raw, (w, h))
    assert status == "ok" and blocks[0].bbox == [0, 0, w, h]


def test_markdown_headings_tables_lists():
    blocks, status = parse_layout(json.dumps(SAMPLE))
    md = blocks_to_markdown(blocks)
    assert status == "ok"
    assert md.split("\n\n")[:3] == ["Company Ltd", "# Invoice", "Customer: **Ali**"]
    assert "<table>" in md and "- Paid in cash" in md
    assert "Company Ltd" not in blocks_to_markdown(blocks, drop=("Page-header",))


def test_truncated_json_is_salvaged():
    raw = json.dumps(SAMPLE[:4])[:-60]  # bị cắt giữa khối bảng (hết max_new_tokens)
    blocks, status = parse_layout(raw)
    assert status == "repaired"
    assert [b.category for b in blocks] == ["Page-header", "Title", "Text"]


def test_repeated_blocks_are_deduplicated():
    raw = json.dumps(SAMPLE[:3] + [SAMPLE[2]] * 5)  # model lặp lại nguyên khối
    blocks, status = parse_layout(raw)
    assert status == "repaired" and len(blocks) == 3


def test_not_json_is_failed():
    blocks, status = parse_layout("just plain text")
    assert status == "failed" and blocks == []


def test_draw_blocks():
    img = Image.new("RGB", (1000, 1000), "white")
    out = draw_blocks(img, [Block("Table", "", [100, 100, 500, 500])])
    assert out.size == img.size and out.getpixel((100, 300)) != (255, 255, 255)


def test_hf_vlm_layout_output():
    from ocrbench.adapters.hf_vlm import HFVLMAdapter

    a = HFVLMAdapter(model_id="x", output_format="layout_json", layout_drop=["Page-header"])
    a._processor = types.SimpleNamespace(image_processor=types.SimpleNamespace(min_pixels=3136, max_pixels=11289600))
    extra = {}
    text = a._layout(json.dumps(SAMPLE), (1008, 1008), extra)
    assert text.startswith("# Invoice") and extra["layout"] == "ok"
    assert extra["blocks"][0] == {"category": "Page-header", "bbox": [10, 10, 500, 60]}
    assert "text" not in extra["blocks"][0]  # không chép chữ hai lần vào predictions.jsonl


# --- bản vá dots_vision -------------------------------------------------------------------------------

def test_flash_attn_stub():
    from ocrbench.adapters import patches

    saved = sys.modules.pop("flash_attn", None)
    try:
        try:
            import flash_attn  # noqa: F401
            pytest.skip("máy này có flash_attn thật")
        except ImportError:
            pass
        patches.install_flash_attn_stub()
        from flash_attn import flash_attn_varlen_func

        with pytest.raises(RuntimeError):
            flash_attn_varlen_func()
    finally:
        sys.modules.pop("flash_attn", None)
        if saved is not None:
            sys.modules["flash_attn"] = saved


def test_per_image_attention_equals_masked():
    """Tính attention riêng từng ảnh = attention có mặt nạ khối chéo của code gốc (y hệt về toán)."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from ocrbench.adapters.patches import per_image_sdpa_forward

    mod = types.ModuleType("fake_dots_vision")
    mod.apply_rotary_pos_emb_vision = lambda t, freqs: t * freqs.cos().unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
    sys.modules[mod.__name__] = mod

    class VisionSdpaAttention(torch.nn.Module):
        def __init__(self, dim=32, heads=4):
            super().__init__()
            self.num_heads, self.qkv, self.proj = heads, torch.nn.Linear(dim, dim * 3), torch.nn.Linear(dim, dim)

    VisionSdpaAttention.__module__ = mod.__name__
    torch.manual_seed(0)
    m = VisionSdpaAttention()
    cu = torch.tensor([0, 30, 42, 70], dtype=torch.int32)  # 3 ảnh
    x = torch.randn(70, 32)
    rope = torch.randn(70, 4)
    with torch.no_grad():
        q, k, v = m.qkv(x).reshape(70, 3, 4, -1).permute(1, 0, 2, 3).unbind(0)
        q = mod.apply_rotary_pos_emb_vision(q.unsqueeze(0), rope).squeeze(0)
        k = mod.apply_rotary_pos_emb_vision(k.unsqueeze(0), rope).squeeze(0)
        mask = torch.zeros(1, 70, 70, dtype=torch.bool)
        for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
            mask[..., a:b, a:b] = True
        ref = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), mask)
        ref = m.proj(ref.transpose(0, 1).reshape(70, -1))
        out = per_image_sdpa_forward(m, x, cu, rope)
    assert torch.allclose(ref, out, atol=1e-5)
