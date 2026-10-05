"""Vá code `trust_remote_code` của vài model để chạy được trên T4 (không FlashAttention, không bf16).

Mỗi bản vá KHÔNG đổi phép tính của model (chỉ đổi cách tính / kiểu số chính xác hơn). Bật bằng
`params.patches: [tên]` của adapter hf_vlm.

dots_vision (dots.ocr, dots.mocr):
1. File modeling_dots_vision.py `import flash_attn` ngay đầu file → chưa cài thì nạp model lỗi.
   Đặt một module `flash_attn` giả (gọi tới là báo lỗi) — bản vá 2 bảo đảm không đường nào gọi tới.
2. Bộ mã hoá ảnh mặc định flash_attention_2 → đổi sang "sdpa" trong config.
3. Attention "sdpa" gốc dựng mặt nạ đầy đủ N×N cho mọi patch (N ~ 15.000 với trang A4 200 DPI) → PyTorch
   rơi về kernel math, cần ~5 GB cho ma trận attention mỗi lớp → hết VRAM T4. Mặt nạ đó chỉ để các ảnh
   khác nhau không nhìn thấy nhau, nên tính attention RIÊNG cho từng ảnh, không mặt nạ, là y hệt về toán,
   và dùng được kernel tiết kiệm bộ nhớ.
4. forward() của bộ mã hoá ảnh ép đầu vào về bfloat16 (T4 không có) → chạy bộ mã hoá ảnh ở float32
   (chính xác hơn bf16, tránh tràn số của fp16; ~1,2 tỉ tham số × 4 byte ≈ 4,8 GB), phần ngôn ngữ giữ fp16.
"""

from __future__ import annotations

import functools
import importlib.machinery
import sys
import types


def install_flash_attn_stub() -> None:
    try:
        import flash_attn  # noqa: F401

        return
    except ImportError:
        pass
    mod = types.ModuleType("flash_attn")
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)

    def _missing(*a, **k):
        raise RuntimeError("flash_attn không có trên máy này (bản vá dots_vision lẽ ra phải chuyển sang sdpa)")

    mod.flash_attn_varlen_func = _missing
    mod.flash_attn_func = _missing
    sys.modules["flash_attn"] = mod


def per_image_sdpa_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    """Thay VisionSdpaAttention.forward của dots: attention theo từng ảnh (đoạn cu_seqlens), không mặt nạ."""
    import torch
    import torch.nn.functional as F

    apply_rope = sys.modules[type(self).__module__].apply_rotary_pos_emb_vision
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q = apply_rope(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rope(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    outs = []
    bounds = cu_seqlens.tolist()
    for a, b in zip(bounds[:-1], bounds[1:]):
        # (1, heads, len, dim): dạng 4 chiều để PyTorch chọn kernel tiết kiệm bộ nhớ
        qi, ki, vi = (t[a:b].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
        outs.append(oi.squeeze(0).transpose(0, 1))
    attn_output = torch.cat(outs, dim=0).reshape(seq_length, -1)
    return self.proj(attn_output)


def dots_vision_before_load(config) -> None:
    install_flash_attn_stub()
    vc = getattr(config, "vision_config", None)
    if vc is not None:
        if isinstance(vc, dict):
            vc["attn_implementation"] = "sdpa"
        else:
            vc.attn_implementation = "sdpa"


def dots_vision_after_load(model, vision_dtype) -> None:
    tower = model.vision_tower
    tower.to(vision_dtype)
    n = 0
    for m in tower.modules():
        if type(m).__name__ == "VisionSdpaAttention":
            m.forward = types.MethodType(per_image_sdpa_forward, m)
            n += 1
    if n == 0:
        raise RuntimeError("dots_vision: không thấy lớp VisionSdpaAttention nào — code model đã đổi, cần sửa bản vá")
    orig = tower.forward

    @functools.wraps(orig)
    def forward(hidden_states, grid_thw, bf16=True):  # bỏ qua cờ ép bf16 của code gốc
        return orig(hidden_states.to(vision_dtype), grid_thw, bf16=False)

    tower.forward = forward


PATCHES = {"dots_vision": (dots_vision_before_load, dots_vision_after_load)}
