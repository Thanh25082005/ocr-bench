"""Chẩn đoán vì sao dots.mocr đọc ra chữ không liên quan tới ảnh (chạy trên Kaggle, ~5 phút, 1 GPU).

    python inference/diagnose_dots.py                 # in báo cáo, ghi /kaggle/working/diag/report.txt

Kiểm tra lần lượt (ảnh nhỏ: góc trên hoá đơn mẫu 01, để nhanh và đủ bộ nhớ cho attention gốc):
  1. phiên bản transformers / torch; processor chèn đúng số token ảnh chưa
  2. bộ mã hoá ảnh: bản vá (attention từng ảnh) so với attention GỐC có mặt nạ của họ — sai khác bao nhiêu
  3. generate theo đường gốc của họ: bước đầu có nhận pixel_values không, vision_tower có được gọi không
  4. generate với inputs_embeds tự dựng (ảnh chắc chắn được đưa vào) — so chữ đọc ra
  5. phần ngôn ngữ fp16 so với fp32 (cùng inputs_embeds)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path("/kaggle/working/diag") if Path("/kaggle/working").exists() else ROOT / "diag"
LINES: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def main(model_dir: str | None = None, max_new: int = 120):
    import torch
    import transformers
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from ocrbench.adapters.hf_vlm import _alias_dir
    from ocrbench.adapters.patches import dots_vision_after_load, dots_vision_before_load, per_image_sdpa_forward

    log("transformers", transformers.__version__, "| torch", torch.__version__,
        "| cuda", torch.cuda.is_available() and torch.cuda.get_device_name(0))
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    path = model_dir or _alias_dir("dots-studio/dots.mocr", "DotsMOCR")
    img = Image.open(ROOT / "inference/samples/01_hoa_don_tieng_anh.png").convert("RGB")
    img = img.crop((0, 0, img.width, img.height // 3))  # phần đầu hoá đơn: tên công ty, số hoá đơn
    img.thumbnail((896, 896))
    prompt = "Extract the text content from this image."

    proc = transformers.AutoProcessor.from_pretrained(path, trust_remote_code=True)
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
    n_img = int((inp["input_ids"][0] == 151665).sum())
    need = int(inp["image_grid_thw"].prod() // 4)
    log(f"[1] token ảnh trong input: {n_img} · cần: {need} · {'OK' if n_img == need else 'SAI'} · keys {list(inp.keys())}")
    inp.pop("mm_token_type_ids", None)  # transformers mới thêm, model của dots không nhận
    inp = inp.to(dev)

    cfg = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)
    dots_vision_before_load(cfg)
    t = time.perf_counter()
    if os.environ.get("TINY"):  # chỉ để thử script trên máy không GPU: model nhỏ, trọng số ngẫu nhiên
        cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 64, 128
        cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
        vc = cfg.vision_config
        vc.num_hidden_layers, vc.embed_dim, vc.hidden_size, vc.intermediate_size, vc.num_attention_heads = 2, 64, 64, 128, 4
        model = transformers.AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, attn_implementation="sdpa")
        model = model.to(device=dev, dtype=torch.float16 if dev != "cpu" else torch.float32).eval()
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, config=cfg, attn_implementation="sdpa", dtype=torch.float16, device_map=dev,
            trust_remote_code=True).eval()
    log(f"nạp model {time.perf_counter() - t:.0f}s")
    tower = model.vision_tower
    tower.to(torch.float32)
    attn_mods = [m for m in tower.modules() if type(m).__name__ == "VisionSdpaAttention"]
    pv, grid = inp["pixel_values"].to(torch.float32), inp["image_grid_thw"]

    # [2] attention gốc (mặt nạ, dùng chính forward gốc của lớp) so với bản vá
    with torch.no_grad():
        ref = tower(pv, grid, bf16=False)
        for m in attn_mods:
            m.forward = types.MethodType(per_image_sdpa_forward, m)
        new = tower(pv, grid, bf16=False)
        for m in attn_mods:  # trả lại forward gốc cho các bước sau (dots_vision_after_load sẽ vá lại)
            del m.forward
    d = (ref - new).abs().max().item()
    log(f"[2] bộ mã hoá ảnh: |gốc − vá| lớn nhất = {d:.3e} · |gốc| TB = {ref.abs().mean().item():.3e} · "
        f"{'OK' if d < 1e-3 else 'KHÁC'} · NaN/inf: {not torch.isfinite(ref).all().item()}")

    dots_vision_after_load(model, torch.float32)
    tok = proc.tokenizer

    def decode(out, start):
        return tok.decode(out[0, start:], skip_special_tokens=True).replace("\n", " ⏎ ")

    # [3] đường gốc của họ
    seen = []
    h1 = model.register_forward_pre_hook(
        lambda m, a, k: seen.append((k.get("pixel_values") is not None,
                                     None if k.get("cache_position") is None else int(k["cache_position"][0]))),
        with_kwargs=True)
    vt = []
    h2 = tower.register_forward_hook(lambda m, i, o: vt.append(tuple(o.shape)))
    try:
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        log(f"[3] đường gốc: bước đầu có pixel_values = {seen[0][0]} (cache_position[0] = {seen[0][1]}) · "
            f"vision_tower gọi {len(vt)} lần {vt[:1]}")
        log("    chữ:", decode(out, inp["input_ids"].shape[1])[:400])
    except Exception as e:
        log(f"[3] đường gốc LỖI: {type(e).__name__}: {e}")
        log("    " + traceback.format_exc().splitlines()[-3])
    h1.remove()
    h2.remove()

    # [4] tự dựng inputs_embeds (chắc chắn có ảnh) rồi generate bằng phần ngôn ngữ Qwen2 (bỏ qua code generate của dots)
    from transformers import Qwen2ForCausalLM

    ids = inp["input_ids"]
    model.prepare_inputs_for_generation = types.MethodType(Qwen2ForCausalLM.prepare_inputs_for_generation, model)
    model.forward = types.MethodType(Qwen2ForCausalLM.forward, model)
    with torch.no_grad():
        emb = model.prepare_inputs_embeds(ids, pv, grid, ids == model.config.image_token_id)
        out = model.generate(input_ids=ids, inputs_embeds=emb, attention_mask=inp["attention_mask"],
                             max_new_tokens=max_new, do_sample=False)
    start = ids.shape[1] if out.shape[1] > max_new else 0
    log("[4] inputs_embeds tự dựng (fp16):", decode(out, start)[:400])

    # [5] cùng inputs_embeds, phần ngôn ngữ fp32
    try:
        model.model.to(torch.float32)
        model.lm_head.to(torch.float32)
        with torch.no_grad():
            out = model.generate(input_ids=ids, inputs_embeds=emb.float(), attention_mask=inp["attention_mask"],
                                 max_new_tokens=max_new, do_sample=False)
        log("[5] inputs_embeds tự dựng (fp32):", decode(out, ids.shape[1] if out.shape[1] > max_new else 0)[:400])
    except Exception as e:
        log(f"[5] fp32 LỖI: {type(e).__name__}: {e}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    log(f"→ đã ghi {OUT / 'report.txt'}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or [None]))
