"""Model thị giác-ngôn ngữ (VLM) chạy bằng Hugging Face transformers.

Dùng được cho phần lớn model OCR dạng VLM: họ Qwen2/2.5/3-VL (Baseer, Qari, amad-vlm,
sherif1313...), HunyuanOCR, Gemma 3... Cấu hình ví dụ:

    adapter: hf_vlm
    params:
      model_id: amad-iq/amad-vlm6
      prompt: "Extract the text in the image. Give me the final text, nothing else."
      max_new_tokens: 4096
      quantization: 4bit          # null | 4bit | 8bit (bitsandbytes) để nhét model to vào GPU nhỏ
      adapter_id: null            # LoRA/QLoRA adapter (PEFT) đặt lên model gốc
      dtype: auto                 # auto | float16 | bfloat16 | float32
      max_image_side: 1600        # thu nhỏ ảnh quá lớn (tiết kiệm VRAM), null = giữ nguyên
      processor_kwargs: {max_pixels: 1605632}

Model trả kết quả dạng BỐ CỤC (dots.ocr / dots.mocr, xem ocrbench/layout.py):

      output_format: layout_json  # JSON [{bbox, category, text}] → văn bản Markdown + extra.blocks (bbox ảnh gốc)
      layout_drop: []             # loại khối bỏ khỏi văn bản, vd. [Page-header, Page-footer]
      local_alias: DotsMOCR       # nạp qua thư mục tên không có dấu chấm (code remote của dots cần vậy)
      patches: [dots_vision]      # vá code model để chạy trên T4, xem adapters/patches.py

Lưu ý với T4 (Kaggle): không có bf16 thật, nên dtype=auto sẽ chọn float16. Một số model
huấn luyện bằng bf16 có thể tràn số ở fp16 (ra chữ rỗng hoặc ký tự rác); khi đó thử
dtype: float32 (chậm, tốn VRAM gấp đôi) hoặc quantization: 4bit.
"""

from __future__ import annotations

from .base import Adapter, Prediction


class HFVLMAdapter(Adapter):
    defaults = {
        "model_id": None,
        "processor_id": None,  # mặc định = model_id; đặt khi processor nằm ở repo khác (vd. repo LoRA)
        "adapter_id": None,
        "model_class": "auto",  # auto, hoặc tên class trong transformers (vd. HunYuanVLForConditionalGeneration)
        "dtype": "auto",
        "quantization": None,
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "trust_remote_code": False,
        "prompt": "Extract all text in the image. Return only the text.",
        "system_prompt": None,
        # processor: apply_chat_template(tokenize=False) rồi processor(text, images) — kiểu Qwen2-VL
        # template:  apply_chat_template(tokenize=True) với ảnh trong message — kiểu model mới
        "input_mode": "processor",
        "max_new_tokens": 2048,
        "generation_kwargs": {},
        "processor_kwargs": {},
        "model_kwargs": {},
        "strip_think": True,
        "json_field": None,
        # tài liệu nhiều trang: "joint" = mọi trang trong MỘT lần gọi (model phải giữ ngữ cảnh qua các trang,
        # vd. nhớ tiêu đề cột ở trang 1 khi đọc tiếp bảng ở trang 2); "per_page" = đọc từng trang rồi ghép
        "multi_page": "joint",
        # --- tối ưu tốc độ (dùng cho triển khai; mặc định TẮT để benchmark giữ nguyên cách chạy) ---
        "batch_size": 1,          # số trang chạy chung một lượt sinh token (predict_batch); T4 + model ~3B: thử 4
        "stop_on_loop": False,    # dừng sinh khi phát hiện vòng lặp (đoạn ≤ loop_max_period token lặp liên tiếp)
        "loop_max_period": 60,
        "loop_min_span": 240,     # vòng lặp phải dài ít nhất chừng này token mới dừng (tránh cắt nhầm bảng nhiều ô trống)
        # --- kết quả dạng bố cục / model cần vá ---
        "output_format": "text",  # text | layout_json
        "layout_drop": [],
        "local_alias": None,
        "patches": [],
        "vision_dtype": "float32",  # dtype bộ mã hoá ảnh khi dùng bản vá dots_vision
    }

    def load(self):
        import torch
        import transformers

        p = self.params
        if not p["model_id"]:
            raise ValueError("hf_vlm cần tham số 'model_id'")
        self._torch = torch

        dtype = self._pick_dtype(torch, p["dtype"])
        self._dtype = dtype
        kwargs = dict(device_map=p["device_map"], trust_remote_code=p["trust_remote_code"], **p["model_kwargs"])
        if p["attn_implementation"]:
            kwargs["attn_implementation"] = p["attn_implementation"]
        # transformers >= 4.56 đổi torch_dtype -> dtype
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = dtype
        if p["quantization"] in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
                if p["quantization"] == "4bit"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
        elif p["quantization"]:
            raise ValueError(f"quantization không hợp lệ: {p['quantization']!r} (null | 4bit | 8bit)")

        from .patches import PATCHES

        unknown = [x for x in p["patches"] if x not in PATCHES]
        if unknown:
            raise ValueError(f"patches không có: {unknown}. Có: {sorted(PATCHES)}")
        model_path = _alias_dir(p["model_id"], p["local_alias"]) if p["local_alias"] else p["model_id"]
        if p["patches"]:
            config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=p["trust_remote_code"])
            for name in p["patches"]:
                PATCHES[name][0](config)
            kwargs["config"] = config
        self._processor = transformers.AutoProcessor.from_pretrained(
            p["processor_id"] or model_path, trust_remote_code=p["trust_remote_code"], **p["processor_kwargs"]
        )
        self._model = self._model_class(transformers, p["model_class"]).from_pretrained(model_path, **kwargs)
        for name in p["patches"]:
            PATCHES[name][1](self._model, getattr(torch, p["vision_dtype"]))
        if p["adapter_id"]:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, p["adapter_id"])
        self._model.eval()

    @staticmethod
    def _pick_dtype(torch, name):
        if name != "auto":
            return getattr(torch, name)
        if not torch.cuda.is_available():
            return torch.float32
        # is_bf16_supported() trả True cả khi chỉ giả lập (T4), nên kiểm tra compute capability
        return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    @staticmethod
    def _model_class(transformers, name):
        if name != "auto":
            return getattr(transformers, name)
        for cand in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            if hasattr(transformers, cand):
                return getattr(transformers, cand)
        return transformers.AutoModelForCausalLM

    def _messages(self, item, images, embed_images: bool):
        p = self.params
        msgs = []
        if p["system_prompt"]:
            msgs.append({"role": "system", "content": [{"type": "text", "text": p["system_prompt"]}]})
        parts = [{"type": "image", "image": im} if embed_images else {"type": "image"} for im in images]
        prompt = self.prompt_for(item)
        # prompt rỗng = chỉ gửi ảnh (vd. CHURRO: toàn bộ chỉ dẫn nằm ở system prompt)
        msgs.append({"role": "user", "content": [*parts, *([{"type": "text", "text": prompt}] if prompt else [])]})
        return msgs

    def predict(self, image, item):
        return self._generate([image], item)

    def predict_batch(self, images, items):
        """Nhiều trang độc lập, MỘT lượt generate (đệm trái). Chỉ hỗ trợ input_mode=processor."""
        p = self.params
        if len(images) == 1 or p["input_mode"] != "processor":
            return [self._generate([im], it) for im, it in zip(images, items)]
        torch = self._torch
        texts = [self._processor.apply_chat_template(self._messages(it, [None], embed_images=False),
                                                     add_generation_prompt=True, tokenize=False) for it in items]
        tok = getattr(self._processor, "tokenizer", self._processor)
        old_side, tok.padding_side = tok.padding_side, "left"
        try:
            inputs = self._processor(text=texts, images=list(images), return_tensors="pt", padding=True)
        finally:
            tok.padding_side = old_side
        return self._run(inputs, n_rows=len(images), sizes=[im.size for im in images])

    def predict_pages(self, images, item):
        if self.params["multi_page"] == "per_page":
            return super().predict_pages(images, item)
        pred = self._generate(images, item)
        pred.extra.update(pages=len(images), page_mode="joint")
        return pred

    def _generate(self, images, item):
        torch, p = self._torch, self.params
        if p["input_mode"] == "template":
            inputs = self._processor.apply_chat_template(
                self._messages(item, images, embed_images=True),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            text = self._processor.apply_chat_template(
                self._messages(item, images, embed_images=False), add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(text=[text], images=list(images), return_tensors="pt")
        return self._run(inputs, n_rows=1, sizes=[images[0].size] if len(images) == 1 else None)[0]

    def _run(self, inputs, n_rows: int, sizes=None):
        torch, p = self._torch, self.params
        inputs = inputs.to(self._model.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self._dtype)
        start = inputs["input_ids"].shape[1]
        tok = getattr(self._processor, "tokenizer", self._processor)
        gen = {"max_new_tokens": p["max_new_tokens"], "do_sample": False, **p["generation_kwargs"]}
        if n_rows > 1 and tok.pad_token_id is not None:
            gen.setdefault("pad_token_id", tok.pad_token_id)
        loop = None
        if p["stop_on_loop"]:
            from transformers import StoppingCriteria, StoppingCriteriaList

            loop = _LoopStop(torch, start, n_rows, p["loop_max_period"], p["loop_min_span"])

            class _Wrap(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):
                    return loop(input_ids, scores)

            gen["stopping_criteria"] = StoppingCriteriaList([_Wrap()])
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen)
        new_tokens = out[:, start:]
        texts = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        stop_ids = {i for i in (tok.pad_token_id, tok.eos_token_id) if i is not None}
        preds = []
        for r in range(n_rows):
            row = new_tokens[r].tolist()
            n_new = next((i for i, t in enumerate(row) if t in stop_ids), len(row))
            extra = {"new_tokens": n_new, "hit_max_tokens": n_new >= p["max_new_tokens"]}
            if loop is not None and loop.stopped[r]:
                extra["stopped_loop"] = True  # trang đã bị lặp: kết quả cần người duyệt
            text = self.postprocess(texts[r])
            if p["output_format"] == "layout_json":
                text = self._layout(text, sizes[r] if sizes else None, extra)
            preds.append(Prediction(text, extra=extra))
        return preds

    def _layout(self, raw: str, size, extra: dict) -> str:
        from ..layout import blocks_to_markdown, parse_layout

        ip = getattr(self._processor, "image_processor", None)
        size_cfg = getattr(ip, "size", None)
        size_cfg = dict(size_cfg) if isinstance(size_cfg, dict) else {}
        min_px = getattr(ip, "min_pixels", None) or size_cfg.get("shortest_edge") or 3136
        max_px = getattr(ip, "max_pixels", None) or size_cfg.get("longest_edge") or 11289600
        blocks, status = parse_layout(raw, size, min_pixels=min_px, max_pixels=max_px)
        extra["layout"] = status
        extra["blocks"] = [b.to_dict(with_text=False) for b in blocks]
        if status == "failed":
            return raw  # không đọc được khối nào: giữ nguyên để người duyệt / chấm điểm vẫn thấy chữ
        return blocks_to_markdown(blocks, drop=tuple(self.params["layout_drop"]))

    def close(self):
        del self._model
        self._torch.cuda.empty_cache()


class _LoopStop:
    """Dừng sinh token cho từng dòng của batch khi đuôi kết quả là một đoạn ngắn (≤ max_period token) lặp liên tiếp,
    tổng dài ≥ min_span token. Trang như vậy đã hỏng; dừng sớm tiết kiệm hàng nghìn token vô ích."""

    def __init__(self, torch, start: int, n_rows: int, max_period: int, min_span: int, every: int = 16,
                 long_max_period: int = 0, long_min_reps: int = 8):
        self.torch, self.start, self.max_period, self.min_span, self.every = torch, start, max_period, min_span, every
        # vòng lặp dài (vd. cả một dòng chữ Ả Rập có tashkeel ~100 token lặp mãi): chu kỳ max_period+1..long_max_period,
        # phải lặp liên tiếp ≥ long_min_reps lần (đòi nhiều lần hơn để không cắt nhầm bảng có nhiều dòng trống giống nhau)
        self.long_max_period, self.long_min_reps = long_max_period, long_min_reps
        self.stopped = [False] * n_rows
        self.calls = 0

    def __call__(self, input_ids, scores, **kwargs):
        torch = self.torch
        self.calls += 1
        done = torch.tensor(self.stopped, device=input_ids.device)
        gen = input_ids[:, self.start:]
        if self.calls % self.every or gen.shape[1] < self.min_span:
            return done
        tail = gen[:, -max(self.min_span, 2 * self.max_period, self.long_max_period * self.long_min_reps):].tolist()
        for r, t in enumerate(tail):
            if self.stopped[r]:
                continue
            for period in range(1, max(self.max_period, self.long_max_period) + 1):
                reps = -(-self.min_span // period)  # làm tròn lên
                if period > self.max_period:
                    reps = max(reps, self.long_min_reps)
                span = period * reps
                if span > len(t):
                    if period > self.max_period:
                        break
                    continue
                seg = t[-span:]
                if seg == seg[:period] * reps:
                    self.stopped[r] = True
                    break
        return torch.tensor(self.stopped, device=input_ids.device)


def _alias_dir(model_id: str, alias: str) -> str:
    """Tải model về cache HF rồi trỏ một thư mục tên `alias` (không dấu chấm) tới bản đó.
    Code remote của dots.ocr/dots.mocr lấy tên thư mục làm tên module Python, nên 'dots.mocr' bị lỗi import.
    Dùng symlink nên `ocrbench clean-cache` xoá cache HF là xoá luôn trọng số."""
    import os
    import tempfile
    from pathlib import Path

    from huggingface_hub import snapshot_download

    if os.path.isdir(model_id):
        src = Path(model_id)
    else:
        src = Path(snapshot_download(model_id))
    link = Path(tempfile.gettempdir()) / "ocrbench_models" / alias
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == src.resolve():
            return str(link)
        link.unlink()
    link.symlink_to(src, target_is_directory=True)
    return str(link)
