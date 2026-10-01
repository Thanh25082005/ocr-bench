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

        self._processor = transformers.AutoProcessor.from_pretrained(
            p["processor_id"] or p["model_id"], trust_remote_code=p["trust_remote_code"], **p["processor_kwargs"]
        )
        self._model = self._model_class(transformers, p["model_class"]).from_pretrained(p["model_id"], **kwargs)
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
        msgs.append({"role": "user", "content": [*parts, {"type": "text", "text": self.prompt_for(item)}]})
        return msgs

    def predict(self, image, item):
        return self._generate([image], item)

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
        inputs = inputs.to(self._model.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self._dtype)

        gen = {"max_new_tokens": p["max_new_tokens"], "do_sample": False, **p["generation_kwargs"]}
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen)
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        n_new = int(new_tokens.shape[1])
        return Prediction(
            self.postprocess(text),
            extra={"new_tokens": n_new, "hit_max_tokens": n_new >= p["max_new_tokens"]},
        )

    def close(self):
        del self._model
        self._torch.cuda.empty_cache()
