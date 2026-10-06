"""dots.ocr / dots.mocr chạy GIỐNG SOURCE GỐC (https://github.com/studio-dots-ai/dots.ocr, commit 36d7248).

Đi đúng luồng của họ, gọi đúng hàm của họ (bản chép nguyên văn trong ocrbench/_vendor/dots_ocr):

    ảnh ─► [fitz_preprocess: ảnh → PDF → render lại theo dpi] ─► fetch_image(min/max_pixels)
        ─► prompt theo chế độ (dict_promptmode_to_prompt; grounding: pre_process_bboxes)
        ─► _inference_with_hf: chat template + qwen_vl_utils.process_vision_info + processor + generate
        ─► post_process_output (JSON → bbox ảnh gốc; JSON hỏng → OutputCleaner)
        ─► layoutjson2md (Markdown; no_page_hf = bỏ đầu/chân trang như bản dùng để chấm benchmark của họ)

Khác source (chỉ để chạy được trên Kaggle T4, không đổi phép tính của model):
- sdpa thay flash_attention_2, float16 thay bfloat16 cho phần ngôn ngữ, bộ mã hoá ảnh float32
  (bản vá dots_vision trong adapters/patches.py, đã so khớp với bản gốc);
- nạp qua thư mục tên không dấu chấm (README của họ cũng yêu cầu vậy);
- stop_on_loop (tùy chọn, mặc định bật): dừng sinh khi model lặp vòng — JSON dở dang vẫn qua OutputCleaner của họ;
- strip_images: bỏ ảnh base64 mà layoutjson2md nhúng cho khối Picture khỏi văn bản để chấm điểm.
- manual_embeds (mặc định bật): tự ghép embedding ảnh vào chuỗi đầu vào (đúng hàm prepare_inputs_embeds của họ)
  rồi sinh bằng phần ngôn ngữ Qwen2. Code generate của họ chỉ đưa ảnh vào khi `cache_position[0] == 0`
  — transformers mới không truyền cache_position nữa → model bị lỗi hoặc KHÔNG nhìn thấy ảnh (đọc ra chữ bịa).
  Kết quả y hệt đường gốc khi đường gốc chạy đúng.

Chế độ (prompt_mode): prompt_layout_all_en (mặc định) · prompt_layout_only_en · prompt_ocr · prompt_grounding_ocr (cần bbox)
· prompt_web_parsing · prompt_scene_spotting · prompt_image_to_svg · prompt_general.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .base import Adapter, Prediction

_VENDOR = str(Path(__file__).resolve().parent.parent / "_vendor")
LAYOUT_MODES = ("prompt_layout_all_en", "prompt_layout_only_en", "prompt_grounding_ocr")
_DATA_IMG = re.compile(r"!\[[^\]]*\]\(data:image/[^)]*\)")


def dots_utils():
    """Module của dots_ocr (bản chép nguyên văn)."""
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    from dots_ocr.utils import consts, format_transformer, image_utils, layout_utils, prompts

    return consts, prompts, image_utils, layout_utils, format_transformer


class DotsAdapter(Adapter):
    defaults = {
        "model_id": "dots-studio/dots.mocr",
        "local_alias": "DotsMOCR",
        "prompt_mode": "prompt_layout_all_en",
        "bbox": None,
        "fitz_preprocess": False,   # True = như CLI gốc mặc định với ảnh (render lại 200 DPI; ảnh thường to gấp ~2 lần)
        "dpi": 200,
        "min_pixels": None,
        "max_pixels": None,
        "max_new_tokens": 24000,    # như _inference_with_hf của họ
        "no_page_hf": False,        # True = văn bản bỏ Page-header/Page-footer (file _nohf.md của họ)
        "strip_images": True,
        "stop_on_loop": True,
        "loop_max_period": 60,
        "loop_min_span": 600,       # bảng HTML nhiều ô trống lặp hợp lệ, đừng cắt nhầm
        "dtype": "auto",
        "vision_dtype": "float32",
        "attn_implementation": "sdpa",
        "device_map": "auto",
        "manual_embeds": True,
        # extra.blocks có cả chữ từng khối (để dựng DOCX giữ nguyên bố cục); benchmark để False cho gọn predictions
        "blocks_with_text": False,
        # chặn chạy vòng tới max_new_tokens với ảnh nhỏ: tối đa k token sinh ra cho mỗi token ảnh (+512).
        # Mỗi token ảnh = 28×28 px chứa vài ký tự → k=4 không cắt trang thật. None = tắt (như source gốc)
        "adaptive_max_tokens": None,
    }

    def load(self):
        import torch
        import transformers

        from .hf_vlm import HFVLMAdapter, _alias_dir
        from .patches import PATCHES

        p = self.params
        consts, prompts, *_ = dots_utils()
        if p["prompt_mode"] not in prompts.dict_promptmode_to_prompt:
            raise ValueError(f"prompt_mode không có: {p['prompt_mode']}. Có: {list(prompts.dict_promptmode_to_prompt)}")
        self._torch = torch
        path = _alias_dir(p["model_id"], p["local_alias"]) if p["local_alias"] else p["model_id"]
        config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)
        before, after = PATCHES["dots_vision"]
        before(config)
        dtype = HFVLMAdapter._pick_dtype(torch, p["dtype"])
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kwargs = {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": dtype}
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            path, config=config, attn_implementation=p["attn_implementation"], device_map=p["device_map"],
            trust_remote_code=True, **kwargs)
        after(self._model, getattr(torch, p["vision_dtype"]))
        self._model.eval()
        if p["manual_embeds"]:
            import types

            from transformers import Qwen2ForCausalLM

            # sinh chữ bằng đường của Qwen2 (inputs_embeds đã có ảnh) — không qua prepare_inputs_for_generation của dots
            self._model.prepare_inputs_for_generation = types.MethodType(
                Qwen2ForCausalLM.prepare_inputs_for_generation, self._model)
            self._model.forward = types.MethodType(Qwen2ForCausalLM.forward, self._model)
        self._processor = transformers.AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=True)
        from qwen_vl_utils import process_vision_info

        self._process_vision_info = process_vision_info

    # --- giống DotsOCRParser._inference_with_hf ---
    def _inference_with_hf(self, image, prompt) -> tuple[str, dict]:
        torch, p = self._torch, self.params
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                 return_tensors="pt")
        inputs.pop("mm_token_type_ids", None)  # transformers mới thêm khoá này, model của dots không nhận
        inputs = inputs.to(self._model.device)
        start = inputs["input_ids"].shape[1]
        if p["manual_embeds"]:
            ids = inputs["input_ids"]
            with torch.inference_mode():
                emb = self._model.prepare_inputs_embeds(ids, inputs["pixel_values"], inputs["image_grid_thw"],
                                                        ids == self._model.config.image_token_id)
            gen_inputs = {"input_ids": ids, "inputs_embeds": emb, "attention_mask": inputs["attention_mask"]}
        else:
            gen_inputs = dict(inputs)
        max_new = p["max_new_tokens"]
        if p["adaptive_max_tokens"]:
            n_img = int((inputs["input_ids"] == self._model.config.image_token_id).sum())
            max_new = min(max_new, int(p["adaptive_max_tokens"]) * n_img + 512)
        gen = {"max_new_tokens": max_new}
        loop = None
        if p["stop_on_loop"]:
            from transformers import StoppingCriteria, StoppingCriteriaList

            from .hf_vlm import _LoopStop

            loop = _LoopStop(torch, start, 1, p["loop_max_period"], p["loop_min_span"])

            class _Wrap(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):
                    return loop(input_ids, scores)

            gen["stopping_criteria"] = StoppingCriteriaList([_Wrap()])
        with torch.inference_mode():
            generated_ids = self._model.generate(**gen_inputs, **gen)
        ids = inputs["input_ids"]
        if generated_ids.shape[1] >= start and torch.equal(generated_ids[:, :start].to(ids.device), ids):
            trimmed = [generated_ids[0, start:]]
        else:  # bản transformers chỉ trả phần mới sinh
            trimmed = [generated_ids[0]]
        response = self._processor.batch_decode(trimmed, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0]
        n_new = int(trimmed[0].shape[0])
        extra = {"new_tokens": n_new, "hit_max_tokens": n_new >= max_new, "max_new_tokens": max_new}
        if loop is not None and loop.stopped[0]:
            extra["stopped_loop"] = True
        return response, extra

    # --- giống DotsOCRParser._parse_single_image (phần không ghi file) ---
    def parse_image(self, origin_image, prompt_mode: str | None = None, bbox=None) -> dict:
        """Trả về dict: response (chữ thô), cells (list khối hoặc None), filtered, md, md_nohf, extra."""
        consts, prompts, image_utils, layout_utils, fmt = dots_utils()
        p = self.params
        prompt_mode = prompt_mode or p["prompt_mode"]
        bbox = bbox if bbox is not None else p["bbox"]
        min_pixels, max_pixels = p["min_pixels"], p["max_pixels"]
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or consts.MIN_PIXELS
            max_pixels = max_pixels or consts.MAX_PIXELS
        if p["fitz_preprocess"]:
            image = image_utils.get_image_by_fitz_doc(origin_image, target_dpi=p["dpi"])
            image = image_utils.fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = image_utils.fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        prompt = prompts.dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == "prompt_grounding_ocr":
            if bbox is None:
                raise ValueError("prompt_grounding_ocr cần bbox [x1, y1, x2, y2]")
            prompt = prompt + str(layout_utils.pre_process_bboxes(
                origin_image, [list(bbox)], input_width=image.width, input_height=image.height,
                min_pixels=min_pixels, max_pixels=max_pixels)[0])
        response, extra = self._inference_with_hf(image, prompt)
        return postprocess_response(response, prompt_mode, origin_image, image, min_pixels, max_pixels, extra)

    def predict(self, image, item):
        r = self.parse_image(image.convert("RGB"))
        text = r["md_nohf"] if self.params["no_page_hf"] and r["md_nohf"] is not None else r["md"]
        if self.params["strip_images"]:
            text = _DATA_IMG.sub("", text).strip()
        extra = r["extra"]
        if self.params["blocks_with_text"] and r.get("cells"):
            extra["blocks"] = [{"category": c.get("category"), "bbox": c.get("bbox"), "text": c.get("text", "")}
                               for c in r["cells"]]
        return Prediction(text, extra=extra)

    def close(self):
        del self._model
        self._torch.cuda.empty_cache()


def postprocess_response(response, prompt_mode, origin_image, image, min_pixels, max_pixels, extra=None) -> dict:
    """Phần hậu xử lý của _parse_single_image, tách riêng để kiểm thử được không cần GPU."""
    _, _, _, layout_utils, fmt = dots_utils()
    extra = dict(extra or {})
    out = {"response": response, "cells": None, "filtered": False, "md": response, "md_nohf": None}
    if prompt_mode in LAYOUT_MODES:
        cells, filtered = layout_utils.post_process_output(response, prompt_mode, origin_image, image,
                                                           min_pixels=min_pixels, max_pixels=max_pixels)
        if filtered and prompt_mode != "prompt_layout_only_en":  # JSON hỏng: OutputCleaner trả về chuỗi
            out.update(filtered=True, md=cells, md_nohf=cells)
            extra["layout"] = "filtered"
        else:
            out["cells"] = cells
            extra["layout"] = "ok"
            extra["blocks"] = [{"category": c.get("category"), "bbox": c.get("bbox")} for c in cells]
            if prompt_mode != "prompt_layout_only_en":
                out["md"] = fmt.layoutjson2md(origin_image, cells, text_key="text")
                out["md_nohf"] = fmt.layoutjson2md(origin_image, cells, text_key="text", no_page_hf=True)
            else:
                out["md"] = ""
    out["extra"] = extra
    return out


def parse_file(adapter: DotsAdapter, input_path, output_dir, prompt_mode: str | None = None, bbox=None,
               dpi: int = 200) -> list[dict]:
    """Giống DotsOCRParser.parse_file: ghi <output>/<tên>/ các file .json, .jpg, .md, _nohf.md mỗi trang
    và <output>/<tên>.jsonl — để so trực tiếp với kết quả của tool gốc."""
    _, _, image_utils, layout_utils, _ = dots_utils()
    from dots_ocr.utils.doc_utils import load_images_from_pdf

    prompt_mode = prompt_mode or adapter.params["prompt_mode"]
    input_path = Path(input_path)
    filename, ext = input_path.stem, input_path.suffix.lower()
    save_dir = Path(output_dir).resolve() / filename
    save_dir.mkdir(parents=True, exist_ok=True)
    if ext == ".pdf":
        pages, source = load_images_from_pdf(str(input_path), dpi=dpi), "pdf"
    elif ext in {".jpg", ".jpeg", ".png"}:
        pages, source = [image_utils.fetch_image(str(input_path))], "image"
    else:
        raise ValueError(f"không hỗ trợ {ext} (chỉ .pdf, .jpg, .jpeg, .png — như tool gốc)")
    results = []
    for i, origin in enumerate(pages):
        r = adapter.parse_image(origin, prompt_mode, bbox)
        name = f"{filename}_page_{i}" if source == "pdf" else filename
        res = {"page_no": i, "file_path": str(input_path), **{k: v for k, v in r["extra"].items() if k != "blocks"}}
        if prompt_mode in LAYOUT_MODES:
            (save_dir / f"{name}.json").write_text(
                json.dumps(r["response"] if r["filtered"] else r["cells"], ensure_ascii=False), encoding="utf-8")
            res["layout_info_path"] = str(save_dir / f"{name}.json")
            img = origin
            if not r["filtered"]:
                try:
                    img = layout_utils.draw_layout_on_image(origin, r["cells"])
                except Exception as e:  # như tool gốc: vẽ lỗi thì lưu ảnh gốc
                    print(f"Error drawing layout on image: {e}")
            img.save(save_dir / f"{name}.jpg")
            res["layout_image_path"] = str(save_dir / f"{name}.jpg")
        else:
            origin.save(save_dir / f"{name}.jpg")
            res["layout_image_path"] = str(save_dir / f"{name}.jpg")
        if prompt_mode != "prompt_layout_only_en":
            (save_dir / f"{name}.md").write_text(r["md"], encoding="utf-8")
            res["md_content_path"] = str(save_dir / f"{name}.md")
            if r["md_nohf"] is not None and not r["filtered"]:
                (save_dir / f"{name}_nohf.md").write_text(r["md_nohf"], encoding="utf-8")
                res["md_content_nohf_path"] = str(save_dir / f"{name}_nohf.md")
        if r["filtered"]:
            res["filtered"] = True
        results.append(res)
    with open(Path(output_dir).resolve() / f"{filename}.jsonl", "w", encoding="utf-8") as w:
        for res in results:
            w.write(json.dumps(res, ensure_ascii=False) + "\n")
    return results
