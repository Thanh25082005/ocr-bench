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
- fix_bboxes: bbox ngược / ra ngoài ảnh được sắp lại + kẹp vào trang trước layoutjson2md (bản gốc crop lỗi → mất trang).
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
import time
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
        # tham số sinh thêm, vd. {repetition_penalty: 1.05}; {} = greedy như source gốc (họ không đặt gì thêm)
        "generation_kwargs": {},
        "no_page_hf": False,        # True = văn bản bỏ Page-header/Page-footer (file _nohf.md của họ)
        "strip_images": True,
        "stop_on_loop": True,
        "loop_max_period": 60,
        "loop_min_span": 600,       # bảng HTML nhiều ô trống lặp hợp lệ, đừng cắt nhầm
        "loop_long_max_period": 0,  # >0: bắt cả vòng lặp chu kỳ dài tới chừng này token (phải lặp ≥ 8 lần); 0 = tắt
        "max_time": None,           # giây: giới hạn thời gian sinh mỗi trang (web demo); None = không giới hạn
        # trang bị dừng giữa chừng (lặp / hết thời gian / hết token): bỏ phần lặp, đóng JSON dở dang để GIỮ khối cuối
        # (thường là đoạn văn dài đọc đúng tới chỗ bắt đầu lặp) thay vì vứt cả khối. False = như source gốc
        "repair_truncated": False,
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
        # PHÓNG TO VÙNG "KHUNG ĐỀU + CHỮ CHÉP": ở form điền tay, dots có lúc bỏ bám dòng, sinh một dãy khung cùng cột,
        # cao bằng nhau, cách đều (ảnh thật: 22 khung cao 22 px) rồi chép chữ / con số của dòng này sang dòng khác
        # ("1983 02081" ở 3 ô, "מר (ה) 3/7/1983" ở 3 ô). Không phải vòng lặp token nên stop_on_loop không bắt.
        # Bật: dãy ≥ zoom_min_run khung như vậy có chữ chép (copied_cells) → cắt vùng đó, dựng lại bố cục ở
        # zoom_min_pixels, thay vào trang nếu bớt chép. Đo: cùng trang, form đọc đúng tên / số căn cước / ngày, hết
        # chép, +~55 giây. Vùng nhỏ (1–2 chữ) KHÔNG phóng: thử thì model trôi sang chữ khác (Urdu, chữ Hán).
        # Đã thử và loại: repetition_penalty 1.05 (vẫn khung đều + chép), giảm còn 3,5 MP (lặp thật, mất nửa form).
        "zoom_repeats": False,
        "zoom_min_run": 6,
        "zoom_min_pixels": 2000000,
        # khối có chữ (hoặc một chuỗi số dài) lặp ≥ chừng này lần mà không sửa được → ghi chú "cần soát"; 0 = tắt
        "flag_copies": 3,
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
        gen = {**p["generation_kwargs"], "max_new_tokens": max_new}
        if p["max_time"]:
            gen["max_time"] = float(p["max_time"])
        loop = None
        if p["stop_on_loop"]:
            from transformers import StoppingCriteria, StoppingCriteriaList

            from .hf_vlm import _LoopStop

            loop = _LoopStop(torch, start, 1, p["loop_max_period"], p["loop_min_span"],
                             long_max_period=p["loop_long_max_period"])
            tok = getattr(self._processor, "tokenizer", self._processor)
            layout_loop = _LayoutLoopStop(tok, start)

            class _Wrap(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):
                    return loop(input_ids, scores) | layout_loop(input_ids)

            gen["stopping_criteria"] = StoppingCriteriaList([_Wrap()])
        t_gen = time.perf_counter()
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
        if p["max_time"] and n_new < max_new and time.perf_counter() - t_gen >= float(p["max_time"]) \
                and not (loop is not None and loop.stopped[0]):
            extra["hit_max_time"] = True
        if loop is not None and loop.stopped[0]:
            extra["stopped_loop"] = True
        return response, extra

    # --- giống DotsOCRParser._parse_single_image (phần không ghi file) ---
    def parse_image(self, origin_image, prompt_mode: str | None = None, bbox=None, min_pixels=None, max_pixels=None,
                    zoom: bool = True) -> dict:
        """Trả về dict: response (chữ thô), cells (list khối hoặc None), filtered, md, md_nohf, extra."""
        consts, prompts, image_utils, layout_utils, fmt = dots_utils()
        p = self.params
        prompt_mode = prompt_mode or p["prompt_mode"]
        bbox = bbox if bbox is not None else p["bbox"]
        min_pixels = min_pixels if min_pixels is not None else p["min_pixels"]
        max_pixels = max_pixels if max_pixels is not None else p["max_pixels"]
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
        if p["repair_truncated"] and prompt_mode in LAYOUT_MODES and prompt_mode != "prompt_layout_only_en" and (
                extra.get("stopped_loop") or extra.get("hit_max_time") or extra.get("hit_max_tokens")):
            fixed = close_truncated_layout(response)
            if fixed != response:
                response, extra["truncated_repaired"] = fixed, True
        out = postprocess_response(response, prompt_mode, origin_image, image, min_pixels, max_pixels, extra)
        if prompt_mode == "prompt_layout_all_en" and out["cells"]:
            if zoom and p["zoom_repeats"]:
                self._zoom(origin_image, out)
            if p["flag_copies"]:
                n = len(copied_cells(out["cells"], p["flag_copies"]))
                if n:
                    out["extra"]["copied_cells"] = n
        return out

    def _zoom(self, origin_image, out: dict) -> None:
        """Vùng "khung đều + chữ chép" (grid_runs) → dựng lại bố cục riêng vùng đó ở độ phân giải cao, thay vào trang
        nếu bớt chép. Dựng lại Markdown khi có thay đổi."""
        _, _, _, _, fmt = dots_utils()
        p, cells = self.params, out["cells"]
        runs = [r for r in grid_runs(cells, p["zoom_min_run"])
                if set(r) & set(copied_cells(cells, p["flag_copies"] or 3))]
        if not runs:
            return
        t = time.perf_counter()
        replaced = 0
        for run in reversed(runs):  # thay từ cuối lên để chỉ số các dãy phía trước không lệch
            box = region_of([cells[i]["bbox"] for i in run], *origin_image.size)
            sub = self.parse_image(origin_image.crop(box), min_pixels=p["zoom_min_pixels"], max_pixels=None, zoom=False)
            ex = sub["extra"]
            if not sub["cells"] or ex.get("stopped_loop") or ex.get("hit_max_tokens") or ex.get("hit_max_time"):
                continue
            new = [dict(c, bbox=[c["bbox"][0] + box[0], c["bbox"][1] + box[1], c["bbox"][2] + box[0],
                                 c["bbox"][3] + box[1]]) for c in sub["cells"]]
            merged = replace_run(cells, run, new)
            if len(copied_cells(merged, p["flag_copies"] or 3)) < len(copied_cells(cells, p["flag_copies"] or 3)):
                cells[:] = merged
                replaced += 1
        out["extra"].update(zoom_regions=len(runs), zoom_replaced=replaced,
                            zoom_seconds=round(time.perf_counter() - t, 1))
        if replaced:
            out["extra"]["blocks"] = [{"category": c.get("category"), "bbox": c.get("bbox")} for c in cells]
            out["md"] = fmt.layoutjson2md(origin_image, cells, text_key="text")
            out["md_nohf"] = fmt.layoutjson2md(origin_image, cells, text_key="text", no_page_hf=True)

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


_CELL = re.compile(r'"category":\s*"([^"]+)"\s*,\s*"text":\s*"((?:[^"\\\\]|\\\\.)*)"')


def layout_loop(text: str, min_repeats: int = 25, min_chars: int = 6) -> bool:
    """Đuôi JSON bố cục là MỘT cặp (category, text) lặp liên tiếp ≥ min_repeats lần (toạ độ có thể khác nhau).
    dots hay rơi vào vòng này (vd. 500 lần cùng một ô bảng tới hết 24000 token, ~40 phút/trang trên T4). Bộ làm
    sạch của họ (OutputCleaner) vốn XOÁ các bản trùng → dừng sớm cho ra cùng kết quả, chỉ tiết kiệm thời gian."""
    cells = _CELL.findall(text)
    if len(cells) < min_repeats:
        return False
    last = cells[-1]
    if len(last[1]) < min_chars:
        return False
    n = 0
    for c in reversed(cells):
        if c != last:
            break
        n += 1
    return n >= min_repeats


def _strip_repeated_tail(text: str, min_repeats: int = 3, min_span: int = 30, max_period: int = 4000) -> str:
    """Đuôi là một đoạn lặp liên tiếp ≥ min_repeats lần (và dài ≥ min_span ký tự, để không đụng tới dấu chấm dẫn
    "....." hợp lệ) → chỉ giữ một bản (chu kỳ ngắn nhất tìm được)."""
    for period in range(1, min(max_period, len(text) // min_repeats) + 1):
        unit = text[-period:]
        if text.endswith(unit * max(min_repeats, -(-min_span // period))):
            while text.endswith(unit * 2):
                text = text[:-period]
            return text
    return text


def close_truncated_layout(response: str) -> str:
    """JSON bố cục bị dừng giữa chừng → bỏ phần lặp ở đuôi, đóng chuỗi "text" / khối / mảng đang dở. Không sửa
    được thành JSON hợp lệ → trả nguyên chuỗi cũ (bộ làm sạch phía sau vẫn nhặt các khối hoàn chỉnh như trước)."""
    try:
        json.loads(response)
        return response
    except json.JSONDecodeError:
        pass
    s = _strip_repeated_tail(response.rstrip())
    if (len(s) - len(s.rstrip("\\"))) % 2:  # dấu \ thoát dở ở cuối
        s = s[:-1]
    s = re.sub(r"\\u[0-9a-fA-F]{0,3}$", "", s)
    candidates = [s + tail for tail in ('"}]', '}]', ']')]
    if not s.lstrip().startswith("["):
        candidates += ["[" + c for c in candidates]
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and data and all(isinstance(d, dict) for d in data):
            return c
    return response


class _LayoutLoopStop:
    """StoppingCriteria cho dots: mỗi `every` bước giải mã phần đuôi đã sinh, dừng khi `layout_loop`."""

    def __init__(self, tokenizer, start: int, every: int = 64, tail_tokens: int = 6000):
        self.tok, self.start, self.every, self.tail = tokenizer, start, every, tail_tokens
        self.calls = 0
        self.stopped = False

    def __call__(self, input_ids):
        import torch

        self.calls += 1
        if not self.stopped and self.calls % self.every == 0 and input_ids.shape[1] - self.start > 200:
            gen = input_ids[0, max(self.start, input_ids.shape[1] - self.tail):]
            self.stopped = layout_loop(self.tok.decode(gen, skip_special_tokens=True))
        return torch.tensor([self.stopped], device=input_ids.device)


_TEXT_CATS = {"Text", "Title", "Section-header", "List-item", "Caption", "Footnote", "Page-header", "Page-footer"}
_LONG_TOKEN = re.compile(r"[^\s]*\d[^\s]*")


def copied_cells(cells: list, min_count: int = 3) -> list[int]:
    """Chỉ số các khối chữ nghi bị chép: chữ (bỏ khoảng trắng thừa) giống hệt ở ≥ min_count khối, hoặc chung một
    "từ" có chữ số dài ≥ 5 ký tự (vd. "02081", "3/7/1983") với ≥ min_count - 1 khối khác. Trùng 2 lần không tính
    (nhãn lặp hợp lệ: hai chỗ "توقيع", hai nhân chứng "شاهد"). Bảng / công thức / hình không xét."""
    from collections import Counter, defaultdict

    norm = {i: " ".join(str(c.get("text") or "").split()) for i, c in enumerate(cells)
            if c.get("category") in _TEXT_CATS and str(c.get("text") or "").strip()}
    counts = Counter(norm.values())
    out = {i for i, t in norm.items() if counts[t] >= min_count}
    owners = defaultdict(set)
    for i, t in norm.items():
        for tok in _LONG_TOKEN.findall(t):
            if len(tok) >= 5:
                owners[tok].add(i)
    for ids in owners.values():
        if len(ids) >= min_count:
            out |= ids
    return sorted(out)


def grid_runs(cells: list, min_run: int = 6, tol: int = 4) -> list[list[int]]:
    """Dãy ≥ min_run khối chữ LIỀN NHAU trong thứ tự đọc, cùng cột (x1, x2 lệch ≤ tol), cao bằng nhau và nối tiếp
    đều đặn (bước = chiều cao, lệch ≤ tol) — dấu hiệu dots sinh khung theo nhịp thay vì bám dòng thật."""
    def ok(a, b):
        (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a["bbox"], b["bbox"]
        h = ay2 - ay1
        return (a.get("category") in _TEXT_CATS and b.get("category") in _TEXT_CATS
                and abs(ax1 - bx1) <= tol and abs(ax2 - bx2) <= tol and abs((by2 - by1) - h) <= tol
                and abs((by1 - ay1) - h) <= tol)

    runs, cur = [], [0] if cells else []
    for i in range(1, len(cells)):
        if ok(cells[i - 1], cells[i]):
            cur.append(i)
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur = [i]
    if len(cur) >= min_run:
        runs.append(cur)
    return runs


def region_of(bboxes: list, width: int, height: int, pad: float = 0.5) -> list[int]:
    """Khung bao các khối, đệm pad × chiều cao khối trung vị mỗi phía, kẹp vào trang."""
    hs = sorted(b[3] - b[1] for b in bboxes)
    d = int(pad * hs[len(hs) // 2])
    return [max(0, min(b[0] for b in bboxes) - d), max(0, min(b[1] for b in bboxes) - d),
            min(width, max(b[2] for b in bboxes) + d), min(height, max(b[3] for b in bboxes) + d)]


def replace_run(cells: list, run: list[int], new: list) -> list:
    """Thay các khối trong dãy run bằng new (đặt đúng chỗ dãy cũ trong thứ tự đọc). Khối mới trùng ≥ 50% diện tích
    với một khối giữ lại (vd. chữ ký ngay mép vùng cắt đã có ở lượt đầu) bị bỏ để không thành hai khối."""
    def overlap(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / max(1, (a[2] - a[0]) * (a[3] - a[1]))

    drop = set(run)
    kept = [c for i, c in enumerate(cells) if i not in drop]
    new = [c for c in new if not any(overlap(c["bbox"], k["bbox"]) >= 0.5 for k in kept)]
    return cells[:run[0]] + new + [c for i, c in enumerate(cells) if i > run[0] and i not in drop]


def fix_bboxes(cells: list, width: int, height: int) -> int:
    """Bbox model trả về bị ngược (x2 < x1 / y2 < y1) hoặc thò ra ngoài ảnh → sắp lại + kẹp vào trang (sửa tại chỗ).
    Không sửa thì layoutjson2md của họ crop khối Picture lỗi ValueError → mất cả trang. Trả về số khối đã sửa."""
    n = 0
    for c in cells:
        b = c.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        x1, x2 = sorted(min(max(int(v), 0), width) for v in (b[0], b[2]))
        y1, y2 = sorted(min(max(int(v), 0), height) for v in (b[1], b[3]))
        if x2 == x1:
            x1, x2 = (x1 - 1, x1) if x1 >= width else (x1, x1 + 1)
        if y2 == y1:
            y1, y2 = (y1 - 1, y1) if y1 >= height else (y1, y1 + 1)
        if [x1, y1, x2, y2] != list(b):
            c["bbox"] = [x1, y1, x2, y2]
            n += 1
    return n


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
            n_fixed = fix_bboxes(cells, *origin_image.size)
            if n_fixed:
                extra["bbox_fixed"] = n_fixed
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
