"""Giao diện chung cho mọi model OCR.

Muốn thêm model mới chỉ cần viết một class kế thừa `Adapter`, cài đặt `load()` và
`predict()`, rồi khai báo trong config:

    adapter: "my_package.my_module:MyAdapter"
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field

from PIL import Image

from ..dataset import Item


@dataclass
class Prediction:
    text: str
    confidence: float | None = None
    extra: dict = field(default_factory=dict)


class Adapter:
    #: Tham số mặc định; tham số trong config sẽ ghi đè.
    defaults: dict = {}

    def __init__(self, **params):
        self.params = {**self.defaults, **params}

    def load(self) -> None:
        """Nạp model (gọi một lần trước khi chạy)."""

    def predict(self, image: Image.Image, item: Item) -> Prediction:
        raise NotImplementedError

    def predict_pages(self, images: list[Image.Image], item: Item) -> Prediction:
        """Tài liệu nhiều trang. Mặc định: đọc từng trang rồi nối kết quả theo thứ tự.
        Model đọc được nhiều ảnh một lúc (VLM) nên ghi đè để nhận cả tài liệu trong một lần."""
        preds = [self.predict(img, item) for img in images]
        confs = [p.confidence for p in preds if p.confidence is not None]
        extra = {"pages": len(images), "page_mode": "per_page"}
        for p in preds:
            for k, v in p.extra.items():
                if k in ("hit_max_tokens", "stopped_loop"):
                    extra[k] = extra.get(k, False) or v
                elif k == "new_tokens":
                    extra[k] = extra.get(k, 0) + v
        return Prediction("\n\n".join(p.text for p in preds), (sum(confs) / len(confs)) if confs else None, extra)

    def predict_batch(self, images: list[Image.Image], items: list[Item]) -> list[Prediction]:
        """Nhiều trang độc lập trong một lượt. Mặc định: lần lượt từng trang; VLM ghi đè để chạy batch thật."""
        return [self.predict(im, it) for im, it in zip(images, items)]

    def close(self) -> None:
        """Giải phóng tài nguyên (tùy chọn)."""

    # --- tiện ích dùng chung -------------------------------------------------

    def prompt_for(self, item: Item) -> str:
        """`prompt` có thể là một chuỗi, hoặc dict chọn theo category / gt_type:

            prompt:
              default: "Extract all text..."
              table_html: "Convert tables to HTML..."
              handwriting_ar: "..."
        """
        p = self.params.get("prompt", "")
        if isinstance(p, dict):
            return p.get(item.category) or p.get(item.gt_type) or p.get("default", "")
        return p

    def postprocess(self, text: str) -> str:
        if self.params.get("strip_think", True):
            if "</think>" in text:
                text = text.rsplit("</think>", 1)[1]
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        field_name = self.params.get("json_field")
        if field_name:
            text = _extract_json_field(text, field_name)
        return text.strip()


def _extract_json_field(text: str, field_name: str) -> str:
    """Một số model (vd. Baseer) trả về JSON; lấy riêng trường chứa văn bản."""
    import json

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            value = json.loads(m.group(0)).get(field_name)
            if isinstance(value, str):
                return value
        except (json.JSONDecodeError, AttributeError):
            pass
    return text


BUILTIN = {
    "dummy": "ocrbench.adapters.dummy:DummyAdapter",
    "tesseract": "ocrbench.adapters.tesseract:TesseractAdapter",
    "easyocr": "ocrbench.adapters.easyocr_adapter:EasyOCRAdapter",
    "paddleocr": "ocrbench.adapters.paddle:PaddleOCRAdapter",
    "paddleocr_vl": "ocrbench.adapters.paddle:PaddleOCRVLAdapter",
    "surya": "ocrbench.adapters.surya_adapter:SuryaAdapter",
    "hf_vlm": "ocrbench.adapters.hf_vlm:HFVLMAdapter",
    "openai_api": "ocrbench.adapters.openai_api:OpenAIAPIAdapter",
    "command": "ocrbench.adapters.command:CommandAdapter",
}


def create_adapter(spec: str, params: dict | None = None) -> Adapter:
    target = BUILTIN.get(spec, spec)
    if ":" not in target:
        raise ValueError(f"adapter không hợp lệ: {spec!r}. Dùng một trong {sorted(BUILTIN)} hoặc 'module:Class'")
    module_name, cls_name = target.split(":", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    if not issubclass(cls, Adapter):
        raise TypeError(f"{target} phải kế thừa ocrbench.adapters.base.Adapter")
    return cls(**(params or {}))
