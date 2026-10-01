"""PaddleOCR 3.x (PP-OCR) và PaddleOCR-VL.

Trên T4 nên dùng engine="transformers" (không cần cài paddlepaddle-gpu);
vLLM cần GPU Compute Capability >= 8.0 nên không dùng được trên T4.
"""

from __future__ import annotations

import numpy as np

from .base import Adapter, Prediction


def _bgr(image):
    """PaddleOCR nhận mảng numpy theo thứ tự kênh BGR (kiểu OpenCV)."""
    return np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])


def _res_get(res, key):
    try:
        return res[key]
    except (KeyError, TypeError):
        return (getattr(res, "json", None) or {}).get("res", {}).get(key)


class PaddleOCRAdapter(Adapter):
    defaults = {
        "lang": "ar",
        "device": "gpu",
        "engine": None,
        "init_kwargs": {},
    }

    def load(self):
        from paddleocr import PaddleOCR

        kwargs = dict(
            lang=self.params["lang"],
            device=self.params["device"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        if self.params["engine"]:
            kwargs["engine"] = self.params["engine"]
        kwargs.update(self.params["init_kwargs"])
        self._ocr = PaddleOCR(**kwargs)

    def predict(self, image, item):
        texts, scores = [], []
        for res in self._ocr.predict(_bgr(image)):
            texts += list(_res_get(res, "rec_texts") or [])
            scores += [float(s) for s in (_res_get(res, "rec_scores") or [])]
        return Prediction("\n".join(texts), confidence=(sum(scores) / len(scores)) if scores else None)


class PaddleOCRVLAdapter(Adapter):
    defaults = {"engine": "transformers", "device": None, "init_kwargs": {}}

    def load(self):
        from paddleocr import PaddleOCRVL

        kwargs = {}
        if self.params["engine"]:
            kwargs["engine"] = self.params["engine"]
        if self.params["device"]:
            kwargs["device"] = self.params["device"]
        kwargs.update(self.params["init_kwargs"])
        self._pipe = PaddleOCRVL(**kwargs)

    def predict(self, image, item):
        parts = []
        for res in self._pipe.predict(_bgr(image)):
            md = getattr(res, "markdown", None) or {}
            parts.append(md.get("markdown_texts", "") if isinstance(md, dict) else str(md))
        return Prediction("\n\n".join(p for p in parts if p))
