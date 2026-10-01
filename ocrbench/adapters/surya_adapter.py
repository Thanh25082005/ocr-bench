"""Surya OCR (pip install surya-ocr).

Hỗ trợ API Surya 2 (SuryaInferenceManager, kết quả dạng `blocks` có html) và API cũ
(FoundationPredictor + DetectionPredictor, kết quả dạng `text_lines`).
Lưu ý: Surya 2 tự khởi động vLLM hoặc llama-server; vLLM không chạy ổn trên T4.
Có thể đặt biến môi trường SURYA_INFERENCE_BACKEND trong mục `env` của config.
"""

from __future__ import annotations

from .base import Adapter, Prediction


class SuryaAdapter(Adapter):
    defaults = {"use_layout": False}

    def load(self):
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor

            self._manager = SuryaInferenceManager()
            self._rec = RecognitionPredictor(self._manager)
            self._layout = None
            if self.params["use_layout"]:
                from surya.layout import LayoutPredictor

                self._layout = LayoutPredictor(self._manager)
            self._api = 2
        except ImportError:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            self._rec = RecognitionPredictor(FoundationPredictor())
            self._det = DetectionPredictor()
            self._api = 1

    def predict(self, image, item):
        if self._api == 2:
            if self._layout is not None:
                page = self._rec([image], self._layout([image]))[0]
            else:
                page = self._rec([image])[0]
            blocks = page.blocks
            text = "\n".join(b.html for b in blocks if getattr(b, "html", None))
            confs = [b.confidence for b in blocks if getattr(b, "confidence", None) is not None]
        else:
            page = self._rec([image], det_predictor=self._det)[0]
            text = "\n".join(line.text for line in page.text_lines)
            confs = [line.confidence for line in page.text_lines if line.confidence is not None]
        return Prediction(text, confidence=(sum(confs) / len(confs)) if confs else None)
