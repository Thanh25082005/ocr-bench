"""Tesseract. Trên Kaggle: apt-get install -y tesseract-ocr tesseract-ocr-ara && pip install pytesseract"""

from __future__ import annotations

from .base import Adapter, Prediction


class TesseractAdapter(Adapter):
    # with_confidence: chạy thêm image_to_data để lấy độ tin cậy (chậm gấp đôi)
    defaults = {"lang": "ara+eng", "psm": 3, "config": "", "with_confidence": False}

    def load(self):
        import pytesseract

        self._tess = pytesseract
        self._tess.get_tesseract_version()  # báo lỗi sớm nếu chưa cài binary

    def predict(self, image, item):
        cfg = f"--psm {self.params['psm']} {self.params['config']}".strip()
        text = self._tess.image_to_string(image, lang=self.params["lang"], config=cfg)
        conf = None
        if self.params["with_confidence"]:
            data = self._tess.image_to_data(
                image, lang=self.params["lang"], config=cfg, output_type=self._tess.Output.DICT
            )
            confs = [float(c) for c in data["conf"] if float(c) >= 0]
            conf = (sum(confs) / len(confs) / 100) if confs else None
        return Prediction(text, confidence=conf)
