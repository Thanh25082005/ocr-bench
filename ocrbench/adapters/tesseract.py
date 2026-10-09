"""Tesseract. Trên Kaggle: apt-get install -y tesseract-ocr tesseract-ocr-ara && pip install pytesseract"""

from __future__ import annotations

from .base import Adapter, Prediction


class TesseractAdapter(Adapter):
    # with_confidence: chạy thêm image_to_data để lấy độ tin cậy (chậm gấp đôi)
    # layout: trả thêm extra.blocks = khung chữ Tesseract tìm được (giao diện web vẽ khung bố cục, như dots):
    #   "line" = mỗi dòng một khối · "paragraph" (hoặc true) = mỗi đoạn một khối (Tesseract hay gộp đoạn rất thô).
    #   Loại khối chữ luôn là "Text" (Tesseract không phân loại tiêu đề / bảng). Benchmark để False.
    # pictures: cùng layout, tìm thêm khối "Picture" (con dấu, vân tay, chữ ký, tem — ocrbench/pictures.py, cần scipy);
    #   DOCX cắt các vùng này từ trang gốc thành ảnh.
    defaults = {"lang": "ara+eng", "psm": 3, "config": "", "with_confidence": False, "layout": False,
                "pictures": False}

    def load(self):
        import pytesseract

        self._tess = pytesseract
        self._tess.get_tesseract_version()  # báo lỗi sớm nếu chưa cài binary

    def predict(self, image, item):
        cfg = f"--psm {self.params['psm']} {self.params['config']}".strip()
        text = self._tess.image_to_string(image, lang=self.params["lang"], config=cfg)
        conf = None
        extra = {}
        if self.params["with_confidence"] or self.params["layout"]:
            data = self._tess.image_to_data(
                image, lang=self.params["lang"], config=cfg, output_type=self._tess.Output.DICT
            )
            if self.params["with_confidence"]:
                confs = [float(c) for c in data["conf"] if float(c) >= 0]
                conf = (sum(confs) / len(confs) / 100) if confs else None
            if self.params["layout"]:
                extra["blocks"] = text_blocks(data, by_line=self.params["layout"] == "line")
                if self.params["pictures"]:
                    from ..pictures import add_pictures, find_pictures

                    words = [([data["left"][i], data["top"][i], data["left"][i] + data["width"][i],
                               data["top"][i] + data["height"][i]], float(data["conf"][i]),
                              (data["block_num"][i], data["par_num"][i], data["line_num"][i]), w)
                             for i, w in enumerate(data["text"]) if (w or "").strip()]
                    extra["blocks"] = add_pictures(extra["blocks"], find_pictures(image, words))
        return Prediction(text, confidence=conf, extra=extra)


def text_blocks(data: dict, by_line: bool = False) -> list[dict]:
    """image_to_data → một khối mỗi đoạn (block_num, par_num), hoặc mỗi dòng nếu by_line, theo thứ tự Tesseract đọc.
    bbox = hình bao các từ của khối trên ảnh gốc; chữ = các dòng của khối nối bằng xuống dòng."""
    paras: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word or float(data["conf"][i]) < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        key = (data["block_num"][i], data["par_num"][i]) + ((data["line_num"][i],) if by_line else ())
        p = paras.setdefault(key, {"bbox": [x, y, x + w, y + h], "lines": {}})
        b = p["bbox"]
        p["bbox"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
        p["lines"].setdefault(data["line_num"][i], []).append(word)
    return [{"category": "Text", "bbox": p["bbox"], "text": "\n".join(" ".join(ws) for ws in p["lines"].values())}
            for p in paras.values()]
