"""EasyOCR. pip install easyocr"""

from __future__ import annotations

import numpy as np

from .base import Adapter, Prediction


class EasyOCRAdapter(Adapter):
    defaults = {"langs": ["ar", "en"], "gpu": True, "paragraph": True}

    def load(self):
        import easyocr

        self._reader = easyocr.Reader(self.params["langs"], gpu=self.params["gpu"])

    def predict(self, image, item):
        results = self._reader.readtext(np.array(image), detail=1, paragraph=self.params["paragraph"])
        texts = [r[1] for r in results]
        # paragraph=True không trả về confidence
        confs = [r[2] for r in results if len(r) > 2]
        return Prediction("\n".join(texts), confidence=(sum(confs) / len(confs)) if confs else None)
