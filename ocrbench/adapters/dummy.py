"""Model giả lập để kiểm tra pipeline mà không cần GPU.

mode: oracle (trả đúng đáp án), empty, noisy (xóa/đổi ngẫu nhiên `noise` tỉ lệ ký tự),
      loop (đáp án + một đoạn lặp vô hạn), fail (luôn báo lỗi).
"""

from __future__ import annotations

import random
import time

from .base import Adapter, Prediction


class DummyAdapter(Adapter):
    defaults = {"mode": "oracle", "noise": 0.05, "seed": 0, "sleep": 0.0}

    def predict_pages(self, images, item):
        return self.predict(images[0], item)  # đáp án đã gồm mọi trang

    def predict(self, image, item):
        p = self.params
        if p["sleep"]:
            time.sleep(p["sleep"])
        mode = p["mode"]
        if mode == "oracle":
            return Prediction(item.gt)
        if mode == "empty":
            return Prediction("")
        if mode == "fail":
            raise RuntimeError("dummy fail")
        if mode == "loop":
            return Prediction(item.gt + " lorem ipsum" * 40)
        if mode == "noisy":
            rng = random.Random(f"{p['seed']}:{item.id}")
            out = []
            for ch in item.gt:
                r = rng.random()
                if r < p["noise"] / 2:
                    continue
                out.append("x" if r < p["noise"] else ch)
            return Prediction("".join(out))
        raise ValueError(f"dummy mode không hợp lệ: {mode}")
