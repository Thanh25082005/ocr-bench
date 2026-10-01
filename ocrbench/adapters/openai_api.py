"""Gọi một server tương thích OpenAI (vLLM, llama.cpp server, Strata...).

    params:
      base_url: http://127.0.0.1:8000/v1
      model: ten-model
      api_key_env: OPENAI_API_KEY   # tên biến môi trường chứa key (tùy chọn)
      prompt: "Extract all text..."
"""

from __future__ import annotations

import base64
import io
import os

from .base import Adapter, Prediction


class OpenAIAPIAdapter(Adapter):
    defaults = {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "default",
        "api_key_env": None,
        "prompt": "Extract all text in the image. Return only the text.",
        "max_tokens": 4096,
        "timeout": 600,
        "extra_body": {},
    }

    def load(self):
        import requests

        self._session = requests.Session()
        key = os.environ.get(self.params["api_key_env"] or "", "none")
        self._session.headers["Authorization"] = f"Bearer {key}"

    def predict(self, image, item):
        return self._call([image], item)

    def predict_pages(self, images, item):
        pred = self._call(images, item)  # mọi trang trong một yêu cầu
        pred.extra.update(pages=len(images), page_mode="joint")
        return pred

    def _call(self, images, item):
        parts = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        body = {
            "model": self.params["model"],
            "messages": [{"role": "user", "content": [*parts, {"type": "text", "text": self.prompt_for(item)}]}],
            "max_tokens": self.params["max_tokens"],
            "temperature": 0,
            **self.params["extra_body"],
        }
        r = self._session.post(
            self.params["base_url"].rstrip("/") + "/chat/completions", json=body, timeout=self.params["timeout"]
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        return Prediction(self.postprocess(text))
