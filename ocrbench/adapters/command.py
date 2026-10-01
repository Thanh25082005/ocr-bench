"""Chạy một lệnh bất kỳ cho mỗi ảnh và lấy stdout làm kết quả.

Dùng cho các model có script riêng (vd. code của một repo trên GitHub):

    params:
      cmd: "python /kaggle/working/Ketaba-OCR-LoRA/infer.py --image {image}"
      timeout: 300

Các biến thay thế: {image} {id} {category}
{image} là file PNG tạm của trang đang đọc (đã xoay đúng chiều, đã thu nhỏ nếu có max_image_side).
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

from .base import Adapter, Prediction


class CommandAdapter(Adapter):
    defaults = {"cmd": None, "timeout": 600, "cwd": None}

    def load(self):
        if not self.params["cmd"]:
            raise ValueError("adapter 'command' cần tham số 'cmd'")

    def predict(self, image, item):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            image.save(path)
            args = [
                a.format(image=str(path), id=item.id, category=item.category)
                for a in shlex.split(self.params["cmd"])
            ]
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=self.params["timeout"], cwd=self.params["cwd"]
            )
        if proc.returncode != 0:
            raise RuntimeError(f"lệnh thoát với mã {proc.returncode}: {proc.stderr.strip()[-500:]}")
        return Prediction(self.postprocess(proc.stdout))
