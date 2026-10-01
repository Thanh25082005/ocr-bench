"""Làm "xấu" ảnh tài liệu sạch cho giống ảnh chụp điện thoại hoặc bản scan kém.

Gồm: giấy ngả màu, sáng không đều, nghiêng, méo phối cảnh, mờ, nhiễu, giảm độ phân giải,
nén JPEG. Mức "medium" đọc vẫn dễ với người; mức "hard" thì khó nhưng người vẫn đọc được.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter

LEVELS = {
    "medium": dict(rot=1.5, persp=0.015, blur=(0.4, 0.9), noise=(3, 7), scale=(0.65, 0.85), jpeg=(55, 75), light=0.15),
    "hard": dict(rot=3.0, persp=0.03, blur=(0.8, 1.4), noise=(6, 12), scale=(0.5, 0.7), jpeg=(35, 55), light=0.3),
}


def _perspective_coeffs(src, dst):
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(src).reshape(8)
    return np.linalg.solve(np.array(a, dtype=float), b).tolist()


def degrade(img: Image.Image, rng: random.Random, level: str = "medium") -> tuple[Image.Image, dict]:
    p = LEVELS[level]
    img = img.convert("RGB")
    w, h = img.size
    info = {"level": level}

    # 1. giấy ngả vàng / xám
    tint = np.array([1.0, rng.uniform(0.95, 1.0), rng.uniform(0.86, 0.97)])
    arr = np.asarray(img, dtype=np.float32) * tint

    # 2. ánh sáng không đều: tối dần về một góc
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.hypot(w, h)
    shade = 1.0 - p["light"] * dist / dist.max()
    arr = np.clip(arr * shade[..., None], 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    bg = tuple(int(v) for v in arr[2, 2])

    # 3. méo phối cảnh (cầm điện thoại không thẳng)
    d = p["persp"]
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    moved = [(x + rng.uniform(-d, d) * w, y + rng.uniform(-d, d) * h) for x, y in corners]
    img = img.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(corners, moved), Image.BICUBIC, fillcolor=bg)

    # 4. nghiêng
    angle = rng.uniform(-p["rot"], p["rot"])
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg)
    info["rotation_deg"] = round(angle, 2)

    # 5. mờ + giảm độ phân giải
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(*p["blur"])))
    s = rng.uniform(*p["scale"])
    img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    info["scale"] = round(s, 2)

    # 6. nhiễu cảm biến
    arr = np.asarray(img, dtype=np.float32)
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    arr += nrng.normal(0, rng.uniform(*p["noise"]), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 7. nén JPEG
    q = rng.randint(*p["jpeg"])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    info["jpeg_quality"] = q
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), info
