"""Tiền xử lý ảnh trước khi đưa vào model (tùy chọn, bật bằng `params.preprocess: true`).

Thích ứng: mỗi bước chỉ chạy khi ảnh THẬT SỰ có vấn đề đó; ảnh sạch được trả về nguyên từng pixel
(info rỗng). Nhờ vậy bật tiền xử lý không thể làm hỏng trang vốn đã tốt.

Các bước (theo thứ tự):
1. flatten  — sáng không đều / giấy ngả màu: khớp mặt nền bậc 2 cho từng kênh màu rồi chia cho nó.
              Mặt bậc 2 trơn nên không tạo quầng quanh ô tô màu, logo, con dấu.
2. contrast — chữ nhạt (khoảng sáng–tối hẹp): kéo giãn tuyến tính.
3. deskew   — trang nghiêng 0,3°–5°: dò góc bằng hình chiếu theo hàng (projection profile).
4. upscale  — ảnh nhỏ (cạnh dài < 1200 px): phóng lên tới 1600 px (tối đa ×2), chữ nhỏ có thêm điểm ảnh.

Cố ý KHÔNG làm: nhị phân hóa đen/trắng và lọc nhiễu (làm mất chấm / dấu phụ tiếng Ả Rập, VLM đọc
kém hơn), nắn phối cảnh, xoay 90°/180°.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULTS = {
    "flatten": True,
    "flatten_min": 0.10,      # nền chênh sáng ≥ 10% hoặc lệch màu ≥ 6% thì làm phẳng
    "tint_min": 0.06,
    "contrast": True,
    "contrast_max_range": 140,  # p99.5 − p0.5 (thang 0–255) dưới ngưỡng này thì kéo giãn
    "deskew": True,
    "deskew_min": 0.3,        # độ
    "deskew_max": 5.0,
    "deskew_min_height": 300,  # ảnh thấp hơn (một dòng chữ) thì không dò góc: không đủ tin cậy
    # chỉ xoay khi chắc chắn: độ nét hình chiếu tăng ≥ 2,5 lần so với không xoay, HOẶC tăng ≥ 1,5 lần và
    # nửa trái, nửa phải trang cho cùng góc (±0,5°). Chữ tay xiên trên biểu mẫu thẳng không qua được
    # (đo trên bộ dev: 0/132 biểu mẫu thẳng bị xoay nhầm; 39/48 trang xoay thử được chỉnh đúng góc)
    "deskew_sure_gain": 2.5,
    "deskew_halves_gain": 1.5,
    "deskew_halves_tol": 0.5,
    "upscale": True,
    "upscale_below": 1200,
    "upscale_to": 1600,
    "upscale_max_factor": 2.0,
}


def config_of(value) -> dict | None:
    """`params.preprocess`: true / "auto" = mặc định; dict = ghi đè từng bước; false / None = tắt."""
    if not value:
        return None
    if value is True or value == "auto":
        return dict(DEFAULTS)
    if isinstance(value, dict):
        unknown = set(value) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"preprocess: không có tùy chọn {sorted(unknown)}. Có: {sorted(DEFAULTS)}")
        return {**DEFAULTS, **value}
    raise ValueError(f"preprocess phải là true/false/'auto'/dict, nhận {value!r}")


def preprocess(img: Image.Image, cfg) -> tuple[Image.Image, dict]:
    """Trả về (ảnh, info). info chỉ có các bước đã thực sự chạy; info rỗng = ảnh không đổi."""
    c = cfg if isinstance(cfg, dict) and set(cfg) == set(DEFAULTS) else config_of(cfg)
    info: dict = {}
    if c is None:
        return img, info
    img = img.convert("RGB")
    if c["flatten"]:
        img = _flatten(img, c, info)
    if c["contrast"]:
        img = _contrast(img, c, info)
    if c["deskew"]:
        img = _deskew(img, c, info)
    if c["upscale"]:
        img = _upscale(img, c, info)
    return img, info


# --- 1. làm phẳng nền ---------------------------------------------------------

def _background_surfaces(img: Image.Image, grid: int = 16):
    """Độ sáng nền từng ô lưới (phân vị 90 mỗi kênh) → khớp mặt bậc 2 cho mỗi kênh.
    Trả về (hàm tính mặt nền trên lưới bất kỳ, độ chênh sáng, độ lệch màu)."""
    small = img.copy()
    small.thumbnail((512, 512), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    h, w, _ = a.shape
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    pts, vals = [], []
    for i in range(grid):
        for j in range(grid):
            cell = a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].reshape(-1, 3)
            if len(cell) == 0:
                continue
            pts.append(((ys[i] + ys[i + 1]) / 2 / h, (xs[j] + xs[j + 1]) / 2 / w))
            vals.append(np.percentile(cell, 90, axis=0))
    pts, vals = np.array(pts), np.array(vals)
    lum = vals.mean(axis=1)
    keep = lum >= np.median(lum) * 0.75  # bỏ ô bị khối tối (ô tô màu, ảnh, logo) chiếm hết
    if keep.sum() < 10:
        return None, 0.0, 0.0
    yy, xx = pts[keep, 0], pts[keep, 1]

    def basis(y, x):
        return np.stack([np.ones_like(y), y, x, y * y, x * x, x * y], axis=-1)

    coef, *_ = np.linalg.lstsq(basis(yy, xx), vals[keep], rcond=None)

    def surface(hh: int, ww: int) -> np.ndarray:
        gy, gx = np.meshgrid((np.arange(hh) + 0.5) / hh, (np.arange(ww) + 0.5) / ww, indexing="ij")
        return np.clip(basis(gy, gx) @ coef, 1.0, None)

    s = surface(64, 64)
    s_lum = s.mean(axis=2)
    uneven = float((s_lum.max() - s_lum.min()) / s_lum.max())
    means = s.reshape(-1, 3).mean(axis=0)
    tint = float((means.max() - means.min()) / means.max())
    return surface, uneven, tint


def _flatten(img, c, info):
    surface, uneven, tint = _background_surfaces(img)
    if surface is None or (uneven < c["flatten_min"] and tint < c["tint_min"]):
        return img
    w, h = img.size
    # tính mặt nền ở độ phân giải thấp rồi phóng lên (mặt trơn nên không mất gì)
    sh, sw = max(1, h // 8), max(1, w // 8)
    s = surface(sh, sw)
    s_full = np.stack([np.asarray(Image.fromarray(s[..., k].astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
                       for k in range(3)], axis=-1)
    a = np.asarray(img, dtype=np.float32) * (250.0 / s_full)
    info["flatten"] = {"uneven": round(uneven, 3), "tint": round(tint, 3)}
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# --- 2. tăng tương phản -----------------------------------------------------------

def _contrast(img, c, info):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(g, [0.5, 99.5])
    if hi - lo >= c["contrast_max_range"] or hi - lo < 20:  # đủ tương phản, hoặc trang gần như trống
        return img
    a = (np.asarray(img, dtype=np.float32) - lo) * (235.0 / (hi - lo)) + 10
    info["contrast"] = {"range": round(float(hi - lo), 1)}
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# --- 3. chỉnh nghiêng -------------------------------------------------------------

def _otsu(g: np.ndarray) -> float:
    hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return float(np.nanargmax(var))


def _profile_score(ink: Image.Image, angle: float) -> float:
    rows = np.asarray(ink.rotate(angle, resample=Image.NEAREST, fillcolor=0), dtype=np.float32).sum(axis=1)
    return float(np.sum(np.diff(rows) ** 2))


def _ink(img: Image.Image) -> Image.Image | None:
    g = img.convert("L")
    g.thumbnail((1000, 1000), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    ink = a < _otsu(a)
    frac = ink.mean()
    if frac < 0.005 or frac > 0.4:
        return None
    return Image.fromarray((ink * 255).astype(np.uint8))


def estimate_skew(img: Image.Image, max_angle: float = 5.0, with_gain: bool = False):
    """Góc (độ, theo chiều của PIL.rotate) làm các dòng chữ nằm ngang nhất. None = không đủ chữ để dò.
    with_gain=True: trả về (góc, độ nét hình chiếu ở góc đó / độ nét khi không xoay)."""
    ink_img = _ink(img)
    if ink_img is None:
        return (None, 0.0) if with_gain else None
    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.25)
    scores = [_profile_score(ink_img, x) for x in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - 0.25, best + 0.25 + 1e-6, 0.05)
    scores = [_profile_score(ink_img, x) for x in fine]
    angle = round(float(fine[int(np.argmax(scores))]), 2)
    if not with_gain:
        return angle
    return angle, max(scores) / max(_profile_score(ink_img, 0.0), 1.0)


def _deskew(img, c, info):
    if img.height < c["deskew_min_height"]:
        return img
    angle, gain = estimate_skew(img, c["deskew_max"], with_gain=True)
    if angle is None or abs(angle) < c["deskew_min"] or abs(angle) >= c["deskew_max"] - 0.05:
        return img  # không đủ chữ, gần như thẳng, hoặc chạm biên dò (không tin được)
    if gain < c["deskew_sure_gain"]:
        if gain < c["deskew_halves_gain"]:
            return img
        w, h = img.size
        tol = c["deskew_halves_tol"]
        halves = [estimate_skew(img.crop(box), c["deskew_max"]) for box in ((0, 0, w // 2, h), (w // 2, 0, w, h))]
        if any(x is None or abs(x - angle) > tol for x in halves) or abs(halves[0] - halves[1]) > tol:
            return img  # hai nửa trang không cùng góc: nghiêng cục bộ (vd. chữ tay xiên), không phải cả trang
    border = np.concatenate([np.asarray(img)[[0, -1]].reshape(-1, 3), np.asarray(img)[:, [0, -1]].reshape(-1, 3)])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    info["deskew"] = {"angle": angle, "gain": round(gain, 2)}
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill)


# --- 4. phóng ảnh nhỏ ------------------------------------------------------------

def _upscale(img, c, info):
    long_side = max(img.size)
    if long_side >= c["upscale_below"]:
        return img
    s = min(c["upscale_max_factor"], c["upscale_to"] / long_side)
    info["upscale"] = {"factor": round(s, 2)}
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
