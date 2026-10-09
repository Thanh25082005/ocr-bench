"""Tìm vùng HÌNH trên trang (con dấu, vân tay, chữ ký, tem, logo) cho model chỉ đọc chữ (Tesseract...).

Không dùng model: xử lý ảnh cổ điển, chạy CPU ~0,5 giây/trang.
1. Tờ giấy = vùng sáng liền lớn nhất (bỏ nền tối quanh ảnh chụp điện thoại).
2. Mực = điểm tối hơn hẳn nền giấy, hoặc có màu (dấu đỏ, mực tím).
3. Mỗi cụm mực liền là một ứng viên; giữ cụm LỚN hơn hẳn cỡ chữ (chiều cao ≥ 2 dòng, hoặc cả hai cạnh ≥ 1,5 dòng),
   bỏ: nét mảnh (dấu /), đường kẻ, khung bảng (thưa, có ≥ 2 đường ngang + 2 đường dọc chạy suốt),
   cụm mà phần lớn mực nằm trong các từ OCR đã đọc chắc chắn (chữ to / chữ dính nhau) hoặc trong dải dòng của
   chúng (chữ viết tay điền vào form, cạnh nhãn in), vết ố nhạt màu.
4. Gộp cụm chồng nhau; cụm sát nhau chỉ gộp khi CÙNG màu (tem + dấu tím đóng đè lên tem thành một, dấu đỏ bên
   cạnh vẫn riêng); nét tối gộp rộng hơn (chữ ký hay đứt nét).

DocLayout-YOLO (đã thử) chỉ ra một khung "figure" cho cả nửa dưới trang giấy tờ cũ → không tách được từng dấu.
Hạn chế: chữ ký mà Tesseract "đọc" thành chữ với độ tin cậy cao thì vẫn là Text.
"""

from __future__ import annotations

import numpy as np


def find_pictures(image, words: list[tuple[list[int], float, object, str]], min_conf: float = 40) -> list[list[int]]:
    """image: PIL. words: [(bbox [x1, y1, x2, y2], độ tin cậy 0–100, mã dòng, chữ)] từ OCR. → bbox các vùng hình."""
    from scipy import ndimage as ndi

    a = np.asarray(image.convert("RGB")).astype(np.float32) / 255
    H, W = a.shape[:2]
    gray = a.mean(2)
    mx = a.max(2)
    sat = (mx - a.min(2)) / np.maximum(mx, 1e-6)

    bright = gray > 0.45
    lab, n = ndi.label(ndi.binary_opening(bright, iterations=3))
    if n:
        sizes = ndi.sum(bright, lab, range(1, n + 1))
        paper = ndi.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
        paper = ndi.binary_erosion(paper, iterations=max(2, W // 200))  # mép giấy / bóng đổ
    else:
        paper = np.ones_like(bright)
    if not paper.any():
        return []
    bg = float(np.median(gray[paper]))
    colored = (sat > 0.28) & (gray < bg * 0.97)
    ink = ((gray < bg * 0.72) | colored) & paper

    sure = [w for w in words if w[1] >= min_conf]
    lh = float(np.median([w[0][3] - w[0][1] for w in sure])) if sure else H / 50  # cỡ chữ
    lh = max(lh, 10.0)  # ảnh nhỏ / không đọc chắc được từ nào: đừng coi từng ký tự là hình
    normal = [w for w in words if w[0][3] - w[0][1] <= 1.8 * lh]  # "từ" cao bất thường = OCR đọc nhầm dấu / chữ ký
    textmask = np.zeros_like(ink)
    for (x1, y1, x2, y2), c, _, _ in normal:
        if c >= min_conf:
            textmask[y1:y2, x1:x2] = True
    # dải dòng: mọi từ OCR xếp vào một dòng có "neo" là từ in đọc chắc (≥ 60, ≥ 3 chữ/số) — chữ viết tay điền vào
    # form ("Full Name: Sarah") nằm trên dải này; dòng chỉ gồm chữ rác đọc ra từ chữ ký ('0', '=') không có neo
    anchored = {k for _, c, k, t in normal if c >= 60 and sum(ch.isalnum() for ch in t) >= 3}
    lines: dict = {}
    for (x1, y1, x2, y2), _, k, _ in normal:
        if k in anchored:
            b = lines.setdefault(k, [x1, y1, x2, y2])
            lines[k] = [min(b[0], x1), min(b[1], y1), max(b[2], x2), max(b[3], y2)]
    linemask = np.zeros_like(ink)
    for x1, y1, x2, y2 in lines.values():
        linemask[y1:y2, x1:x2] = True
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = colored & (r > g + 0.12) & (r > b + 0.05)
    blue = colored & (b > g + 0.05) & ~red

    lab, _ = ndi.label(ndi.binary_closing(ink, iterations=1))
    seeds = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (h >= 2 * lh or min(h, w) >= 1.5 * lh) or min(h, w) < 0.8 * lh:
            continue
        m = lab[sl] == i
        px = int(m.sum())
        fill = px / (h * w)
        if fill < 0.05 and min(h, w) < 2 * lh:
            continue
        if max(w, h) / max(1, min(w, h)) > 8 and fill > 0.5:  # đường kẻ
            continue
        if fill < 0.25 and (m.sum(1) > 0.6 * w).sum() >= 2 and (m.sum(0) > 0.6 * h).sum() >= 2:  # khung bảng
            continue
        if (m & textmask[sl]).sum() >= 0.5 * px:  # phần lớn là chữ đã đọc được
            continue
        if (m & linemask[sl]).sum() >= 0.5 * px:  # chữ viết tay trên cùng dòng với chữ in ("Full Name: Sarah")
            continue
        if gray[sl][m].mean() > 0.66 * bg:  # vết ố / gỉ nhạt màu (giấy cũ: 0,71·nền; dấu, vân tay, chữ ký: ≤ 0,51)
            continue
        nr, nb = int((m & red[sl]).sum()), int((m & blue[sl]).sum())
        kind = "red" if nr > 0.3 * px else "blue" if nb > 0.3 * px else "dark"
        seeds.append([xs.start, ys.start, xs.stop, ys.stop, kind])

    def near(p, q, d):
        return p[0] - d < q[2] and q[0] - d < p[2] and p[1] - d < q[3] and q[1] - d < p[3]

    merged = True
    while merged:
        merged = False
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                p, q = seeds[i], seeds[j]
                gap = (2 * lh if p[4] == "dark" else 0.5 * lh) if p[4] == q[4] else 0
                if near(p, q, gap):
                    kind = p[4] if p[4] == q[4] else ("dark" if "dark" in (p[4], q[4]) else p[4])
                    seeds[i] = [min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3]), kind]
                    del seeds[j]
                    merged = True
                    break
            if merged:
                break
    return [s[:4] for s in seeds]


def _inside(b, box) -> float:
    """Tỉ lệ diện tích khung b nằm trong box."""
    ix = max(0, min(b[2], box[2]) - max(b[0], box[0]))
    iy = max(0, min(b[3], box[3]) - max(b[1], box[1]))
    return ix * iy / max(1, (b[2] - b[0]) * (b[3] - b[1]))


def add_pictures(blocks: list[dict], pictures: list[list[int]]) -> list[dict]:
    """Chèn khối Picture vào danh sách khối chữ theo thứ tự đọc (trước khối chữ đầu tiên nằm thấp hơn nó);
    bỏ khối chữ nằm phần lớn trong một hình (chữ rác OCR đọc ra từ con dấu / chữ ký)."""
    out = [b for b in blocks if not (b.get("bbox") and any(_inside(b["bbox"], p) > 0.5 for p in pictures))]
    for p in sorted(pictures, key=lambda p: p[1]):
        k = next((i for i, b in enumerate(out) if b.get("bbox") and b["bbox"][1] > p[1]), len(out))
        out.insert(k, {"category": "Picture", "bbox": [int(v) for v in p], "text": ""})
    return out
