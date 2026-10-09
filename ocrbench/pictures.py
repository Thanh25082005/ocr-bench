"""Tìm vùng HÌNH trên trang (con dấu, vân tay, chữ ký, tem, logo) cho model chỉ đọc chữ (Tesseract...).

find_pictures — xử lý ảnh cổ điển, không model, CPU ~0,5 giây/trang:
1. Tờ giấy = vùng sáng liền lớn nhất (bỏ nền tối quanh ảnh chụp điện thoại).
2. Mực = điểm tối hơn hẳn nền giấy, hoặc có màu (dấu đỏ, mực tím).
3. Mỗi cụm mực liền là một ứng viên; giữ cụm LỚN hơn hẳn cỡ chữ (chiều cao ≥ 2 dòng, hoặc cả hai cạnh ≥ 1,5 dòng),
   bỏ: nét mảnh (dấu /), đường kẻ, khung bảng (thưa, có ≥ 2 đường ngang + 2 đường dọc chạy suốt),
   cụm mà phần lớn mực nằm trong các từ OCR đã đọc chắc chắn (chữ to / chữ dính nhau) hoặc trong dải dòng của
   chúng (chữ viết tay điền vào form, cạnh nhãn in), vết ố nhạt màu.
4. Mảng tối liền khối mà ngưỡng mực bỏ qua (vân tay mực nhạt): tìm trên ảnh làm mờ.
5. Gộp cụm chồng nhau; cụm sát nhau chỉ gộp khi CÙNG màu (tem + dấu tím đóng đè lên tem thành một, dấu đỏ bên
   cạnh vẫn riêng); nét tối gộp rộng hơn (chữ ký hay đứt nét).
Giới hạn: trên trang VIẾT TAY, chữ ký cùng cỡ, cùng nét bút với chữ → luật không tách được. Vì vậy:

find_signatures — model phát hiện chữ ký (YOLOS-base, Apache-2.0, huấn luyện trên tech4humans/signature-detection;
CPU ~2,5 giây/trang, tải ~0,5 GB lần đầu). So trên các trang thử: bắt đúng chữ ký ở trang in, form và trang viết tay
toàn bộ, không nhầm chữ viết tay điền form. Bản YOLOv8 cùng bộ dữ liệu bị loại vì giấy phép AGPL-3.0.
DocLayout-YOLO (đã thử) chỉ ra một khung "figure" cho cả nửa dưới trang giấy tờ cũ → không tách được từng dấu.

combine — ghép hai nguồn thành khối "Picture" / "Signature" (xem docstring).
"""

from __future__ import annotations

import threading

import numpy as np

SIGNATURE_MODEL = "mdefrance/yolos-base-signature-detection"


def find_pictures(image, words: list[tuple[list[int], float, object, str]] = (), min_conf: float = 40,
                  text_mask=None, line_height: float | None = None) -> list[dict]:
    """image: PIL. Vùng chữ lấy từ MỘT trong hai nguồn:
    - words: [(bbox [x1, y1, x2, y2], độ tin cậy 0–100, mã dòng, chữ)] từ OCR (Tesseract), hoặc
    - text_mask (mảng bool H×W, True = vùng dòng chữ, vd. đa giác dòng của kraken) + line_height (cỡ chữ, px).
    → [{"bbox", "kind": red | blue | dark | tone, "fill": tỉ lệ điểm mực trong khung}]."""
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
    cols, rows = np.flatnonzero(paper.any(0)), np.flatnonzero(paper.any(1))
    edges = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)  # mép tờ giấy
    colored = (sat > 0.28) & (gray < bg * 0.97)
    ink = ((gray < bg * 0.72) | colored) & paper

    if text_mask is not None:
        lh = max(float(line_height or H / 50), 10.0)
        textmask = linemask = np.asarray(text_mask, dtype=bool)
    else:
        textmask, linemask, lh = _text_from_words(words, min_conf, (H, W))
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
        cx, cy = (xs.start + xs.stop) / 2, (ys.start + ys.stop) / 2
        near_edge = min(cx - edges[0], edges[2] - cx, cy - edges[1], edges[3] - cy) < 0.12 * (edges[2] - edges[0])
        if fill > 0.6 and max(h, w) < 1.3 * min(h, w) and np.median(gray[sl][m]) < 0.2 * bg and near_edge:
            continue  # lỗ đục hồ sơ: tròn, đặc, đen tuyệt đối (thấy nền phía sau), sát mép giấy
        nr, nb = int((m & red[sl]).sum()), int((m & blue[sl]).sum())
        kind = "red" if nr > 0.3 * px else "blue" if nb > 0.3 * px else "dark"
        seeds.append([xs.start, ys.start, xs.stop, ys.stop, kind])

    # mảng tối liền khối mà ngưỡng mực bỏ qua (vân tay mực nhạt: ~0,86·nền): làm mờ để nét chữ mảnh nhạt đi, mảng
    # đặc vẫn tối. Chỉ nhận mảng xanh/tím hơn giấy (màu mực lăn tay) hoặc đen thật — vết gỉ / ố thì đỏ-nâu hơn giấy.
    blur = ndi.gaussian_filter(gray, sigma=lh * 0.5)
    tone = (blur < bg * 0.9) & paper
    paper_br = float(np.median((b - r)[paper]))
    lab, _ = ndi.label(tone)
    found = list(seeds)  # mảng chồng lên vùng mực đã thấy (con dấu, tem) thì bỏ — đừng làm cầu nối gộp chúng lại
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if min(h, w) < 2 * lh or max(h, w) > 2.5 * min(h, w):
            continue
        if any(xs.start < s[2] and s[0] < xs.stop and ys.start < s[3] and s[1] < ys.stop for s in found):
            continue
        m = lab[sl] == i
        px = int(m.sum())
        if px < 0.5 * h * w:  # tròn / đặc, không phải dải chữ
            continue
        p5 = float(np.percentile(gray[sl][m], 5))
        if p5 < 0.2 * bg:  # lỗ đục giấy: thấy nền đen phía sau (vân tay không tối thế)
            continue
        bluer = float((b - r)[sl][m].mean()) - paper_br
        # vết mực loang nhạt: KHÔNG có nét đậm (5% điểm tối nhất ≥ 0,5·nền; mảng chữ: 0,26–0,38) và xanh hơn giấy rõ
        # (≥ 0,08; mảng chữ ≤ 0,05) → vân tay, kể cả khi đa giác dòng kraken (rộng) trùm lên nó
        smudge = bluer >= 0.08 and p5 >= 0.5 * bg
        if not smudge and (m & (textmask[sl] | linemask[sl])).sum() >= 0.3 * px:
            continue
        if bluer < 0.04 and blur[sl][m].mean() > 0.75 * bg:
            continue
        seeds.append([xs.start, ys.start, xs.stop, ys.stop, "tone"])

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
    return [{"bbox": [int(v) for v in s[:4]], "kind": s[4], "fill": float(ink[s[1]:s[3], s[0]:s[2]].mean())}
            for s in seeds]


_sig_lock = threading.Lock()
_sig_models: dict = {}


def find_signatures(image, model_id: str = SIGNATURE_MODEL, threshold: float = 0.45) -> list[list[int]]:
    """bbox các chữ ký (model object-detection của transformers; nạp một lần cho cả tiến trình)."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    with _sig_lock:
        if model_id not in _sig_models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _sig_models[model_id] = (AutoImageProcessor.from_pretrained(model_id),
                                     AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval(), device)
        proc, model, device = _sig_models[model_id]
        im = image.convert("RGB")
        with torch.no_grad():
            out = model(**proc(images=im, return_tensors="pt").to(device))
        res = proc.post_process_object_detection(out, threshold=threshold, target_sizes=[im.size[::-1]])[0]
    return [[int(round(v)) for v in b.tolist()] for b in res["boxes"]]


def _text_from_words(words, min_conf: float, shape) -> tuple[np.ndarray, np.ndarray, float]:
    """Từ OCR → (mặt nạ từ đọc chắc, mặt nạ dải dòng, cỡ chữ)."""
    H = shape[0]
    sure = [w for w in words if w[1] >= min_conf]
    lh = float(np.median([w[0][3] - w[0][1] for w in sure])) if sure else H / 50  # cỡ chữ
    lh = max(lh, 10.0)  # ảnh nhỏ / không đọc chắc được từ nào: đừng coi từng ký tự là hình
    normal = [w for w in words if w[0][3] - w[0][1] <= 1.8 * lh]  # "từ" cao bất thường = OCR đọc nhầm dấu / chữ ký
    textmask = np.zeros(shape, dtype=bool)
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
    linemask = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in lines.values():
        linemask[y1:y2, x1:x2] = True
    return textmask, linemask, lh


def inside(b, box) -> float:
    """Tỉ lệ diện tích khung b nằm trong box."""
    ix = max(0, min(b[2], box[2]) - max(b[0], box[0]))
    iy = max(0, min(b[3], box[3]) - max(b[1], box[1]))
    return ix * iy / max(1, (b[2] - b[0]) * (b[3] - b[1]))


def _area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _union(a, b) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def combine(pictures: list[dict], signatures: list[list[int]] | None) -> list[dict]:
    """→ [{"category": "Picture" | "Signature", "bbox"}].
    - Chữ ký nằm gọn (≥ 60%) trong một hình lớn hơn gấp đôi (cụm tem + dấu) → là một phần của hình đó, bỏ.
    - Hình chồng lên chữ ký (≥ 50% hình nằm trong, hoặc ≥ 50% chữ ký nằm trong hình) → gộp vào khối Signature.
    - Có model chữ ký (signatures không phải None): hình kiểu NÉT BÚT — mực thưa (fill < 0,25), dài (cạnh dài ≥ 1,8
      cạnh ngắn) — mà model không nhận là chữ ký → chữ viết tay luật nhận nhầm, bỏ. Con dấu / logo / vân tay gần
      tròn hoặc đặc nên giữ."""
    pics = [dict(p) for p in pictures]
    sigs: list[list[int]] = []
    for s in signatures or []:
        if any(inside(s, p["bbox"]) >= 0.6 and _area(p["bbox"]) >= 2 * _area(s) for p in pics):
            continue
        for p in [p for p in pics if inside(p["bbox"], s) >= 0.5 or inside(s, p["bbox"]) >= 0.5]:
            s = _union(s, p["bbox"])
            pics.remove(p)
        for t in [t for t in sigs if inside(t, s) >= 0.5 or inside(s, t) >= 0.5]:
            s = _union(s, t)
            sigs.remove(t)
        sigs.append(s)
    if signatures is not None:
        def stroke(p):
            x1, y1, x2, y2 = p["bbox"]
            return p["kind"] != "tone" and p["fill"] < 0.25 and max(x2 - x1, y2 - y1) >= 1.8 * min(x2 - x1, y2 - y1)

        pics = [p for p in pics if not stroke(p)]
    return ([{"category": "Picture", "bbox": p["bbox"]} for p in pics]
            + [{"category": "Signature", "bbox": s} for s in sigs])


def add_pictures(blocks: list[dict], regions: list[dict]) -> list[dict]:
    """Chèn khối hình (từ combine) vào danh sách khối chữ theo thứ tự đọc (trước khối chữ đầu tiên nằm thấp hơn nó);
    bỏ khối chữ nằm phần lớn trong một hình (chữ rác OCR đọc ra từ con dấu / chữ ký)."""
    out = [b for b in blocks if not (b.get("bbox") and any(inside(b["bbox"], r["bbox"]) > 0.5 for r in regions))]
    for r in sorted(regions, key=lambda r: r["bbox"][1]):
        k = next((i for i, b in enumerate(out) if b.get("bbox") and b["bbox"][1] > r["bbox"][1]), len(out))
        out.insert(k, {"category": r["category"], "bbox": [int(v) for v in r["bbox"]], "text": ""})
    return out
