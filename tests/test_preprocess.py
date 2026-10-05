"""Tiền xử lý ảnh: ảnh sạch giữ nguyên, ảnh nghiêng / tối được chỉnh; worker dùng lại kết quả khi ảnh không đổi."""

import json

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw, ImageFont

from ocrbench.preprocess import config_of, estimate_skew, preprocess


def _page(lines=18, w=1400, h=1800):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default(size=30)
    for i in range(lines):
        d.text((90, 120 + i * 80), "The quick brown fox jumps over the lazy dog 0123456789", fill="black", font=font)
    return img


def test_clean_page_untouched():
    img = _page()
    out, info = preprocess(img, True)
    assert info == {}
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_disabled_is_noop():
    img = _page().rotate(2, expand=True, fillcolor="white")
    assert preprocess(img, False) == (img, {})


@pytest.mark.parametrize("angle", [-2.4, -0.8, 1.2, 3.0])
def test_deskew_corrects_rotation(angle):
    img = _page().rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
    out, info = preprocess(img, True)
    assert abs(info["deskew"]["angle"] + angle) <= 0.3
    assert abs(estimate_skew(out)) <= 0.3


def test_small_skew_ignored():
    img = _page().rotate(0.1, resample=Image.BICUBIC, expand=True, fillcolor="white")
    assert "deskew" not in preprocess(img, True)[1]


def test_flatten_uneven_light_and_tint():
    img = _page()
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape[:2]
    shade = 1.0 - 0.3 * (np.mgrid[0:h, 0:w][1] / w)  # tối dần sang phải
    a = a * shade[..., None] * np.array([1.0, 0.95, 0.85])  # giấy ngả vàng
    dark = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    out, info = preprocess(dark, True)
    assert "flatten" in info
    o = np.asarray(out, dtype=np.float32)
    left, right = np.percentile(o[:, : w // 5], 90), np.percentile(o[:, -w // 5:], 90)
    assert abs(left - right) < 12  # nền hai bên đã đều
    bg = o[o.mean(axis=2) > 200].mean(axis=0)
    assert bg.max() - bg.min() < 10  # hết ngả màu


def test_upscale_small_image():
    out, info = preprocess(_page().resize((600, 700)), True)  # cạnh dài 700 → ×2 (trần)
    assert info["upscale"]["factor"] == 2.0 and out.size == (1200, 1400)
    out, info = preprocess(_page().resize((700, 900)), True)  # cạnh dài 900 → tới 1600
    assert info["upscale"]["factor"] == 1.78 and max(out.size) == 1600


def test_config_validation():
    assert config_of(None) is None and config_of(False) is None
    assert config_of(True)["deskew"] and config_of("auto")["flatten"]
    assert config_of({"upscale": False})["upscale"] is False
    with pytest.raises(ValueError):
        config_of({"binarize": True})


def test_worker_reuses_unchanged_pages(tmp_path):
    """preprocess_reuse: mẫu ảnh không đổi lấy kết quả model gốc; mẫu bị chỉnh thì chạy model."""
    from ocrbench.cli import main

    data = tmp_path / "data"
    data.mkdir()
    _page().save(data / "clean.png")
    _page().rotate(2.0, resample=Image.BICUBIC, expand=True, fillcolor="white").save(data / "tilted.png")
    rows = [{"id": f"c/{n}", "image": f"{n}.png", "category": "c", "gt": "x", "doc_id": n, "split": "dev"}
            for n in ("clean", "tilted")]
    (data / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    base = {"name": "echo", "adapter": "dummy", "params": {"mode": "oracle"}}
    pre = {"name": "echo_pre", "adapter": "dummy",
           "params": {"mode": "oracle", "preprocess": True, "preprocess_reuse": "echo"}}
    bad = {"name": "echo_bad", "adapter": "dummy",
           "params": {"mode": "oracle", "noise": 0.5, "preprocess": True, "preprocess_reuse": "echo"}}
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"dataset": str(data / "manifest.jsonl"), "output_dir": str(tmp_path / "runs"),
                                   "models": [base, pre, bad]}))
    main(["run", "--config", str(cfg), "--models", "echo", "--inline", "--no-score"])
    # sửa kết quả gốc để biết chắc bản ghi được chép lại chứ không phải chạy lại
    p = tmp_path / "runs/dev/echo/predictions.jsonl"
    recs = [json.loads(x) for x in p.read_text().splitlines()]
    for r in recs:
        r["text"] = "FROM_BASE"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    main(["run", "--config", str(cfg), "--models", "echo_pre", "--inline", "--no-score"])
    out = {json.loads(x)["id"]: json.loads(x) for x in (tmp_path / "runs/dev/echo_pre/predictions.jsonl").read_text().splitlines()}
    assert out["c/clean"]["text"] == "FROM_BASE" and out["c/clean"]["extra"]["reused_from"] == "echo"
    assert out["c/tilted"]["text"] != "FROM_BASE" and "deskew" in out["c/tilted"]["extra"]["preprocess"][0]

    with pytest.raises(SystemExit):  # tham số khác model gốc → từ chối
        main(["run", "--config", str(cfg), "--models", "echo_bad", "--inline", "--no-score"])
