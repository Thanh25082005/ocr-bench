"""Tiến trình tách dòng kraken (blla) cho adapter kraken_layout.

Chạy bằng python của venv kraken (inference/setup_kraken.sh) — kraken kéo bản torch riêng nên không cài chung với
ocrbench. File này KHÔNG import ocrbench (venv không có), chỉ kraken + PIL + thư viện chuẩn.

Giao thức (một tiến trình sống suốt phiên web, model nạp một lần):
- stdin: mỗi dòng một JSON {"image": đường dẫn ảnh, "text_direction": "horizontal-rl"}
- stdout: mỗi kết quả một dòng '@@KRAKEN ' + JSON; dòng đầu tiên là {"ready": true} sau khi nạp model.
  {"lines": [{"baseline": [[x, y]...], "boundary": [[x, y]...], "regions": [id...]}], "regions": [{"id", "type",
   "boundary"}], "seconds": ...} theo THỨ TỰ ĐỌC của kraken, hoặc {"error": "..."}.
  Mọi thứ thư viện in ra stdout bị chuyển sang stderr để không lẫn vào giao thức.

Tự kiểm tra: python kraken_worker.py --check ảnh.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
PREFIX = "@@KRAKEN "


def _points(seq) -> list[list[int]]:
    return [[int(round(float(x))), int(round(float(y)))] for x, y in (seq or [])]


def segment(model, path: str, text_direction: str, device: str) -> dict:
    from kraken import blla
    from PIL import Image

    t = time.perf_counter()
    with Image.open(path) as im:
        seg = blla.segment(im.convert("RGB"), text_direction=text_direction, model=model, device=device)
    regions = [{"id": r.id, "type": kind, "boundary": _points(r.boundary)}
               for kind, rs in (seg.regions or {}).items() for r in rs]
    lines = [{"baseline": _points(ln.baseline), "boundary": _points(ln.boundary), "regions": list(ln.regions or [])}
             for ln in seg.lines]
    return {"lines": lines, "regions": regions, "seconds": round(time.perf_counter() - t, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model tách dòng (.mlmodel / .safetensors); mặc định blla của kraken")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--height", type=int,
                    help="chiều cao ảnh đưa vào mạng (model mặc định: 1800). 1400 trên CPU: nhanh hơn ~40%%, số dòng gần "
                         "như không đổi (đo trên 3 trang thật); 1200 bắt đầu lệch dòng + lỗi dựng đa giác")
    ap.add_argument("--check", help="tách dòng một ảnh, in số dòng rồi thoát")
    a = ap.parse_args()

    out, sys.stdout = sys.stdout, sys.stderr  # thư viện in gì ra stdout cũng không phá giao thức
    from importlib import resources

    from kraken.lib import vgsl

    t = time.perf_counter()
    model = vgsl.TorchVGSLModel.load_model(a.model or str(resources.files("kraken").joinpath("blla.mlmodel")))
    if a.height:
        model.input = (model.input[0], model.input[1], a.height, model.input[3])
    if a.check:
        r = segment(model, a.check, "horizontal-rl", a.device)
        print(f"   kraken: nạp model {time.perf_counter() - t - r['seconds']:.1f}s · {len(r['lines'])} dòng, "
              f"{len(r['regions'])} vùng trong {r['seconds']}s", file=out)
        return

    def send(obj: dict) -> None:
        out.write(PREFIX + json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    send({"ready": True, "load_seconds": round(time.perf_counter() - t, 2)})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            send(segment(model, req["image"], req.get("text_direction", "horizontal-rl"), a.device))
        except Exception as e:  # một trang hỏng không giết tiến trình
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
