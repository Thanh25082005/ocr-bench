"""Bố cục bằng kraken: tách dòng (blla) + chữ ký / con dấu / vân tay + đọc chữ từng dòng.

Vì sao: Tesseract tách dòng kém trên chữ viết tay Ả Rập (ảnh thật của khách: bỏ trắng cả khối tên nhân chứng, chỗ
ký, đoạn xác nhận cuối trang). blla của kraken được huấn luyện cho tài liệu lịch sử / viết tay, đọc phải→trái:
cùng trang đó ra 41 dòng phủ gần hết chữ (Tesseract 24). Không cần GPU.

Ba lớp:
1. kraken tách dòng → đa giác + đường chân chữ từng dòng, theo thứ tự đọc. Chạy trong venv riêng
   (inference/setup_kraken.sh; kraken kéo torch riêng) qua tiến trình con ocrbench/kraken_worker.py, model nạp một lần.
2. Phần tử không phải chữ (ocrbench/pictures.py): Signature (model YOLOS) + Picture (con dấu, tem, logo, vân tay).
   Mặt nạ vùng chữ = đa giác dòng kraken (phủ cả chữ viết tay → luật không nhầm chữ viết tay thành hình). Dòng kraken
   nằm trong chữ ký / con dấu (kraken coi nét chữ ký là "dòng") bị bỏ.
3. Đọc chữ từng dòng: recognizer "tesseract" (cắt đúng đa giác dòng, psm 7) hoặc "none" (chỉ bố cục; DOCX khi đó cắt
   mỗi dòng thành ảnh). Model đọc tốt hơn cho chữ viết tay chọn sau (experiments/real_docs_probe).

Khối trả về: mỗi dòng một khối "Text" (bbox + polygon), xen khối Picture / Signature theo thứ tự đọc.
Tốc độ CPU (2 luồng): tách dòng ~23–28 giây/trang ở seg_height 1400, đọc chữ Tesseract ~5–15 giây.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .base import Adapter, Prediction

PREFIX = "@@KRAKEN "
WORKER = Path(__file__).resolve().parents[1] / "kraken_worker.py"
# cỡ chữ (chiều cao từ Tesseract) ≈ 0,58 × chiều cao đa giác dòng kraken (đo trên 3 trang: 0,57–0,59)
LINE_TO_TEXT = 0.58


def default_python() -> str | None:
    """python của venv kraken: $OCRBENCH_KRAKEN_PYTHON, rồi chỗ inference/setup_kraken.sh cài."""
    if os.environ.get("OCRBENCH_KRAKEN_PYTHON"):
        return os.environ["OCRBENCH_KRAKEN_PYTHON"]
    for venv in (Path("/kaggle/working/venvs/kraken"), Path.home() / "venvs" / "kraken"):
        if (venv / "bin" / "python").exists():
            return str(venv / "bin" / "python")
    return None


class KrakenWorker:
    """Tiến trình con kraken_worker.py; mỗi trang một yêu cầu, chết thì khởi động lại một lần."""

    def __init__(self, python: str, model: str | None = None, device: str = "cpu", height: int | None = None,
                 timeout: float = 600):
        self.cmd = [python, str(WORKER), "--device", device]
        if model:
            self.cmd += ["--model", model]
        if height:
            self.cmd += ["--height", str(height)]
        self.timeout = timeout
        self.proc = None
        self.lock = threading.Lock()
        self.log = Path(tempfile.gettempdir()) / f"ocrbench_kraken_{os.getpid()}.log"

    def start(self) -> None:
        env = {**os.environ, "PYTHONNOUSERSITE": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
        self._log = open(self.log, "a", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                     text=True, encoding="utf-8", bufsize=1, env=env)
        self.replies: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.replies), daemon=True).start()
        ready = self._reply()
        if not ready.get("ready"):
            raise RuntimeError(f"kraken không khởi động được: {ready.get('error')}")

    @staticmethod
    def _pump(proc, replies) -> None:
        for raw in proc.stdout:
            if raw.startswith(PREFIX):
                replies.put(json.loads(raw[len(PREFIX):]))
        replies.put({"error": f"tiến trình kraken đã dừng (mã {proc.wait()})"})

    def _reply(self) -> dict:
        try:
            return self.replies.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            return {"error": f"kraken quá {self.timeout:.0f} giây không trả lời"}

    def _tail(self) -> str:
        try:
            return "\n".join(self.log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
        except OSError:
            return ""

    def segment(self, path: str, text_direction: str) -> dict:
        with self.lock:
            for attempt in range(2):
                if self.proc is None or self.proc.poll() is not None:
                    self.start()
                try:
                    self.proc.stdin.write(json.dumps({"image": path, "text_direction": text_direction}) + "\n")
                    self.proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    self.proc = None
                    continue
                r = self._reply()
                if "error" in r and self.proc is not None and self.proc.poll() is not None and attempt == 0:
                    continue  # tiến trình chết giữa chừng (vd. hết RAM): khởi động lại, thử lại một lần
                if "error" in r:
                    raise RuntimeError(f"kraken: {r['error']}\n{self._tail()}")
                return r
        raise RuntimeError(f"kraken không chạy được\n{self._tail()}")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None


class KrakenLayoutAdapter(Adapter):
    defaults = {
        "python": None,                 # python của venv kraken; None = default_python()
        "seg_model": None,              # model tách dòng; None = blla mặc định của kraken
        "seg_height": 1400,             # chiều cao ảnh vào mạng (model: 1800). GPU: đặt 1800 hoặc null
        "device": "cpu",                # cpu | cuda (kraken)
        "text_direction": "horizontal-rl",
        "recognizer": "tesseract",      # đọc chữ từng dòng: tesseract | none
        "lang": "ara+eng",
        "psm": 7,                       # Tesseract: một dòng
        "ocr_workers": None,            # số tiến trình Tesseract song song; None = số nhân CPU được cấp
        "min_line_ink": 0.72,           # bỏ dòng không có điểm nào tối hơn 0,72·nền giấy (watermark, vết mờ); None = giữ
        "pictures": True,               # con dấu, tem, logo, vân tay (ocrbench/pictures.py)
        "signatures": True,             # model chữ ký (true | tên model HF | false)
        "signature_threshold": 0.45,
        "timeout": 600,
    }

    def load(self):
        py = self.params["python"] or default_python()
        if not py:
            raise RuntimeError("Chưa có venv kraken: chạy `bash inference/setup_kraken.sh` "
                               "(hoặc đặt params.python / biến OCRBENCH_KRAKEN_PYTHON)")
        self.worker = KrakenWorker(py, self.params["seg_model"], self.params["device"], self.params["seg_height"],
                                   self.params["timeout"])
        self.worker.start()
        if self.params["recognizer"] == "tesseract" and not shutil.which("tesseract"):
            raise RuntimeError("Chưa cài tesseract (apt-get install -y tesseract-ocr tesseract-ocr-ara)")
        n = self.params["ocr_workers"] or len(os.sched_getaffinity(0))
        self.pool = ThreadPoolExecutor(max(1, int(n)))

    def close(self):
        self.worker.close()
        self.pool.shutdown(wait=False)

    def predict(self, image, item):
        from ..pictures import SIGNATURE_MODEL, add_pictures, combine, find_pictures, find_signatures, inside

        image = image.convert("RGB")
        sig = self.params["signatures"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "page.png"
            image.save(path)
            # model chữ ký chạy trong tiến trình này trong lúc kraken (tiến trình con) tách dòng
            sigs_job = self.pool.submit(find_signatures, image, SIGNATURE_MODEL if sig is True else sig,
                                        self.params["signature_threshold"]) if sig and self.params["pictures"] else None
            seg = self.worker.segment(str(path), self.params["text_direction"])
        lines = parse_lines(seg)
        if self.params["min_line_ink"]:
            lines = drop_faint_lines(image, lines, self.params["min_line_ink"])
        regions = []
        if self.params["pictures"]:
            mask, lh = text_mask(image.size, lines)
            sigs = sigs_job.result() if sigs_job else None
            regions = combine(find_pictures(image, text_mask=mask, line_height=lh), sigs)
        # dòng kraken nằm trong chữ ký / con dấu: nét vẽ, không phải chữ → không đọc
        lines = [ln for ln in lines if not any(inside(ln["bbox"], r["bbox"]) > 0.5 for r in regions)]
        texts = self.recognize(image, lines)
        blocks = [{"category": "Text", "bbox": ln["bbox"], "polygon": ln["polygon"], "text": t}
                  for ln, t in zip(lines, texts)]
        blocks = add_pictures(blocks, regions)
        return Prediction("\n\n".join(t for t in texts if t),
                          extra={"blocks": blocks, "kraken_seconds": seg.get("seconds"), "lines": len(lines)})

    def recognize(self, image, lines: list[dict]) -> list[str]:
        if self.params["recognizer"] == "none" or not lines:
            return ["" for _ in lines]
        return list(self.pool.map(lambda ln: tesseract_line(line_crop(image, ln), self.params["lang"],
                                                            self.params["psm"]), lines))


def tesseract_line(crop, lang: str, psm: int, timeout: float = 60) -> str:
    """Một dòng → chữ. Gọi thẳng tesseract với OMP_THREAD_LIMIT=1: mặc định mỗi tiến trình mở nhiều luồng OpenMP, chạy
    song song vài tiến trình trên 2 nhân thì tranh nhau tới mức một dòng mất > 2 phút (một mình: 0,1–0,5 giây)."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "line.png"
        crop.save(path)
        try:
            r = subprocess.run(["tesseract", str(path), "stdout", "-l", lang, "--psm", str(psm)], capture_output=True,
                               text=True, encoding="utf-8", timeout=timeout,
                               env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        except subprocess.TimeoutExpired:
            return ""
    return " ".join(r.stdout.split())


def parse_lines(seg: dict) -> list[dict]:
    """Kết quả kraken_worker → [{"polygon", "bbox", "baseline"}] theo thứ tự đọc; bỏ dòng suy biến."""
    out = []
    for ln in seg.get("lines", []):
        poly = ln.get("boundary") or []
        if len(poly) < 3:
            continue
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        bbox = [max(0, min(xs)), max(0, min(ys)), max(xs), max(ys)]
        if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
            continue
        out.append({"polygon": poly, "bbox": bbox, "baseline": ln.get("baseline") or []})
    return out


def drop_faint_lines(image, lines: list[dict], ratio: float = 0.72) -> list[dict]:
    """Bỏ "dòng" không có nét mực đậm: 2% điểm tối nhất trong đa giác vẫn sáng hơn ratio·nền giấy. Đo trên ảnh thật:
    watermark báo ("اليوم السابع") 0,76–0,87·nền, dòng chữ thật (cả bút bi xanh nhạt) ≤ 0,65·nền."""
    from PIL import Image, ImageDraw

    gray = np.asarray(image.convert("L")).astype(np.float32) / 255
    paper = gray[gray > 0.45]
    bg = float(np.median(paper)) if paper.size else 1.0
    out = []
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox"]
        m = Image.new("L", (x2 - x1 + 1, y2 - y1 + 1), 0)
        ImageDraw.Draw(m).polygon([(x - x1, y - y1) for x, y in ln["polygon"]], fill=255)
        g = gray[y1:y2 + 1, x1:x2 + 1][(np.asarray(m) > 0)[: gray.shape[0] - y1, : gray.shape[1] - x1]]
        if g.size and np.percentile(g, 2) <= ratio * bg:
            out.append(ln)
    return out


def text_mask(size: tuple[int, int], lines: list[dict]) -> tuple[np.ndarray, float]:
    """(mặt nạ vùng chữ = đa giác các dòng cỡ thường, cỡ chữ ước tính). Dòng cao bất thường (> 2,5 × trung vị: thường
    là nét chữ ký / con dấu kraken coi là dòng) không tính là chữ."""
    from PIL import Image, ImageDraw

    W, H = size
    hs = [ln["bbox"][3] - ln["bbox"][1] for ln in lines]
    med = float(np.median(hs)) if hs else H / 30
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for ln, h in zip(lines, hs):
        if h <= 2.5 * med:
            d.polygon([tuple(p) for p in ln["polygon"]], fill=255)
    return np.asarray(m) > 0, LINE_TO_TEXT * med


def line_crop(image, ln: dict, pad: int = 3, min_height: int = 48):
    """Ảnh một dòng: cắt theo khung, tô trắng ngoài đa giác (chặn chân / đầu chữ dòng bên cạnh), phóng nếu thấp."""
    from PIL import Image, ImageDraw

    x1, y1, x2, y2 = ln["bbox"]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(image.width, x2 + pad), min(image.height, y2 + pad)
    crop = image.crop((x1, y1, x2, y2))
    m = Image.new("L", crop.size, 0)
    ImageDraw.Draw(m).polygon([(x - x1, y - y1) for x, y in ln["polygon"]], fill=255)
    out = Image.new("RGB", crop.size, (255, 255, 255))
    out.paste(crop, mask=m)
    if out.height < min_height:
        k = min(3.0, min_height / max(1, out.height))
        out = out.resize((int(out.width * k), int(out.height * k)), Image.BICUBIC)
    return out
