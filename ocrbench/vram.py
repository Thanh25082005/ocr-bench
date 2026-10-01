"""`ocrbench vram`: kiểm tra GPU, ước lượng VRAM từng model cần, và chọn cấu hình chạy được.

Thứ tự thử (ít mất chất lượng nhất trước), dừng ở bậc đầu tiên vừa:

    A  fp16, 1 GPU
    B  fp16, chia 2 GPU                  (không mất chất lượng)
    C  fp16, chia 2 GPU + giảm ảnh       (có thể đọc sai chữ nhỏ)
    D  8-bit, 1 GPU                      (mất rất ít)
    E  8-bit, chia 2 GPU
    F  4-bit, 1 GPU                      (mất rõ hơn: bắt buộc đo)
    G  4-bit, chia 2 GPU
    H  4-bit + đẩy bớt lớp sang RAM      (rất chậm, chỉ để lấy số liệu tham khảo)

Đây là ƯỚC LƯỢNG để chọn bậc bắt đầu. Nếu chạy thật vẫn hết bộ nhớ thì xuống bậc kế tiếp.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass, field

GB_PER_BILLION = {None: 2.0, "8bit": 1.1, "4bit": 0.6}  # bộ nhớ trọng số cho mỗi tỉ tham số
OVERHEAD_GB = {"normal": 2.5, "long": 7.0}  # ảnh, bộ nhớ đệm, kích hoạt; "long" = tài liệu 2–3 trang
SMALL_IMAGES_SAVING_GB = 1.5  # giảm ảnh xuống ~1600 px tiết kiệm khoảng chừng này
USABLE = 0.9  # chỉ dùng 90% VRAM trống, chừa chỗ cho phân mảnh bộ nhớ
SPLIT_EFFICIENCY = 0.92  # chia 2 GPU không chia đều tuyệt đối

LIGHT_ADAPTERS = {"dummy", "tesseract", "easyocr", "paddleocr", "paddleocr_vl", "surya"}
EXTERNAL_ADAPTERS = {"openai_api", "command"}


@dataclass
class Option:
    code: str
    label: str
    changes: dict
    need_gb: float
    fits: bool
    note: str = ""


@dataclass
class Plan:
    model: str
    params_b: float | None
    free_gb: list[float]
    options: list[Option] = field(default_factory=list)

    @property
    def recommended(self) -> Option | None:
        return next((o for o in self.options if o.fits), None)


def free_vram_gb() -> list[float]:
    """VRAM còn trống của từng GPU (GB). Rỗng nếu không có GPU / không có nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    free = []
    for line in out.splitlines():
        total, used = (float(x) for x in line.split(","))
        free.append(round((total - used) / 1024, 1))
    return free


def param_count_b(model_id: str) -> float | None:
    """Số tỉ tham số, lấy từ thông tin safetensors trên Hugging Face."""
    url = f"https://huggingface.co/api/models/{model_id}?expand[]=safetensors"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        return round(data["safetensors"]["total"] / 1e9, 2)
    except Exception:
        return None


def need_gb(params_b: float, quant: str | None, long_docs: bool) -> float:
    return round(params_b * GB_PER_BILLION[quant] + OVERHEAD_GB["long" if long_docs else "normal"], 1)


QUANT_LEVEL = {None: 0, "8bit": 1, "4bit": 2}
CODE_OF = {(None, 1): "A", (None, 2): "B", ("8bit", 1): "D", ("8bit", 2): "E", ("4bit", 1): "F", ("4bit", 2): "G"}


def make_plan(model: str, params_b: float, free: list[float], long_docs: bool = False,
              min_quant: str | None = None) -> Plan:
    """min_quant: model đã khai báo sẵn mức nén (vd. bản 4-bit để so sánh) thì không đề xuất bậc nhẹ nén hơn."""
    one = max(free) * USABLE if free else 0.0
    two = sum(sorted(free, reverse=True)[:2]) * USABLE * SPLIT_EFFICIENCY if len(free) >= 2 else 0.0
    small_img = {"max_image_side": 1600}
    p = Plan(model, params_b, free)
    for code, label, quant, gpus, changes, saving in [
        ("A", "fp16, 1 GPU", None, 1, {}, 0),
        ("B", "fp16, chia 2 GPU", None, 2, {"gpus": 2}, 0),
        ("C", "fp16, chia 2 GPU + giảm ảnh (1600 px)", None, 2, {"gpus": 2, **small_img}, SMALL_IMAGES_SAVING_GB),
        ("D", "8-bit, 1 GPU", "8bit", 1, {"quantization": "8bit"}, 0),
        ("E", "8-bit, chia 2 GPU", "8bit", 2, {"quantization": "8bit", "gpus": 2}, 0),
        ("F", "4-bit, 1 GPU", "4bit", 1, {"quantization": "4bit"}, 0),
        ("G", "4-bit, chia 2 GPU", "4bit", 2, {"quantization": "4bit", "gpus": 2}, 0),
    ]:
        if QUANT_LEVEL[quant] < QUANT_LEVEL[min_quant]:
            continue
        need = round(need_gb(params_b, quant, long_docs) - saving, 1)
        cap = one if gpus == 1 else two
        p.options.append(Option(code, label, changes, need, cap > 0 and need <= cap))
    p.options.append(Option(
        "H", "4-bit + đẩy bớt lớp sang RAM", {"quantization": "4bit",
                                             "model_kwargs": {"max_memory": {0: f"{int(one)}GiB", "cpu": "24GiB"}}},
        need_gb(params_b, "4bit", long_docs), bool(free), "rất chậm; chỉ chạy tập con để lấy số liệu tham khảo"))
    return p


def _variant_yaml(spec, opt: Option) -> str:
    """Mục config cho bản đã giảm, tên mới theo luật 'đổi tham số thì đổi tên'."""
    import yaml

    params = dict(spec.params)
    params.pop("quantization", None)  # mức nén lấy theo bậc được chọn
    gpus = opt.changes.get("gpus", 1)
    for k, v in opt.changes.items():
        if k == "gpus":
            continue
        if k == "model_kwargs":
            params["model_kwargs"] = {**params.get("model_kwargs", {}), **v}
        else:
            params[k] = v
    suffix = {"A": "", "B": "__2gpu", "C": "__2gpu_side1600", "D": "__8bit", "E": "__8bit_2gpu", "F": "__4bit",
              "G": "__4bit_2gpu", "H": "__4bit_offload"}[opt.code]
    entry = {"name": spec.name + suffix, "adapter": spec.adapter}
    if gpus > 1:
        entry["gpus"] = gpus
    entry["params"] = params
    return yaml.safe_dump([entry], allow_unicode=True, sort_keys=False)


def report_plans(cfg, specs, free: list[float], long_docs: bool, params_lookup=param_count_b) -> int:
    """In kế hoạch cho từng model. Trả về số model không có cấu hình nào vừa (trừ bậc H)."""
    print(f"GPU trống: {free if free else 'KHÔNG TÌM THẤY GPU'} (GB)  ·  chế độ: "
          f"{'tài liệu dài (2–3 trang)' if long_docs else 'tài liệu thường'}\n")
    impossible = 0
    for spec in specs:
        if spec.adapter in LIGHT_ADAPTERS:
            print(f"== {spec.name} ({spec.adapter}): model nhỏ, không cần kiểm tra VRAM.\n")
            continue
        if spec.adapter in EXTERNAL_ADAPTERS:
            print(f"== {spec.name} ({spec.adapter}): chạy ngoài tool (server/script), tự kiểm tra VRAM của nơi chạy.\n")
            continue
        model_id = spec.params.get("model_id")
        params_b = params_lookup(model_id) if model_id else None
        if params_b is None:
            print(f"== {spec.name}: KHÔNG lấy được số tham số của '{model_id}' từ Hugging Face. "
                  "Xem trang model, rồi tính tay theo docs trong AGENT_PROMPT.md mục 6.\n")
            continue
        cur_quant = spec.params.get("quantization")
        plan = make_plan(spec.name, params_b, free, long_docs, min_quant=cur_quant)
        print(f"== {spec.name} ({model_id}): {params_b} tỉ tham số"
              + (f"; config hiện tại đang dùng quantization={cur_quant}" if cur_quant else ""))
        print(f"   {'Bậc':<4}{'Cấu hình':<40}{'Cần (GB)':>9}  Vừa?")
        rec = plan.recommended
        for o in plan.options:
            mark = "  ← BẮT ĐẦU TỪ ĐÂY" if o is rec else ""
            print(f"   {o.code:<4}{o.label:<40}{o.need_gb:>9}  {'CÓ' if o.fits else 'không'}{mark}"
                  + (f"  ({o.note})" if o.note else ""))
        if rec is None:
            print("   → Không có cấu hình nào vừa. Đặt enabled: false và ghi lý do vào nhật ký.\n")
            impossible += 1
            continue
        if rec.code == "H":
            impossible += 1
        if rec.code == CODE_OF.get((cur_quant, min(spec.gpus, 2))) and "max_image_side" not in rec.changes:
            print("   → Cấu hình hiện tại đã vừa. Không cần thay đổi.\n")
        else:
            print(f"   → Thêm mục sau vào config (tên mới), chạy thử --per-category 1. Hết bộ nhớ thì xuống bậc kế tiếp:")
            for line in _variant_yaml(spec, rec).splitlines():
                print("     " + line)
            print()
    return impossible
