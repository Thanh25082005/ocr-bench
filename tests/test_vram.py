import pytest

from ocrbench.config import load_config
from ocrbench.vram import make_plan, need_gb, report_plans

T4X2 = [15.0, 15.0]


def test_small_model_fits_one_gpu():
    assert make_plan("m", 2.2, T4X2).recommended.code == "A"


def test_8b_model_needs_two_gpus_in_fp16():
    p = make_plan("amad", 8.29, T4X2)
    assert p.recommended.code == "B" and not p.options[0].fits


def test_8b_on_single_gpu_falls_to_quantization():
    assert make_plan("amad", 8.29, [15.0]).recommended.code == "D"


def test_long_docs_need_more():
    assert need_gb(4.4, None, True) > need_gb(4.4, None, False)
    assert make_plan("qwen4b", 4.4, T4X2, long_docs=True).recommended.code in ("A", "B")


def test_huge_model_goes_to_4bit_or_offload():
    assert make_plan("big", 32, T4X2).recommended.code in ("F", "G")
    assert make_plan("huge", 70, T4X2).recommended.code == "H"


def test_report_suggests_renamed_variant(capsys):
    cfg = load_config("configs/kaggle_example.yaml")
    specs = [cfg.model("amad_vlm6"), cfg.model("tesseract")]
    report_plans(cfg, specs, [15.0], False, params_lookup=lambda mid: 8.29)
    out = capsys.readouterr().out
    assert "amad_vlm6__8bit" in out and "BẮT ĐẦU TỪ ĐÂY" in out and "không cần kiểm tra VRAM" in out


def test_declared_quantization_is_respected(capsys):
    p = make_plan("amad4", 8.29, T4X2, min_quant="4bit")
    assert {o.code for o in p.options} == {"F", "G", "H"} and p.recommended.code == "F"
    cfg = load_config("configs/kaggle_example.yaml")
    report_plans(cfg, [cfg.model("amad_vlm6_4bit")], T4X2, False, params_lookup=lambda mid: 8.29)
    assert "Cấu hình hiện tại đã vừa" in capsys.readouterr().out
