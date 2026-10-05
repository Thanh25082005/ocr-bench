"""Chạy toàn bộ pipeline (make-manifest -> run -> score) bằng model giả lập, không cần GPU."""

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from ocrbench.cli import main
from ocrbench.config import load_config
from ocrbench.dataset import ManifestError, load_manifest
from ocrbench.worker import load_predictions

SAMPLES = {
    "printed_en": [("invoice01__p1", "Total amount due: 1,250.00 USD"), ("letter02__p1", "Dear Sir, thank you.")],
    "printed_ar": [("hoso03__p1", "بسم الله الرحمن الرحيم"), ("hoso03__p2", "المبلغ الإجمالي ١٢٥٠ ريال")],
    "handwriting_ar": [("w05__l1", "كتب الطالب الدرس"), ("w06__l1", "ذهب إلى المدرسة")],
}
TABLE = ("table07__p1", "<table><tr><td>Item</td><td>السعر</td></tr><tr><td>Pen</td><td>٥</td></tr></table>")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    for cat, samples in SAMPLES.items():
        (data / cat).mkdir(parents=True)
        for name, text in samples:
            Image.new("RGB", (64, 32), "white").save(data / cat / f"{name}.png")
            (data / cat / f"{name}.txt").write_text(text, encoding="utf-8")
    (data / "tables").mkdir()
    Image.new("RGB", (64, 32), "white").save(data / "tables" / f"{TABLE[0]}.png")
    (data / "tables" / f"{TABLE[0]}.html").write_text(TABLE[1], encoding="utf-8")

    cfg = {
        "dataset": "data/manifest.jsonl",
        "output_dir": "runs",
        "models": [
            {"name": "oracle", "adapter": "dummy", "params": {"mode": "oracle"}},
            {"name": "noisy", "adapter": "dummy", "params": {"mode": "noisy", "noise": 0.2}},
            {"name": "looper", "adapter": "dummy", "params": {"mode": "loop"}},
            {"name": "broken", "adapter": "dummy", "params": {"mode": "fail"}, "enabled": False},
        ],
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # holdout_ratio=0 để mọi mẫu nằm ở dev
    main(["make-manifest", str(data), "-o", str(data / "manifest.jsonl"), "--holdout-ratio", "0"])
    return tmp_path


def test_manifest(project):
    items = load_manifest(project / "data/manifest.jsonl")
    assert len(items) == 7
    by_id = {it.id: it for it in items}
    assert by_id["tables/table07__p1"].gt_type == "table_html"
    # hai trang của cùng hồ sơ có cùng doc_id
    assert by_id["printed_ar/hoso03__p1"].doc_id == by_id["printed_ar/hoso03__p2"].doc_id


def test_manifest_split_is_by_document(tmp_path):
    data = tmp_path / "d" / "cat"
    data.mkdir(parents=True)
    for doc in range(40):
        for page in range(3):
            Image.new("RGB", (8, 8)).save(data / f"doc{doc}__p{page}.png")
            (data / f"doc{doc}__p{page}.txt").write_text("x")
    main(["make-manifest", str(tmp_path / "d"), "-o", str(tmp_path / "m.jsonl")])
    items = load_manifest(tmp_path / "m.jsonl")
    splits = {}
    for it in items:
        splits.setdefault(it.doc_id, set()).add(it.split)
    assert all(len(s) == 1 for s in splits.values())
    assert {"dev", "holdout"} == set().union(*splits.values())


def test_leakage_is_rejected(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    lines = [
        {"id": "a1", "image": "a.png", "category": "c", "gt": "x", "doc_id": "D", "split": "dev"},
        {"id": "a2", "image": "a.png", "category": "c", "gt": "x", "doc_id": "D", "split": "holdout"},
    ]
    (tmp_path / "m.jsonl").write_text("\n".join(json.dumps(r) for r in lines))
    with pytest.raises(ManifestError, match="rò rỉ"):
        load_manifest(tmp_path / "m.jsonl")


def test_run_and_report(project, capsys):
    cfg_path = str(project / "config.yaml")
    main(["run", "--config", cfg_path])
    out = project / "runs" / "dev"
    report = (out / "_report" / "report.md").read_text(encoding="utf-8")

    scores = {
        m: [json.loads(line) for line in (out / m / "scores.jsonl").read_text().splitlines()]
        for m in ("oracle", "noisy", "looper")
    }
    assert all(r["cer"] == 0 for r in scores["oracle"])
    table_row = next(r for r in scores["oracle"] if r["gt_type"] == "table_html")
    assert table_row["teds"] == pytest.approx(1.0)
    assert sum(r["cer"] for r in scores["noisy"]) > 0
    assert all("repetition" in r["flags"] for r in scores["looper"])
    # oracle phải đứng đầu bảng tổng quan
    overview = report.split("## Tổng quan")[1].split("##")[0]
    assert overview.index("oracle") < overview.index("noisy")
    assert (out / "_report" / "summary.csv").exists()
    # model bị tắt (enabled: false) không chạy
    assert not (out / "broken").exists()


def test_resume_skips_done(project):
    cfg_path = str(project / "config.yaml")
    main(["run", "--config", cfg_path, "--models", "oracle", "--limit", "3", "--no-score"])
    pred = project / "runs/dev/oracle/predictions.jsonl"
    assert len(pred.read_text().splitlines()) == 3
    main(["run", "--config", cfg_path, "--models", "oracle", "--no-score"])
    lines = pred.read_text().splitlines()
    assert len(lines) == 7  # chỉ chạy thêm 4 mẫu còn lại, không chạy lại 3 mẫu cũ


def test_failing_model_reports_errors(project):
    cfg_path = str(project / "config.yaml")
    with pytest.raises(SystemExit):
        main(["run", "--config", cfg_path, "--models", "broken", "--inline", "--no-score"])
    preds = load_predictions(project / "runs/dev/broken/predictions.jsonl")
    assert all(p["error"] for p in preds.values())


def test_changed_config_under_same_name_is_refused(project, capsys):
    cfg_path = str(project / "config.yaml")
    with pytest.raises(SystemExit):
        main(["run", "--config", cfg_path, "--models", "broken", "--inline", "--no-score"])
    cfg = yaml.safe_load((project / "config.yaml").read_text())
    cfg["models"][3]["params"]["mode"] = "oracle"  # sửa cấu hình nhưng giữ tên "broken"
    (project / "config.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit):
        main(["run", "--config", cfg_path, "--models", "broken", "--retry-errors", "--inline", "--no-score"])
    assert "TỪ CHỐI CHẠY" in capsys.readouterr().out
    # đặt tên mới thì chạy bình thường
    cfg["models"][3]["name"] = "broken__v2"
    (project / "config.yaml").write_text(yaml.safe_dump(cfg))
    main(["run", "--config", cfg_path, "--models", "broken__v2", "--inline", "--no-score"])
    assert not any(p["error"] for p in load_predictions(project / "runs/dev/broken__v2/predictions.jsonl").values())


def test_retry_errors_with_same_config(project):
    cfg_path = str(project / "config.yaml")
    main(["run", "--config", cfg_path, "--models", "oracle", "--inline", "--no-score"])
    pred = project / "runs/dev/oracle/predictions.jsonl"
    lines = [json.loads(x) for x in pred.read_text().splitlines()]
    lines[0].update(text="", error="CUDA out of memory (giả lập lỗi tạm thời)")
    pred.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n")
    main(["run", "--config", cfg_path, "--models", "oracle", "--retry-errors", "--inline", "--no-score"])
    assert not any(p["error"] for p in load_predictions(pred).values())


def test_report_warns_when_dataset_changed(project):
    cfg_path = str(project / "config.yaml")
    main(["run", "--config", cfg_path, "--models", "oracle", "--inline"])
    gt = next((project / "data/printed_en").glob("*.txt"))
    gt.write_text("đáp án đã bị sửa", encoding="utf-8")
    main(["score", "--config", cfg_path])
    assert "dữ liệu đã thay đổi" in (project / "runs/dev/_report/report.md").read_text(encoding="utf-8")


def test_holdout_requires_confirmation(project):
    with pytest.raises(SystemExit, match="holdout"):
        main(["run", "--config", str(project / "config.yaml"), "--split", "holdout"])


def test_parallel_slots(project):
    """--gpus với 2 slot: hai model chạy song song mà vẫn ra kết quả đầy đủ."""
    main(["run", "--config", str(project / "config.yaml"), "--models", "oracle,noisy", "--gpus", "0,1", "--no-score"])
    for m in ("oracle", "noisy"):
        assert len(load_predictions(project / f"runs/dev/{m}/predictions.jsonl")) == 7


def test_config_rejects_bad_fields(tmp_path):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"dataset": "m.jsonl", "models": [{"name": "a", "adapter": "dummy", "gpu": 1}]}))
    with pytest.raises(ValueError, match="gpu"):
        load_config(tmp_path / "c.yaml")


def test_per_category_sample_is_stable_and_stratified(project):
    from ocrbench.dataset import load_manifest

    m = project / "data/manifest.jsonl"
    a = load_manifest(m, per_category=1)
    b = load_manifest(m, per_category=1)
    assert [it.id for it in a] == [it.id for it in b]
    assert sorted(it.category for it in a) == sorted({it.category for it in load_manifest(m)})

    main(["run", "--config", str(project / "config.yaml"), "--models", "oracle", "--per-category", "1"])
    assert len(load_predictions(project / "runs/dev/oracle/predictions.jsonl")) == 4


def test_single_model_is_split_across_gpus(project, capsys):
    """Một model 1-GPU với --gpus 0,1: 2 tiến trình, mỗi tiến trình một nửa mẫu, chung một file kết quả."""
    import json as _json
    from ocrbench.worker import shard_of
    cfg_path = str(project / "config.yaml")
    main(["run", "--config", cfg_path, "--models", "oracle", "--gpus", "0,1", "--no-score"])
    out = capsys.readouterr().out
    assert "chia mẫu cho 2 GPU" in out and out.count("✔ oracle: kết thúc") == 1
    lines = (project / "runs/dev/oracle/predictions.jsonl").read_text().splitlines()
    ids = [_json.loads(x)["id"] for x in lines]
    assert len(ids) == 7 and len(set(ids)) == 7  # đủ, không trùng, không hỏng dòng
    meta = _json.loads((project / "runs/dev/oracle/meta.json").read_text())
    assert sorted(s["shard"] for s in meta["sessions"]) == ["1/2", "2/2"]
    assert {shard_of(i, 2) for i in ids} == {0, 1}
    # chạy lại: không phần nào phải chạy thêm
    main(["run", "--config", cfg_path, "--models", "oracle", "--gpus", "0,1", "--no-score"])
    assert len((project / "runs/dev/oracle/predictions.jsonl").read_text().splitlines()) == 7


def test_holdout_excluded_items_skipped_only_in_holdout(tmp_path, monkeypatch):
    """Mẫu holdout đã bị xem: không tính khi chấm holdout; dev và dấu vân tay không đổi."""
    import json

    from ocrbench import dataset as D

    (tmp_path / "a.png").write_bytes(b"")
    rows = [{"id": f"c/{i}", "image": "a.png", "category": "c", "gt": "x", "doc_id": f"d{i}",
             "split": "holdout" if i < 3 else "dev"} for i in range(5)]
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    fp = D.fingerprint(D.load_manifest(m))
    ex = tmp_path / "ex.yaml"
    ex.write_text("excluded:\n  - id: c/1\n    reason: test\n")
    monkeypatch.setattr(D, "HOLDOUT_EXCLUDED_FILE", ex)
    assert [it.id for it in D.load_manifest(m, split="holdout")] == ["c/0", "c/2"]
    assert len(D.load_manifest(m, split="dev")) == 2
    assert D.fingerprint(D.load_manifest(m)) == fp


def test_real_holdout_exclusions_are_valid_ids():
    from ocrbench.dataset import holdout_excluded

    ids = holdout_excluded()
    assert "pub_tables_en/pubtabnet__552595" in ids and all("/" in i for i in ids)
