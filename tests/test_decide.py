import yaml

from ocrbench.cli import main
from ocrbench.config import load_config
from ocrbench.decide import decide


def test_decide_screening_and_final(tmp_path):
    from tests.test_pipeline import SAMPLES, TABLE  # dùng lại dữ liệu mẫu
    from PIL import Image

    data = tmp_path / "data"
    for cat, samples in SAMPLES.items():
        (data / cat).mkdir(parents=True)
        for name, text in samples * 1:
            Image.new("RGB", (32, 16)).save(data / cat / f"{name}.png")
            (data / cat / f"{name}.txt").write_text(text, encoding="utf-8")
    (data / "tables").mkdir()
    Image.new("RGB", (32, 16)).save(data / "tables" / f"{TABLE[0]}.png")
    (data / "tables" / f"{TABLE[0]}.html").write_text(TABLE[1], encoding="utf-8")
    main(["make-manifest", str(data), "-o", str(data / "manifest.jsonl"), "--holdout-ratio", "0"])
    cfg = {"dataset": "data/manifest.jsonl", "output_dir": "runs", "models": [
        {"name": "good", "adapter": "dummy", "params": {"mode": "oracle"}},
        {"name": "looper", "adapter": "dummy", "params": {"mode": "loop"}},
        {"name": "bad", "adapter": "dummy", "params": {"mode": "noisy", "noise": 0.6}},
    ]}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    main(["run", "--config", str(tmp_path / "c.yaml"), "--inline", "--no-score"])
    c = load_config(tmp_path / "c.yaml")

    dec = decide(c, "dev", "screening")
    for cat, r in dec["categories"].items():
        st = {e["model"]: e for e in r["models"]}
        assert st["looper"]["status"] == "loại" and "không ổn định" in st["looper"]["reason"]
        assert "good" in r["finalists"]
    tbl = dec["categories"]["tables"]
    assert "cell_exact" in tbl["metric"]

    fin = decide(c, "dev", "final")
    assert all(r["winner"] == "good" for r in fin["categories"].values())
