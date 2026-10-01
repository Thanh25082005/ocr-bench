"""status.md + đẩy lên git + khôi phục, dùng một repo git cục bộ thay cho GitHub."""

import subprocess

import yaml

from ocrbench.cli import main
from ocrbench.config import load_config
from ocrbench.status import build_status, push_status, restore
from ocrbench.worker import load_predictions
from tests.test_pipeline import project  # noqa: F401  (dùng lại fixture dữ liệu mẫu)


def _bare_repo(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _enable_push(project, remote, every_min=30):
    cfg = yaml.safe_load((project / "config.yaml").read_text())
    cfg["status"] = {"push": True, "remote": str(remote), "every_min": every_min}
    (project / "config.yaml").write_text(yaml.safe_dump(cfg))
    return str(project / "config.yaml")


def _branch_files(remote):
    out = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "results"],
                         capture_output=True, text=True, check=True).stdout
    return set(out.split())


def test_status_contents(project):
    cfg_path = str(project / "config.yaml")
    main(["run", "--config", cfg_path, "--models", "oracle,noisy", "--inline", "--no-score"])
    md = build_status(load_config(cfg_path), running=["noisy"])
    assert "| oracle |" in md and "▶ đang chạy" in md and "7/7" in md and "0.0%" in md


def test_each_finished_model_is_pushed(project, tmp_path):
    remote = _bare_repo(tmp_path)
    cfg_path = _enable_push(project, remote)
    main(["run", "--config", cfg_path, "--models", "oracle,noisy", "--no-score"])  # chạy qua tiến trình con
    files = _branch_files(remote)
    assert {"status.md", "config.yaml", "runs/dev/oracle/predictions.jsonl",
            "runs/dev/noisy/predictions.jsonl", "runs/dev/noisy/meta.json"} <= files
    log = subprocess.run(["git", "--git-dir", str(remote), "log", "--format=%s", "results"],
                         capture_output=True, text=True).stdout
    assert "oracle kết thúc" in log and "noisy kết thúc" in log


def test_restore_after_crash(project, tmp_path):
    remote = _bare_repo(tmp_path)
    cfg_path = _enable_push(project, remote)
    main(["run", "--config", cfg_path, "--models", "oracle", "--inline", "--no-score"])
    (project / "EXPERIMENTS.md").write_text("# nhật ký\n\n## mục 1\n- Việc tiếp theo: giai đoạn 2\n")
    push_status(load_config(cfg_path), note="ghi nhật ký")

    # "server sập": mất hết thư mục làm việc, chỉ còn bộ dữ liệu và một config mới tinh
    import shutil
    shutil.rmtree(project / "runs")
    shutil.rmtree(project / ".ocrbench-results")
    (project / "EXPERIMENTS.md").unlink()

    restored = restore(load_config(cfg_path))
    assert len(load_predictions(project / "runs/dev/oracle/predictions.jsonl")) == 7
    assert "giai đoạn 2" in (project / "EXPERIMENTS.md").read_text()
    assert restored
    # chạy lại không phải chạy lại mẫu nào
    main(["run", "--config", cfg_path, "--models", "oracle", "--inline", "--no-score"])
    assert len((project / "runs/dev/oracle/predictions.jsonl").read_text().splitlines()) == 7


def test_push_failure_does_not_stop_run(project, tmp_path, capsys):
    cfg_path = _enable_push(project, tmp_path / "khong-ton-tai.git")
    main(["run", "--config", cfg_path, "--models", "oracle", "--inline", "--no-score"])
    assert "ĐẨY STATUS THẤT BẠI" in capsys.readouterr().out
    assert len(load_predictions(project / "runs/dev/oracle/predictions.jsonl")) == 7
    assert (project / "runs/status.md").exists()
