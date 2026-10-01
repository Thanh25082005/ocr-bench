"""Vòng /goal: hàng đợi → chạy → benchmark → đẩy → xóa trọng số → GOAL: ĐẠT; khóa chống sửa."""

import subprocess

import pytest
import yaml

from ocrbench import goal
from ocrbench.benchmark import write_benchmark
from ocrbench.cli import main
from ocrbench.config import load_config
from ocrbench.status import push_status
from tests.test_pipeline import project  # noqa: F401


@pytest.fixture
def goal_env(project, tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    cfg = yaml.safe_load((project / "config.yaml").read_text())
    cfg["status"] = {"push": True, "remote": str(remote)}
    (project / "config.yaml").write_text(yaml.safe_dump(cfg))
    queue = tmp_path / "queue.yaml"
    queue.write_text(yaml.safe_dump({"split": "dev", "max_skips": 1, "max_error_rate": 0.01,
                                     "queue": ["oracle", "noisy", "broken"]}))
    monkeypatch.setattr(goal, "QUEUE_FILE", queue)
    monkeypatch.setattr(goal, "LOCK_FILE", tmp_path / "GOAL_LOCK.sha256")
    monkeypatch.setattr(goal, "CHECK_GIT", False)
    c = load_config(project / "config.yaml")
    goal.write_lock(c)
    return project, str(project / "config.yaml"), queue


def _state(cfg_path, name):
    return next(s for s in goal.evaluate(load_config(cfg_path))[0] if s.name == name).state


def test_full_goal_loop(goal_env, capsys):
    project, cfg_path, _ = goal_env
    text, ok = goal.report(load_config(cfg_path))
    assert not ok and "VIỆC TIẾP THEO (oracle)" in text and "GOAL: CHƯA ĐẠT" in text

    main(["run", "--config", cfg_path, "--models", "oracle,noisy", "--no-score"])
    assert _state(cfg_path, "oracle") == "CHẠY XONG, CHƯA GHI BENCHMARK"

    write_benchmark(load_config(cfg_path))
    bench = (project / "BENCHMARK.md").read_text(encoding="utf-8")
    assert "| oracle |" in bench and "| noisy |" in bench
    assert bench.index("| oracle |") < bench.index("| noisy |")  # xếp theo CER
    assert _state(cfg_path, "oracle") == "ĐÃ GHI BENCHMARK, CHƯA LÊN GITHUB"

    push_status(load_config(cfg_path), note="benchmark")
    assert _state(cfg_path, "oracle") == "HOÀN THÀNH"  # model không phải Hugging Face: không có trọng số để xóa

    # model hỏng: chạy (qua tiến trình con để có run.log làm bằng chứng) rồi ghi vào danh sách bỏ qua
    with pytest.raises(SystemExit):
        main(["run", "--config", cfg_path, "--models", "broken", "--no-score"])
    (project / "goal_skips.yaml").write_text(yaml.safe_dump([{
        "model": "broken", "reason_code": "LOAD_FAILED_3X", "detail": "3 lần sửa đều lỗi", "evidence_variant": "broken"}]))
    write_benchmark(load_config(cfg_path))
    push_status(load_config(cfg_path), note="bỏ qua broken")
    text, ok = goal.report(load_config(cfg_path))
    assert ok and "GOAL: ĐẠT" in text, text
    assert "Model bị bỏ qua" in (project / "BENCHMARK.md").read_text(encoding="utf-8")


def test_invalid_skip_is_rejected(goal_env):
    project, cfg_path, _ = goal_env
    (project / "goal_skips.yaml").write_text(yaml.safe_dump([{"model": "broken", "reason_code": "KHO_QUA"}]))
    text, ok = goal.report(load_config(cfg_path))
    assert not ok and "reason_code không hợp lệ" in text


def test_tampering_breaks_lock(goal_env):
    project, cfg_path, queue = goal_env
    q = yaml.safe_load(queue.read_text())
    q["queue"].remove("broken")  # agent tự bỏ model khó khỏi hàng đợi
    queue.write_text(yaml.safe_dump(q))
    text, ok = goal.report(load_config(cfg_path))
    assert not ok and "KHÓA" in text and "queue.yaml đã bị thay đổi" in text


def test_clean_cache_only_after_push(goal_env, monkeypatch):
    project, cfg_path, _ = goal_env
    deleted = []

    class Rev:
        commit_hash = "abc"

    class Repo:
        repo_id = "org/model-x"
        revisions = [Rev()]

    class Strategy:
        expected_freed_size_str = "4.2G"

        def execute(self):
            deleted.append(True)

    class Info:
        repos = [Repo()]

        def delete_revisions(self, *h):
            return Strategy()

    monkeypatch.setattr(goal, "hf_repo_ids", lambda cfg, name: {"org/model-x"} if name == "oracle" else set())
    monkeypatch.setattr(goal, "cached_hf_repos", lambda ids: sorted(ids))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: Info())

    main(["run", "--config", cfg_path, "--models", "oracle", "--inline", "--no-score"])
    with pytest.raises(RuntimeError, match="Từ chối xóa"):
        goal.clean_cache(load_config(cfg_path), "oracle")
    write_benchmark(load_config(cfg_path))
    push_status(load_config(cfg_path))
    assert _state(cfg_path, "oracle") == "ĐÃ LÊN GITHUB, CHƯA XÓA TRỌNG SỐ"
    assert "4.2G" in goal.clean_cache(load_config(cfg_path), "oracle") and deleted
