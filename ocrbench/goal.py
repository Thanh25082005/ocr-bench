"""Chế độ /goal: chạy lần lượt một hàng đợi model, mỗi model xong thì ghi benchmark, đẩy lên GitHub, xóa
trọng số khỏi ổ đĩa rồi sang model tiếp theo.

- Hàng đợi:      goal/queue.yaml (trong repo, bị khóa sha256).
- Bỏ qua model:  WORK/goal_skips.yaml (agent ghi, có lý do và bằng chứng; tối đa `max_skips`).
- Kiểm tra:      `ocrbench goal-check` in trạng thái từng model, VIỆC TIẾP THEO, và `GOAL: ĐẠT` khi xong.
- Khóa:          goal/GOAL_LOCK.sha256 — sha256 của code chấm điểm, hàng đợi và dấu vân tay bộ test.
                 Chỉ con người được tạo lại (`ocrbench goal-check --write-lock`).

Một model (mục trong hàng đợi, vd. `baseer`) được tính bằng mọi mục config tên `baseer` hoặc `baseer__...`
(bản đã giảm VRAM, bản cho tài liệu dài...).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config
from .dataset import fingerprint, load_manifest, sample_per_category
from .worker import load_predictions

CODE_DIR = Path(__file__).resolve().parents[1]
QUEUE_FILE = CODE_DIR / "goal" / "queue.yaml"
LOCK_FILE = CODE_DIR / "goal" / "GOAL_LOCK.sha256"
LOCKED_FILES = [
    "goal/queue.yaml",
    "ocrbench/goal.py",
    "ocrbench/benchmark.py",
    "ocrbench/metrics.py",
    "ocrbench/normalize.py",
    "ocrbench/score.py",
    "ocrbench/decide.py",
    "ocrbench/dataset.py",
    "ocrbench/worker.py",
    "ocrbench/status.py",
]
LONG_CATEGORIES = {"syn_longtable", "syn_longtext"}
SKIP_REASONS = {
    "OOM_ALL_LEVELS": "hết bộ nhớ ở mọi bậc của AGENT_PROMPT mục 6",
    "LOAD_FAILED_3X": "không nạp/chạy được sau 3 lần sửa theo AGENT_PROMPT mục 7",
    "GATED_OR_LICENSE": "model yêu cầu quyền truy cập / giấy phép mà con người chưa cấp",
    "DEPENDENCY_CONFLICT": "xung đột thư viện không giải quyết được bằng venv riêng",
}
HF_ADAPTERS = {"hf_vlm"}
CHECK_GIT = True  # kiểm tra code của tool không bị sửa (tắt trong test)


def goal_items(cfg: Config, q: dict) -> list:
    """Mẫu mà goal yêu cầu: mọi mẫu nhóm thường + (tập con cố định của) nhóm tài liệu dài.
    Tập con chọn bằng đúng hàm của `--per-category`, nên `ocrbench run --per-category N` chạy đúng các mẫu này."""
    items = load_manifest(cfg.dataset, split=q["split"])
    normal = [it for it in items if it.category not in LONG_CATEGORIES]
    long = [it for it in items if it.category in LONG_CATEGORIES]
    if q.get("long_per_category"):
        long = sample_per_category(long, int(q["long_per_category"]))
    return normal + long


def load_queue() -> dict:
    q = yaml.safe_load(QUEUE_FILE.read_text(encoding="utf-8"))
    q.setdefault("max_skips", 3)
    q.setdefault("max_error_rate", 0.01)
    q.setdefault("split", "dev")
    q.setdefault("long_per_category", None)  # tài liệu dài: chỉ chấm N mẫu cố định mỗi nhóm (None = tất cả)
    return q


# --- khóa -------------------------------------------------------------------------------


def compute_lock(cfg: Config) -> dict[str, str]:
    lock = {f: hashlib.sha256((QUEUE_FILE if f == "goal/queue.yaml" else CODE_DIR / f).read_bytes()).hexdigest()
            for f in LOCKED_FILES}
    lock["dataset_fingerprint"] = fingerprint(load_manifest(cfg.dataset))
    return lock


def write_lock(cfg: Config) -> Path:
    lock = compute_lock(cfg)
    LOCK_FILE.write_text("".join(f"{v}  {k}\n" for k, v in lock.items()))
    return LOCK_FILE


def check_lock(cfg: Config) -> list[str]:
    """Danh sách vi phạm (rỗng = hợp lệ)."""
    if not LOCK_FILE.exists():
        return [f"thiếu {LOCK_FILE.relative_to(CODE_DIR)}"]
    expected = dict(reversed(line.split(None, 1)) for line in LOCK_FILE.read_text().splitlines() if line.strip())
    expected = {k.strip(): v for k, v in expected.items()}
    actual = compute_lock(cfg)
    problems = [f"{k} đã bị thay đổi" for k in expected if actual.get(k) != expected[k]]
    if not CHECK_GIT:
        return problems
    try:
        dirty = subprocess.run(["git", "-C", str(CODE_DIR), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        if dirty:
            problems.append("code của tool có thay đổi chưa commit:\n      " + dirty.replace("\n", "\n      "))
    except (OSError, subprocess.SubprocessError):
        pass
    return problems


# --- trạng thái từng model ------------------------------------------------------------------


@dataclass
class ItemState:
    name: str
    variants: list[str] = field(default_factory=list)
    normal: tuple[str, int, int, int] | None = None  # (biến thể, số mẫu có kết quả, tổng, số lỗi)
    long: tuple[str, int, int, int] | None = None
    in_benchmark: bool = False
    pushed: bool = False
    cache_ids: list[str] = field(default_factory=list)  # repo HF còn trên ổ đĩa
    skipped: dict | None = None
    state: str = ""
    next_action: str = ""


def _variants(cfg: Config, name: str) -> list[str]:
    return [m.name for m in cfg.models if m.name == name or m.name.startswith(name + "__")]


def _coverage(cfg: Config, split: str, variant: str, items) -> tuple[str, int, int, int]:
    preds = load_predictions(Path(cfg.output_dir) / split / variant / "predictions.jsonl")
    ok = sum(1 for it in items if it.id in preds and not preds[it.id].get("error"))
    errors = sum(1 for it in items if it.id in preds and preds[it.id].get("error"))
    return variant, ok + errors, len(items), errors


def _best(cands):
    # nhiều mẫu có kết quả nhất, ít lỗi nhất
    return max(cands, key=lambda c: (c[1], -c[3])) if cands else None


def hf_repo_ids(cfg: Config, name: str) -> set[str]:
    ids = set()
    for v in _variants(cfg, name):
        spec = cfg.model(v)
        if spec.adapter in HF_ADAPTERS:
            ids |= {spec.params.get(k) for k in ("model_id", "adapter_id", "processor_id")} - {None}
    return ids


def cached_hf_repos(ids: set[str]) -> list[str]:
    if not ids:
        return []
    try:
        from huggingface_hub import scan_cache_dir

        return sorted(r.repo_id for r in scan_cache_dir().repos if r.repo_id in ids)
    except Exception:
        return []


def _results_repo_has(cfg: Config, rel: str, needle: str | None = None) -> bool:
    p = cfg.work_dir / ".ocrbench-results" / rel
    return p.exists() and (needle is None or needle in p.read_text(encoding="utf-8"))


def _pushed(cfg: Config) -> bool:
    """Commit mới nhất của repo kết quả cục bộ đã có trên GitHub chưa."""
    from .status import _git, _remote, results_repo

    repo = results_repo(cfg)
    if not (repo / ".git").exists():
        return False
    head = _git(repo, "rev-parse", "HEAD", check=False).stdout.strip()
    remote = _remote(cfg)
    out = _git(repo, "ls-remote", "origin", cfg.status.branch, remote=remote, check=False).stdout.split()
    return bool(head) and bool(out) and out[0] == head


def load_skips(cfg: Config) -> dict[str, dict]:
    p = cfg.work_dir / "goal_skips.yaml"
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    return {d["model"]: d for d in data if isinstance(d, dict) and "model" in d}


def _skip_problem(cfg: Config, s: dict) -> str | None:
    if s.get("reason_code") not in SKIP_REASONS:
        return f"reason_code không hợp lệ ({s.get('reason_code')}); chỉ nhận: {', '.join(SKIP_REASONS)}"
    if not str(s.get("detail", "")).strip():
        return "thiếu 'detail' mô tả các bước đã thử"
    ev = s.get("evidence_variant")
    if not ev:
        return "thiếu 'evidence_variant' (tên mục model có run.log chứng minh lỗi)"
    has_log = (Path(cfg.output_dir) / "dev" / ev / "run.log").exists() or \
        _results_repo_has(cfg, f"runs/dev/{ev}/run.log.tail")
    if s["reason_code"] != "GATED_OR_LICENSE" and not has_log:
        return f"không thấy run.log của '{ev}' làm bằng chứng"
    return None


def evaluate(cfg: Config) -> tuple[list[ItemState], list[str]]:
    q = load_queue()
    split = q["split"]
    items = goal_items(cfg, q)
    normal_items = [it for it in items if it.category not in LONG_CATEGORIES]
    long_items = [it for it in items if it.category in LONG_CATEGORIES]
    skips = load_skips(cfg)
    pushed = _pushed(cfg)
    bench = cfg.work_dir / "BENCHMARK.md"
    bench_text = bench.read_text(encoding="utf-8") if bench.exists() else ""
    problems = []
    states = []
    known = {m.name for m in cfg.models}
    for name in q["queue"]:
        st = ItemState(name)
        if name not in known and not _variants(cfg, name):
            st.state = "LỖI CONFIG"
            st.next_action = f"thêm mục model '{name}' vào config (lấy từ configs/kaggle_example.yaml)"
            states.append(st)
            continue
        st.variants = _variants(cfg, name)
        st.normal = _best([_coverage(cfg, split, v, normal_items) for v in st.variants if not v.endswith("__long")])
        st.long = _best([_coverage(cfg, split, v, long_items) for v in st.variants])
        st.in_benchmark = f"| {name} |" in bench_text
        st.cache_ids = cached_hf_repos(hf_repo_ids(cfg, name))
        if name in skips:
            prob = _skip_problem(cfg, skips[name])
            if prob:
                problems.append(f"goal_skips.yaml mục '{name}': {prob}")
            else:
                st.skipped = skips[name]
        st.pushed = pushed and st.in_benchmark and _results_repo_has(cfg, "BENCHMARK.md", f"| {name} |")
        _decide(st, q, cfg)
        states.append(st)
    n_skip = sum(1 for s in states if s.skipped)
    if n_skip > q["max_skips"]:
        problems.append(f"bỏ qua {n_skip} model, vượt giới hạn {q['max_skips']}: DỪNG và hỏi con người")
    return states, problems


def _complete(cov, max_err) -> bool:
    return cov is not None and cov[1] == cov[2] and cov[3] <= max_err * cov[2]


def _decide(st: ItemState, q: dict, cfg: Config) -> None:
    C = f"--config {cfg.path}"
    if st.skipped:
        st.state = f"BỎ QUA ({st.skipped['reason_code']})"
        if st.cache_ids:
            st.next_action = f"ocrbench clean-cache {C} --item {st.name} --force   # xóa trọng số model đã bỏ qua"
        return
    maxe = q["max_error_rate"]
    normal_ok, long_ok = _complete(st.normal, maxe), _complete(st.long, maxe)
    if st.normal is None or st.normal[1] == 0:
        st.state = "CHƯA BẮT ĐẦU"
        st.next_action = f"bước A–C của docs/GOAL_PROMPT.md cho '{st.name}' (vram → chạy thử → chạy nhóm thường)"
    elif not normal_ok:
        v, done, total, err = st.normal
        st.state = f"ĐANG CHẠY nhóm thường ({done}/{total}, lỗi {err})"
        cmd = "--retry-errors " if done == total else ""
        st.next_action = (f"ocrbench run {C} --models {v} --categories <nhóm thường> --gpus 0,1 {cmd}"
                          f"  # trong tmux; lỗi > {maxe:.0%} thì sửa theo AGENT_PROMPT mục 7").strip()
    elif not long_ok:
        done = st.long[1] if st.long else 0
        st.state = f"ĐANG CHẠY tài liệu dài ({done}/{st.long[2] if st.long else '?'})"
        n = q.get("long_per_category")
        sub = f" --per-category {n}" if n else ""
        st.next_action = (f"bước D của docs/GOAL_PROMPT.md cho '{st.name}': ocrbench run {C} --models <biến thể> "
                          f"--categories syn_longtable,syn_longtext{sub} --gpus 0,1")
    elif not st.in_benchmark:
        st.state = "CHẠY XONG, CHƯA GHI BENCHMARK"
        st.next_action = f"ocrbench benchmark {C} && ocrbench status {C} --push --note 'benchmark: {st.name}'"
    elif not st.pushed:
        st.state = "ĐÃ GHI BENCHMARK, CHƯA LÊN GITHUB"
        st.next_action = f"ocrbench status {C} --push --note 'benchmark: {st.name}'"
    elif st.cache_ids:
        st.state = "ĐÃ LÊN GITHUB, CHƯA XÓA TRỌNG SỐ"
        st.next_action = f"ocrbench clean-cache {C} --item {st.name}"
    else:
        st.state = "HOÀN THÀNH"


def report(cfg: Config) -> tuple[str, bool]:
    lock_problems = check_lock(cfg)
    states, problems = evaluate(cfg)
    q = load_queue()
    L = ["== Hàng đợi /goal (goal/queue.yaml) =="]
    for i, s in enumerate(states, 1):
        mark = "✔" if s.state == "HOÀN THÀNH" or s.state.startswith("BỎ QUA") and not s.cache_ids else "·"
        L.append(f" {mark} {i:>2}. {s.name:<22} {s.state}")
    done = sum(1 for s in states if s.state == "HOÀN THÀNH")
    skipped = sum(1 for s in states if s.skipped)
    L.append(f"\nHoàn thành {done}/{len(states)} · bỏ qua {skipped} (tối đa {q['max_skips']})")
    for p in lock_problems:
        L.append(f"✘ KHÓA: {p}")
    for p in problems:
        L.append(f"✘ {p}")
    pending = [s for s in states if not (s.state == "HOÀN THÀNH" or (s.skipped and not s.cache_ids))]
    ok = not pending and not lock_problems and not problems
    if not ok and not lock_problems and pending:
        s = pending[0]
        L.append(f"\nVIỆC TIẾP THEO ({s.name}): {s.next_action}")
    L.append("\nGOAL: ĐẠT" if ok else "\nGOAL: CHƯA ĐẠT" + (" (KHÓA KHÔNG HỢP LỆ — DỪNG và hỏi con người)" if lock_problems else ""))
    return "\n".join(L), ok


def clean_cache(cfg: Config, name: str, force: bool = False) -> str:
    states, _ = evaluate(cfg)
    st = next((s for s in states if s.name == name), None)
    if st is None:
        raise ValueError(f"'{name}' không có trong goal/queue.yaml")
    allowed = st.skipped or st.state in ("ĐÃ LÊN GITHUB, CHƯA XÓA TRỌNG SỐ", "HOÀN THÀNH")
    if not allowed and not force:
        raise RuntimeError(f"Từ chối xóa: '{name}' đang ở trạng thái '{st.state}'. Chỉ xóa khi benchmark của model "
                           "đã lên GitHub (goal-check báo 'ĐÃ LÊN GITHUB, CHƯA XÓA TRỌNG SỐ').")
    ids = set(hf_repo_ids(cfg, name))
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    revs = [rev.commit_hash for r in info.repos if r.repo_id in ids for rev in r.revisions]
    if not revs:
        return f"Không có trọng số nào của '{name}' trong cache Hugging Face."
    strategy = info.delete_revisions(*revs)
    freed = strategy.expected_freed_size_str
    strategy.execute()
    return f"Đã xóa {', '.join(sorted(ids & {r.repo_id for r in info.repos}))} — giải phóng {freed}."
