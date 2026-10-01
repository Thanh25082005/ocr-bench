"""status.md + đẩy kết quả lên GitHub, để server sập lúc nào cũng không mất tiến độ.

- `write_status`: ghi OUT/status.md (tiến độ + kết quả nhanh của từng model).
- `push_status`: chép status.md, config, nhật ký, báo cáo và kết quả thô (predictions.jsonl, meta.json)
  vào nhánh `results` của repo rồi đẩy lên GitHub.
- `restore`: phiên mới thì kéo nhánh `results` về và chép kết quả vào chỗ cũ, để `ocrbench run` chạy tiếp.

Token GitHub (quyền Contents: Read and write) lấy từ biến môi trường GITHUB_TOKEN, hoặc file
~/.config/ocrbench/github_token. Token chỉ được truyền qua header cho từng lệnh git, không ghi vào .git/config.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from .config import Config
from .dataset import fingerprint, load_manifest
from .normalize import normalize
from .worker import load_predictions

TOKEN_FILE = Path.home() / ".config" / "ocrbench" / "github_token"
CODE_DIR = Path(__file__).resolve().parents[1]
RUN_FILES = ("predictions.jsonl", "meta.json")
REPORT_GLOBS = ("report.md", "summary.csv", "decision_*.md", "decision_*.json")
WORK_FILES = ("EXPERIMENTS.md", "FINAL_REPORT.md")


# --- status.md --------------------------------------------------------------------------


def _git_out(*args, cwd=CODE_DIR) -> str:
    try:
        return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _model_state(meta: dict, running: set[str], name: str) -> str:
    if name in running:
        return "▶ đang chạy"
    sessions = meta.get("sessions") or []
    if not sessions:
        return "⚠ dừng giữa chừng (chưa kết thúc lần nào, có thể do sập)"
    return "■ đã dừng" if sessions[-1].get("finished") else "⚠ dừng giữa chừng"


def build_status(cfg: Config, running: list[str] | None = None, note: str | None = None) -> str:
    from .worker import _gpu_names

    running_set = set(running or [])
    all_items = load_manifest(cfg.dataset)
    now = datetime.now(timezone.utc)
    L = ["# Trạng thái ocrbench", "",
         f"- Cập nhật: **{now:%Y-%m-%d %H:%M} UTC**" + (f" · {note}" if note else ""),
         f"- Máy: `{platform.node()}` · GPU: {', '.join(_gpu_names()) or 'không thấy'}",
         f"- Code: `{_git_out('rev-parse', '--short', 'HEAD') or '?'}` · Dấu vân tay dữ liệu: `{fingerprint(all_items)}`",
         f"- Đang chạy: {', '.join(sorted(running_set)) or '(không có)'}", ""]

    for split_dir in sorted(p for p in Path(cfg.output_dir).glob("*") if p.is_dir() and p.name in ("dev", "holdout")):
        split = split_dir.name
        items = [it for it in all_items if it.split == split]
        by_id = {it.id: it for it in items}
        cats = sorted({it.category for it in items})
        models = sorted(p.name for p in split_dir.iterdir() if (p / "predictions.jsonl").exists())
        if not models:
            continue
        L += [f"## Split `{split}` ({len(items)} mẫu)", "",
              "| Model | Trạng thái | Đã chạy | Lỗi | CER norm (TB các mẫu đã chạy) | s/mẫu | VRAM đỉnh | Lần chạy cuối |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
        per_cat: dict[str, dict[str, list[float]]] = {}
        for m in models:
            preds = load_predictions(split_dir / m / "predictions.jsonl")
            meta_p = split_dir / m / "meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            cers, lat, errors = [], [], 0
            per_cat[m] = {}
            for pid, p in preds.items():
                it = by_id.get(pid)
                if it is None:
                    continue
                if p.get("error"):
                    errors += 1
                ref, hyp = normalize(it.gt, cfg.normalization), normalize(p.get("text") or "", cfg.normalization)
                cer = Levenshtein.distance(ref, hyp) / max(len(ref), 1)
                cers.append(cer)
                per_cat[m].setdefault(it.category, []).append(cer)
                if p.get("latency_s") is not None:
                    lat.append(p["latency_s"])
            last = (meta.get("sessions") or [{}])[-1]
            vram = meta.get("peak_vram_mib")
            L.append(f"| {m} | {_model_state(meta, running_set, m)} | {len(cers)}/{len(items)} | {errors} "
                     f"| {statistics.fmean(cers):.1%} | {statistics.fmean(lat) if lat else 0:.2f} "
                     f"| {f'{vram / 1024:.1f} GB' if vram else '—'} | {last.get('finished') or last.get('started') or '—'} |"
                     if cers else f"| {m} | {_model_state(meta, running_set, m)} | 0/{len(items)} | {errors} | — | — | — | — |")
        L += ["", "CER norm theo nhóm (`số mẫu đã chạy: CER`):", "",
              "| Nhóm | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
        for c in cats:
            cells = []
            for m in models:
                v = per_cat[m].get(c)
                cells.append(f"{len(v)}: {statistics.fmean(v):.1%}" if v else "—")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Số liệu ở đây là CER tính nhanh trên các mẫu đã chạy (chưa có TEDS / ô đúng vị trí). "
          "Số liệu chính thức: `runs/<split>/_report/report.md` và `decision_*.md`.", ""]
    exp = cfg.work_dir / "EXPERIMENTS.md"
    if exp.exists():
        tail = exp.read_text(encoding="utf-8").strip().split("\n## ")[-1]
        L += ["## Mục nhật ký gần nhất (EXPERIMENTS.md)", "", "## " + tail if not tail.startswith("#") else tail, ""]
    return "\n".join(L)


def write_status(cfg: Config, running: list[str] | None = None, note: str | None = None) -> Path:
    path = Path(cfg.output_dir) / "status.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_status(cfg, running, note), encoding="utf-8")
    return path


# --- git -------------------------------------------------------------------------------


def github_token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok and TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text().strip()
    return tok or None


def _remote(cfg: Config) -> str:
    remote = cfg.status.remote or _git_out("remote", "get-url", "origin")
    if not remote:
        raise RuntimeError("không biết đẩy kết quả lên đâu: đặt status.remote trong config")
    return remote


def _git(repo: Path, *args, remote: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["git"]
    tok = github_token()
    if tok and remote and remote.startswith("https://"):
        basic = base64.b64encode(f"x-access-token:{tok}".encode()).decode()
        cmd += ["-c", f"http.extraHeader=Authorization: Basic {basic}"]
    proc = subprocess.run([*cmd, "-C", str(repo), *args], capture_output=True, text=True, timeout=600)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} lỗi: {(proc.stderr or proc.stdout).strip()[-400:]}")
    return proc


def results_repo(cfg: Config) -> Path:
    return cfg.work_dir / ".ocrbench-results"


def ensure_repo(cfg: Config) -> Path:
    """Repo cục bộ của nhánh results; lần đầu thì kéo nhánh từ GitHub về (hoặc tạo mới nếu chưa có)."""
    repo, remote, branch = results_repo(cfg), _remote(cfg), cfg.status.branch
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        _git(repo, "init", "-q")
        _git(repo, "remote", "add", "origin", remote)
        _git(repo, "config", "user.name", f"ocrbench ({platform.node()})")
        _git(repo, "config", "user.email", "ocrbench@users.noreply.github.com")
        if _git(repo, "fetch", "-q", "origin", branch, remote=remote, check=False).returncode == 0:
            _git(repo, "checkout", "-q", "-B", branch, "FETCH_HEAD")
        else:
            _git(repo, "checkout", "-q", "--orphan", branch)
    return repo


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def collect(cfg: Config, repo: Path) -> None:
    out = Path(cfg.output_dir)
    _copy(out / "status.md", repo / "status.md")
    _copy(cfg.path, repo / "config.yaml")
    for name in WORK_FILES:
        if (cfg.work_dir / name).exists():
            _copy(cfg.work_dir / name, repo / name)
    if (cfg.work_dir / "reports").is_dir():
        shutil.copytree(cfg.work_dir / "reports", repo / "reports", dirs_exist_ok=True)
    for split_dir in (p for p in out.glob("*") if p.is_dir()):
        for model_dir in (p for p in split_dir.iterdir() if p.is_dir()):
            dest = repo / "runs" / split_dir.name / model_dir.name
            if model_dir.name == "_report":
                for pattern in REPORT_GLOBS:
                    for f in model_dir.glob(pattern):
                        _copy(f, dest / f.name)
                continue
            for name in RUN_FILES:
                if (model_dir / name).exists():
                    _copy(model_dir / name, dest / name)
            log = model_dir / "run.log"
            if log.exists():  # chỉ giữ phần cuối log cho gọn
                dest.mkdir(parents=True, exist_ok=True)
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]
                (dest / "run.log.tail").write_text("\n".join(tail) + "\n", encoding="utf-8")


def push_status(cfg: Config, running: list[str] | None = None, note: str | None = None) -> str:
    """Ghi status.md rồi đẩy lên GitHub. Trả về mô tả ngắn; lỗi thì ném RuntimeError."""
    write_status(cfg, running, note)
    repo, remote, branch = ensure_repo(cfg), _remote(cfg), cfg.status.branch
    collect(cfg, repo)
    _git(repo, "add", "-A")
    if _git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return "không có gì mới để đẩy"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _git(repo, "commit", "-q", "-m", f"status {stamp}" + (f": {note}" if note else ""))
    if _git(repo, "push", "-q", "origin", f"HEAD:{branch}", remote=remote, check=False).returncode != 0:
        # nhánh trên GitHub có commit mới hơn (vd. từ phiên trước): kéo về, giữ bản của máy này khi trùng file
        _git(repo, "fetch", "-q", "origin", branch, remote=remote)
        _git(repo, "rebase", "-q", "-X", "theirs", "FETCH_HEAD")
        _git(repo, "push", "-q", "origin", f"HEAD:{branch}", remote=remote)
    return f"đã đẩy lên nhánh {branch} (commit {_git(repo, 'rev-parse', '--short', 'HEAD').stdout.strip()})"


def safe_push(cfg: Config, running: list[str] | None = None, note: str | None = None) -> bool:
    """Dùng trong lúc chạy model: luôn ghi status.md; đẩy lên GitHub nếu config bật. Không bao giờ ném lỗi."""
    try:
        if not cfg.status.push:
            write_status(cfg, running, note)
            return True
        msg = push_status(cfg, running, note)
        print(f"✔ STATUS: {msg}", flush=True)
        return True
    except Exception as e:
        print(f"✘ ĐẨY STATUS THẤT BẠI: {e}", flush=True)
        return False


# --- khôi phục -------------------------------------------------------------------------


def restore(cfg: Config) -> list[str]:
    """Kéo nhánh results về và chép kết quả vào thư mục làm việc. Không ghi đè dữ liệu mới hơn ở máy này."""
    repo = ensure_repo(cfg)
    remote = _remote(cfg)
    if _git(repo, "fetch", "-q", "origin", cfg.status.branch, remote=remote, check=False).returncode == 0:
        _git(repo, "reset", "-q", "--hard", "FETCH_HEAD")
    done = []
    runs = repo / "runs"
    if runs.is_dir():
        for f in runs.rglob("*"):
            if not f.is_file() or f.name == "run.log.tail":
                continue
            dest = Path(cfg.output_dir) / f.relative_to(runs)
            if f.name == "predictions.jsonl" and dest.exists():
                if len(dest.read_text().splitlines()) >= len(f.read_text().splitlines()):
                    continue  # máy này đã có nhiều kết quả hơn
            elif dest.exists():
                continue
            _copy(f, dest)
            done.append(str(dest))
    for name in WORK_FILES:
        if (repo / name).exists() and not (cfg.work_dir / name).exists():
            _copy(repo / name, cfg.work_dir / name)
            done.append(str(cfg.work_dir / name))
    if (repo / "reports").is_dir() and not (cfg.work_dir / "reports").exists():
        shutil.copytree(repo / "reports", cfg.work_dir / "reports")
        done.append(str(cfg.work_dir / "reports"))
    if (repo / "config.yaml").exists() and (repo / "config.yaml").read_bytes() != cfg.path.read_bytes():
        shutil.copy2(cfg.path, cfg.path.with_suffix(".yaml.bak"))
        shutil.copy2(repo / "config.yaml", cfg.path)
        done.append(f"{cfg.path} (bản cũ lưu ở {cfg.path.with_suffix('.yaml.bak').name})")
    return done
