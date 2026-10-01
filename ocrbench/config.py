"""Đọc file cấu hình YAML (xem configs/kaggle_example.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .normalize import NormConfig

_NAME_OK = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class ModelSpec:
    name: str
    adapter: str
    params: dict = field(default_factory=dict)
    python: str | None = None  # python của venv riêng nếu model cần thư viện xung đột
    gpus: int = 1  # số GPU model cần (vd. 2 cho model lớn chia trên 2×T4)
    env: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class StatusConfig:
    """Tự động ghi status.md và đẩy kết quả lên GitHub (phòng khi server sập)."""
    push: bool = False  # đẩy lên GitHub sau mỗi model chạy xong và định kỳ trong lúc chạy
    every_min: int = 30  # chu kỳ đẩy trong lúc model đang chạy
    remote: str | None = None  # mặc định: remote 'origin' của repo code
    branch: str = "results"


@dataclass
class Config:
    path: Path
    dataset: Path
    output_dir: Path
    normalization: NormConfig
    models: list[ModelSpec]
    status: StatusConfig = field(default_factory=StatusConfig)

    @property
    def work_dir(self) -> Path:
        """Thư mục chứa output_dir; nơi để EXPERIMENTS.md, FINAL_REPORT.md, reports/."""
        return Path(self.output_dir).parent

    def model(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"không có model '{name}' trong {self.path}. Có: {[m.name for m in self.models]}")

    def select(self, names: list[str] | None) -> list[ModelSpec]:
        if names:
            return [self.model(n) for n in names]
        return [m for m in self.models if m.enabled]


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (base / p).resolve()

    if "dataset" not in raw:
        raise ValueError(f"{path}: thiếu 'dataset' (đường dẫn tới manifest.jsonl)")
    models = []
    for m in raw.get("models") or []:
        unknown = set(m) - {"name", "adapter", "params", "python", "gpus", "env", "enabled"}
        if unknown:
            raise ValueError(f"model {m.get('name')}: trường không hợp lệ {sorted(unknown)}")
        if not _NAME_OK.match(str(m.get("name", ""))):
            raise ValueError(f"tên model không hợp lệ: {m.get('name')!r} (chỉ dùng chữ, số, _ . -)")
        models.append(
            ModelSpec(
                name=m["name"],
                adapter=m["adapter"],
                params=m.get("params") or {},
                python=m.get("python"),
                gpus=int(m.get("gpus", 1)),
                env={k: str(v) for k, v in (m.get("env") or {}).items()},
                enabled=bool(m.get("enabled", True)),
            )
        )
    st = raw.get("status") or {}
    unknown = set(st) - {"push", "every_min", "remote", "branch"}
    if unknown:
        raise ValueError(f"status: trường không hợp lệ {sorted(unknown)}")
    status = StatusConfig(push=bool(st.get("push", False)), every_min=int(st.get("every_min", 30)),
                          remote=st.get("remote"), branch=str(st.get("branch", "results")))
    names = [m.name for m in models]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"tên model bị trùng: {sorted(dup)}")
    return Config(
        path=path,
        dataset=resolve(raw["dataset"]),
        output_dir=resolve(raw.get("output_dir", "runs")),
        normalization=NormConfig.from_dict(raw.get("normalization")),
        models=models,
        status=status,
    )
