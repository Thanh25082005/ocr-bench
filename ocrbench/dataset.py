"""Bộ dữ liệu test: đọc manifest.jsonl và tạo manifest từ thư mục.

Mỗi dòng của manifest.jsonl là một mẫu:

    {"id": "hoso12__p3", "image": "images/hoso12__p3.png", "category": "handwriting_ar",
     "gt_file": "gt/hoso12__p3.txt", "doc_id": "hoso12", "split": "dev", "lang": "ar"}

- image: một ảnh, hoặc danh sách ảnh cho tài liệu nhiều trang (["p1.png", "p2.png"]).
- image, gt_file: đường dẫn tương đối so với thư mục chứa manifest (hoặc tuyệt đối).
- gt: có thể ghi thẳng nội dung đáp án thay cho gt_file.
- gt_type: "text" hoặc "table_html" (mặc định suy ra từ đuôi file: .html -> table_html).
- doc_id: mã tài liệu / người viết. Dùng để chia dev/holdout không rò rỉ.
- split: "dev" hoặc "holdout".
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
GT_EXTS = (".html", ".htm", ".txt", ".md")
SPLITS = ("dev", "holdout")


@dataclass
class Item:
    id: str
    image: Path
    category: str
    gt: str
    gt_type: str = "text"
    doc_id: str = ""
    split: str = "dev"
    lang: str | None = None
    meta: dict = field(default_factory=dict)
    images: list[Path] = field(default_factory=list)  # tài liệu nhiều trang: mọi trang theo thứ tự

    def __post_init__(self):
        if not self.images:
            self.images = [self.image]

    @property
    def n_pages(self) -> int:
        return len(self.images)


class ManifestError(ValueError):
    pass


def _gt_type_for(path: str | None, explicit: str | None) -> str:
    if explicit:
        if explicit not in ("text", "table_html"):
            raise ManifestError(f"gt_type không hợp lệ: {explicit!r} (chỉ nhận 'text' hoặc 'table_html')")
        return explicit
    if path and Path(path).suffix.lower() in (".html", ".htm"):
        return "table_html"
    return "text"


def load_manifest(
    path: str | Path,
    split: str | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
    per_category: int | None = None,
    sample_seed: str = "ocrbench",
) -> list[Item]:
    path = Path(path)
    root = path.parent
    items: list[Item] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"{path}:{lineno}: JSON lỗi: {e}") from e
            for key in ("id", "image", "category"):
                if key not in rec:
                    raise ManifestError(f"{path}:{lineno}: thiếu trường '{key}'")
            if rec["id"] in seen:
                raise ManifestError(f"{path}:{lineno}: id bị trùng: {rec['id']}")
            seen.add(rec["id"])

            # "image" là một đường dẫn, hoặc danh sách đường dẫn cho tài liệu nhiều trang
            raw_images = rec["image"] if isinstance(rec["image"], list) else [rec["image"]]
            images = [Path(p) if Path(p).is_absolute() else root / p for p in raw_images]
            if not images:
                raise ManifestError(f"{path}:{lineno}: 'image' rỗng")
            if "gt" in rec:
                gt = rec["gt"]
            elif "gt_file" in rec:
                gt_path = Path(rec["gt_file"])
                if not gt_path.is_absolute():
                    gt_path = root / gt_path
                gt = gt_path.read_text(encoding="utf-8")
            else:
                raise ManifestError(f"{path}:{lineno}: cần 'gt' hoặc 'gt_file'")

            rec_split = rec.get("split", "dev")
            if rec_split not in SPLITS:
                raise ManifestError(f"{path}:{lineno}: split không hợp lệ: {rec_split!r}")
            known = {"id", "image", "category", "gt", "gt_file", "gt_type", "doc_id", "split", "lang"}
            items.append(
                Item(
                    id=str(rec["id"]),
                    image=images[0],
                    images=images,
                    category=rec["category"],
                    gt=gt,
                    gt_type=_gt_type_for(rec.get("gt_file"), rec.get("gt_type")),
                    doc_id=str(rec.get("doc_id") or rec["id"]),
                    split=rec_split,
                    lang=rec.get("lang"),
                    meta={k: v for k, v in rec.items() if k not in known},
                )
            )

    _check_doc_leakage(items)
    if split:
        items = [it for it in items if it.split == split]
    if categories:
        wanted = set(categories)
        items = [it for it in items if it.category in wanted]
    if per_category:
        items = sample_per_category(items, per_category, sample_seed)
    if limit:
        items = items[:limit]
    return items


def sample_per_category(items: list[Item], n: int, seed: str = "ocrbench") -> list[Item]:
    """Lấy tối đa n mẫu mỗi nhóm, chọn ngẫu nhiên nhưng cố định (theo hash của id):
    mọi model nhận đúng cùng một tập con, và các mẫu trải đều trên nhiều tài liệu."""
    def key(it: Item) -> bytes:
        return hashlib.sha256(f"{seed}:{it.id}".encode()).digest()

    chosen: set[str] = set()
    for cat in {it.category for it in items}:
        group = sorted((it for it in items if it.category == cat), key=key)
        chosen.update(it.id for it in group[:n])
    return [it for it in items if it.id in chosen]


def _check_doc_leakage(items: list[Item]) -> None:
    """Một tài liệu (doc_id) không được nằm ở cả dev lẫn holdout."""
    splits_by_doc: dict[str, set[str]] = {}
    for it in items:
        splits_by_doc.setdefault(it.doc_id, set()).add(it.split)
    leaked = sorted(d for d, s in splits_by_doc.items() if len(s) > 1)
    if leaked:
        raise ManifestError(
            f"{len(leaked)} doc_id nằm ở cả dev và holdout (rò rỉ dữ liệu), ví dụ: {leaked[:5]}"
        )


def fingerprint(items: list[Item]) -> str:
    """Dấu vân tay của bộ dữ liệu (id, nhóm, split, đáp án, tên ảnh).
    Sửa bất kỳ đáp án / cách chia dev-holdout / danh sách mẫu nào cũng làm dấu vân tay đổi.
    Không tính nội dung byte của ảnh: ảnh dựng bằng Chromium có thể lệch vài byte giữa hai phiên bản
    trình duyệt dù nội dung như nhau, nên hai máy dựng cùng một bộ test vẫn có cùng dấu vân tay."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.id):
        imgs = ",".join(p.name for p in it.images)
        gt_hash = hashlib.sha256(it.gt.encode("utf-8")).hexdigest()
        h.update(f"{it.id}\t{it.category}\t{it.split}\t{it.gt_type}\t{gt_hash}\t{imgs}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def summarize(items: list[Item]) -> str:
    by_cat = Counter((it.category, it.split) for it in items)
    docs = Counter()
    for cat in {it.category for it in items}:
        docs[cat] = len({it.doc_id for it in items if it.category == cat})
    cats = sorted({it.category for it in items})
    lines = [f"{'category':<24}{'dev':>8}{'holdout':>10}{'doc_id':>9}", "-" * 51]
    for cat in cats:
        lines.append(f"{cat:<24}{by_cat[(cat, 'dev')]:>8}{by_cat[(cat, 'holdout')]:>10}{docs[cat]:>9}")
    lines.append("-" * 51)
    lines.append(
        f"{'TỔNG':<24}{sum(v for (c, s), v in by_cat.items() if s == 'dev'):>8}"
        f"{sum(v for (c, s), v in by_cat.items() if s == 'holdout'):>10}"
        f"{len({it.doc_id for it in items}):>9}"
    )
    return "\n".join(lines)


def _split_for_doc(doc_id: str, holdout_ratio: float, seed: str) -> str:
    h = hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()
    return "holdout" if int.from_bytes(h[:8], "big") / 2**64 < holdout_ratio else "dev"


def make_manifest(
    root: str | Path,
    out: str | Path,
    holdout_ratio: float = 0.5,
    seed: str = "ocrbench",
    doc_sep: str = "__",
) -> list[dict]:
    """Tạo manifest từ cấu trúc thư mục:

        root/<category>/<doc_id>__<trang>.png   + cùng tên .txt (văn bản) hoặc .html (bảng)

    Phần trước `doc_sep` trong tên file là doc_id; không có thì cả tên file là doc_id.
    dev/holdout được chia theo doc_id bằng hash cố định, nên chạy lại vẫn ra cùng kết quả
    và các trang của cùng một tài liệu luôn nằm cùng một phía.
    """
    root = Path(root).resolve()
    out = Path(out).resolve()
    records = []
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(p for p in cat_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            gt_file = next((img.with_suffix(ext) for ext in GT_EXTS if img.with_suffix(ext).exists()), None)
            if gt_file is None:
                print(f"[bỏ qua] không có đáp án cho {img.relative_to(root)}")
                continue
            doc_id = img.stem.split(doc_sep)[0] if doc_sep in img.stem else img.stem
            doc_id = f"{cat_dir.name}/{doc_id}"
            records.append(
                {
                    "id": f"{cat_dir.name}/{img.stem}",
                    "image": _rel(img, out.parent),
                    "gt_file": _rel(gt_file, out.parent),
                    "category": cat_dir.name,
                    "doc_id": doc_id,
                    "split": _split_for_doc(doc_id, holdout_ratio, seed),
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def _rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)
