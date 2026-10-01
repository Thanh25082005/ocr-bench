"""Các chỉ số chấm điểm: CER, WER, TEDS (bảng) và cờ phát hiện lỗi bất thường."""

from __future__ import annotations

import re
import warnings

from rapidfuzz.distance import Levenshtein

from .normalize import NormConfig, normalize, to_plain_text


def edit_stats(ref: str, hyp: str) -> dict:
    ref_words, hyp_words = ref.split(), hyp.split()
    return {
        "char_edits": Levenshtein.distance(ref, hyp),
        "ref_chars": len(ref),
        "word_edits": Levenshtein.distance(ref_words, hyp_words),
        "ref_words": len(ref_words),
    }


def error_rate(edits: int, ref_len: int) -> float:
    """Không giới hạn ở 1.0: model sinh thừa nhiều chữ thì tỉ lệ lỗi có thể > 100%."""
    if ref_len == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / ref_len


# --- Cờ bất thường -------------------------------------------------------------

# Một đoạn 2-80 ký tự lặp liền nhau từ 10 lần trở lên: dấu hiệu model bị kẹt vòng lặp.
_REPEAT = re.compile(r"(.{2,80}?)(?:\s*\1){9,}", re.S)


def find_repetition(text: str, min_span: int = 100) -> str | None:
    for m in _REPEAT.finditer(text):
        unit = m.group(1)
        if len(m.group(0)) >= min_span and any(ch.isalpha() for ch in unit):
            return unit
    return None


def flags_for(ref_norm: str, hyp_norm: str, hyp_raw: str, error: str | None) -> list[str]:
    flags = []
    if error:
        flags.append("error")
    elif not hyp_norm and ref_norm:
        flags.append("empty")
    # bỏ thẻ HTML/Markdown trước: bảng có nhiều ô trống liên tiếp không phải là kẹt vòng lặp
    if find_repetition(to_plain_text(hyp_raw)):
        flags.append("repetition")
    if len(ref_norm) >= 20:
        ratio = len(hyp_norm) / len(ref_norm)
        if ratio > 1.5:
            flags.append("too_long")
        elif ratio < 0.5 and hyp_norm:
            flags.append("too_short")
    return flags


# --- TEDS cho bảng ---------------------------------------------------------------

_warned_teds = False


def _teds_available() -> bool:
    global _warned_teds
    try:
        import apted  # noqa: F401
        import lxml  # noqa: F401
    except ImportError:
        if not _warned_teds:
            warnings.warn("Chưa cài 'apted' và 'lxml' nên bỏ qua TEDS. Cài bằng: pip install 'ocrbench[teds]'")
            _warned_teds = True
        return False
    return True


def markdown_tables_to_html(text: str) -> str:
    """Chuyển các bảng Markdown (| a | b |) trong văn bản thành <table> HTML."""
    tables, rows = [], []

    def flush():
        if rows:
            body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
            tables.append(f"<table>{body}</table>")
            rows.clear()

    for line in text.splitlines():
        st = line.strip()
        if st.startswith("|") and st.count("|") >= 2:
            if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?", st):
                continue
            rows.append([c.strip() for c in st.strip("|").split("|")])
        else:
            flush()
    flush()
    return "".join(tables)


def extract_tables_html(text: str) -> str:
    found = re.findall(r"<table\b.*?</table>", text, flags=re.S | re.I)
    if found:
        return "".join(found)
    return markdown_tables_to_html(text)


def _build_tree(html_text: str, norm: NormConfig):
    from apted.helpers import Tree
    from lxml import html as lh

    class Node(Tree):
        def __init__(self, tag, colspan=1, rowspan=1, content="", children=None):
            self.tag, self.colspan, self.rowspan, self.content = tag, colspan, rowspan, content
            self.children = children or []

    def span(cell, attr):
        try:
            return int(cell.get(attr, 1))
        except ValueError:
            return 1

    root = Node("body")
    if not html_text.strip():
        return root, 1
    doc = lh.document_fromstring(f"<html><body>{html_text}</body></html>")
    n = 1
    for table in doc.body.xpath(".//table[not(ancestor::table)]"):
        t = Node("table")
        n += 1
        for row in table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr"):
            r = Node("tr")
            n += 1
            for cell in row.xpath("./td|./th"):
                content = normalize(cell.text_content(), norm)
                r.children.append(Node("td", span(cell, "colspan"), span(cell, "rowspan"), content))
                n += 1
            t.children.append(r)
        root.children.append(t)
    return root, n


def teds(gt_html: str, pred_text: str, norm: NormConfig, structure_only: bool = False) -> float | None:
    """TEDS (Tree-Edit-Distance-based Similarity), 1.0 = giống hệt. None nếu không tính được."""
    if not _teds_available():
        return None
    from apted import APTED, Config

    class TedsConfig(Config):
        def rename(self, a, b):
            if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
                return 1.0
            if a.tag == "td" and not structure_only and (a.content or b.content):
                return Levenshtein.normalized_distance(a.content, b.content)
            return 0.0

        def children(self, node):
            return node.children

    gt_tree, n_gt = _build_tree(extract_tables_html(gt_html), norm)
    if n_gt <= 1:
        return None
    pred_tree, n_pred = _build_tree(extract_tables_html(pred_text), norm)
    if n_pred <= 1:
        return 0.0
    dist = APTED(pred_tree, gt_tree, TedsConfig()).compute_edit_distance()
    return max(0.0, 1.0 - dist / max(n_gt, n_pred))


# --- Bảng: ô có bị lệch hàng / lệch cột không -------------------------------------------


def table_rows(html_text: str, norm: NormConfig) -> list[list[str]] | None:
    """Mọi hàng của mọi bảng (theo thứ tự), mỗi ô đã chuẩn hóa. None nếu thiếu lxml."""
    try:
        from lxml import html as lh
    except ImportError:
        return None
    if not html_text.strip():
        return []
    doc = lh.document_fromstring(f"<html><body>{html_text}</body></html>")
    rows = []
    for table in doc.body.xpath(".//table[not(ancestor::table)]"):
        for row in table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr"):
            rows.append([normalize(c.text_content(), norm) for c in row.xpath("./td|./th")])
    return rows


def _drop_repeated_headers(rows: list[list[str]], header: list[str]) -> list[list[str]]:
    """Bảng nhiều trang thường in lại hàng tiêu đề ở đầu mỗi trang; đó không phải dữ liệu."""
    return [r for i, r in enumerate(rows) if i == 0 or r != header]


def _align_rows(g: list[list[str]], p: list[list[str]]) -> list[tuple[int, int]]:
    """Căn hàng đáp án với hàng model đọc (quy hoạch động, tối đa số ô khớp), cho phép bỏ/thêm hàng."""
    def match(a, b):
        return sum(1 for x, y in zip(a, b) if x == y)

    n, m = len(g), len(p)
    S = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        gi = g[i - 1]
        for j in range(1, m + 1):
            S[i][j] = max(S[i - 1][j - 1] + match(gi, p[j - 1]), S[i - 1][j], S[i][j - 1])
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if S[i][j] == S[i - 1][j - 1] + match(g[i - 1], p[j - 1]) and match(g[i - 1], p[j - 1]) > 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif S[i][j] == S[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _quarters(values_by_pos: list[tuple[int, float]], n: int) -> list[float | None]:
    """Gom (vị trí, điểm) thành 4 phần đều nhau theo vị trí trong tài liệu: đầu → cuối."""
    buckets: list[list[float]] = [[], [], [], []]
    for pos, v in values_by_pos:
        buckets[min(3, pos * 4 // max(n, 1))].append(v)
    return [sum(b) / len(b) if b else None for b in buckets]


def table_cell_metrics(gt_html: str, pred_text: str, norm: NormConfig) -> dict | None:
    """Đo giá trị trong bảng có bị lệch không.

    - cell_exact: tỉ lệ ô đúng giá trị VÀ đúng vị trí (hàng, cột) như đáp án.
    - cell_aligned: tỉ lệ ô đúng sau khi căn lại hàng (bỏ qua việc thiếu/thừa hàng).
      cell_aligned cao mà cell_exact thấp = model bỏ sót hoặc thêm hàng làm các hàng sau bị đẩy lệch.
    - rows_wrong_ncols: số hàng có số ô khác đáp án, thường do bỏ sót ô trống nên giá trị bị dồn cột.
    - pos_q: tỉ lệ ô đúng (sau căn hàng) theo 4 phần của bảng, từ đầu đến cuối.
    """
    g = table_rows(extract_tables_html(gt_html), norm)
    if not g:
        return None
    p = table_rows(extract_tables_html(pred_text), norm) or []
    if p:
        p = _drop_repeated_headers(p, g[0])
    total = sum(len(r) for r in g) or 1
    exact = sum(1 for i, r in enumerate(g) for j, v in enumerate(r) if i < len(p) and j < len(p[i]) and p[i][j] == v)
    pairs = _align_rows(g, p)
    matched = {i: sum(1 for x, y in zip(g[i], p[j]) if x == y) for i, j in pairs}
    aligned = sum(matched.values())
    per_row = [(i, matched.get(i, 0) / max(len(g[i]), 1)) for i in range(len(g))]
    return {
        "cell_exact": exact / total,
        "cell_aligned": aligned / total,
        "gt_rows": len(g),
        "pred_rows": len(p),
        "rows_wrong_ncols": sum(1 for i, j in pairs if len(g[i]) != len(p[j])),
        "pos_q": _quarters(per_row, len(g)),
    }


def merge_tables_html(text: str, norm: NormConfig) -> str:
    """Gộp các bảng (vd. một bảng bị ngắt qua nhiều trang) thành một, bỏ hàng tiêu đề lặp lại."""
    from lxml import html as lh

    tables = extract_tables_html(text)
    if not tables:
        return ""
    doc = lh.document_fromstring(f"<html><body>{tables}</body></html>")
    rows = doc.body.xpath(".//table[not(ancestor::table)]/tr|.//table[not(ancestor::table)]/*/tr")
    out, header = [], None
    for r in rows:
        key = [normalize(c.text_content(), norm) for c in r.xpath("./td|./th")]
        if header is None:
            header = key
        elif key == header:
            continue
        out.append(lh.tostring(r, encoding="unicode"))
    return "<table>" + "".join(out) + "</table>"


# --- Văn bản: càng về cuối càng sót? ----------------------------------------------------


def position_recall(ref_norm: str, hyp_norm: str, words_per_chunk: int = 8, threshold: float = 85) -> list[float | None]:
    """Cắt đáp án thành các đoạn ~8 từ liên tiếp, xem mỗi đoạn có xuất hiện (gần đúng) trong
    kết quả model không, rồi gom theo 4 phần của tài liệu. Phần cuối thấp hơn hẳn phần đầu
    = model đọc sót / cắt ngang / mất ngữ cảnh khi tài liệu dài."""
    from rapidfuzz import fuzz

    words = ref_norm.split()
    if not words:
        return [None] * 4
    chunks = [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    found = [(k, 1.0 if fuzz.partial_ratio(ch, hyp_norm) >= threshold else 0.0) for k, ch in enumerate(chunks)]
    return _quarters(found, len(chunks))
