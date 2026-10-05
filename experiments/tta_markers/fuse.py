"""Ghép ký hiệu chú thích nhỏ (†, ‡, §, *...) từ các lượt đọc phóng to vào bảng của lượt đọc thường.

Thuần Python + lxml, không gọi model → kiểm thử được không cần GPU.

Ý tưởng (test-time augmentation + bỏ phiếu, chỉ cho KÝ HIỆU, không đụng con số):
- lượt GỐC (cấu hình web demo): giữ nguyên cấu trúc bảng và mọi con số;
- các lượt PHỤ (ảnh phóng ×4, ×5, dịch lề...): model hay đọc ra ký hiệu nhỏ mà lượt gốc bỏ mất;
- với mỗi ô của lượt gốc có giá trị v: tìm trong CÙNG DÒNG (khớp theo nhãn ở ô đầu) của lượt phụ một token
  = <ký hiệu> + v. Ký hiệu chỉ gồm ký tự trong MARKS (hoặc <sup>chữ thường</sup> trong HTML).
  Phần thêm là chữ số (vd. "146" so với "46") → KHÔNG sửa.
- chỉ thêm khi ≥ min_votes lượt phụ cùng đưa ra đúng ký hiệu đó.
"""

from __future__ import annotations

import re

MARKS = set("†‡§¶*#‖")
_SUP_LETTER = re.compile(r"^[a-e]$")


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def table_rows(md: str) -> list[list]:
    """Các dòng của mọi bảng HTML trong kết quả: list[list[lxml element ô]] (dùng để sửa tại chỗ)."""
    from lxml import html as lh

    rows = []
    for m in re.finditer(r"<table.*?</table>", md or "", flags=re.S | re.I):
        frag = lh.fragment_fromstring(m.group(0), create_parent="div")
        for tr in frag.xpath(".//tr"):
            cells = tr.xpath("./td|./th")
            if cells:
                rows.append(cells)
    return rows


def _cell_tokens(cell) -> list[str]:
    """Token của một ô HTML; <sup>a</sup>105 → '\\u2063a\\u2063105' (dấu \\u2063 đánh dấu chữ mũ)."""
    parts = []
    text = cell.text or ""
    parts.append(text)
    for ch in cell:
        if isinstance(ch.tag, str) and ch.tag.lower() == "sup":
            sup = _norm(ch.text_content())
            parts.append(("\u2063" + sup + "\u2063") if _SUP_LETTER.match(sup) or (sup and all(c in MARKS for c in sup))
                         else sup)
        else:
            parts.append(ch.text_content())
        parts.append(ch.tail or "")
    joined = re.sub("\u2063\\s+", "\u2063", "".join(parts))  # ký hiệu mũ dính vào số theo sau (bỏ xuống dòng)
    return joined.split()


def detail_lines(md: str) -> list[tuple[str, list[str]]]:
    """Kết quả lượt phụ → [(nhãn dòng, token)] — từ bảng HTML (ô đầu = nhãn) hoặc từ dòng chữ thường."""
    out = []
    for cells in table_rows(md):
        label = _norm(cells[0].text_content())
        toks = [t for c in cells[1:] for t in _cell_tokens(c)]
        out.append((label, toks))
    plain = re.sub(r"<table.*?</table>", "\n", md or "", flags=re.S | re.I)
    for line in plain.split("\n"):
        line = _norm(re.sub(r"<[^>]+>", " ", line))
        if line:
            out.append((line, line.split()))
    return out


def _marker_of(token: str, value: str) -> str | None:
    """'†105' với value '105' → '†'; '\u2063a\u2063105' → '<sup>a</sup>'; '146' với '46' → None."""
    if token == value or not token.endswith(value):
        return None
    prefix = token[: -len(value)]
    if prefix.startswith("\u2063") and prefix.endswith("\u2063") and _SUP_LETTER.match(prefix.strip("\u2063")):
        return f"<sup>{prefix.strip(chr(0x2063))}</sup>"
    prefix = prefix.replace("\u2063", "")
    if prefix and all(c in MARKS for c in prefix):
        return prefix
    return None


def _find_row(label: str, lines: list[tuple[str, list[str]]]):
    """Dòng của lượt phụ có cùng nhãn (bảng: nhãn trùng; chữ thường: dòng bắt đầu bằng nhãn)."""
    if not label:
        return None
    for lab, toks in lines:
        if lab == label:
            return toks
    for lab, toks in lines:
        if lab.startswith(label + " "):
            return lab[len(label):].split()
    return None


def fuse_markers(base_md: str, detail_mds: list[str], min_votes: int = 1) -> tuple[str, list[dict]]:
    """→ (kết quả gốc đã thêm ký hiệu, danh sách thay đổi). Chỉ sửa bảng HTML của lượt gốc."""
    from lxml import html as lh

    details = [detail_lines(d) for d in detail_mds]
    changes = []

    def fix_table(m):
        frag = lh.fragment_fromstring(m.group(0), create_parent="div")
        for tr in frag.xpath(".//tr"):
            cells = tr.xpath("./td|./th")
            if len(cells) < 2:
                continue
            label = _norm(cells[0].text_content())
            for cell in cells[1:]:
                value = _norm(cell.text_content())
                if not value or not any(c.isdigit() for c in value) or " " in value:
                    continue  # chỉ ô một giá trị có chữ số
                votes: dict[str, int] = {}
                for lines in details:
                    toks = _find_row(label, lines)
                    if not toks:
                        continue
                    found = {mk for t in toks if (mk := _marker_of(t, value))}
                    if len(found) == 1:  # đúng một ký hiệu cho giá trị này trong dòng — không mơ hồ
                        mk = found.pop()
                        votes[mk] = votes.get(mk, 0) + 1
                if not votes:
                    continue
                mk, n = max(votes.items(), key=lambda kv: kv[1])
                if n >= min_votes and len(votes) == 1:
                    for ch in list(cell):
                        cell.remove(ch)
                    cell.text = None
                    cell.append(lh.fragment_fromstring(f"<span>{mk}{value}</span>"))
                    changes.append({"row": label, "value": value, "marker": mk, "votes": n})
        html = lh.tostring(frag, encoding="unicode")
        return html[len("<div>"):-len("</div>")].replace("<span>", "").replace("</span>", "")

    fused = re.sub(r"<table.*?</table>", fix_table, base_md or "", flags=re.S | re.I)
    return fused, changes


def gt_markers(gt_html: str) -> list[tuple[str, str, str]]:
    """Ký hiệu đứng trước số trong đáp án: [(nhãn dòng, giá trị, ký hiệu)]."""
    out = []
    for cells in table_rows(gt_html):
        label = _norm(cells[0].text_content())
        for c in cells[1:]:
            for t in _cell_tokens(c):
                m = re.match(r"^(.*?)(\d[\d.,]*)$", t)
                if m and (mk := _marker_of(t, m.group(2))):
                    out.append((label, m.group(2), mk))
    return out


def markers_in(md: str) -> set[tuple[str, str, str]]:
    return set(gt_markers(md))
