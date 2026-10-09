"""dots: phát hiện khung đều + chữ chép, phóng to vùng (ocrbench/adapters/dots.py) — không cần GPU."""

from ocrbench.adapters.dots import copied_cells, grid_runs, region_of, replace_run


def _cell(x1, y1, x2, y2, text="", cat="Text"):
    return {"category": cat, "bbox": [x1, y1, x2, y2], "text": text}


def _grid_page():
    """Như trang thật: tiêu đề, 8 khung cùng cột cao 22 px nối tiếp đều (chữ viết tay bị chép), chữ ký ở dưới."""
    cells = [_cell(375, 88, 609, 118, "وكالة دورية", "Section-header")]
    texts = ["מר (ה) 3/7/1983", "הידוע (ה) לי", "על פי ח.ז. 1983 02081", "שניתן מאת 1983 02081",
             "ב 1983 02081", "מר (ה) 3/7/1983", "הידוע (ה) לי", "מר (ה) 3/7/1983"]
    cells += [_cell(484, 1000 + 22 * k, 863, 1022 + 22 * k, t) for k, t in enumerate(texts)]
    cells.append(_cell(476, 1170, 681, 1260, cat="Picture"))
    return cells


def test_copied_cells_needs_three_or_shared_long_number():
    cells = _grid_page()
    assert copied_cells(cells) == [1, 3, 4, 5, 6, 8]  # "מר (ה) 3/7/1983" ×3, "02081" ở 3 khối
    # nhãn lặp 2 lần là hợp lệ (hai chỗ ký, hai nhân chứng) → không tính
    assert copied_cells([_cell(0, 0, 9, 9, "توقيع الوكالة"), _cell(0, 20, 9, 29, "توقيع الوكالة")]) == []
    # ô bảng trùng nhau là bình thường
    assert copied_cells([_cell(0, 0, 9, 9, "0.00", "Table")] * 4) == []


def test_grid_runs_only_regular_same_column_stacks():
    cells = _grid_page()
    assert grid_runs(cells) == [list(range(1, 9))]
    # dòng thật: cao / bước không đều, lệch cột → không phải dãy khung đều
    real = [_cell(500 + 7 * k, 1000 + 25 * k + (k % 3) * 9, 860 - 5 * k, 1023 + 25 * k + (k % 2) * 12, "x")
            for k in range(8)]
    assert grid_runs(real) == []
    assert grid_runs(cells[:5]) == []  # ngắn hơn min_run


def test_region_of_pads_and_clamps():
    assert region_of([[484, 1000, 863, 1022], [484, 1022, 863, 1044]], 870, 1600) == [473, 989, 870, 1055]


def test_replace_run_keeps_order_and_drops_duplicate_of_kept_block():
    cells = _grid_page()
    new = [_cell(480, 995, 860, 1030, "מר (ה) ג'האד"), _cell(480, 1030, 860, 1060, "על פי ת.ז. 98302081"),
           _cell(476, 1172, 681, 1190, cat="Picture")]  # mép chữ ký đã có ở lượt đầu → bỏ
    out = replace_run(cells, list(range(1, 9)), new)
    assert [c.get("text") for c in out] == ["وكالة دورية", "מר (ה) ג'האד", "על פי ת.ז. 98302081", ""]
    assert out[-1]["bbox"] == [476, 1170, 681, 1260] and len(copied_cells(out)) == 0
