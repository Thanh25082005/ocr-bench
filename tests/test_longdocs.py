"""Tài liệu dài / nhiều trang: chỉ số lệch bảng, độ phủ theo vị trí, pipeline nhiều ảnh."""

import json
import random
import re

import pytest
import yaml
from PIL import Image

from ocrbench.adapters.base import Adapter, Prediction
from ocrbench.cli import main
from ocrbench.metrics import merge_tables_html, position_recall, table_cell_metrics, teds
from ocrbench.normalize import NormConfig
from ocrbench.testset.longdocs import TABLE_BINS, TEXT_BINS, long_table, long_text, perturb_numbers

N = NormConfig()
HDR = "<tr><th>Date</th><th>Ref</th><th>Debit</th><th>Credit</th></tr>"
ROWS = [f"<tr><td>d{i}</td><td>R{i}</td><td>{'' if i % 2 else i}</td><td>{i if i % 2 else ''}</td></tr>" for i in range(40)]
GT = "<table>" + HDR + "".join(ROWS) + "</table>"


def test_dropped_row_shows_as_shift():
    pred = "<table>" + HDR + "".join(ROWS[:10] + ROWS[11:]) + "</table>"
    m = table_cell_metrics(GT, pred, N)
    assert m["cell_aligned"] > 0.95 and m["cell_exact"] < 0.4  # chênh lệch lớn = lệch hàng
    assert m["pred_rows"] == m["gt_rows"] - 1


def test_dropped_empty_cell_shows_as_column_shift():
    pred = "<table>" + HDR + "".join(r.replace("<td></td>", "", 1) for r in ROWS) + "</table>"
    m = table_cell_metrics(GT, pred, N)
    assert m["rows_wrong_ncols"] == 40


def test_repeated_header_across_pages_is_not_penalized():
    pred = "<table>" + HDR + "".join(ROWS[:20]) + "</table><table>" + HDR + "".join(ROWS[20:]) + "</table>"
    assert table_cell_metrics(GT, pred, N)["cell_exact"] == 1.0
    assert teds(GT, merge_tables_html(pred, N), N) == pytest.approx(1.0)


def test_position_recall_detects_truncation():
    ref = " ".join(f"w{i}" for i in range(400))
    q = position_recall(ref, " ".join(f"w{i}" for i in range(200)))
    assert q[0] == 1.0 and q[3] == 0.0


@pytest.mark.parametrize("b", range(len(TABLE_BINS)))
@pytest.mark.parametrize("lang", ["ar", "en", "mixed"])
def test_long_table_pages_match_ground_truth(b, lang):
    pages, gt, gt_type, meta = long_table(lang, random.Random(f"{b}{lang}"), "", b)
    assert len(pages) == TABLE_BINS[b][1] and gt.count("<table") == 1
    pred = "\n".join(re.sub(r"<style>.*?</style>", "", p, flags=re.S) for p in pages)
    assert table_cell_metrics(gt, pred, N)["cell_exact"] == 1.0
    assert table_cell_metrics(gt, pred, N)["gt_rows"] == meta["n_rows"] + 1


def test_long_text_paging():
    parts = [("p", f"Paragraph {i} " + "word " * 60) for i in range(40)]
    parts.insert(5, ("h2", "Section"))
    for b, (label, n_pages, budget) in enumerate(TEXT_BINS):
        pages, gt, gt_type, meta = long_text("en", ("Title", parts), random.Random(0), "", b)
        assert len(pages) == n_pages and gt.startswith("Title")
        assert meta["length_bin"] == label


def test_perturb_numbers_keeps_shape():
    out, n = perturb_numbers("سنة ١٩٥٢ و 2001", random.Random(1), p=1.0)
    assert n == 2 and re.fullmatch(r"سنة [٠-٩]{4} و \d{4}", out)


class PageCounter(Adapter):
    """Adapter thử: không ghi đè predict_pages, nên mỗi trang được đọc riêng rồi ghép."""

    def predict(self, image, item):
        return Prediction(f"page-{image.size[0]}")


def test_multi_page_item_through_pipeline(tmp_path):
    for k, w in enumerate((100, 200, 300)):
        Image.new("RGB", (w, 50), "white").save(tmp_path / f"p{k}.png")
    rec = {"id": "doc", "image": ["p0.png", "p1.png", "p2.png"], "category": "long", "gt": "page-100 page-200 page-300",
           "length_bin": "3 trang"}
    (tmp_path / "m.jsonl").write_text(json.dumps(rec))
    cfg = {"dataset": "m.jsonl", "output_dir": "runs", "models": [
        {"name": "counter", "adapter": "tests.test_longdocs:PageCounter"},
        {"name": "oracle", "adapter": "dummy"}]}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    main(["run", "--config", str(tmp_path / "c.yaml"), "--inline"])
    pred = json.loads((tmp_path / "runs/dev/counter/predictions.jsonl").read_text())
    assert pred["text"] == "page-100\n\npage-200\n\npage-300" and pred["extra"]["pages"] == 3
    score = json.loads((tmp_path / "runs/dev/oracle/scores.jsonl").read_text())
    assert score["cer"] == 0 and score["pages"] == 3
    assert "Tài liệu dài" in (tmp_path / "runs/dev/_report/report.md").read_text()
