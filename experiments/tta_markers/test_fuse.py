"""Kiểm thử bước ghép ký hiệu (không cần GPU). Xoá cả thư mục experiments/tta_markers là xoá luôn test này."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("lxml")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fuse import fuse_markers, gt_markers  # noqa: E402

BASE = ("<table><tr><th></th><th>n</th><th>%</th></tr>"
        "<tr><td>Neurological</td><td>105</td><td>1.9</td></tr>"
        "<tr><td>Autoimmune</td><td>88</td><td>1.6</td></tr>"
        "<tr><td>Infection</td><td>46</td><td>0.8</td></tr></table>")


def test_marker_from_plain_text_pass():
    detail = "Neurological †105 1.9\n\nAutoimmune ‡88 1.6\n\nInfection 146 0.8"
    fused, ch = fuse_markers(BASE, [detail])
    assert {(c["row"], c["marker"] + c["value"]) for c in ch} == {("Neurological", "†105"), ("Autoimmune", "‡88")}
    assert "<td>†105</td>" in fused and "<td>‡88</td>" in fused
    assert "<td>46</td>" in fused  # "146": phần thêm là chữ số → không sửa


def test_marker_from_html_pass_and_sup_letter():
    detail = ("<table><tr><td>Neurological</td><td><sup>a</sup>105</td><td>1.9</td></tr>"
              "<tr><td>Autoimmune</td><td>*88</td><td>1.6</td></tr></table>")
    fused, ch = fuse_markers(BASE, [detail])
    assert "<sup>a</sup>105" in fused and "*88" in fused and len(ch) == 2


def test_votes_and_conflicts():
    one = "Neurological †105 1.9"
    other = "Neurological ‡105 1.9"
    assert fuse_markers(BASE, [one], min_votes=2)[1] == []          # 1 phiếu < 2
    assert len(fuse_markers(BASE, [one, one], min_votes=2)[1]) == 1  # 2 phiếu
    assert fuse_markers(BASE, [one, other])[1] == []                 # hai lượt mâu thuẫn → không sửa


def test_no_change_when_row_not_found_or_same():
    assert fuse_markers(BASE, ["Something else †105"])[1] == []
    fused, ch = fuse_markers(BASE, [BASE])
    assert ch == [] and "<td>105</td>" in fused


def test_gt_markers():
    gt = "<table><tr><td>Neurological</td><td> <sup> † </sup> 105 </td><td>1.9</td></tr></table>"
    assert gt_markers(gt) == [("Neurological", "105", "†")]
    gt_nl = "<table><tr><td>\n Infection\n</td><td>\n <sup>\n § \n </sup>\n 46\n </td></tr></table>"  # dạng PubTabNet
    assert gt_markers(gt_nl) == [("Infection", "46", "§")]
