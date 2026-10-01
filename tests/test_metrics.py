import pytest

from ocrbench.metrics import (
    edit_stats,
    error_rate,
    extract_tables_html,
    find_repetition,
    flags_for,
    markdown_tables_to_html,
    teds,
)
from ocrbench.normalize import NormConfig, normalize, normalize_raw, to_plain_text

NORM = NormConfig()


def test_cer_wer_basic():
    s = edit_stats("hello world", "helo world")
    assert s["char_edits"] == 1 and s["ref_chars"] == 11
    assert s["word_edits"] == 1 and s["ref_words"] == 2
    assert error_rate(0, 0) == 0.0
    assert error_rate(5, 0) == 1.0
    # sinh thừa chữ: CER có thể vượt 100%
    s = edit_stats("ab", "ab" + "x" * 10)
    assert error_rate(s["char_edits"], s["ref_chars"]) == 5.0


def test_arabic_normalization():
    # tashkeel và tatweel bị bỏ, chữ số Ả Rập-Ấn -> ASCII
    assert normalize("كَتَبَ", NORM) == "كتب"
    assert normalize("كـــتب", NORM) == "كتب"
    assert normalize("رقم ١٢٣", NORM) == "رقم 123"
    assert normalize("١٬٢٥٠٫٥٠ ريال", NORM) == "1,250.50 ريال"
    # dạng trình bày (presentation form) được NFKC gộp về chữ gốc
    assert normalize("ﻻ", NORM) == "لا"
    # ký tự điều khiển hướng chữ bị bỏ
    assert normalize("‏مرحبا‎", NORM) == "مرحبا"
    # mặc định KHÔNG gộp hamza
    assert normalize("أحمد", NORM) != normalize("احمد", NORM)
    assert normalize("أحمد", NormConfig(unify_alef=True)) == "احمد"


def test_raw_keeps_diacritics():
    assert normalize_raw("كَتَبَ") == "كَتَبَ"


def test_markdown_and_html_stripped():
    md = "# Title\n\n**Bold** text\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert normalize(md, NORM) == "Title Bold text a b 1 2"
    html = "<table><tr><td>a</td><td>b</td></tr></table>"
    assert normalize(html, NORM) == "a b"
    # dấu < > trong văn bản thường không bị coi là thẻ
    assert to_plain_text("x < 5 and y > 3") == "x < 5 and y > 3"


def test_unknown_norm_option_rejected():
    with pytest.raises(ValueError):
        NormConfig.from_dict({"strip_diacritic": True})


def test_repetition_detection():
    assert find_repetition("normal text " * 3) is None
    assert find_repetition("start " + "وقال الرجل " * 30) is not None
    # lặp số / dấu chấm (mục lục, bảng toàn số 0) không bị coi là kẹt vòng lặp
    assert find_repetition("0.00 " * 50) is None
    assert find_repetition("." * 300) is None


def test_flags():
    empty_cells = "<table>" + "<tr>" + "<td></td>" * 40 + "</tr>" + "</table>"
    assert "repetition" not in flags_for("x", "x", empty_cells, None)
    assert "repetition" in flags_for("x", "x", "<table>" + "<tr><td>loop text</td></tr>" * 30 + "</table>", None)
    ref = "a" * 100
    assert flags_for(ref, "", "", None) == ["empty"]
    assert "too_long" in flags_for(ref, "a" * 200, "a" * 200, None)
    assert "too_short" in flags_for(ref, "a" * 30, "a" * 30, None)
    assert flags_for(ref, "", "", "RuntimeError: x") == ["error"]


GT_TABLE = "<table><tr><th>Name</th><th>الاسم</th></tr><tr><td>1</td><td>٢</td></tr></table>"


def test_teds_identical_and_markdown_pred():
    assert teds(GT_TABLE, GT_TABLE, NORM) == pytest.approx(1.0)
    md_pred = "| Name | الاسم |\n|---|---|\n| 1 | 2 |"
    # th/td coi như nhau; ٢ và 2 giống nhau sau chuẩn hóa
    assert teds(GT_TABLE, md_pred, NORM) == pytest.approx(1.0)


def test_teds_penalizes_errors():
    wrong_content = "<table><tr><td>Nmae</td><td>الاسم</td></tr><tr><td>1</td><td>9</td></tr></table>"
    missing_row = "<table><tr><td>Name</td><td>الاسم</td></tr></table>"
    full = teds(GT_TABLE, wrong_content, NORM)
    assert 0.5 < full < 1.0
    assert teds(GT_TABLE, wrong_content, NORM, structure_only=True) == pytest.approx(1.0)
    assert teds(GT_TABLE, missing_row, NORM) < 0.8
    assert teds(GT_TABLE, "no table here", NORM) == 0.0


def test_colspan_counts_as_structure():
    gt = '<table><tr><td colspan="2">x</td></tr></table>'
    pred = "<table><tr><td>x</td></tr></table>"
    assert teds(gt, pred, NORM, structure_only=True) < 1.0


def test_extract_tables():
    text = "Intro\n<table><tr><td>a</td></tr></table>\nmore"
    assert extract_tables_html(text) == "<table><tr><td>a</td></tr></table>"
    assert markdown_tables_to_html("| a | b |\n|--|--|\n| 1 | 2 |") == (
        "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    )


def test_adapter_postprocess():
    from ocrbench.adapters.base import Adapter

    a = Adapter()
    assert a.postprocess("<think>hmm</think>\nالنص") == "النص"
    assert a.postprocess("draft </think> a </think> final") == "final"
    b = Adapter(json_field="full_text")
    out = '```json\n{"subject": "x", "keywords": [], "full_text": "المبلغ ١٠٠"}\n```'
    assert b.postprocess(out) == "المبلغ ١٠٠"
    # JSON hỏng thì giữ nguyên văn bản, không mất kết quả
    assert b.postprocess('{"full_text": "abc"') == '{"full_text": "abc"'
