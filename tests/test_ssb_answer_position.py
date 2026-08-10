"""SSB answer_position 解析修复(2026-08-10,CD-006 反馈两条):

  * P2 — sheet 名含逗号(单引号包裹)被裸 split(",") 切坏(130-9);
  * P1 伴生 — 裸整列区间 ``A:G`` 在 int("") 上炸(283-32);
  * 引号计数为奇(数据 typo,283-32 缺前引号)回退裸切分。

等价性红线:无引号、无裸列的存量输入,切分与展开行为与历史实现逐字节一致。
端到端用真 openpyxl workbook(golden-vs-golden 必须满分)。
"""
import openpyxl
import pytest

from trace_opd.envs.spreadsheetbench.evaluator import (
    _generate_cell_names,
    _split_regions,
    compare_workbooks,
)


# ---------- 切分:修复行为 ----------

def test_quoted_sheet_name_with_commas_is_one_region():
    # 130-9 实录:sheet 名 `b2b, sez, de` 是任务语义的一部分
    assert _split_regions("'b2b, sez, de'!A5:V10") == ["'b2b, sez, de'!A5:V10"]


def test_mixed_quoted_and_plain_regions():
    assert _split_regions("'a, b'!A1:B2,Sheet2!C3") == ["'a, b'!A1:B2", "Sheet2!C3"]


def test_odd_quote_count_falls_back_to_plain_split():
    # 283-32 实录:字段开头缺一个引号(上游数据 typo)——裸切分后每段的
    # strip("'\"") 恰好消化散落引号,两段均可解析
    assert _split_regions("Sheet3'!A:G,'Sheet4'!A:G") == ["Sheet3'!A:G", "'Sheet4'!A:G"]


# ---------- 切分:存量等价红线 ----------

@pytest.mark.parametrize("ap", [
    "Sheet1!A1:B2,C3",
    "A1:B2",
    "Sheet1!A1,Sheet2!B2,Sheet3!C3",
    "  A1 , B2 ",          # 空白由调用方 strip(与历史一致,切分不动空白)
    "",
])
def test_plain_inputs_split_byte_identical_to_historical(ap):
    assert _split_regions(ap) == ap.split(",")


# ---------- 裸整列区间展开 ----------

def test_bare_column_range_expands_over_max_row():
    assert _generate_cell_names("A:C", max_row=2) == [
        "A1", "A2", "B1", "B2", "C1", "C2",
    ]


def test_numbered_ranges_ignore_max_row():
    # 存量等价:带行号的区间不受 max_row 参数影响
    assert _generate_cell_names("A1:B2", max_row=99) == ["A1", "A2", "B1", "B2"]
    assert _generate_cell_names("D7", max_row=99) == ["D7"]


def test_half_bare_range_still_raises():
    # `A1:G` 不是合法 Excel 区间,维持历史 fail 行为(不静默猜语义)
    with pytest.raises(ValueError):
        _generate_cell_names("A1:G", max_row=5)


# ---------- 端到端:真 workbook ----------

def _write_wb(path, sheet_name, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    for r in rows:
        ws.append(r)
    wb.save(path)
    wb.close()


def test_comma_sheet_golden_vs_golden_is_perfect(tmp_path):
    p = tmp_path / "g.xlsx"
    _write_wb(p, "b2b, sez, de", [[1, "x"], [2, "y"]])
    ok, msg = compare_workbooks(str(p), str(p), "'b2b, sez, de'!A1:B2")
    assert ok and msg == ""


def test_bare_column_range_end_to_end(tmp_path):
    g, q = tmp_path / "g.xlsx", tmp_path / "p.xlsx"
    _write_wb(g, "S", [[1, 2], [3, 4]])
    _write_wb(q, "S", [[1, 2], [3, 4]])
    ok, _ = compare_workbooks(str(g), str(q), "S!A:B")
    assert ok
    # pred 比 golden 多写一行 → 必须进比较并判负(max_row 取两侧较大者)
    _write_wb(q, "S", [[1, 2], [3, 4], [5, 6]])
    ok, msg = compare_workbooks(str(g), str(q), "S!A:B")
    assert not ok and "A3" in msg


def test_mismatch_reports_pred_only(tmp_path):
    # GROUND-TRUTH SAFETY 红线在新路径下仍成立:reason 只报 pred 值
    g, q = tmp_path / "g.xlsx", tmp_path / "p.xlsx"
    _write_wb(g, "a, b", [[42]])
    _write_wb(q, "a, b", [[7]])
    ok, msg = compare_workbooks(str(g), str(q), "'a, b'!A1")
    assert not ok
    assert "pred=7" in msg and "42" not in msg
