"""SpreadsheetBench cell evaluation + test-case discovery + preview.

逐规则移植自 skills_evolve/css/envs/spreadsheetbench/{evaluator,testcases,
codegen}.py(其自身是官方 RUCKBReasoning/SpreadsheetBench
`evaluation/evaluation.py` 的移植)。语义与旧实现逐字节一致(对拍测试
tests/test_ssb_evaluator.py):

  * 数值(int/float/数值字符串)统一 ``round(float(v), 2)`` 定点量化
    (是量化不是容差);
  * ``datetime.time`` 字符串化去微秒;``datetime.datetime`` 转 Excel 序列
    日取整;bool 提升为 float(1 vs True 对齐);
  * ``""`` 与 ``None`` 等价;此外类型不同即不等;
  * 只比 answer_position 区域;格式/样式不比(官方注释掉的行为一致);
  * 2026-08-10(CD-006 反馈):多区域切分改为引号感知(sheet 名可含逗号,
    130-9)+ 裸整列区间 ``A:G`` 按已用行数展开(283-32);无引号、无裸列
    的存量输入行为逐字节不变(tests/test_ssb_answer_position.py 钉住);
  * data_only=True 读缓存值 ⇒ agent 用 openpyxl 写公式得不到缓存值,读出
    None——必须写计算结果值(scaffold 明示;与旧实现同语义)。

GROUND-TRUTH SAFETY(继承旧纪律):所有失败 reason 只报 agent 自己的
pred 值,绝不泄露 golden 值——这些文本会进轨迹与训练侧观察。
"""
from __future__ import annotations

import datetime
import glob as _glob
import os


# ---------- value transform / compare(官方移植) ----------

def _datetime_to_float(dt: datetime.datetime) -> float:
    excel_start_date = datetime.datetime(1899, 12, 30)
    delta = dt - excel_start_date
    return delta.days + delta.seconds / 86400.0


def _transform_value(v):
    if isinstance(v, bool):
        return round(float(v), 2)
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, datetime.time):
        return str(v)[:-3]
    if isinstance(v, datetime.datetime):
        return round(_datetime_to_float(v), 0)
    if isinstance(v, str):
        try:
            return round(float(v), 2)
        except ValueError:
            return v
    return v


def _compare_cell_value(v1, v2) -> bool:
    v1 = _transform_value(v1)
    v2 = _transform_value(v2)
    if (v1 == "" and v2 is None) or (v1 is None and v2 == ""):
        return True
    if (v1 == "" and v2 == "") or (v1 is None and v2 is None):
        return True
    if type(v1) is not type(v2):
        return False
    return v1 == v2


# ---------- answer_position 多区域切分(2026-08-10 CD-006 反馈修复) ----------

def _split_regions(answer_position: str) -> list[str]:
    """按逗号切多区域 —— 但 Excel sheet 名自身可含逗号(单引号包裹,
    verified-400 任务 130-9 `'b2b, sez, de'!A5:V10` 是任务语义的一部分),
    引号内的逗号不是分隔符。引号计数为奇 = 配对已坏(数据侧 typo,如
    283-32 `Sheet3'!A:G,...` 缺前引号),状态机会吞掉真分隔符,此时回退
    裸切分 —— 每段的 strip("'\\"") 恰好消化散落的引号。无引号输入两条
    路径与历史 `split(",")` 逐字节一致。"""
    s = answer_position or ""
    if s.count("'") % 2 != 0:
        return s.split(",")
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    for ch in s:
        if ch == "'":
            in_quote = not in_quote
        if ch == "," and not in_quote:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


# ---------- range 解析(官方移植) ----------

def _col_num2name(n: int) -> str:
    name = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        name = chr(65 + r) + name
    return name


def _col_name2num(name: str) -> int:
    num = 0
    for c in name:
        num = num * 26 + (ord(c) - ord("A") + 1)
    return num


def _parse_range(range_str: str):
    start_cell, end_cell = range_str.split(":")
    sc = "".join(ch for ch in start_cell if ch.isalpha())
    sr = "".join(ch for ch in start_cell if ch.isdigit())
    ec = "".join(ch for ch in end_cell if ch.isalpha())
    er = "".join(ch for ch in end_cell if ch.isdigit())
    return (_col_name2num(sc), int(sr)), (_col_name2num(ec), int(er))


def _generate_cell_names(range_str: str, max_row: int | None = None):
    if ":" not in range_str:
        return [range_str]
    start_cell, end_cell = range_str.split(":")
    if (max_row is not None
            and not any(ch.isdigit() for ch in start_cell)
            and not any(ch.isdigit() for ch in end_cell)):
        # 裸整列区间 `A:G`(verified-400 任务 283-32):无行号,历史实现在
        # int("") 上炸。语义 = 整列,按两侧 workbook 的已用行数展开
        # (2026-08-10 CD-006 反馈修复)。
        range_str = f"{start_cell.strip()}1:{end_cell.strip()}{max_row}"
    (sc, sr), (ec, er) = _parse_range(range_str)
    cols = [_col_num2name(i) for i in range(sc, ec + 1)]
    return [f"{c}{r}" for c in cols for r in range(sr, er + 1)]


def _cell_level_compare(wb_gt, wb_proc, sheet_name: str, cell_range: str):
    if sheet_name not in wb_proc.sheetnames:
        return False, f"worksheet not found: {sheet_name}"
    ws_gt = wb_gt[sheet_name]
    ws_proc = wb_proc[sheet_name]
    # 取两侧较大的已用行数:pred 比 golden 多写行时也要进比较(多写的行
    # golden 侧读出 None,与空等价规则一致地参与判定)。
    max_row = max(ws_gt.max_row or 1, ws_proc.max_row or 1)
    for cn in _generate_cell_names(cell_range, max_row):
        cg = ws_gt[cn]
        cp = ws_proc[cn]
        if not _compare_cell_value(cg.value, cp.value):
            # GROUND-TRUTH SAFETY: never leak the golden value.
            return False, f"value@{sheet_name}!{cn}: pred={cp.value!r} (did not match)"
    return True, ""


# ---------- public API ----------

def compare_workbooks(gt_file: str, proc_file: str, answer_position: str) -> tuple[bool, str]:
    """(ok, msg) — single test-case comparison, official semantics."""
    import openpyxl  # noqa: PLC0415 — heavy import stays lazy

    if not os.path.exists(proc_file):
        return False, "file not exist"
    try:
        wb_gt = openpyxl.load_workbook(filename=gt_file, data_only=True)
        wb_proc = openpyxl.load_workbook(filename=proc_file, data_only=True)
    except Exception as e:  # noqa: BLE001
        return False, f"load error: {e}"
    try:
        ok_all = True
        msg_first = ""
        for scr in _split_regions(answer_position):
            scr = scr.strip()
            if not scr:
                continue
            if "!" in scr:
                sheet_name, cell_range = scr.split("!", 1)
                sheet_name = sheet_name.strip().strip("'\"")
            else:
                sheet_name = wb_gt.sheetnames[0]
                cell_range = scr
            cell_range = cell_range.strip().strip("'\"")
            ok, msg = _cell_level_compare(wb_gt, wb_proc, sheet_name, cell_range)
            if not ok:
                ok_all = False
                if not msg_first:
                    msg_first = msg
        return ok_all, msg_first
    finally:
        wb_gt.close()
        wb_proc.close()


def evaluate(pred_path: str, gold_path: str, answer_position: str) -> dict:
    """Single test-case verdict; soft/hard aggregation is the caller's job."""
    ok, msg = compare_workbooks(gold_path, pred_path, answer_position)
    return {"ok": ok, "reason": msg}


# ---------- test-case discovery(旧 testcases.py 移植) ----------

def find_test_cases(task_dir: str) -> list[tuple[str, str, str]]:
    """[(case_no, input_path, answer_path), ...]; 支持四种发布命名:
    ``{no}_{id}_input/answer``(original)、``{no}_{id}_init/golden``
    (verified_400)、bare ``initial/golden``(verified_400 无前缀)、
    bare ``input/target``(Spreadsheet-RL, arXiv:2605.22642)。"""
    cases: list[tuple[str, str, str]] = []
    inputs = sorted(_glob.glob(os.path.join(task_dir, "*_input.xlsx")))
    for ip in inputs:
        no = os.path.basename(ip).split("_", 1)[0]
        ap = ip.replace("_input.xlsx", "_answer.xlsx")
        if os.path.exists(ap):
            cases.append((no, ip, ap))
    inits = sorted(_glob.glob(os.path.join(task_dir, "*_init.xlsx")))
    for ip in inits:
        no = os.path.basename(ip).split("_", 1)[0]
        ap = ip.replace("_init.xlsx", "_golden.xlsx")
        if os.path.exists(ap):
            cases.append((no, ip, ap))
    if not cases:
        bare_init = os.path.join(task_dir, "initial.xlsx")
        bare_gold = os.path.join(task_dir, "golden.xlsx")
        if os.path.exists(bare_init) and os.path.exists(bare_gold):
            cases.append(("1", bare_init, bare_gold))
    if not cases:
        # Spreadsheet-RL:golden = target.xlsx(官方 sample_meta.py 同款);
        # 同目录的 output.xlsx 是 oracle 中间产物,判分不用。
        srl_init = os.path.join(task_dir, "input.xlsx")
        srl_gold = os.path.join(task_dir, "target.xlsx")
        if os.path.exists(srl_init) and os.path.exists(srl_gold):
            cases.append(("1", srl_init, srl_gold))
    return cases


# ---------- workbook preview(旧 codegen._preview_workbook 移植) ----------

def preview_workbook(path: str, max_rows: int = 5, max_cols: int = 20) -> str:
    """全 sheet 首行预览(确定性:同文件同输出;initial_obs 的组成部分)。"""
    import openpyxl  # noqa: PLC0415

    wb = openpyxl.load_workbook(path, data_only=False)
    chunks: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if not hasattr(ws, "iter_rows"):
            # Chartsheet 等无单元格网格的 sheet(SRL Charting 任务实测;
            # SSB verified 无此形态,存量 initial_obs 逐字节不变)
            chunks.append(f"## Sheet: {sheet_name}  (chart sheet — no cell grid)")
            chunks.append("")
            continue
        chunks.append(
            f"## Sheet: {sheet_name}  "
            f"(dim={ws.dimensions}, max_row={ws.max_row}, max_col={ws.max_column})"
        )
        for row in ws.iter_rows(
            min_row=1,
            max_row=min(ws.max_row, max_rows),
            max_col=min(ws.max_column, max_cols),
            values_only=False,
        ):
            cells = []
            for cell in row:
                v = cell.value
                if v is None:
                    cells.append(f"{cell.coordinate}=")
                else:
                    s = str(v)
                    if len(s) > 40:
                        s = s[:37] + "..."
                    cells.append(f"{cell.coordinate}={s}")
            chunks.append(" | ".join(cells))
        if ws.max_row > max_rows:
            chunks.append(f"... ({ws.max_row - max_rows} more rows)")
        chunks.append("")
    wb.close()
    return "\n".join(chunks)
