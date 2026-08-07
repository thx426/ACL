# SpreadsheetBench(verified 400)

方案与四项用户裁定:`docs/spreadsheetbench-integration-plan.md`。
本包 = 按 `envs/_template/` 契约的第二个真实 benchmark。

- **协议**:原生 function-calling(`run_python(code)` / `submit()`),react
  自写(不经 text_action_react);解析用 core 通用件 `parse_tool_calls`;
  反馈按 Qwen3 模板 tool 渲染形态包裹 `<tool_response>`(服务器真 tokenizer
  对拍冒烟后生效,不等价则升级 TemplateDelta 变体)。
- **评估**:`evaluator.py` = 官方 evaluation.py 语义(经旧 skills_evolve
  移植,行为快照测试锁定);soft(case 通过比例)→ `score`,hard → `won`。
  **评分前公式重算**(`recalc.py`,官方 open_spreadsheet.py 语义,
  LibreOffice headless)——写公式与写静态值都是合法解法;soffice 缺失
  装配期即死(部署前置:装 LibreOffice 或设 `SSB_SOFFICE`)。
- **红线**:评估结果不进 executor 流(submit 反馈恒 "Submitted.");golden
  不进沙箱/extra_info,gt_text=None;所有 reason 不泄期望值。
- **数据**:`build_dataset.py` 在数据所在服务器同环境跑(union 400 →
  seed42 → train 200 / valid 200);xlsx 根走 `SSB_DATA`。
- **配置**:`configs/experiments/a800-ssb-s1-selfres.yaml`(v4 自体)/
  `a800-ssb-s1-extres.yaml`(外部 35B researcher)——两路长期并存。
