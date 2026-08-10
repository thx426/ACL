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
  LibreOffice headless)——写公式与写静态值都是合法解法,但**两者的可靠性
  不同**(见下节);soffice 缺失装配期即死(部署前置:装 LibreOffice 或设
  `SSB_SOFFICE`;官方 README 要求 **7.5+**)。
- **红线**:评估结果不进 executor 流(submit 反馈恒 "Submitted.");golden
  不进沙箱/extra_info,gt_text=None;所有 reason 不泄期望值。
- **数据**:`build_dataset.py` 在数据所在服务器同环境跑(union 400 →
  seed42 → train 200 / valid 200);xlsx 根走 `SSB_DATA`。
- **配置**:默认起点是 `configs/experiments/ssb-selfres-default.yaml`(完整机制:
  v3.3 × v4 自体研究员)与 `ssb-grpo-default.yaml`(对照臂)。两者 executor 侧
  逐键相同,差异只有机制开关。新实验从默认出发,只覆盖 run 身份与机器档
  (`n_gpus`/`ulysses_sp`;注意 `agent_num_workers` **必须等于**
  `train_batch_size`,config.py 有校验)。research 侧一律用 v4 设计值,
  不做成本缩减(用户裁定 2026-08-09)。

## reward 可靠性:只跑 372 个任务(用户裁定 2026-08-09)

**SpreadsheetBench 的数据集就是 `ssb_reliable372_seed42`(train 184 / valid 188),
不再有 verified-400 的概念。** 全 400 里 28 个任务的 reward 信号是坏的,已剔除。

判定 = 六门审计 `scripts/ssb_verify_reward_signal.py`,全程走生产
`SpreadsheetbenchTaskAdapter._score_output`,底本是 `build_dataset` 的实际产物
(即训练时真正喂进去的 `answer_position`):

| 门 | 构造的 pred | 期望 | 失败含义 |
|---|---|---|---|
| ① 正向 | golden 原样(含公式) | score==1.0 | 完美答案也拿不到分,reward 恒零 |
| ② 反向-未操作 | init 原样 | <1.0 | 不做事就满分,任务是空的 |
| ③ 反向-扰动 | golden 答案区改坏 | <1.0 | 比对没真比 |
| ④ 确定性 | 同 pred 连评两次 | 一致 | 重算不确定,reward 是噪声 |
| ⑤ answer 区非空 | — | ≥1 非空 | **仅提示**:"清空某区域"是合法任务形态 |
| ⑥ 单调 | — | ①>② | ①②的推论 |

**只验正向是不够的** —— `54196`(init 就等于 golden)、`73-45`(答案区空且扰动仍
满分)两个"恒过"任务,正向门完全通过,只有反向门抓得到。它们组内 rollout 全 1,
advantage 同样全零,还会把 val 抬成假象。

剔除的 28 个(清单与逐任务根因见 `unreliable_tasks.json`,重审用同名脚本):

| 类别 | 数 | 说明 |
|---|---|---|
| Excel 365 新函数(`_xlfn.`) | 14 | XLOOKUP/UNIQUE/LET/TEXTJOIN/AGGREGATE/TEXTSPLIT/FILTERXML… LibreOffice 不认识 ⇒ `#NAME?` |
| 数组公式/结构化引用 | 7 | `Table1[#This Row]` 之类 ⇒ `#VALUE!`/`#N/A`/None |
| 其他公式语义差异 | 5 | 查找失败、数值算错(54144: 37.46→17.46) |
| 反向恒过 | 2 | 54196 / 73-45 |

LibreOffice **6.4.7.2 与 7.6.7.2 实测逐任务同结果** ⇒ 与版本无关,换版本无解。
官方 README 要求 7.5+,故部署用 7.6.7.2 并把它前置到 PATH(`find_soffice()` 走
`shutil.which`,不前置会静默用回系统版本)。

**解法偏置**(口径事实,非 bug):纯静态值答案不进重算(`workbook_has_formulas`
判定),写公式要过 LibreOffice 重算 —— 372 版上两种形态六门全绿,但全 400 上
实测可靠率分别是 99.7% 与 90.5%。官方文档既未讨论后端差异,也未说明论文
baseline 用的哪个后端。

**372 版六门实测**:①②③④⑥ 全绿 0 违例;⑤ 的 3 例(`168-17`/`414-20`/`534-26`)
已查实是"删除行列"类任务 —— golden 目标区本就该空,生产 score golden=1.0 /
init=0.0,信号正确,故 ⑤ 不作剔除依据(用户 2026-08-09 确认)。

**生成与门禁**
- 唯一入口:`build_dataset.py`(默认剔除;`--keep-unreliable` 仅用于生成审计底本)
- 启动门禁:`cfg_probe.py <cfg> ssbdata` 按 **manifest 记录的剔除集合与当前清单
  逐个比对**,作废的 400/373 版与审计底本一律被拒
- 清单复核:`ssb_verify_reward_signal.py --verify-list`(六门口径;只验正向的
  脚本会把反向剔除的任务误判成"已变可靠")
