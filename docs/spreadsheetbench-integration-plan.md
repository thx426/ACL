# SpreadsheetBench(verified 400)任务环境引入方案(v2 定稿)

状态:用户四项裁定已定(2026-08-07),实施中。
契约骨架:`envs/_template/README.md` 交互点总表 + 设计文档 §9。

## 0. 用户裁定(本方案的约束)

1. **交互协议 = 原生 function-call 工具调用**(`<action>` 文本块是 ALFWorld
   特定协议,SSB 不用):两个工具 `run_python(code)` + `submit()`。
2. **划分**:seed 42,union(400)→sort(id)→shuffle→ **train 200 / test 200**
   (test 200 即训练期 val 集;旧 140/60/200 弃用)。
3. **步数预算**:可配置,默认 **20**(episode caps 走 yaml)。
4. **实施**:机制侧全量引入 + v4 自进化全流程训练机制同步引入;
   **researcher 双模式必须都保留**——policy 自体(research_rl.enabled)与
   外部大模型端点(现有 roles.researcher 路径),SSB 上两路都要能跑。

## 1. 旧资产考古(skills_evolve/css/envs/spreadsheetbench,1673 行)

- 数据:`data/spreadsheetbench_split_shuffled_seed42/*/items.json`
  (union(400) 清单);xlsx 本体在服务器
  `/home/wushang/workspace/data/spreadsheetbench_split_t2s_aligned`
  (本地无;实施后由 CD-006 确认在位 + case 数分布)。
- item schema:id / instruction / spreadsheet_path / instruction_type
  (Cell-Level / Sheet-Level)/ answer_position(+answer_sheet)。
- **evaluator.py = 官方 evaluation.py 逐规则移植**(round-2 量化、类型
  严格、空值等价、range 解析)→ 逐字节移植 + 对拍测试。
- testcases.py:三种命名((`{no}_{id}_input/answer`、`{no}_{id}_init/
  golden`、bare `initial/golden`)→ 移植。
- 多 case 语义:agent 只在 case1 input 上工作,单一 output 对全部 case 的
  golden 逐一评;**soft=n_pass/n_cases → ActionOutcome.score;hard=全过 →
  won**(与 v2 协议抽象完美对齐)。
- 评估反馈红线:评估结果永不进 executor 可见流(旧 POST-EXECUTION
  VERIFICATION 仅训练侧诊断)→ 继承。
- 隔离纪律:episode 只见当前任务 case1 input 的拷贝;golden 只被 adapter
  评估路径读取(SSB-2 核查文档的 per-instance isolation 教训)。

## 2. 工具调用协议设计(function-call 三层)

- **工具集**(用户裁定,窄面):
  - `run_python(code: str)`:episode 工作目录内执行 Python(openpyxl/
    pandas 可用);subprocess 隔离 + timeout + 输出截断;返回 stdout+stderr。
  - `submit()`:终局。检查 `output.xlsx` → 逐 case 评估 → done=True,
    score=soft;feedback 恒 "Submitted."(零评估信息)。output 缺失 ⇒ 不
    终局,feedback 提示缺失。
- **prompt 层**:工具 schema 以 Qwen 官方 chat-template tools 渲染的同款
  文本格式(`<tools>` JSON + `<tool_call>` 说明)内嵌 `executor_scaffold()`
  ——分布内形态(4B 原生 FC 能力直接可用),`render_ids`/episode prompt
  构造零改动。guidance(特权文档)仍经 `inject_guidance` 统一 header 追加。
- **解析层**:core/protocol.py 新增通用件 `parse_tool_calls(text)`
  (hermes `<tool_call>{json}</tool_call>`,取合法 JSON 调用列表)。这是
  **模板家族级**协议件(非任务级),与 parse_action 平级;SSB 的 react
  用它分派工具,每轮取第一个合法调用(单动作语义,多余调用忽略并在
  feedback 注明)。
- **反馈层**:Qwen3 模板把 tool 响应渲染为 user 块内
  `<tool_response>\n{content}\n</tool_response>`——feedback 文本自带包裹,
  经现有 user 结构段进流,**TemplateDelta/episode 零机制改动**。
  模板逐字节等价性列入服务器冒烟清单(真 tokenizer 对拍;不等价则回此节
  升级 TemplateDelta 变体,机制侧改动已知且有界)。
- 无合法 tool_call ⇒ `format_reminder()`(本环境习语,acted=False,
  防 spiral 计数照常)。normalize_action:strip-only(代码大小写敏感)。

## 3. TaskAdapter 实现要点(envs/spreadsheetbench/adapter.py)

- **EnvHandle = episode 工作目录**:open_episode = mkdtemp + copy case1
  input → `input.xlsx`(固定相对名);所有路径用相对 CWD 固定名
  (input.xlsx / output.xlsx),命令流零绝对路径 ⇒ initial_obs 确定性与
  v4 重放可行性。close = 清理目录。
- **react**:parse_tool_calls → run_python(subprocess,cwd=workdir,净化
  env,timeout,截断)| submit(评估编排)分派;valid = returncode==0
  (观测字段);consumed_env_step = 每次工具执行。
- **replay_action**:action_norm 存 JSON 编码的工具调用;重放 = 重新分派
  执行,返回 (feedback, ended)。确定性依赖代码无时间/随机——机制侧逐步
  校验兜底(漂移作废)。
- **initial_obs**(build_dataset 捕获 = live 生成,同源函数):instruction
  + instruction_type + answer_position + spreadsheet 预览(移植旧渲染,
  确定性)+ 路径约定。
- **executor_scaffold**:角色(spreadsheet 专家)+ 工具 schema 文本
  (§2)+ 输出约定 + RESTRICTIONS(只碰工作目录)。
- **task_brief**(researcher 面板):目标/环境机制(Python+openpyxl)/
  工具文法/评估口径(answer_position 区域、多 case、soft-hard)/步数预算。
- **gt_text = None**:golden 是测试答案本体,绝不进 researcher(区别于
  ALFWorld 的过程性 walkthrough)。researcher 信息来源 = bare 轨迹 + 自做
  实验(run_experiment 的 won/score 是环境真值反馈,合法)。
- task_type = instruction_type(SFT 分层/分组统计)。

## 4. 机制交互点核对

| 交互点 | SSB 落点 | 机制改动 |
| --- | --- | --- |
| 装配 | registry 一行 + `SpreadsheetbenchCfg`(data_root/exec_concurrency/python_timeout_s/obs 截断) | 零(新 section 照 AlfworldCfg 模式) |
| episode 预算 | yaml episode 段:max_env_steps=20(可配) | 零 |
| 工具解析 | core `parse_tool_calls`(模板家族级通用件) | **新增 helper(+测试)** |
| 反馈流 | tool_response 包裹文本走 user 结构段 | 零(待服务器模板对拍确认) |
| score | soft 保真;won=hard | 零(v2 抽象已就绪) |
| v4 三段/续写 | 24 行三段、分位切分、半轮承接协议无关 | 零 |
| researcher 双模式 | enabled 开关切 policy 自体 / 外部端点(现有两路) | 零 |
| SFT 冷启动 | 通用 sftgen + task_type 二类分层 | 零 |
| CPU 并发 | python 子进程并发门(semaphore,builder 内 per-worker 折算)——CPU 饥饿教训 | 零(builder 职责) |

## 5. 工程清单

1. core:`protocol.parse_tool_calls` + 测试;
2. `envs/spreadsheetbench/`:adapter.py(react/评估编排/scaffold/brief)、
   evaluator.py + testcases.py(移植+对拍)、preview.py、build_dataset.py
   (items.json 400 → seed42 → train 200/test 200 parquet)、README.md;
3. registry 一行 + `SpreadsheetbenchCfg` + 实验 yaml(v4 自体 与 外部
   researcher 两份变体);
4. 测试:评估器对拍/契约面/episode 端到端(真 subprocess+构造 xlsx)/
   续写端到端/tool_call 解析/隔离红线(golden 不可达、评估信息不外泄);
5. 交 CD-006 清单:数据在位与 case 分布、openpyxl 进训练环境(走 env
   入口纪律)、真 tokenizer 的 tool 模板对拍冒烟、35B/4B bare 探针。

## 5.5 公式陷阱(旧项目 48 探针实证;用户确认的历史细节点)

评估口径(官方 evaluation.py 语义)用 openpyxl `data_only=True` 读**缓存
值**,而 agent 用 openpyxl 写入的公式没有缓存值 ⇒ 公式单元格被读成
None,**写公式的解法结构性必挂**。旧项目实证(skills_evolve
`explorer_trajectories_review/06_ss_global_residual_48probe_ledger.md`):

- 任务 53647(指令字面要求 create formulas)连败 23 探针,失败签名恒
  `pred=None`(agent 坚持写公式);
- 泛泛的"写静态值"指令**不够**(多个 static-value 探针仍 0/2);
- 翻转要素 = 显式因果解释("harness 不重算公式 → 公式被读 empty/None →
  失败")+ MUST 静态值规则 + 字面指令覆盖授权 —— probe_046 generic 段
  (49036 同款 8/8 稳定)。

落地:上述三要素文案已进 `executor_scaffold`(勿削弱措辞,源注释标注
台账出处);**task_brief 同步告知 researcher 该评估机理**(否则 researcher
可能产出教 executor 写公式的有害文档)。口径声明:此为评估语义设计事实
(与旧 +14.5pp 实证同口径),非缺陷;若未来需对齐官方论文数字,须另核
官方管线是否含 Excel/WPS 重算步骤,报告口径分开。

## 6. 科学定位与已知风险

- 第二环境角色:机制跨环境泛化验证(探索型 vs 代码操作型);旧 +14.5pp
  (9B,skill 文档)为特权文档有效性同域参考。
- 死记风险:单 case 任务上文档可编码答案值拿满 reward(合法但非泛化);
  多 case 抗死记;报告按 case 数分层。
- 4B 底线未知(旧实证 9B):bare 探针先行,SFT 冷启动预期必要。
- 观察截断:大 workbook 预览/输出截断是主要 inspection 瓶颈(SSB-2 核查
  同款教训),截断参数入配置。
