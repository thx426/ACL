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
- **prompt 层(2026-08-07 用户裁定升级:通用 FC 格式 + 模板原生渲染)**:
  工具 schema 以 **OpenAI tools 格式**结构化定义为单一真源(adapter 新契约
  `tools()`,`envs/spreadsheetbench` 的 `TOOLS`),prompt 构造经
  `render_ids(..., tools=...)` 交 **chat template 原生渲染**——逐字节
  分布内,scaffold 绝不手写渲染块(原手写 `_TOOLS_BLOCK` 已废除,其
  模板一致性对拍风险随之消灭)。guidance(特权文档)仍经
  `inject_guidance` 统一 header 追加。文本动作协议 benchmark 的
  `tools()` 返回 None,`render_ids` 行为逐字节不变(ALFWorld 线零影响)。
- **解析层**:core/protocol.py 新增通用件 `parse_tool_calls(text)`
  (hermes `<tool_call>{json}</tool_call>`,取合法 JSON 调用列表)。这是
  **模板家族级**协议件(非任务级),与 parse_action 平级;SSB 的 react
  用它分派工具,每轮取第一个合法调用(单动作语义,多余调用忽略并在
  feedback 注明)。
- **反馈层**:Qwen3 模板把 tool 响应渲染为 user 块内
  `<tool_response>\n{content}\n</tool_response>`——feedback 文本自带包裹,
  经现有 user 结构段进流,**TemplateDelta/episode 零机制改动**。
  模板逐字节等价性列入服务器冒烟清单(真 tokenizer 对拍,**范围只剩
  反馈侧**——prompt 侧已由模板原生渲染保证;不等价则回此节升级
  TemplateDelta 变体,机制侧改动已知且有界)。
- **解析层为何不交引擎**(用户问询 2026-08-07 的裁定记录):引擎 FC 把
  `<tool_call>` 解析成结构化 `tool_calls` 并丢弃原始文本,训练还原时
  模板重渲染的 JSON(键序/空白/转义)与模型原始输出可能不逐字节一致 ⇒
  历史轮 token 漂移。本架构 token-in-token-out:生成的原始 token 原样留
  在训练流(mask=1),`parse_tool_calls` 只读不还原——训练保真最优。
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

## 5.5 公式重算设施(用户确认的历史细节点;2026-08-07 二次修正定稿)

**真相链**:官方评估管线在比对前有「打开-重算-另存」步骤
(``evaluation/open_spreadsheet.py``:Linux/macOS 用 LibreOffice headless
``--convert-to "xlsx:Calc MS Excel 2007 XML"``,Windows 用 Excel COM),公式
在打开时被计算、值写入缓存,然后才 ``data_only=True`` 比对。**第一代
(skills_evolve)移植漏掉了这一步** ⇒ 公式解法结构性必挂(48 探针台账的
「禁公式/写静态值」提示词是该环境缺陷下的 workaround 考古,不是应继承的
机理);第二代(experience_internalization_claude)按用户裁定 2026-07-16
「对齐官方;公式解法必须可判分」补齐 ``recalc.py``。

**本环境落地**(自第二代移植适配,``envs/spreadsheetbench/recalc.py``):

- 评分前对**产出 workbook 字节**重算(golden 不重算——官方发布件自带
  缓存值);重算一次、全 case 复用;临时拷贝,沙箱权威文件不动;
- 便宜预检(zip 扫 worksheet XML ``<f`` 标签),无公式输出不付 soffice
  往返;每次调用独立 UserInstallation profile(并发不抢锁);
  ``SAL_DISABLE_OPENCL/SKIA=1`` 强制纯 CPU(第二代实测 LibreOffice 默认
  抓 ~412MiB 显存,推理引擎吃满的机器上是 OOM 隐患);
- 重算失败降级按原字节评 + 响亮日志;**soffice 完全缺失 ⇒ 装配期 fail
  loud**(比第二代运行时降级更严:无后端时公式解法被静默判 0 是 reward
  语义漂移,不是可降级小故障);
- scaffold 与 task_brief 中性声明「公式会被重算,公式与静态值都是合法
  解法」,并教一个真实 caveat:openpyxl 读公式格返回公式串非值(agent
  自检时会看到)。

回归锁:``test_formula_solution_passes_via_recalc``(真 soffice:agent 写
``=A1+A2`` → 重算 → 评分 1.0)。**部署前置:训练机必须装 LibreOffice**
(PATH 或 ``SSB_SOFFICE``)。可比性口径:旧 +14.5pp 实证产生于第一代
「无重算」口径,与本环境非同口径,只作粗参考。

## 6. 科学定位与已知风险

- 第二环境角色:机制跨环境泛化验证(探索型 vs 代码操作型);旧 +14.5pp
  (9B,skill 文档)为特权文档有效性同域参考。
- 死记风险:单 case 任务上文档可编码答案值拿满 reward(合法但非泛化);
  多 case 抗死记;报告按 case 数分层。
- 4B 底线未知(旧实证 9B):bare 探针先行,SFT 冷启动预期必要。
- 观察截断:大 workbook 预览/输出截断是主要 inspection 瓶颈(SSB-2 核查
  同款教训),截断参数入配置。
