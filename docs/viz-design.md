# TRACE-OPD 可视化观察系统 — 设计文档

> 需求源:`docs/collab/cd005-cd006/req-visualization-system.md`(CD-006,2026-07-24)
> + 用户两轮定位升级(2026-07-24):①系统同时是**论文/报告的证据基础设施**
> (图表、数字、案例的产地);②**必须机制通用**,不得锚定特定环境任务类型。
> 本文是实现真源;代码在 `src/trace_opd/viz/`。

## 0. 一句话

只读扫描 `runs/` 既有产物 → SQLite 索引 → 零依赖 Web 服务,打通
「曲线 → step → 轨迹 → researcher 文档 →(P3)token 蒸馏」双向下钻;
每个聚合数字自带 provenance,每个视图可寻址(deep-link),曲线可导出出版级
图(P1);环境语义全部隔离在观察插件里,内核只认机制协议层 schema。

## 1. 通用性架构(三层,第一性约束)

**切分线与机制自身的 TaskAdapter 抽象同构**(design.md §5):

| 层 | 内容 | 环境耦合 |
|---|---|---|
| **内核** | 产物扫描/索引、轨迹渲染、通用事件(见下)、成功/失败**分层**聚合、下钻链路、文档渲染、曲线、provenance/deep-link、审计 | **零**——只消费协议层 schema(§2) |
| **观察插件** | 失败环节分类器、领域事件标记、task family 解析 | 每环境一个,**可缺省** |
| **降级** | 无插件时:失败分类="unclassified",family=task_id 首段启发,通用事件照常 | 新环境第一天全功能可用 |

**内核可用的通用事件**(全部来自协议层结构化字段,不做环境文本匹配):
`acted=False`(格式失败步,协议层判定)、`valid=False`(环境 no-op,
**适配层已结构化**,内核不认识 "Nothing happens" 这种词)、动作复读
(`action_norm` 重复,纯序列算法)、`won`、`truncated`、`outcome∈{SUCCESS,
FAILED,ENV_ERROR}`。

**机制变体通用性**(第二维,防锚定第一轮实验形态):指标采用
**schema 发现 + 语义注册表**——数据里有什么键就可查可画(未知键不丢),
已知键(`rollout_extra` 族、distillation 族等)在注册表带语义注释与推荐读法;
step 阶段分解从 timing 键发现,不硬编码"research→rollout→train"三段。

**通用性验收(测试锁死)**:fixture 造一个假想第二环境("gridworld",动作词表
north/grab/…,无插件)的产物树,断言内核全功能(索引/分层/轨迹/曲线/下钻)通过。

## 2. 数据契约(机制产物 schema,已对代码核实)

系统**只读**消费 `runs/<run>/` 下既有产物;来源代码为准:
`train/agent_loop.py::_dump_rollout` / `docstore/archive.py` / `core/types.py`。

```
rollouts/step_<N:05d>[_val]/<task_id_safe>_<hex8>.json     # task_id_safe = task_id.replace("/","__")
  {task_id, split, validate, global_step, outcome, won, env_steps,
   assistant_turns, invalid_format, truncated, doc_words,
   research: {research_error, research_experiments, research_episodes,
              research_turns, research_final_success, doc_missing,
              research_archive},                            # ← 反向指针(run 相对路径,可空)
   trajectory: {episode_id, task_id, guidance_hash, sample_idx, initial_obs,
                steps: [{t, thought, action_raw, action_norm, observation,
                         valid, acted}], outcome, won, error, meta},
   ts}
research/<task_id_safe>/<ts>_v<step>[-k]/
  doc.md            # 可缺(doc_missing)
  session.jsonl     # researcher 会话逐消息
  episodes.jsonl    # 每行 {trajectory, fact_card, narration, narration_skip, narration_conflict}
  meta.json         # {task_id, ts, policy_version, finalized, no_finalize, turns_used,
                    #  experiments_used, episodes_consumed, doc_words, self_assessment,
                    #  flags, last_exp_successes, last_exp_attempts, guidance_log,
                    #  session_error, researcher_prompt_tokens, researcher_completion_tokens,
                    #  researcher_model, tool_protocol, error}
tape/<role>.<pid>.jsonl · logs/train_*.log · launch_manifest_<ts>.json
```

**Join 键**:`(task_id, global_step)` 天然主键;`research.research_archive`
轨迹→研究目录;`meta.policy_version` = 训练 step。产物 append-only /
immutable(重跑带新 uuid/时间戳),增量索引因此可靠。

**曲线双源**:①dump 重算(P0)——`rollouts` 表 SQL 聚合即得 score、
env_steps、invalid_format、truncated、doc_words、research 族的逐 step 曲线
(与 swanlab `rollout_extra/*` 同口径:train 非 val、mean);②swanlab(P1)
——训练内部指标(loss/entropy/grad_norm/蒸馏族/显存/timing)只存在于
swanlab,经其 Python API 拉取缓存进 SQLite(在线不可达时曲线区标注缺源,
不阻塞其余功能;服务器实测后定接口细节)。

## 3. 技术形态:零新增依赖(有意为之)

**后端**:Python 标准库——`http.server.ThreadingHTTPServer` + 手写路由表、
`sqlite3` 索引、`json/pathlib/urllib`。**否决 FastAPI**:它要动服务器 venv
(训练环境纪律),而 pydantic/OpenAPI 对单人只读内网 API 无收益。零依赖 ⇒
162 上 `python -m trace_opd.viz` 直接起,不碰任何环境。
**前端**:no-build 原生 ES modules + 自写 SVG 图表模块(线图/柱状/堆叠,
数百 step 一屏内,无需缩放平移;不引 CDN/node 链,无供应链与断网问题)+
自写简版 markdown 渲染(doc.md 是受控结构)。CSS variables 深/浅主题。
**部署**:数据在哪服务在哪(162),默认绑 `127.0.0.1`,SSH 隧道访问
(沿用仓库安全纪律);纯 CPU 进程,与在跑实验完全正交。
**只读安全**:所有文件访问 `resolve()` 后强制在 runs-root 内(防路径穿越);
索引/缓存写到独立 `--data-dir`(默认 `~/.trace_opd_viz/`),**绝不写 runs/**。

## 4. 索引方案(SQLite)

表(P0):
- `rollout(run, file PK, step, validate, split, task_id, family, outcome, won,
  env_steps, assistant_turns, invalid_format, truncated, doc_words,
  research_final_success, research_error, doc_missing, research_archive,
  fail_label, plugin_ver, ts)` — 过滤/聚合字段全进库;thought 等大文本
  **不进库**(详情按需读文件)。`fail_label` 是插件分类结果(缓存,连同
  `plugin_ver`;插件升版自动重算)。
- `research(run, dir PK, task_id, family, policy_version, ts, doc_present,
  doc_words, experiments_used, episodes_consumed, final_success, finalized,
  session_error, researcher_model)`。
- `scan_state(run, last_scan_ts, …)`。
P1 增:`metric(run, step, key, value, source)`(swanlab 缓存)、FTS5 全文表。

**增量扫描**:文件 immutable ⇒ 差量 = 新路径;step/research 目录 mtime 剪枝
(目录未变不 listdir);损坏/半写 JSON 跳过并计数(下轮重试),不崩扫描。
规模账:444 step × 128 ≈ 5.7 万 dump,首扫全解析分钟级、增量秒级;SQL 聚合
在此规模远低于交互延迟阈。

## 5. 观察插件接口

```python
class ObservationPlugin(Protocol):
    name: str; version: str                    # version 变更 → 索引自动重算 fail_label
    def matches(self, task_id: str) -> bool    # 按任务命中(一个 run 可混多环境)
    def task_family(self, task_id: str) -> str
    def classify_failure(self, traj: dict) -> str | None   # 失败环节标签(won 样本不调用)
    def step_tags(self, step: dict) -> list[str]           # 领域事件标记(渲染用)
    def traj_tags(self, traj: dict) -> list[str]           # 轨迹级模式标签(如穷举搜索)
```
注册表在 `viz/plugins.py`;ALFWorld 实现 `viz/plugins_alfworld.py` v1 按
CD-006 口径:`never_moved`(从不成功 go-to)/ `moved_no_take`(移动了没拿到)
/ `took_no_finish`(拿到了没完成),附 `exhaustive_search` 标签(多容器
遍历无收获);**口径经 changelog 提请 CD-006 校验**,对齐其离线脚本后升 v2。
无插件命中 → `unclassified` + family 启发(task_id 首段)。

## 6. Provenance / deep-link / figure spec(一份机制,三处收益)

一切聚合都经**可序列化查询**(dataclass:run/step 范围/split/过滤器/分组/
指标/口径版本)。同一份序列化:
- **provenance**:聚合响应恒带 `{value, n, refs:[构成样本的 run 相对路径…],
  criteria:{query…, plugin_ver}}`;前端任何数字可点开来源清单、一键复制
  **引用串**(`38.3% (49/128; runs/<run>/rollouts/step_9/; classifier=alfworld@v1)`)
  ——落实"数字进报告必须带可追溯路径"铁律(报告数字从系统拷出即自带出处);
- **deep-link**:前端 hash 路由编码同一查询,每个视图/异常点可寻址、可进报告脚注;
- **figure spec**(P1):查询 + 样式(尺寸/色板/标签,ICLR 单双栏预设)存 JSON,
  matplotlib 后端管线重渲出版级 PDF/SVG;数据长了/修稿改样式,同 spec 一键重出。

## 7. 页面与 API(P0 面)

| 页面(hash 路由) | 内容 | API |
|---|---|---|
| `#/runs` | run 列表(进度/最新 step/健康摘要) | `GET /api/runs` |
| `#/run/<r>` | 总览:健康行、**完整性审计**(每 step dump 计数 vs 众数,缺失标红)、曲线格(dump 重算族;分 train/val,val 分 split) | `/api/run/<r>/overview` `/api/run/<r>/curves?keys&split` |
| `#/run/<r>/step/<n>` | 该 step 轨迹总览:分层统计条(成功/失败**恒分层**,防聚合口径误判)、失败环节分布、列表(筛选:结果/family/环节/split/副本),点击进轨迹 | `/api/run/<r>/step/<n>/rollouts?…` |
| `#/run/<r>/rollout/<file>` | 轨迹细察:逐 turn 折叠(thought 默认收起)、通用+插件事件标记、meta、**→research 文档**跳转 | `/api/run/<r>/rollout?file=` |
| `#/run/<r>/research?dir=` | doc.md 渲染 + meta(实验数/末次成功率/guidance_log/model)+ **→该 (task,step) 的 rollouts** 反向跳转 | `/api/run/<r>/research?dir=` |
| `#/run/<r>/task/<id>` | 同任务跨 step 历史(每 step 成功率、env_steps 分层、入口到各轨迹/各版文档)= "训练前后对比"工具化 | `/api/run/<r>/task/<id>` |

曲线点可点击 → 跳对应 step 页(验收用例 1 的下钻链)。

## 8. 分期(需求 P0-P3 + 定位升级增量)

- **P0(本期)**:内核+插件架构、扫描索引、§7 全部页面/API、provenance 与
  deep-link 内建、完整性审计、ALFWorld 插件 v1、fixture(alfworld+gridworld)
  与通用性测试。
- **P1**:swanlab 双源接入(metric 缓存表)、研究层完整版(session/episodes
  渲染、双向 join 完整化)、**跨 run 对比**(同指标多 run 叠加、三件套对照表、
  同任务跨 run 并置;消融判决工具,升格为一等功能)、图表导出管线 v1
  (曲线类)+ figure spec、step 时间线(timing 键发现)。
- **P2**:交叉分析(失败环节演化堆叠、文档特权增益按 family 收敛、任务类型
  剖面、research 资源画像)、**证据案例册**(收藏证据卡+标注+**冻结数据副本**
  到案例册目录+导出 markdown/LaTeX 片段)、指标工作台(自由 x/y/分组、相位图、
  存书签——P1–P5 判决作为预存书签的用法而非专用功能)、轨迹结构化 diff、
  FTS5 全文检索、SFT 数据源视角(会话质量分布/可入选预览)。
- **P3**:token 级蒸馏视图——**依赖机制侧采样式落盘**(全量 ~400MB/step 不
  可行;每 step 抽样轨迹存 top-k id/prob 或高分歧 token 邻域),按需求 §7
  预留通道单独与 CD-006/用户协商后立项。

**不做**(克制边界):自动报告生成;通用 BI/notebook;任意绘图 DSL(导出只
覆盖系统内视图类型);实验控制;多租户。swanlab 已做的指标追踪不重造,只
消费。

## 9. 验收(需求 §8 七条全保留,新增三条)

⑧ gridworld fixture(无插件)下内核全功能通过(通用性);
⑨ 任一聚合数字可复制出带样本路径与口径版本的引用串(provenance);
⑩ 任一曲线视图可导出 PDF 且由保存的 spec 重渲(P1 验收)。

## 10. 运行方式

```bash
python -m trace_opd.viz --runs-root <…>/runs [--port 8642] [--host 127.0.0.1] \
                        [--data-dir ~/.trace_opd_viz] [--rescan-interval 30]
```
服务器上直接跑(零依赖);本地访问 `ssh -L 8642:127.0.0.1:8642 <server>`。
端口默认 8642(避开仓库既有端口约定;冲突时 `--port` 覆盖)。
