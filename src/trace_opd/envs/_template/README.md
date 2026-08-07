# 新 benchmark 引入模板(参考实现 + 操作指南)

本目录是一个**完整可运行**的参考 benchmark(确定性玩具「信号接线板」),
与 ALFWorld 零共享。它的存在目的:把「机制核心 ↔ 任务特定」的全部交互点
落成可复制的代码,`tests/test_template_env.py` 用真实机制(episode 运行器、
v4 前缀续写、registry 装配)驱动它作为分离性的机器证明。

契约唯一真源:`src/trace_opd/core/adapter.py`(TaskAdapter / EnvFactory /
EnvHandle / ActionOutcome)。机制核心对任务协议零假定——动作语法(文本标签
/ JSON 工具调用 / 代码补丁)、环境执行接口、评估反馈形态(二元/部分分)
全部封装在 `react` / `replay_action` 之后。

## 机制 ↔ 任务交互点总表

| 交互点 | 机制侧(不改) | 任务侧(你实现) |
| --- | --- | --- |
| 任务装配 | `envs/registry.py::make_env_factory` 按 `environment.name` 分派 | registry 加一行 builder;`EnvFactory.adapter_from_item(extra_info)` |
| 数据集 | verl 读 parquet;agent_loop/run_eval/sftgen 把 `extra_info` 交给工厂 | 自带构建脚本,产出 `build_dataset.py` 注释的行契约(extra_info 必须自足) |
| executor 提示词 | guidance 经 `types.inject_guidance` 统一 header 注入 | `executor_scaffold()`(含动作文法与输出协议;锁定,researcher 不可改) |
| 每轮交互 | episode 循环:token 流、预算、防 spiral、记账 | `react(env, text) -> ActionOutcome`(解析+执行+反馈+评估,全权) |
| 动作协议 | 文本单动作可复用件 `protocol.text_action_react` | 委托它(实现 normalize/invalid/reminder 三件),或自写 react |
| 评估反馈 | `ActionOutcome.score: float` 保真进 Trajectory;GRPO 现口径 won=score>=1.0 | 终局时给 [0,1] 标量(bool 也成立;部分分一等公民) |
| 确定性 | 开局比对 initial_obs;v4 续写重放逐步比对反馈,漂移作废 | `open_episode` 确定性;`replay_action(env, action) -> (feedback, ended)` |
| researcher 面板 | 通用模板对交互形态零假定 | `task_brief()` 说清目标/机制/动作文法/评估口径(绝不含 GT) |
| 特权信息 | GT 开关只路由到 researcher;SFT 数据红线校验 | `gt_text()`(无 GT 返回 None) |
| SFT 分层 | `sftgen_researcher` 按 `task_type` 桶均匀抽样 | `task_type` 属性(无分类返回常量) |
| 叙事提示 | narrator 通用契约 | `domain_notes()`(通常空串) |

## 引入操作序列

1. **复制本目录** → `envs/<name>/`,实现 `adapter.py` 全部成员(逐成员注释
   即契约说明;红线:initial_obs 确定性、gt_text 信息状态、score 保真)。
2. **数据集脚本**:照 `build_dataset.py` 的行契约产 train/valid parquet
   (`agent_name` 恒为 `trace_opd`,写错 = verl 静默路由到别的 loop)。
3. **注册**:`envs/registry.py` 的 `_BUILDERS` 加一行(需要专属配置时照
   `AlfworldCfg` 模式加 config section,builder 里做 per-worker 资源折算)。
4. **配置**:实验 yaml 写 `environment: {name: <name>}`(默认 alfworld;
   非默认值自动进 objective_fingerprint——换环境=换优化目标)。
5. **测试**:照 `tests/test_template_env.py` 写同款四组:契约完备、
   episode 端到端(赢局/部分分/格式失败提醒)、v4 前缀续写端到端、registry
   装配。**另加两类定向自查**(2026-08-07 审阅实证:逐环境复制最易走样的
   两处)——①资源记账平衡:episode 失败路径上 slot/gate 恰好释放一次
   (参照 `test_slot_balance_on_boot_failure`);②漂移哨兵真触发:
   `env.initial_obs` 必须 live 重构造(不得回传捕获值,否则比对恒真空转),
   并用被改动的资产断言 ENV_ERROR(参照 `test_initial_obs_drift_detected`)。
   全绿后机制侧零改动即可跑训练/评估/SFT 三条流水;起跑前置用统一预检:
   `python -m trace_opd.envs.preflight --config <yaml> --parquet <valid>`。

## 机制侧已知边界(诚实声明,勿绕过)

- **反馈的消息形态**:episode 的 token 流把环境反馈作为 user 结构段拼接
  (`TemplateDelta`)。React 抽象覆盖「反馈是一段文本」的所有协议;若目标
  benchmark 需要 chat 模板原生 tool-role 消息(如 OpenAI tools 格式的
  `role:"tool"`),需扩展 TemplateDelta 的结构段变体——接线点在
  `core/episode.py` 的流构造与 `train/backends.py`,属机制演进,先与机制
  分支对齐。
- **部分分进 reward**:`score` 已保真到 Trajectory/ContinuationOutcome,
  但当前训练 reward(GRPO 二元 outcome)与 v4 续写聚合读的是派生 won。
  把连续 score 接进 reward 是训练口径决策(agent_loop 的 rm_scores 组装
  与 `continuation.aggregate_reward`),不是协议改动。
- **episode 预算语义**:`MAX_ENV_STEPS`/`MAX_ASSISTANT_TURNS` 是机制侧
  统一预算;`ActionOutcome.consumed_env_step` 决定哪类交互计入前者(免费
  查询类动作可不计,但要在 task_brief 里向 researcher 说清)。
