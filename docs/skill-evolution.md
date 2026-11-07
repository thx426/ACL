# 错误定位、局部修订、独立验证

本分支增加可选的 `research.evolution` 模式。研究员继续使用同一个
`ResearchSession`、工具协议、执行模型、轨迹叙述和 `fetch_raw`。旧配置默认关闭
该模式。开启后，`finalize` 只能输出已通过验收的 Skill。研究员不能用最终文档
绕过验收。这里的 Skill 是注入执行模型的经验文档。

## 三个阶段

1. **错误定位。** `run_experiment()` 固定当前 Skill 和执行模型。每个配置的提示
   采样 `samples` 次，默认 4 次。开发阶段允许两个提示。程序保存完整轨迹，并按
   步骤顺序寻找首次有证据支持的错误。研究员可以用 `record_error` 添加诊断说明。
   诊断必须引用真实轨迹和步骤。程序不能确认的诊断保存为 `uncertain`，不能作为
   修订依据。终局失败本身不能证明最后一步是首错。
2. **局部修订。** `propose_revision` 每轮只接受一个 `error_type`。每条规则都有
   触发条件、检查步骤、执行动作、完成条件、恢复方法和来源错误编号。新规则追加
   到初始 Skill；已有规则只能在相同错误类型下更新。其他规则保持原文。候选版本
   在验证前封存，保存旧、新文本及 SHA-256。规则来源必须是当前 Skill 的发现轨迹，
   不能引用验证轨迹或其他任务。
3. **独立验证。** `validate_revision` 由程序产生旧、新 Skill 的成对执行。两臂
   使用相同提示、温度、预算和种子；各阶段使用不同的新种子。程序检查每次生成的
   policy version。局部续跑、原始起点的完整执行和保留任务分别统计。只有所有适用
   条件通过时，程序才替换当前 Skill。拒绝或中断均保留旧版。

多提示只用于发现阶段。所有验证使用 `validation_prompt`，不使用研究员提供的
提示或自报分数。执行模型仍由现有 `EngineGenerate` 提供。此功能不增加模型 API。

## 错误记录与可确认范围

每条记录保存 `task_id`、`episode_id`、轨迹 SHA-256、Skill 指纹、任务要求、
动作前观察、动作、环境反馈、步骤、错误类型、验证器、判定依据和研究员说明。
`step` 是从 1 开始的执行模型轮次，包含无法解析动作的轮次。

支持四种错误类型：`illegal_action`、`missing_precondition`、`wrong_choice`、
`repeated_loop`。内置检查只能确认无法解析的动作或适配器标记为无效的动作。
这类记录使用 `illegal_action`，不声称已经发现更早的语义原因。单纯出现三次相同
动作和反馈只能产生“不确定”的循环候选，因为隐藏状态可能仍在变化。

环境可以实现可选方法 `diagnose_error(trajectory, index)`。`index` 从 0 开始。
该方法只能读取当前步骤及其之前的轨迹。返回 `None`，或返回
`{"error_type": "missing_precondition", "reason": "可核查的环境依据"}`。
该方法应使用确定性的环境规则，不能使用研究员自己的结论。它可以确认遗漏前提、
错误选择或重复循环，也可以细分通用的无效动作。没有对应验证器时，这些判断保持
`uncertain`。增加新环境时，应同时测试其判定规则。

## 局部验证与替换条件

适配器必须显式声明 `supports_exact_replay = True` 才能执行局部对照。
本分支为 ALFWorld 启用此能力。程序从原始起点重放首错之前的真实动作，并逐步
核对观察。它保留原执行模型的 token 前缀，以及已经消耗的环境步、轮次和 token
预算。旧、新 Skill 都从同一个错误发生前的状态续跑。观察不一致、提前终止、
策略版本变化或基础设施错误都会使该对照无效，且阻止替换。

局部 `correct_action_rate` 指第一步动作可执行且没有被独立检查器判为错误的比例。
它是针对已支持错误类型的纠正率，不代表对所有动作都有完整的最优动作判定器。
局部成功率是从该状态继续执行到成功的比例。环境未声明精确重放时，报告显式记录
`unsupported`；完整执行和保留任务检查仍必须执行。

默认替换条件如下。比例差的单位为 0 到 1，`0.01` 表示一个百分点。

- 所有旧、新对照都必须取得完整的有效样本；不能丢弃基础设施故障后仍通过。
- 所有测得的局部对照：第一步纠正率严格提高，后续成功率不下降。
- 从原始起点完整执行：目标错误的轨迹发生率至少下降 `min_error_reduction`。
- 同一组完整执行：整题成功率至少提高 `min_success_gain`。
- 每个保留任务：成功率下降不超过 `max_holdout_regression`。
- 发现与验证阶段的执行模型版本保持一致，候选文本和配置未被修改。

默认三个阈值依次为 `0.01`、`0.01`、`0.0`。四次采样时，实际成功率以 0.25 为
步长变化。该门槛用于版本筛选，不提供统计显著性或跨任务泛化保证。正式研究应在
运行前选择样本数和阈值，并另外保留最终测试集。

## 启用

将 [配置片段](../configs/examples/skill-evolution.yaml) 中的 `evolution` 合并到
现有实验配置的 `research` 中。初始 Skill 默认是空文档，即裸执行模型。
也可以在 `initial_skill` 中提供已有 Skill。一个会话内的每轮会继承上次通过的版本；
此改动不引入跨训练 step 的文档自动回读。

`holdout_tasks` 是 JSONL 文件。每行是环境工厂所需的一个 dataset `extra_info`
对象。例如 ALFWorld 需要 `task_id`、`gamefile`、`initial_obs` 等字段。应从独立的
开发保留集生成该文件，并排除本次发现阶段的任务。不要使用最终测试集。
会话开始时固定文件内容。保留任务不能重复。缺少该文件、工厂或有效任务时，默认
拒绝替换。确实只评估单任务时，可以显式设置 `require_holdout: false`；此时报告
没有跨任务回归检查，不能据此声称满足跨任务替换条件。

现有 launcher 支持配置覆盖。例如，在已完成环境和模型设置的服务器上：

```bash
python scripts/launch_train.py \
  --config configs/experiments/a800-pgs-s2-sglang-researcher.yaml \
  --set research.evolution.enabled=true \
  --set research.evolution.holdout_tasks=/absolute/path/development_holdout.jsonl \
  --set 'research.evolution.discovery_prompts=["", "Check the task requirements before each action."]' \
  --set research.evolution.samples=4 \
  --set research.max_prompt_tokens=8192 \
  --dry-run
```

正式对照的发现阶段也应使用固定提示：`discovery_prompts: [""]`。上述双提示只用于
开发。`validation_prompt` 默认空字符串，即环境原有固定 scaffold。开启本功能后
研究员工具说明变长，应根据真实 tokenizer 检查 prompt 上限。自体研究员模式还需
满足已有的模型上下文和训练 padding 校验，不能只增大一个预算。

`run_experiment` 的一次调用消耗一个实验名额；一次 `validate_revision` 也消耗一个。
验证包含所有必要的对照，执行次数单独记录在 `episodes_consumed`。默认一个来源
错误、一个保留任务、四次采样时，验证需 24 次执行：局部 8 次、整题 8 次、保留
任务 8 次。环境不支持局部重放时为 16 次。`max_rounds` 默认 3。

## 归档与测试

外部研究员的现有归档目录新增以下文件：

- `errors.jsonl`：可核查的错误记录，包括不确定诊断。
- `skill_validations.jsonl`：每轮测量值、阈值、拒绝原因和最终决定。
- `skill_evolution.json`：完整配置、模型版本、所有候选 Skill、所有轨迹和替换历史。

原有 `session.jsonl`、`episodes.jsonl` 和 `doc.md` 继续保存。自体研究员的 rollout
记录增加 `skill_evolution` 字段，保存同一份完整审计数据。未通过验证的候选不进入
下游 Skill 文档，但仍保留在审计数据中。相关配置及保留任务内容进入训练目标指纹。

本地测试不调用真实模型或环境：

```bash
python -m pytest -q tests/test_skill_evolution.py tests/test_episode.py \
  tests/test_episode_edges.py tests/test_research.py tests/test_research_edges.py
```

测试覆盖旧流程兼容、首错证据、不确定诊断、单错误类型、策略版本变化、种子隔离、
精确重放漂移、局部成功但整题失败、保留任务退化、验收中断和归档。真实成功率和
服务器上的预算适配需要实际实验验证。

本次实现的针对性回归为 100 项通过、1 项跳过。全量检查为 599 项通过、14 项
跳过、2 项失败；之后增加的两项边界测试已包含在针对性回归中。两项全量失败也在
原始 `main`（`b4c7d619`）复现：`test_maxopen` 的既有 epsilon 校验预期未满足，
以及 WebShop 测试依赖仓库外的 `experiments/bench_env/classic_bench/webshop_service.py`。
