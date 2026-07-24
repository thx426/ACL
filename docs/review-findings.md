# Code-review 研判与处置记录(Codex gpt-5.6-sol ultra,2026-07-23)

> 逐条独立验证后的裁决。verdict:**采纳**(已修)/ **部分采纳**(修其成立部分,注明取舍)/
> **不采纳**(注明理由)/ **决策**(非 bug,记录设计裁决)。
> 修复落点:trace_opd 代码 + verl trace-opd 分支 2 个新 commit(69e8129 validate 传递、
> 71d5dec P0-1 对齐修复)。测试 48→65 全绿。

## P0(全部采纳)

| # | 验证结论 | 处置 |
|---|---|---|
| P0-1 teacher 首 token 错位 | **属实**。逐行核验 `extract_prompt_logprobs`(丢 None 行 0、尾补 dummy ⇒ 返回行 i=位置 i 的预测)与 loss 切片 `[P-1:P+R-1]`:旧 patch 从 T 切 ⇒ 首个 response token 吃零行,其余 R-1 个 token 对齐正确(这解释了 h3 结果仍强——每样本仅 1 个坏 token,~0.02% 污染)。**h3 全量 run 带着此 bug** | verl commit 71d5dec:改切 `[T-1:-1]`,布局 (P-1) 零行 \| R 有效行 \| 1 dummy;运行时哨兵断言(首行必须含负 logprob);`tests/test_teacher_alignment.py` 纯 python 重演三段语义锁死(含旧错位演示与非特权直通对齐验证) |
| P0-2 V1 global_steps wrapper | **场景链路属实**(V1 默认开、`assign_non_tensor_data` 包装、TQ 传 wrapper 给 get_trajectory_info)。tensordict 取值是否自动解包有版本不确定性 → 防御性修复不赌行为 | `agent_loop._as_int()` duck-typing 解包(`.data`)+ int 化失败置 None;`write_json` 加 `default=str`(审计文件宁存字符串不丢文件);测试 `test_archive_meta_survives_unserializable_policy_version` |
| P0-3 降级边界不完整 | **属实**:try 只包 session.run(),装配路径裸奔 | 两层修复:①系统性错误 fail-fast——launcher 启动前探活 researcher endpoint(/models + model 名校验),config 加载时枚举/组合校验(backend、tool_protocol、gt、roles 齐备性、engine+fncall 禁止);②per-sample 全链路(router/session/prompt/gt 装配)入 `_research` 的 try,任何失败降级 doc=None 并计数 |

## P1

| # | verdict | 处置 |
|---|---|---|
| P1-1 并发×8 放大 | **采纳** | 语义定为"yaml 写全局上限":`train.agent_num_workers` 进 yaml 并显式传 verl;launcher 导出 `TRACE_AGENT_WORKERS`;agent loop 内 researcher/narrator max_concurrency 与 alfworld spawn_concurrency 按 worker 数折算(`_per_worker`) |
| P1-2 abort/空输出 | **采纳** | EngineGenerate 透传 `stop_reason`;runner 将 `aborted` 分类为 ENV_ERROR(研究 refund / 训练重试语义);末尾归一化守卫 ENV_ERROR 不被覆盖(测试暴露并修复);空 response 两次后仍 raise(避免 rm_scores[-1] IndexError,已核验该 IndexError 真实存在) |
| P1-3 LR 调度不符 | **部分采纳(方向修正)** | 核验 verl 默认 `lr_scheduler_type: constant`、h3 脚本未传调度参数 ⇒ **h3 实际以 constant+0 warmup 跑**,其 notes 表格"cosine+warmup0.03"系误记。正确做法不是改成 cosine,而是**显式钉死 constant/0**(yaml 字段 + launcher 显式传参),保持与 h3 真实行为可比;yaml 注释写明勘误 |
| P1-4 3553 覆盖 | **部分采纳** | build 强校验 train=3553(EXPECTED_COUNTS);yaml 注释修正为 110 step、drop_last 语义(每 epoch 33 任务不入批,随 shuffle 轮换)。**不做 pad/oversample**:h3 同为 drop_last,同口径可比优先(单变量纪律)。**口径更新 2026-07-23(应 CD-006 feedback,原裁决不变)**:用户评审改 batch=32 后为 **111 step/epoch、2 epoch=222 step、每 epoch 仅 1 任务轮空**(现行口径见 `design.md` §6 与 trace.yaml);本行"110 step/33 任务"是 batch=64 时代数字,仅作评审当时留痕 |
| P1-5 task_type=train | **采纳** | adapter 跳过 split 段解析;`test_task_type_skips_split_segment` |
| P1-6 timeout 泄漏子进程 | **采纳** | boot 走 `shield` + 取消时 done-callback 移交后台线程关闭;finally close 亦 shield(取消态下关闭线程继续) |
| P1-7 中途失败丢记录 | **采纳** | session.run() 不再抛(除 Cancelled):异常返回 partial SessionResult(error 字段+全部 messages/计数);暴露 `snapshot()`/`messages`,超时场景 agent loop 经 snapshot 归档 partial;archive 记 `session_error` |
| P1-8 无探活 | **采纳**(circuit breaker 除外) | launcher preflight(见 P0-3);运行中熔断器不进第一轮——doc_missing/research_error 已逐样本入 dump+日志,冒烟阶段人工监控足够,避免额外干预机制 |
| P1-9 评测分母 | **采纳** | `summarize()`:success_rate=wins/attempts(排除 ENV_ERROR)、单列 env_errors、incomplete 标记 + exit code 2;函数抽出可测 |
| P1-10 输出目录混写 | **采纳** | episodes.jsonl 已存在即 FATAL(`--force` 覆盖删除) |

## P2

| # | verdict | 处置 |
|---|---|---|
| P2-1 obs 漂移无校验 | **采纳(比 h3 更严)** | runner 内 dataset vs live initial_obs 比对,不一致 ⇒ ENV_ERROR(h3 仅 warning;研究者研究错场景是数据污染,fail loud) |
| P2-2 steps 口径 | **采纳** | evalm 记 `env_steps=sum(acted)`(h3 可比)+ assistant_turns |
| P2-3 保留量/turn cap | **采纳** | 结构追加保留量 32→96(保证追加后必有下一轮生成,无悬空 gen_prefix);turn cap 非正常退出置 truncated;两条新测试 |
| P2-4 teacher scaffold 硬编码 | **采纳** | teacher prompt 改 `inject_guidance(adapter.executor_scaffold(), doc)`(与 student 同源,多环境正确);前置 max_model_len 校验,超长降级 bare(计 doc_missing) |
| P2-5 ToolProtocol 语义 | **采纳**(parallel_tool_calls 参数除外) | 单 call/消息客户端强制(执行首个+IGNORED 提示);text 协议 raw_arguments 语义与 fncall 对齐(成功解析存规范 dump,合法空 {} 不再被误拒);malformed_hint 机制(marker 存在但解析失败 → 软纠错消息)。不传 `parallel_tool_calls=false`(vLLM 版本兼容风险,客户端强制已兜底) |
| P2-6 parser TypeError | **采纳** | 捕获 (JSONDecodeError, TypeError, ValueError) |
| P2-7 GT 元数据/fallback | **采纳** | build 记 `gt_source`(annotation/walkthrough/none)进 parquet;archive 记 `gt_injected`;gt.py 全类型防御("plan": null 等落 fallback 而非 AttributeError 中止构建) |
| P2-8 builder 参数真源 | **采纳** | build_datasets 接 `--config` 复用 AlfworldCfg;boot_concurrency 入 yaml |
| P2-9 worker 资源缺口 | **部分采纳** | kill 后 wait/reap + 幂等关闭 stdin/stdout(采纳);**不改非阻塞 IO**:select+readline 在 pilot 6000+ episodes 零事故,半行挂死属极低概率,重写成本/风险不成比例(记录为已知边界) |
| P2-10 research 曲线/计时 | **部分采纳** | 不给 verl _compute_metrics 加聚合 patch(动 verl 面大);替代:per-sample rollout dump(research 字段全量入 dump,离线聚合)+ 每会话 grep 行。计时保持 tool_calls=研究、generate_sequences=学生 rollout 的粗粒度映射,细粒度在 dump 的 wall_s。**口径更新 2026-07-23(用户指令扩指标,原裁决当时理由留痕)**:research 聚合指标改为**上曲线**——发现 `reward_extra_info` 现成通道后聚合仅 10 行(verl `f8f26d5` + `train/curve.py`),"动 verl 面大"前提不再成立;dump/grep 通道保留不变 |
| P2-11 tape 跨进程写 | **采纳** | tape 文件名加 pid 分片(`researcher.<pid>.jsonl`) |
| P2-12 launcher 真源/override | **部分采纳** | 采纳:launch manifest 落盘(resolved cmd+config snapshot+overrides)、passthrough 醒目警告、RoleCfg/顶层 section 未知键 fail、枚举/组合校验。**保留** launcher 内固定 profile 常量(它们是 h3 定版绑定的单一出处,全部入 yaml 会稀释评审焦点;manifest 提供完整审计) |
| P2-13 checkpoint 绑定 | **采纳** | `trainer.default_local_dir=$RUN_DIR/checkpoints`——checkpoint/研究归档/rollout dump/磁带同 run 目录,resume 同源 |
| P2-14 overlong filter | **采纳** | `filter_overlong_prompts=False`(超长走 truncation=error loud fail,符合"无静默过滤"设计;ALFWorld prompt 实际远低于 1024,h3 的 True 从未生效) |

## P3

- **P3-1 测试盲区**:部分采纳——本地新增 17 个测试(对齐语义/episode 边界/partial/单 call/malformed/task_type/评测分母/meta 兜底);真实 V1 TensorDict e2e、Ray 多 actor、真 subprocess 取消属服务器冒烟范畴(本地无 GPU/torch),已列入冒烟清单。
- **P3-2 eval provenance**:采纳——manifest.json(model/served/api/parquet/预算/温度/协议 hash/config 快照)。

## 疑点裁决(设计决策,非 bug)

1. **mid-episode ENV_ERROR 的前缀参与蒸馏**:保留。纯 token 级蒸馏下故障前前缀是有效 on-policy 数据;outcome 已入 extra_fields/dump 可审计。低频(pilot 零 ENV_ERROR)。
2. **external stop-only 空文本**:接受 truncated 语义,不改(greedy 下空输出即模型异常,罕见)。
3. **researcher context 无上限**:27B 64k ctx、h1 实测中位 6 轮,不加限制;tape 记录 usage,冒烟观察。
4. **宽松 action 解析 vs 严格契约**:保持宽松(仅要求可提取 `<action>`),h3 同款、训练/评测/研究同构优先;严格格式是提示词约束而非程序校验。

## 未采纳汇总(3 项,均记录理由)

runtime circuit breaker(P1-8 部分)、worker 非阻塞 IO 重写(P2-9 部分)、
verl metrics 聚合 patch(P2-10 部分)——共同原则:第一轮以最小干预面换取机制
纯度与可比性,观测靠 dump/日志/manifest 补足。
