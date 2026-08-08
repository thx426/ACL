# 9B 训练步生成阶段吞吐诊断与优化(CD-010,2026-08-08)

**数据源**:`TRACE-OPD-Q35/runs/q35_9b_formal` step1(EOS 修复后首个干净步),
A800 实机取证脚本 `/tmp/{timeline,load_curve,research_split2}.py`(时序由
rollout dump `ts−wall_s` 区间与 research 归档 mtime 重构)。

## 0. 结论先行

整步 **20.3 min**(11:10:34→11:30:50)。生成段 13.2 min 的墙钟结构不是算力
不足而是**调度空洞**:步首 2.3 min 引擎 student 并发为 0(全批在等各自任务的
research),步尾 4.7 min 并发 <10(128 槽位利用 <8%,尾巴=最慢 research 任务
+它的 16 条 episodes)。**已落码 P1(bare-first):bare 臂与 research 并行,
预计省 2–3.5 min/step(整步 −10~15%),零效果面**。更大的 P2(research 跨步
预取,整步 −35%)动效果面与显存账,列裁定项。

## 1. Step1 实测分解(9B formal,258 步全程按此外推 ≈5 天)

| 段 | 区间 | 墙钟 | 占比 | 引擎状态 |
|---|---|---|---|---|
| research 独占窗 | 11:10:34→11:12:52 | 2.3 min | 11% | student=0;仅 research eps(16 任务×k2,间歇) |
| student 主体 | →≈11:19 | ~6 min | 30% | 峰值 122 并发(30s 桶),有排队 |
| 长尾 | →11:23:46 | 4.7 min | 23% | 并发<10;尾=最慢任务 research(11:18:47)+其 16 eps |
| 训练段 | →11:30:50 | ~7 min | 34% | old_log_prob+update+sync(生成阶段之外,另议) |

关键读数:episode wall P50=60s/P90=131s/max=335s(bare 50s vs steered 80s);
research 总长 P50=226s/max=493s,其中**实验 episodes 占 62–88%(P50 68%)**
——research 墙钟大头也是本机引擎时间,35B(162)只占约 1/3。

## 2. 根因:任务级 research 串行前置于**全部** 16 条 episodes

v3 路径连 bare 臂(8/16,不消费文档)也在 rollout 前 await research——
原注释自认"只为携带同一份 research_fields(IF-09 schema 恒齐)"。于是:
128 条 bare(P50 50s)被 P50 226s 的 research 前置阻塞 ⇒ 步首空窗;
research 完成后 16 条同时涌入 ⇒ 122 并发峰与排队;最慢任务(research 493s)
拖出低利用长尾。v4 路径(selfres)早已是"bare 立即起跑",v3 没有对齐。

## 3. P1(已落码):bare-first 调度

- `common/aio.py` 拆 `shared_task_handle`(get-or-create 不等待;
  `shared_task` 复用,语义不变);
- `train/agent_loop.py` v3 分支:bare 臂只**触发** producer 即起跑 rollout,
  rollout 后 join fields(回填 `doc_missing`、tool_calls 记真实剩余等待);
  steered 臂照旧 await 后起跑;retired(M2 停产)分支照旧跳过 producer。
- **零效果面依据**:bare 生成与 doc 零数据依赖(doc 恒 "");policy version
  断言在 join 后执行且步内 policy 冻结,对拍语义不变;IF-09 靠 join 保证;
  research_fields 全部消费点(extra_fields/断言/dump)本就在 rollout 之后。
- 测试:`tests/test_aio.py` 新增 2 例(触发不等待/同 producer 共享/陈旧步
  清理),全套 395 passed。
- **预期读数变化**(CD-006 观察口径):步首 student 空窗 2.3min→≈0;
  尾段变 "最慢 research + 8 条 steered";`tool_calls` 均值显著下降
  (bare 行从 ≈research 全长变为剩余等待);generate_sequences 现为纯
  rollout 计时(原含 research 等待,口径变化,跨 run 比较注意)。
  估算净省 **2–3.5 min/step**。

## 4. P2(~~裁定项~~ **已否决**,2026-08-08 用户裁定):research 跨步预取(off-by-one)

> **否决理由**:后续将切换同 policy researcher(v4 自体研究员)——research 行
> 本身是训练数据且 doc 必须出自当前 policy,off-by-one 在机制上不成立。
> 本节仅留痕。v4 语境的吞吐分析另见 `docs/v4-selfres-throughput-analysis.md`。

训练段 ~7 min 完全能藏下 research(P50 226s/max 493s):在 step t 训练期间
预跑 step t+1 的 research ⇒ 生成段只剩 episodes,整步 20.3→≈13 min(−35%)。
两个硬前提,均需用户裁定:
1. **效果面**:doc 将基于 π_t 写、给 π_{t+1} 的 rollout 用(off-by-one);
   `assert_rollout_policy_version` 须预注册放宽为 version∈{t,t+1}。机制语义
   上 researcher 观察的是"接近当前"的 policy,步间漂移小,但这是效果假设。
2. **资源面**:训练期引擎必须活着接 research episodes——sglang 常驻
   (rollout_gpu_mem 0.55)与训练峰值的显存共存要实测(param_offload 在,
   余量未知);若显存不允许,P2 不可行。
不建议的替代:复用 research 探针 episodes 充当 bare 样本(引入 researcher
选择偏差,效果面污染,不提)。

## 5. 已核查、无空间/不动的项

- research 会话逐 turn 串行 + 每消息只执行第一个 tool call:**协议设计**
  (researcher 须见结果再决策),不动;
- 实验内 k=2 episodes 已 `gather` 并发;narration 已并发(2026-07-25 修,
  注释存 duty-cycle 32% 教训);
- SPS draft→verify 逐段依赖是机制固有(verify 决定续写前缀),verify 已
  同 replica 共享 KV;
- verl "全 256 样本完成才训练" 的 barrier:GRPO 组间无依赖,理论可流水,
  但属 verl 架构级改动,收益被 P2 覆盖,不做;
- 训练段 ~7 min 的内部分解(old_log_prob/update/sync)不属生成阶段,
  待 swanlab timing 曲线另议。
