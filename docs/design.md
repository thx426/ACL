# TRACE-OPD 正式机制设计(v1,CD-005)

> 2026-07-23,与用户三轮对齐后的定稿形态。本文是实现的单一设计真源;
> 全部超参数的单一配置真源是 `trace_opd/configs/trace.yaml`(实施完成后先经
> 用户评审再跑)。历史依据(两个 pilot 的实证)只作引用,不在此复述。

## 1. 机制一句话

对每个训练 step 的 batch 任务:**researcher agent 现场做真实实验产出该任务的
经验文档(compute scaling 获取信息)→ 文档作为特权信息构成 self-teacher →
对 student 裸 on-policy rollout 做标准 OPD 蒸馏,把信息内化进 policy 权重**;
按 step 迭代,与 RL 训练循环同构。

## 2. 训练循环(核心形态)

```
训练集 shuffle → 按 train_batch_size 划分 batch → 每个 step:
  batch 内每任务(并行,全部发生在 verl AgentLoop.run() 内):
    1. researcher 会话(默认外部 27B):
         run_experiment(guidance, k) → k 个真实 episodes
           【episodes 调 verl rollout engine = 当前 step 的 policy 权重】
           → fact card + 叙述(默认 policy 自叙)+ 评测结果 反馈
         fetch_raw 取证 | 迭代改进 | finalize → 当次经验文档(现场产物)
    2. student 裸 rollout(同 engine、同权重、同协议、append-only 构流)
    3. teacher_prompt_ids = 裸 system prompt + GUIDANCE_HEADER + 文档
  → self-teacher 打分(现有 verl patch)→ forward_kl_topk 纯蒸馏更新
  每 test_freq step:test 集(seen140+unseen134)裸 greedy 评测;每 save_freq 存 ckpt
```

关键语义:

- **文档无跨步生命周期**:每次都是 researcher 基于当前 policy 行为的现场研究
  产物,用完即归档(归档仅审计/未来 SFT 数据源,永不回注下次研究)。
- **多 epoch 任务重见** = researcher 面对已进化的 policy 重新研究,与上一条自洽。
- **无脉冲/无闸门/无冻结/无输入过滤**:标准 OPD 连续训练;只保留 verl 标准
  的曲线 logging 与定期 checkpoint。

> **循环形态演变(2026-07-24/25,用户裁定,机制语义不变)**:
> ① researcher 由本地 27B sidecar 迁**外部 qwen3.6-35b-a3b 双端点池**(经隧道,
> `PooledClient` 会话粘性+least-active 均衡+故障切换;27B 下线,本地卡归 policy
> 独占;详见 `docs/collab/cd005-cd006/req-external-researcher-pool.md`)。
> ② 研究会话与学生裸 rollout **并行**(学生不依赖 doc,doc 只在 teacher 打分处
> await;并发上限 21→112,3401ba5)。
> ③ self-teacher 打分双后端:`TRACE_TEACHER_SCORING_BACKEND` = `vllm`(原路径,
> rollout 引擎 prompt_logprobs)/ **`fsdp`(主线)**——打分迁训练引擎 no-grad
> 前向,行语义严格复刻,根治 vLLM logprobs 预算外分配(见 §6 R11)。

## 3. 角色 × 后端矩阵

LLM 角色统一经 `backend` 声明解析(`common/config.py` + `train/backends.py`):

| 角色 | 默认 backend | 依据 |
|---|---|---|
| researcher | `external`(OpenAI-compat URL,起步 Qwen3.6-27B) | 外部强模型场景:任意机器起 vLLM server,机制只认 endpoint、只探活不管理;未来 policy 自进化切 `engine` |
| narrator | `engine`(policy 同模型,用户定) | 自进化自包含;可切 `external` 对照 |
| executor(研究实验 episodes) | `engine`(固定) | 权重同步硬要求:研究对象必须是"此刻被训练的 policy" |
| student rollout / teacher | `engine`(固定) | verl 原生 + `self_teacher` patch |

- `engine` 后端 = verl AsyncLLMServerManager(`generate(prompt_ids, sampling_params)`
  token-ids 口径,权重每 step 自动同步)。
- `external` 后端 = `common/llm.py` OpenAI-compat 客户端(三重工具解析防御 + 磁带,
  h1 实证移植重写)。
- researcher 工具交互经 **ToolProtocol 抽象**:`fncall`(vLLM function-calling,
  qwen3_xml,27B 用)与 `text`(纯文本工具协议,为 policy-as-researcher 与未来
  researcher RL 预留;第一轮实现接口与协议、不投入调质量)。

## 4. 协议与同构纪律(SSOT)

- episode 协议统一 **CD-003 定稿口径**(PG-3):多轮对话式、无 admissible list、
  `<reasoning>...</reasoning>\n<action>...</action>` 契约、engine-verbatim 13 命令表、
  format-failure×3 熔断、append-only token 构流(TemplateDelta 硬断言)。
  **步数预算参数化(2026-07-23 用户裁定,偏离 CD-003 的 50)**:trace.yaml
  `episode.max_env_steps=30` / `max_assistant_turns=40`(依据:CD-006 用 h3
  真实分布验证 cap=30 保住训后 94.2% 成功、中位仅 11 步,失败 episode 不再烧满
  预算;触及 PG-3 口径,由 CD-006 走 Proposed Global Decisions 报 master);
  GT walkthrough 提取上限独立保持 50(`GT_EXTRACT_MAX_STEPS`,专家参考解不受
  训练预算截断);最终评测拟另跑一组 50 步与 pilot 严格可比(改 yaml 即可)。
- **一份 EpisodeRunner,三处共用**(`core/episode.py`):研究实验 episodes、
  训练 student rollout、独立评测器,同一构流/解析/熔断实现,仅 GenerateFn
  (engine / external)与 guidance/温度参数不同。researcher 观察到的行为分布
  = 被训练的分布 = 被评测的分布。
- 研究实验 episodes 采样温度 = 训练 rollout 温度(1.0);评测 greedy。
- 注入格式单一真源:`GUIDANCE_HEADER = "## Task guidance"`(PG-7),研究期
  guidance 注入、teacher 文档注入、评测 +doc 臂三处 byte 级同格式。

## 5. 任务抽象(通用泛化)

`core/adapter.py` 的 TaskAdapter 协议(h1 实证形态,按 h3 协议修正 executor 面):
`task_id / task_brief() / executor_scaffold() / initial_obs / gt_text() /
normalize_action() / domain_notes() / open_episode()`。核心机制零环境 import;
`envs/alfworld/` 是第一个实现(worker 子进程 + spawn gate + GT 双源提取)。
训练侧数据构造(`train/build_datasets.py`)走同一协议:parquet 行含
task_id/gamefile/initial_obs/split/**gt_text(预提取)**,GT 是否给 researcher
由运行时开关决定(数据总是带上,离线一次提取,训练时零额外引擎访问)。

## 6. 训练集成(verl)

- verl 分支 `trace-opd`(自 h3-pilot 分支起,同 5 个通用 patch,预计零新增):
  `self_teacher=True` + `extra_fields["teacher_prompt_ids"]` per-sample 特权前缀。
- `train/agent_loop.py::TraceAgentLoop(AgentLoopBase)` 注册名 `trace_opd`:
  上述循环第 1–3 步的完整编排;研究会话任何异常 → 记录并降级为 doc=None
  (teacher=bare,蒸馏信号≈0,中性),**绝不杀训练 step**。
- doc=None 的样本不设 teacher_prompt_ids(bare self-KL≈0,自然中性)+ 计数上报。
- 监控标量经 extra_fields 上报:research_final_success(会话末次实验成功率,
  免费的文档质量过程信号)/ research_experiments / research_episodes /
  doc_words / doc_missing / research_error。
- 训练超参底座沿 h3 稳定 profile(dynamic bsz / lr 1e-6 **constant 调度**
  ——h3 的实际行为,其 notes 表格"cosine"系误记 / fkl_topk64 纯蒸馏 /
  mini_bsz 32),**用户评审裁定的三处偏离(2026-07-23)**:
  ①`response_length=32768`(=评测预算,50 步 episode 永不截断——"不因输出
  长就不训练";联动 MAX_TOKEN_LEN=34816 / ROLLOUT_MAXLEN=40960,即 h3 脚本
  32k 档原生值;h3 的 10240 系当时 52GB/卡 的显存妥协,新拓扑训练阶段
  ~76GB/卡,32k 稳定性列冒烟第一验证项);②`train_batch_size=16`(h3 为
  64;2026-07-23 用户裁定 16×8 组合,经 CD-006 转达);③`rollout_n=8`
  (h3 为 1,用户机制升级):每任务 n 条独立 on-policy 轨迹(温度 1.0)
  共享**同一个**研究会话/文档蒸馏——verl 侧走 `rollout.n` 原生链路
  (interleave repeat、teacher per-sample 打分、mini_batch 内部 ×n 故每
  step 仍 1 次全批更新),trace 侧研究会话按 (step, task) 去重恰跑一次
  (`common/aio.py::shared_task`;去重前提 train_batch_size %
  agent_num_workers == 0,加载时强校验)。researcher(27B,step 墙钟
  大头)算力不随 n 增,同一文档的蒸馏信号 ×8;doc=None 降级时 n 条副本
  同为中性样本。rollout **TP2**
  (h3 为 TP1;长程 KV 余量,冒烟对照后定版)。launcher
  (`scripts/launch_train.py`)从 trace.yaml 单真源生成 verl 命令行,启动前
  探活 researcher endpoint(sidecar 模式加验 `/is_sleeping`),并把 resolved
  命令与配置快照写入 launch manifest。
- 数据口径:全量 3553、batch 16、drop_last ⇒ **222 step/epoch,2 epoch =
  444 step**(每 epoch 仅 1 任务随 shuffle 轮空);每 step 训练轨迹 =
  16 任务 × rollout_n 8 = **128 条**(h3 为 64 条/step,n=1);每 epoch
  研究会话总量仍 3553(每任务恰一次)。test_freq 20 ⇒ 22 个 val 点
  (用户第二次裁定保持 20:间隔 320 任务,时间分辨率比 h3 细一倍,
  代价 val 墙钟 ≈6.6h)。与 h3(batch 64)对比结论时注意口径差。
  评审全部裁决见 `docs/review-findings.md`。

### 6.2 R11:self-teacher 打分迁训练引擎(fsdp 后端,2026-07-26 定版主线)

vLLM `prompt_logprobs` 的 logprobs 分配在 `gpu_memory_utilization` 预算之外
(vllm issue #5907)——rollout 尾部 76-81GB 显存尖峰与 vLLM 进程 torch 驻留爬升
(5→15GB)的共同根因。R11(CD-006 实施、经 codex 深评、生产冒烟通过后收编)把
self-teacher 打分移到训练引擎:rollout 侧跳过 vLLM 调用,特权前缀随
`extra_fields`(`teacher_mode`/`teacher_topk`/`teacher_prefix_ids`,provenance
fail-loud)到 trainer,`update_actor` 入口对每样本 no-grad 前向取 top-K
(`verl/trainer/distillation/teacher_topk_fsdp.py`,行语义严格复刻 vLLM 路径:
prefix+response 打分行、per-sample 零填充/有效/dummy 布局、fp32 logsumexp)。
权重相等性:update_actor 在本 step optimizer 更新之前执行 = vLLM 本会用的权重。
等价性实测:overlap_ratio 0.850(vLLM 基线带 0.828-0.848)、teacher_mass
0.99880、行级 KL 均值 7.7e-04(小于训练信号 3 个数量级);rollout 尾部尖峰
81.2→47.2GB 根除,并顺带消除跨引擎数值 mismatch(旧 P3)。配套:
`logits/temperature` 恒等除法跳过(seqlen×vocab fp32 副本 13.5GiB@22k,
step-16 OOM 触发点,逐位等价);`rollout_max_num_batched_tokens` 与后端联锁
(fsdp→4096 恢复 h3 铁稳值,vllm 回退→2048)。开关默认 fsdp(脚本层),
`vllm` 保留为回退路径。

### 6.3 混合目标 OPD × RL × SFT(2026-07-26 用户需求;默认全关)

设计真源:**`docs/mixed-objective-design.md`**。一句话:总损失
`L = OPD + mix_rl_weight × RL + mix_sft_weight × SFT`,两权重默认 0 ⇒ 与纯 OPD 逐字节
一致。动机是弱模型(1.7B)上 self-teacher 对 student 自身错误前缀打分时,文档的语义
抓地力迅速衰减、KL 起不来(CD-006 五环因果链 + framing 探针的 on-policy 天花板;
SDAR 在同基座同环境复现同一现象)。RL = GRPO(轨迹 won);SFT = 每任务额外 4 条
**带文档** rollout 的成功轨迹,重放到裸 prompt 后做 CE。**定位是正式的可开关可消融
机制(A0-A3 臂),不是 1.7B 补丁**——PG-4 的"锚/RL 联合留后续"与 CD-003 的"真修复=
冻结 teacher + 锚"在此落地,故触及 PG-1/PG-5 的"第一轮纯 OPD",进完成报告
Proposed Global Decisions 报备 master。

### 6.4 信号增强三件套 K1/K2/K3:已移除(2026-07-26 用户裁定)

CD-006 在 1.7B v2 时代实验过的 loss/打分侧三改动(K1 clamp 软化、K2 KL^α
自适应加权、K3 分块保鲜打分)按用户裁定**从机制中整体移除(非关闭)**:
K2 实测 no-op;K3 在 fsdp 打分后端不被支持(只会静默降级)且是步时 2.7×
开销主因(33.0→12.4 分钟);K1 是唯一可能有价值的(早期黄金信号撞
`loss_max_clamp=10` 截顶,observations (73)),留作主线稳定后的**独立立项
A/B**,不作常驻开关。蒸馏 loss 回归纯 `forward_kl_topk` + 硬 clamp(= v1 与
h3 口径,跨版本 loss 曲线可比性恢复)。移除由 `test_vendored_verl.py::
test_signal_enhancement_triplet_removed` 锁死。三件套的实验判决数据仍有效
(K3 曾证深度保持率 3.6%→16.7%——深处信号可救,为后续机制设计保留证据;
见 `docs/experiments/cd006-signal-enhancement-final.md` 与 feedback 底账)。

### 6.1 序列容量伸缩路径(长程任务预案,用户原则:配置不得限制任务)

`response_length / max_prompt_length / rollout_max_model_len /
max_token_len_per_gpu` 是**容量声明**而非机制上限——机制(EpisodeRunner /
append-only 构流 / teacher 打分)对轨迹长度零假设。任务包络增大时:

- **rollout 侧**(episode 采样 + teacher 打分):单序列 KV = len ×
  0.147MB(4B);扩容 = 提 `rollout_tp`(KV 池跨卡)+ 调 `rollout_gpu_mem`
  配比 + 降并发。256K 级:单序列 KV ≈ 38GB ⇒ TP4 每卡 ~9.5GB,可行。
- **训练侧**(真正的墙,FSDP 单序列须整条进一个微批):单序列容量硬约束
  = `max_token_len_per_gpu × ulysses_sp ≥ prompt+response`(加载时强校验);
  显存不足时依次 ①(已有)grad-ckpt + entropy/logits chunking + 双 offload;
  ②**Ulysses 序列并行——已实施为 `train.ulysses_sp` 开关(2026-07-23)**:
  sp>1 时 launcher 自动联动 `use_remove_padding=True` +
  `attn_implementation=flash_attention_2`(SP 的注意力 patch 挂在 FA forward
  上,sdpa 不可配;verl 侧 topk 蒸馏 loss 的 rmpad+SP 路径系上游原生实现
  ——logits processor 按 SP rank 切 teacher 切片、loss all-gather 拼回,
  见 `verl/trainer/distillation/fsdp/losses.py`;bshd 分支明确 no-SP)。
  sp=1(默认)保持 h3 profile(rmpad=False+sdpa)零偏离;sp=2 单序列容量
  69632。**适用性(2026-07-23 校准,CD-006 论证采纳)**:SP 针对的是
  **"单条序列装不下/单条 logits 撑爆"** 型显存墙(§6.1 原始定位:任务包络
  超过单卡容量);**"微批打包多条"型 OOM 不适用**(降打包预算才对症,见
  §9 训练阶段账——smoke3/5 即此型,单条实测仅 5-8k)。**边界**:
  Qwen3-4B(常规 qwen3)架构支持 rmpad,**Qwen3.5/3.6 不支持 rmpad/THD**,
  policy 换代到 3.5/3.6 时此路不通,只剩 ③;注意力头数须整除 sp(verl
  启动强校验);**部署前提:venv_train 需可用 flash-attn(162 未验证,
  冒烟 SP 首验项)**;③多机(`trainer.nnodes`)。
- 首轮观测大的环境(表格类等)按包络调 `max_prompt_length`(fail-loud
  校验,非静默截断)。

## 7. 评测

- 训练内 val:每 test_freq step,test seen+unseen 两 parquet 裸 greedy(RL 范式,
  用户定);step-0 val 即训前基线,与 h3 ①③ 直接可比。**已实证(CD-006
  2026-07-23):冒烟 step-0 val 逐 split 复现 CD-003 `eval_2507_arm1` 基线**
  (seen 0.00%=0/140、unseen 0.75%=1/134、env_steps 一致;invalid_format
  3/0 vs h3 24/9 显著更低)——协议同构实现正确,此为后续所有对照的锚点
  (证据:feedback 2026-07-23 val 条 + `runs/trace_opd_smoke/rollouts/step_00000_val/`)。
- 独立评测器(`evalm/run_eval.py`,token-ids /completions greedy):base 复测
  与训后 checkpoint 正式数字;支持 bare / +doc(诊断)两臂。
- test 集不需要也不使用文档。

## 8. 全流程观测记录与断点恢复(run 目录 = 单一恢复源)

一次 run 的全部产物绑定同一 `TRACE_RUN_DIR`(launcher 生成并把 verl
checkpoint 也钉进来),记录矩阵:

| 交互 | 记录位置 | 内容 |
|---|---|---|
| researcher 会话(每任务每 step) | `research/<task>/<ts>_v<step>/session.jsonl` | 全量消息(含 partial:中途失败/超时也落盘;SFT 冷启动数据源) |
| 研究实验 episodes | 同目录 `episodes.jsonl` | 结构化 Trajectory + **researcher 实际读到的叙述/fact card/conflict** |
| 当次文档与会话元数据 | 同目录 `doc.md` / `meta.json` | 文档、计数、末次实验成功率、gt_injected、session_error、policy_version |
| **student rollout(训练与 val 每样本)** | `rollouts/step_<N>[_val]/<task>_<id>.json` | 完整 Trajectory、outcome/won/env_steps/truncated、research 统计与归档指针 |
| external LLM 请求级磁带 | `tape/<role>.<pid>.jsonl` | 逐请求 messages/响应/usage/延迟/重试(pid 分片防跨进程交织) |
| 训练曲线 | verl tracking(console + **swanlab**;无 key 自动 offline) | 完整清单与读法见本节下方"曲线体系" |
| 训练状态 | `checkpoints/`(trainer.default_local_dir) | 模型+优化器+extra(含数据进度),`resume_mode=auto` |
| 启动审计 | `launch_manifest_<ts>.json` | resolved 命令行、config 快照、手工 override |

**曲线体系(2026-07-23 用户指令定稿;全部零额外算力——训练 rollout 轨迹
自带 won,无须另跑评估)**:

| 曲线 | 含义与读法 |
|---|---|
| `critic/score/mean` | **每 step 训练 batch(16 任务×8 rollout=128 条裸 on-policy 轨迹,温度 1.0)的平均成功率**——训练任务上的即时效果趋势线,最敏感;score 来自 rollout 自身的 won,免费 |
| `rollout_extra/research_final_success/mean` | 研究会话末次实验成功率 = **文档质量过程信号**(doc 注入下的实验成功率) |
| **两线差值** | `research_final_success − score` ≈ **文档的特权增益**;score 上升且差值收敛 = **内化生效的标志形态**(机制核心效果的直接观测) |
| `distillation/loss` | **即 topk forward KL**(teacher‖student):蒸馏信号强度;趋零过快=文档信息耗尽前兆(P 系预测相关) |
| `distillation/student_mass·teacher_mass·overlap_ratio·overlap_token_advantage·loss_min/max` | topk 概率质量与 teacher-student 重合:崩塌前兆观测(overlap 骤变/质量漂移) |
| `actor/entropy` | 熵膨胀早期信号(h3 崩塌参照;metric-only,不进 loss) |
| `rollout_extra/doc_missing·research_error/mean` | 研究降级率/错误率(闭环健康) |
| `rollout_extra/doc_words·research_experiments·research_episodes/mean` | 研究预算消耗与文档形态趋势 |
| `rollout_extra/env_steps·invalid_format·truncated/mean` + `response_length/*`·`num_turns/*` | 轨迹形态:步数包络(cap 30 下)、**格式失败与截断率上升是重复吸引子的先导指纹** |
| `val-core/alfworld/valid_seen\|valid_unseen/reward/mean@1` | 分 split 泛化曲线(三件套锚点,每 20 step);`val-aux/.../env_steps·invalid_format·truncated` 同步分 split |
| `actor/grad_norm`·lr·`perf/*`·timing 族 | verl 原生训练健康与吞吐 |

通道实现:per-sample 标量经 `extra_fields["reward_extra_info"]`
(`train/curve.py` 组装,batch 内键集齐整硬约束)→ train 侧 verl patch
`f8f26d5`+`e8f697c` 聚合为 `rollout_extra/<key>/mean`,val 侧上游原生聚合;
research 细粒度仍以 per-sample dump + `grep 'research '` 为准(曲线是趋势,
dump 是真值)。**执行路径事实(f8f26d5 事故教训,CD-006 2026-07-24 实证)**:
verl 双 trainer 并存,实际执行 **V1**(`TaskRunnerV1`→`v1/trainer_base.py`,
TransferQueue 架构),V0 `ray_trainer.py` 是死路径——f8f26d5 只打 V0 致
train 侧曲线全缺 20 步;e8f697c 镜像到 V1 `_compute_metrics`(与
`critic/score/mean` 同处,padding 样本过滤)。本文引用的训练循环行为
(entropy 在 old_log_prob 段产出、`compute_data_metrics` 免费 score)均
以 V1 路径为准;`tests/test_vendored_verl.py` 断言 vendored 导出物的 V1
路径带全该 patch,防再打偏。

**判停读法:组合签名(CD-006 step-1~12 实证沉淀,2026-07-24)**——机制
定稿无自动闸门(§2),判停靠人工按签名组合判别,远稳健于单看熵:

- **健康内化签名**(step-9 突破实测):`score`↑ + `overlap_ratio`↑ +
  `loss`↓ + `grad_norm`↓ + 复读↓;`teacher_mass_min`↓ 伴随(训练推进到
  更难决策点的指纹,非恶化)。
- **崩塌预期签名**(h3 参照):`score`↓ + `loss`↑(student 追不上发散
  teacher)+ 复读↑ + `response_length` **无界膨胀**——与上组多指标反向,
  不会混淆。
- **早期平滑进度标量**:`overlap_token_advantage`(全程唯一平滑单调改善
  的量,score 仍为 0 时已在动)——比 score 更早、比轨迹分析更省。
- **物理信号**:`response_length` 漂移(均值斜率与 p95)与显存同源
  (activation ∝ 长度)——判停应在物理 OOM 之前触发(§9 显存账)。
- 歧义时以**轨迹抽读定性**(熵升 ≠ 退化:step-5 熵 +27% 实为健康探索;
  长度涨 ≠ 复读:需读 token 是推理还是复读)。

断点恢复:重启同一 launcher 命令 → verl 从最近 checkpoint 恢复权重/优化器/
数据进度;当前 step 重新 rollout(研究会话重新现场进行——on-policy 语义下
这正是正确行为);归档/dump 均为带时间戳追加,不互相破坏。

engine 后端的 LLM 调用不做请求级磁带:研究 episodes 与 student rollout 已有
结构化全轨迹,narration 产物随归档——语义层记录完整,请求层冗余。

## 9. 部署拓扑:162 单机分时复用(用户裁定 2026-07-23)

> **拓扑演变(2026-07-25/26 定版,本节以下为 162 时代历史记录)**:4B 主线迁
> **A800 8 卡**(10.10.252.11;该机两条硬约束:①卡间通信不可靠——默认
> MODE=noshard 全 gloo 零 NCCL,MODE=shard2 最小化 NCCL;②驱动 525 **不可用
> `ulysses_sp>1`/flash-attn**——实证六卡同时 Xid 31 MMU Fault,回退须卸载
> flash-attn)+ **切勿设 `expandable_segments:True`**(与 vLLM CuMemAllocator
> 互斥,实证事故)。researcher = 外部 35B 双端点(§2 演变);162 保留对照线。
> 稳定性定版:R1 打包预算不抬 + R2 resume weights 前清缓存 + R3 每 prompt
> 2700s 超时 + `VLLM_RPC_TIMEOUT=300000` + 空批诊断防御;ckpt 策略
> `save_contents` 去 `hf_model`(需 HF 用 `python -m verl.model_merger` 离线转,
> **probe 工具链取 HF 权重按此路径**)+ `max_actor_ckpt_to_keep=3`(峰值 ckpt
> 由 CD-006 手工 pin,requirements 归档区 R4)。启动入口
> `scripts/run_a800_4b.sh`(联锁与硬约束注释在脚本内)。

A800-SXM 经用户实测确认不可靠(TP4 负载 40min 楔死事故 + 复测仍不可用),
**正式与测试实验一律在 zkgy(10.77.110.162,4×80G:A800=GPU0,1 / A100=GPU2,3,
混卡 PCIe,RAM 503G)单机进行**。用户资源意图:研究阶段 2 卡 27B + 2 卡 4B,
训练阶段 4 卡全给 policy、27B offline。落地形态如下(与意图等效,理由见后):

**空间布局(常驻;27B 摊满 4 卡系用户方向 2026-07-23)**
- 27B researcher:独立 vLLM server,**DP2 × TP2 摊满 4 卡**(vLLM 原生
  `--data-parallel-size 2`,单端口内部负载均衡;每个 engine 实例 TP2 落一个
  同型卡对:实例A@GPU0,1(A800)、实例B@GPU2,3(A100)——混卡"TP 只在
  同型对内"纪律保持),`GPU_MEM=0.52`(**冒烟 OOM 后由 0.60 下调,CD-006
  2026-07-23 部署修订**;显存账见下),64k ctx,`--enable-sleep-mode` +
  `VLLM_SERVER_DEV_MODE=1`(`scripts/serve_researcher.sh`)。对比:TP2 只占
  2 卡时 KV 池仅 ~34GB 且 GPU2,3 闲置 ~45GB;DP2×TP2 摊满后每卡负载对称,
  0.52 下 KV 池 ≈58GB(0.60 时 ≈84GB;`max_concurrency=64` 预计仍够,
  preemption 冒烟观测,不足时按下方账目回调 0.55);不选 TP4:破混卡 TP
  纪律、聚合 decode 吞吐劣于 DP2。已知代价:DP 路由非会话粘性,多轮会话
  prefix-cache 命中率下降(prefill 变贵,量级秒级/轮,冒烟观测)。
- verl(训练任务):4 卡,rollout vLLM **TP2 ⇒ 2 replica**(replica0@GPU0,1、
  replica1@GPU2,3,verl 连续分组天然落同型对),`gpu_memory_utilization=0.35`
  (28GB/卡);FSDP2 param+optimizer 双 offload(h3 同款)。

**每卡显存账(AgentLoop 阶段;冒烟 OOM 后的实测修订版,2026-07-23)**——
初版预估账(27B 0.60 + verl 0.35 ≈76-78GB)实测 78.6GB 且**漏算一整项**:
self-teacher 打分时 vLLM 每个 chunked-prefill 步物化的**全 vocab logits 临时
张量** = `rollout_max_num_batched_tokens × 151936 × 2B`(TP2 `_gather_logits`
all_gather 输出再 +1 份全量),h3 值 4096 ⇒ 峰值 ≈2.3GiB > 当时余量 0.5GB
⇒ **冒烟 step-1 必然 OOM**(replica1 EngineCore 崩溃;h3 独占拓扑余量大,
该项从未暴露)。修订账(现行):

| 项 | 值 |
|---|---|
| 27B 预分配(`GPU_MEM=0.52`) | 41.6GB |
| 27B activation 等超预分配(实测) | ~1.4GB |
| verl rollout(0.35,实测) | 27.3GB |
| CUDA context + ray WorkerDict(实测) | ~1.3GB |
| **小计 / 余量(79.14GB 可用)** | **71.6GB / 余 ~7.5GB** |
| self-teacher logits 峰值(chunk 2048,参数化后) | ≈0.9-1.2GiB ✓ 双保险 |

27B 回调空间:0.55 ⇒ 小计 74.0GB、余 5.1GB,仍 ≥4× 峰值——27B KV 不足
(preemption)时的安全回调位;**0.60 无余地**(余 1.5GB,边际不足)。

**训练阶段显存账(smoke3/5 两次 OOM 后补,2026-07-23)**——两引擎皆睡后
FSDP 拿满 4 卡,但**微批 forward 的全 vocab logits 临时张量 =
打包预算 × 151936 × 4B(fp32)** 是此阶段真正的天花板(此前 34816 预算
⇒ 21.2GiB,叠加实测 FSDP 前反向基线 **70.84GiB**(双 offload 下的
activation/梯度/杂项)必炸——smoke3 死于 log_prob 阶段、smoke5 死于训练
前反向;h3 当年的"52GB 时代 32k OOM"极可能同根因)。修订(现行):
打包预算与单序列容量声明解耦——`ppo_max_token_len_per_gpu=8192`
(logits ≈5.0GiB)/ `log_prob_max_token_len_per_gpu=12288`(≈7.5GiB,
无梯度激活、基线更低,smoke5 实证;该阶段产出 actor/entropy 曲线故
**不可跳过**,虽纯蒸馏不消费 old_log_probs)。可用余量 79.14−70.84
≈8.3GiB,两值均在内。**长尾兜底(verl `6d77dc7`)**:超预算单条自动
独占微批(warning),logits 随该条实际长度增长——cap30 实测轨迹 5-8k,
>~14k 单条才逼近余量,真不够时 loud OOM 而非静默;预算是打包目标、
非梯度语义参数(mini_batch 才定全批一更,微批间梯度累积数学等价)。

**全量 step-5 OOM 修订(2026-07-24,`8192/12288` 定版)**:曾按"训练阶段
仅用 52%"上调 16384(用户 07-23 裁定),step-1 起步验证准(预测 55 实测
56.8GB),但 `max_memory_allocated` 随 step **加速**上涨(56.8→58.6→62.9→
72.2,step-5 撞墙)——涨的是 **activation 而非 logits**:`response_length`
每步输出变长(2988→3346,env_steps 已饱和 30),activation ∝ 序列长度,
固定 token 预算只能压 logits 项。教训:**打包预算须按漂移期上界定,不按
step-1 起点定**;8192 实测基线 38-42GB,给 activation 漂移留 ~37GB
(CD-006 判断足扛全程),16384 裁定作废,yaml 默认维持 8192/12288。
**运行时显存自适应明确不做**(用户裁定 2026-07-24):预算随显存浮动破坏
可复现性与可判读性,取静态保守预算 + 曲线监控。同时认清**预算不是最后
防线**:单条超预算独占微批时该微批的 logits/activation 由单条长度决定
(response 硬上界 32768 的单条仅 logits 就 ~18.5GiB,任何预算不保)——
显存的最后防线是**轨迹长度不失控 = 机制健康本身**,`response_length`
漂移因此列为判停签名的物理信号(§8;"OOM 是崩塌的早期物理信号",
CD-006 观察)。

**时间轴(每 step 自动交替,verl sidecar patch:commit 1e03590)**
- AgentLoop 阶段(研究会话 + 实验 episodes + student rollout):
  `CheckpointEngineManager.update_weights()` 完成权重同步后 POST 27B `/wake_up`
  → 27B 醒、rollout 引擎醒。每卡对称占用见上表。
- 训练阶段(FSDP 前反向):`sleep_replicas()` 先 POST 27B `/sleep?level=1`
  (权重下沉 CPU RAM——DP2 两份 ≈110GB,主机 503G 富余;KV 释放)再睡
  rollout replicas → 4 卡各 ~76GB 归 FSDP(h3 时代仅 ~52GB;response 32k
  的可行性即源于此,冒烟首验)。切换开销:**CD-006 实测(2026-07-23)
  sleep 后四卡 49.35→2.01GB 同降、`/wake_up` 仅 1.9s**(权重确在 CPU RAM,
  level=1 语义正确),相对 step 时长可忽略;dp×sleep×qwen3_5 三重组合
  已实测完全兼容(sleep 为全 DP rank 广播语义),无需回退单实例形态
  (证据:`docs/collab/cd005-cd006/experiment-feedback.md` §10-1 条)。
- 27B 失联语义:wake 失败 → 研究请求失败 → 既有 doc=None 降级+计数;
  sleep 失败 → 训练侧 OOM 即 loud failure。两向都不静默。

**为何不是字面"研究阶段 4B 只占 2 卡 / 独立 4B server"**:实验 episodes 与
叙述的权重必须 = 当前 step 被训练的 policy(机制硬要求,§3 executor 固定
engine 后端)。独立 4B server 权重冻结在旧版本,每 step 推送权重等于重造
verl weight-sync;而 verl colocated 的 rollout 引擎天然逐 step 同权重,其
replica 必须摊满 n_gpus。同理 verl 的睡眠/唤醒是 all-replica 语义,不存在
"训练用 4 卡、研究阶段只激活其中 2 卡的 rollout"的部分唤醒形态——于是
两侧(4B rollout 与 27B)都各自 DP/TP 摊满 4 卡、以显存配比共存 + 逐 step
分时,两个阶段每张卡都无闲置,即用户"充分利用"意图的落地。

**起服顺序契约**:27B 先起(预分配 0.60)→ launcher preflight(`/models` +
`/is_sleeping`)→ verl 后起(预分配 0.35)。调大任一侧显存必须同步调小
另一侧(trace.yaml / serve_researcher.sh 注释互引)。

**一键启动**:`scripts/run_experiment.sh`(162 版)把整条链自动化——预检
(环境/模型/数据/vendored verl/GPU 抢占防护)→ parquet 缺则构建 → 27B
起服或复用(就绪判据=HTTP 真实往返)→ 训练 nohup + 存活确认;幂等重跑
即断点恢复;`SMOKE=1` 走 32 任务 + `total_training_steps=3` + 独立 run 目录。
训练 import 的 verl 是仓库内 vendored 完整代码(`trace_opd/verl/`,
trace-opd 分支 1e03590 的 git archive 导出,来源与 patch 清单见其
`VENDORED.md`;launcher 自动前置 PYTHONPATH 并在 manifest 记录)。

**KV 容量口径**(按架构参数推算,**冒烟实测校准**):4B(36 层×8 KV 头×128
维 bf16)≈0.147MB/token,TP2 后权重每卡 2GB,两 replica KV 合计 ≈104GB
≈70 万 token——response 32k 后单 episode 峰值 KV ≈5GB,满长并发 ~20,
实际均长远短于满长,episodes+student 并发预期充裕(冒烟实测);27B 是
qwen3_5 hybrid linear attention(KV 密度远低于全注意力,h1 实证 0.85util
TP2 下 128 并发×64k 稳),DP2×TP2 共存、现行 `GPU_MEM=0.52` 下 KV 池
≈58GB(0.60 时 ≈84GB;OOM 修订见上文显存账),`max_concurrency=64` 起步
(trace.yaml 用户评审值),冒烟观测 preemption 后再定上下调(不足时 27B
回调 0.55 是账内安全位)。

**跨机形态保留**(机制只认 endpoint):27B 部署任意机器时
`sidecar_sleep: false` + SSH 隧道(vLLM 只绑 127.0.0.1),其余零改动。

## 10. 冒烟计划(全量前必过)

1. 分时链路专项:27B sleep-mode 起服(qwen3_5 × sleep-mode × DP2 三重组合)
   → 手工 `/sleep`→`/is_sleeping`→`/wake_up` 往返 + 真实请求——**服务端侧已
   实测通过(CD-006 2026-07-23:四卡 49.35→2.01GB 同降、wake 1.9s、幂等 ✓,
   无需回退)**;剩余待验:与 verl 合跑数 step,观察 sidecar 日志与显存曲线
   (阶段切换必须见锯齿=分时生效判据);
2. **response 32k 训练稳定性首验**(用户裁定值):打包型 OOM 已由预算解耦
   +长尾 clamp 修复(§9 训练阶段账,smoke3/5 复盘);剩余验证 = 长轨迹
   单条独占微批的实际行为(6d77dc7 warning 频率与显存峰值);若出现**单条
   装不下**型 OOM(>~14k 单条独占仍炸)才轮到 `ulysses_sp=2`(§6.1 适用性,
   先过第 7 项 SP 首验),不行再 16k 逐级回探,两者都先报告用户;
3. 训练链路:4 卡 + 研究内联,16~32 任务小 parquet,数 step 端到端;
   TP1/TP2 各一组对照 step 时间与 preemption 计数(rollout_tp 定版依据);
4. **rollout_n=8 专项**:研究去重正确性——`research/` 归档每 (step, task)
   恰一份、rollout dump 每 step = 16×8=128 条且同任务 8 条共享同一
   research_archive 指针;同任务 8 条轨迹多样性抽检(温度 1.0 下不应全同);
   step 墙钟三段分解(研究 / rollout / 训练)记录,供 n 上调/下调决策;
5. 对照组(同一小任务集,结果报用户裁定全量默认):**gt=on vs off** 各一组
   (研究行为/文档质量);**researcher thinking on/off** 各一组(末次实验
   成功率 + doc 质量——用户已裁定全量跑 non-thinking 以保 SFT 数据分布,
   此对照量化关思考的质量代价,h1 的思考 ON 定版无 non-thinking 实证);
6. **prefix-cache 亲和性验证(v1.3 修复 4B 侧 + v1.5 修复 27B 侧)**:
   4B——同一任务的全部请求(rollout_n 条 student episodes、研究 episodes、
   teacher 打分)应粘同一 rollout replica,并确认两 replica 任务数大致均衡
   (least-loaded 首触分配);27B——`X-data-parallel-rank` 路由后
   `prefix_cache_queries/hits_total` 命中率应从无亲和基线 65.3% 显著上行
   (多轮会话理论 90%+),同时记录两 rank queries 增量比,**偏斜 >1.5×
   反馈 CD-005 议加权哈希**;tape 每请求已记 `dp_rank` 供审计;
7. **Ulysses SP 首验(条件项:32k OOM 或主动扩容前必过)**:venv_train
   flash-attn 真实可用性(import + 真实 varlen 前向,三门纪律)→
   `ulysses_sp=2` 数 step 与 sp=1 对照——蒸馏 loss 曲线应在浮点噪声内一致
   (rmpad+FA2+SP 与 sdpa 数学等价的实证),不一致即停并反馈 CD-005;
8. 实测 step 时间(估算校准)、10240→32768 下的 episode 长度分布与(旧值
   假想)截断率统计、磁带与归档完整性检查、27B KV preemption 观测
   (`max_concurrency` 校准);
9. 三门冒烟纪律(infra 通用纪律 7)先行于任何新服务。

## 11. 未来路线的接口预留(第一轮不做)

- policy-as-researcher:researcher backend 切 engine + ToolProtocol=text;
- **researcher SFT 冷启动(27B 轨迹已按此要求保全,用户 2026-07-23)**:
  - **分布硬约束(用户裁定 2026-07-23):数据源必须在 non-thinking 模式下
    产生**——trace.yaml roles.researcher 已固化
    `extra_body: {chat_template_kwargs: {enable_thinking: false}}`;带
    thinking 的会话即使剥掉 reasoning,其 content 仍是"思考之后"的产物,
    与 non-thinking SFT 分布不匹配,一律作废(冒烟期 23 份带思考归档已
    据此作废,仅留作机制验证证据)。thinking on/off 的质量代价由冒烟
    §10 第 5 项对照量化;
  - 数据源:`research/<task>/<ts>_v<step>/session.jsonl` —— assistant 消息
    保存 vLLM 原始结构(思考字段、`tool_calls` 原样;注意 vllm 0.19.1 起
    思考字段名由 `reasoning_content` 改为 **`reasoning`**,机制代码按原样
    透传不硬编码字段名,消费侧按新名取;arguments 仅规范化为已解析 JSON,
    磁带留原始),tool 结果消息完整,partial 会话也落盘;
  - 过滤信号(meta.json):`finalized / no_finalize / session_error / flags /
    last_exp_successes÷attempts / doc_words / gt_injected / researcher_model /
    tool_protocol / researcher_prompt_tokens / researcher_completion_tokens`;
  - 下游真值 join:`rollouts/step_<N>/<task>_*.json` 的 `research.research_archive`
    字段反向指向会话目录 —— 按 (task_id, global_step) join 即得"该文档
    对应的 student 裸 rollout 是否成功",作最有价值的挑选信号;
  - 请求级兜底:`tape/researcher.<pid>.jsonl` 全量请求/响应/usage,
    `ctx="research:<task>:turnNN"` 可回捞任意会话任意轮。
- researcher RL:文档质量 reward 原语 = doc 注入 K rollout 成功率
  (evalm 组件化);
- 多环境:新增 `envs/<name>/` 实现 TaskAdapter 即可(长程任务接入时
  rollout_tp/response_length 等容量参数按 §9 口径重估)。
