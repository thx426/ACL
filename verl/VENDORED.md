# Vendored verl(trace-opd 分支完整工作树)

> 本目录是 verl 仓库 **trace-opd 分支 HEAD(2ec0a65)的 `git archive` 完整导出**
> (用户要求 2026-07-23:改写适配后的完整 verl 代码随 trace_opd 一体存放/部署)。
> 训练时 `scripts/launch_train.py` 自动把本目录前置进 PYTHONPATH——**实际
> import 的就是这份代码**(优先于 venv 里安装的 verl),启动日志会打印
> `verl from: ...` 供核对。

## 来源

- 上游 base:verl main `a07d3e3`
- 分支:`trace-opd`(开发真源仓库:本机 `../verl`,即 TRACE-OPD 仓库外的克隆)
- 导出方式:`git -C ../verl archive HEAD | tar -x -C trace_opd/verl`(无 .git)

## 携带的全部 patch(自 a07d3e3,旧→新)

| commit | 内容 |
|---|---|
| 2a86f7a | H3:特权 teacher prompt 前缀(`extra_fields["teacher_prompt_ids"]`,context distillation) |
| ab6feae | H3:`compute_topk_loss` 接受 fsdp2 strategy |
| 7821dea | H3:self-teacher 模式——rollout 引擎自打 teacher logprobs(prompt_logprobs) |
| f3e620e | H3:self_teacher 时跳过 teacher pool spec |
| 838e235 | H3:distillation enabled 即构建 distillation_config(self_teacher 无 manager) |
| 69e8129 | trace-opd:validate/global_step 透传到 agent_loop.run |
| 71d5dec | trace-opd:特权 teacher prompt-logprob off-by-one 修复(review P0-1) |
| 1e03590 | trace-opd:CheckpointEngineManager 驱动外部 sidecar server 随 replica 睡/醒(162 单机分时) |
| 79d4252 | trace-opd:self-teacher 打分走 `teacher_routing_key` 粘性路由(同任务副本共享 doc 前缀 prefill;key 恒 pop 不入 batch) |
| f8f26d5 | trace-opd:训练侧聚合 per-sample `reward_extra_info` 数值键为 `rollout_extra/<key>/mean` 曲线(val 侧上游本就聚合,此为镜像) |
| 6d77dc7 | trace-opd:微批 token 预算改为打包目标——超预算长尾单条自动独占微批 + warning(原 assert 会杀整个 run;显存不足仍 loud OOM) |
| e8f697c | trace-opd:`rollout_extra/*` 训练曲线镜像到 **V1 trainer**(实际执行路径 TaskRunnerV1→`v1/trainer_base.py`;f8f26d5 只打在 V0 的 ray_trainer.py,曲线全缺——CD-006 2026-07-24 实证。V0 调用保留) |
| 6f3d089 | trace-opd:**蒸馏显微镜采样落盘**——`TRACE_DISTILL_MICROSCOPE_DIR` 置位时 `compute_forward_kl_topk` 按内容哈希确定性抽样(默认 1/4,`TRACE_DISTILL_SAMPLE_EVERY`)落 per-token {kl, mask, top-k mass, overlap, teacher 截断熵, token ids};哈希与 `trace_opd.common.hashing` 位级一致(离线 join rollout dump);fsdp topk 路径新增 per-token teacher 截断熵;落盘失败绝不杀训练(schema:`docs/analysis-design.md` §2) |
| c48568e | trace-opd(CD-006 热修上收 3401ba5):**R1** 打包预算不再抬到最长序列(抬预算使全批微批变大,峰值 1.3× 最长序列,4B step-5 OOM 放大器;均衡分区器天然让超长条独占微批,修正 6d77dc7)+ **R2** resume weights 前补 `aggressive_empty_cache`(上游只护了 kv_cache 一半,torch caching allocator 不还块则 vLLM `cuMemCreate` 拿不到物理页——step-5 崩溃直接根因;宜提上游 PR)+ `VERL_DIST_BACKEND`/`VERL_FSDP_NO_SHARD` 逃生通道(A800 卡间通信不可靠;默认行为不变) |
| 58cd07e | trace-opd(CD-006 上收 906b66a):**R3** TQ agent loop 每 prompt wall-clock 超时(`asyncio.wait_for`,默认 2700s,`TRACE_AGENT_TASK_TIMEOUT_S=0` 禁用)——vLLM EngineCore 死亡后任务永久 await,`except Exception` 接不住 hang,整个训练循环静默死锁(07-25 第二次 A800 事故);超时取消副本落入既有 failure 分支放行 batch |
| ce96673 | trace-opd(CD-006 上收 d5244ae..44e2bbf):**R11a self-teacher 打分迁训练引擎**(fsdp 后端)——新文件 `trainer/distillation/teacher_topk_fsdp.py`,`TRACE_TEACHER_SCORING_BACKEND=fsdp` 时 rollout 侧跳过 vLLM `prompt_logprobs`(前缀经 extra_fields 到 trainer,update_actor 入口 no-grad 打分),行语义严格复刻 vLLM 路径;根治 vLLM logprobs 预算外分配(issue #5907,rollout 尾部 76-81GB 尖峰);codex 深评九项落实(provenance fail-loud/bare 用裸 prompt/eval_mode 契约/分块 topk 不物化全 log_softmax/前向前全量结构校验);`constants_ppo` 显式转发 backend/timeout/`PYTORCH_CUDA_ALLOC_CONF` 进 Ray runtime env。生产冒烟:overlap 0.850(基线带内)、rollout 尾部 81.2→47.2GB |
| f88db6c | trace-opd(CD-006 上收 617817c):fsdp engine `prepare_model_outputs` 跳过 temperature 恒等除法(`logits / 1.0` 额外分配一份 seqlen×vocab fp32 = 13.5GiB@22k 序列,step-16 OOM 直接触发点;逐位等价;宜提上游 PR)+ fsdp 打分探针 print 到 stdout(Ray worker 不可靠转发 logger 层级) |
| 28a82d5 | trace-opd(CD-006 上收 01f0fae+5f9a29d):蒸馏诊断 metrics 空批防御——hang guard 取消 prompt 后其副本空 response、mask 全 False,`min()/max()` 在空张量上抛错把"损失一个 prompt"放大成杀 run(step-17 两处两杀);两处调用点均判空后按缺席上报,loss 数学不动 |
| 1c6d457 | trace-opd:**清除 K3 残留引用**(rollout 侧 `teacher_doc_ids`/`teacher_block_starts` 透传 + fsdp 打分的 K3 死警告)。信号增强三件套 K1/K2/K3 按用户 2026-07-26 裁定**从机制中移除**:其实现只存在于 vendored 树(CD-006 1.7B v2 时代直改,从未进本真源仓库),本次重导出即自然消失——K2 实证 no-op、K3 在 fsdp 打分后端不被支持且为步时 2.7× 开销主因、K1 留作独立立项 |

| a61c654 | trace-opd:**混合目标 OPD × RL × SFT**(`docs/mixed-objective-design.md`)——`DistillationLossConfig` 新增 `rl_loss_coef`/`sft_loss_coef`(均默认 0 ⇒ 与纯蒸馏逐字节一致);RL 项复用上游 `ppo_loss`(GRPO 任务 reward)但带显式权重(与 `use_task_rewards` 互斥,config 拦);SFT 项 = 新 `sft_ce_loss` 对 `sft_mask` 选中的专家 token 做 CE;三项共用 `loss_agg_mode` 与全局 `batch_num_tokens` 分母 ⇒ 系数按 token 可直接互比。expert 行在 **TQ put 时清空 `response_mask`**(`agent_loop_tq`),故不进 OPD/PG 分子、也不进 `batch_num_tokens` ⇒ 开 SFT 臂不改变 OPD 尺度(消融臂可比);零专家轨迹的 step 贡献恰为 0 而非 NaN。另:expert 行独立 GRPO 组(uid 后缀,防 96% 成功率抬高 bare 组均值使成功的 bare 行拿到负 advantage)、fsdp 打分对其跳过前向(`teacher_mode="sft"`)、`distillation_only` 在任一臂开启时关闭 |
| 93a0109 | trace-opd:**expert 行的观测隔离**——`_compute_metrics` 由 `sample_role` 导出 on-policy mask,`compute_data_metrics`/方差代理/`num_turns`/除 `sft_*` 外的 reward_extra 键一律只读 on-policy 行(否则 `critic/score/mean` 被 96% 成功率的专家行抬高,而它正是判停曲线);吞吐/计时仍计全部行(专家行确实耗算力)。显微镜落盘跳过全零 mask 行(专家行 + hang guard 取消的 prompt),否则离线 join 的 span-vs-mask 完整性自检会误报 mismatch |

| f27eb65 | trace-opd:混合目标的两处归一化定正——①`config.global_batch_info` 只由 `ppo_loss` 写入,而纯 OPD 路径从不调它 ⇒ 现行 OPD 一直是「每微批 token 平均」;开 RL 臂后该 config 跨微批存活,使 OPD 从第二个微批起静默换成全局分母(尺度跳 ~微批数×dp_size)且**只发生在开 RL 的臂** ⇒ 消融失效。现 OPD 项在 `rl_loss_coef>0` 时对清空后的快照求值,各臂钉死纯 OPD 口径。②SFT 项改为自算「每微批 expert token 平均」(clamp 分母)——同族口径使 `sft_loss_coef` 跨臂含义一致,且规避 `agg_loss` token-mean 在「该微批无 expert token」时的 0/0 → NaN |
| c7c2194 | trace-opd:混合目标的三处收口——①`construct_minimal_padding_template` 会 clone 源行**全部**字段再覆盖一份固定清单,`sft_mask` 不在其中 ⇒ 合成的 padding 行带着**别的行的全长 mask**(很可能是真专家行的),而其 responses 已截到 1 token ⇒ SFT loss 形状崩溃,或(宽度恰好相同时)**静默虚增 `sft/expert_tokens`**——正是判定该臂是否生效的指标。现按需收缩(字段存在才改,SFT 关时不凭空造字段);`sft_ce_loss` 另加形状断言,任何未来"合成整行"的路径都会 fail loud 而非错数。②TQ 侧 mask 切分原为无条件执行,SFT 关时仍每行分配零 `sft_mask` 并入 TQ(每 step 数十 MB 纯浪费)⇒ 改由 `sample_role` 标记 gate(trace agent loop 仅在该臂开启时给所有行盖章)⇒ **关臂即零开销**。③`global_batch_info` 快照仅在 RL 臂开启时执行 |
| 6c4e090 | trace-opd:**全 expert 微批的 OPD 除零**(外部审阅发现)——expert 行 `response_mask` 全零,而均衡分区器按 workload 分微批、不看 role ⇒ 「整个微批都是 expert 行」可达;此时 `agg_loss` 的 token-mean 分母 = `loss_mask.sum()` = 0(纯 OPD 路径下 `global_batch_info` 恒为空,只有 `ppo_loss` 会填它)⇒ NaN 进 backward 毒化更新。改为传入 clamp 后的计数:mask 非空时分母**完全不变**(各臂 loss 尺度不动),全空时该微批贡献恰为 0。SFT 项此前已自行处理空批,蒸馏项没有 |
| 6de6693 | trace-opd:**RL 项与 OPD 项同归一化口径**——`ppo_loss` 会从 data 重填 `global_batch_info` ⇒ 其 PG 项按**全局** token 归一,而 OPD 项被钉在**每微批** token 平均(A0 历史口径)。两者的差是**微批数**:每微批平均在 N 个微批上累加成 N×mean,全局平均累加成 1×mean,本 run 形状下 N≈60(已数值验算)⇒ 名义 `mix_rl_weight=1.0` 实为 OPD:RL≈60:1,消融会报出"RL 没用"这个**算术假象**。注:**上游不是缺陷**(`use_task_rewards` 下两项都全局、自洽),不自洽是我们钉 OPD 口径造成的。故 RL 项改走新 `task_reward_pg`(同一 `policy_loss_fn`、同一批 GRPO advantage,但在调用方口径下求值),并顺带不再把 entropy/ref-KL 正则折进被 RL 系数缩放的标量。新增 `rl/nonzero_advantage_frac`(二元 reward 下组内全成功/全失败则 advantage 恒 0,低成功率时多数组如此 ⇒ RL 有效强度随成功率自动长大,直接测而非推断) |
| e8ecd43 | trace-opd:蒸馏项获得显式系数 `opd_loss_coef`(默认 1.0)。上游 `distillation_loss_coef` 仅在 `use_task_rewards` 时生效(纯蒸馏下整体缩放无意义故被强制 1),混合目标里它有意义了 ⇒ 三臂读法对称,且 **0 成为一个真实的消融臂**(A5:RL+SFT 无 OPD,回答"OPD 在弱模型上还有没有净贡献"——此前它被钉死为 1,没有任何臂在检验这件事)。config 校验拦住"三项全 0"。同时更正了那段声称"三项共用全局 `batch_num_tokens`"的注释——实际共用的是每微批 token 平均 |
| 0a6edfb | trace-opd:**fsdp 打分后端单点模式(`teacher_topk=0`)**——estimator 家族(k3 / reverse_kl_gated)每位置只需已实现 token 的 teacher logprob;vLLM 路径自 CD-003 reverse-KL 消融起就支持(`prompt_logprobs=0`),R11a fsdp 后端(现默认)此前无条件 `torch.topk`。新 `_score_rows_realized`:gather 已实现 token logit − fp32 logsumexp,产 `(rows, 1)` 张量(dummy 宽度 `max(k,1)` 与 vLLM 提取器一致);比 top-k 更省(无 topk kernel、传输 1 列)。两处对齐哨兵(fsdp/vLLM)从"只看首行"改为"看前 8 行"——单列下一个饱和 glue token 的 logprob 在 fp32 中可合法取整到 0.0(假警报),错位窗口读到的全零 dummy 行仍必然触发;空批显式拒绝 |
| 6fb4145 | trace-opd:**`reverse_kl_topk`——forward_kl_topk 的稠密方向镜像**。逐 token 为 teacher top-k 支撑 + 尾桶的 (K+1) 类分布真 KL:`Σ p_S(log p_S−log p_T) + (1−m_S)(log(1−m_S)−log(1−m_T))`;尾桶使其非负、且是"student 质量停在 teacher top-k 之外"的 mode-seeking 压力唯一表达点(两侧质量本就为诊断而算)。同数据面、同 clamp、同 chunked 路径、同 overlap/熵诊断 ⇒ 拨此开关的消融**只改一个变量(KL 方向)**;estimator 模式一次改三个(方向/单点 vs 分布/估计器形式)。刻意独立镜像不重构——`compute_forward_kl_topk` 是 4B 主线在跑热路径,保持逐字节不动。已验:手算对拍/恒等零/20 随机抽样非负/尾桶梯度方向(支撑外 logit 压下、支撑内抬起)/chunked 一致 |
| 48c8743 | trace-opd:**`reverse_kl_gated`(SDAR/SEED 置信门控 logp-shift)+ KL 模式分发**。两家 2026 agentic 自蒸馏线的辅助项数学同构(SDAR `sdar_utils.compute_sdar_loss`、SEED `core_algos.compute_opd_loss`,β 均默认 5.0):`Δ_t=logπ_T(y_t)−logπ_S(y_t)`(detach)、`gate=sigmoid(βΔ)`(detach)、`loss=gate·Δ`,梯度只走 `−logπ_S`——每个采样 token 都被以 gate 强度上拉(teacher 背书则全力、否决则指数衰减);非 KL 估计器(论文称 reverse-KL surrogate),但与 k1 不同,teacher 经 detached gate 塑形梯度。纯数学抽为 `_gated_logp_shift` 供测试独立执行,**逐位复现 SEED 官方测试向量**(loss/gate/梯度)。他们的 step 级 `opd_step_mask` 刻意无对应物(SEED hindsight skill 只覆盖部分步;我们的特权文档条件化整条轨迹——response_mask 即 step mask);λ(SDAR 代码默认 0.1、论文 0.01;SEED `opd_loss_coef`)映射到我们的 `opd_loss_coef` 不另设旋钮。另:`compute_topk_loss` 按 loss_mode 分发(megatron 对 reverse 明确 raise)、top-k 外层后处理注册双名(其每一步都方向无关)、`gate_beta` 进 config(有限且 >0:β→0 退化为平坦 0.5、负 β 反转到 teacher 否决 token 上)、`use_policy_gradient` 警告覆盖全部 use_topk 模式、estimator 家族补显微镜落盘(gated 加 gate 列)与 abs_loss 空批防御 |
| 6e524ed | trace-opd:**self-research 通道——policy-as-researcher 的 RL 项与 joint/separate 双更新形态**(docs/self-research-design.md)。research 行(researcher 会话轨迹)沿 SFT 臂纪律过 TQ:`response_mask` 清零 + 独立 `research_mask`(researcher 生成 token)+ `research_advantage` 占位 ⇒ 执行侧三项与其一切分母永远看不见它们;trainer 端 uid 加 `:research` 独立 GRPO 组,advantage = 组内去均值(**不除 std**——K 次成功率在近平组上 std 坍缩、除之放大纯采样噪声)+ 硬 clip ±`research_adv_clip`;按 `research_update_mode` 路由——**joint(默认)**:同一 backward 加带 ratio+clip 的 PG 项,分母 = 批全局 research token 数(`research_num_tokens`,切微批前 all-reduce ⇒ 数值与打包方式无关、完美 on-policy、系数在梯度裁剪下语义保持);**separate**:research 行拆出走同一全局 step 的第二次 update(独立 mini-batch、不推进 lr scheduler、关 topk logits processor、loss 端 `research_only` 短路)——UCOB/SEED 结构。`research_warmup_steps` 内 research 行整体退出 update(文档生产与 teacher 选择照常);teacher 打分对 research 行如 expert 行跳过(超长轨迹零浪费 forward);padding 模板同步缩零两个新字段;`research_loss_coef=0`(默认)零字段零开销逐字节现状 |
| e3bd3bf | trace-opd:research 行退出 on-policy 行为指标(SFT 行同款纪律——expert 行会抬高指标,research 行会拉低:research 会话没有任务结局,其 reward 是文档成功率);`research_rl_` 前缀曲线照 `sft_` 先例全量收集 |
| af6a8d3 | trace-opd:角色拆分后的子批带 `num_mini_batch=1`——拆分批行数不再匹配按 rollout.n 推导的 mini_batch_size,dataloader 会产出零个完整 mini-batch、更新静默变 no-op(生产配置 ppo_mini_batch_size==train_batch_size 下 num_mini_batch=1 精确保持原语义) |
| 13741a2 | trace-opd:**codex 审阅修复(gpt-5.6-sol max effort,3 P0 + 3 P1 + 2 P2)**。P0-1 打分端 M4 结构校验枚举纳入 `research`(原先在跳过分支之前就 raise,任何 research 批必死);P0-2 `distillation_only` 显存捷径在 `research_loss_coef>0` 时保持关闭(research PG 每次 update 都读 log_probs,warmup 的 exec 子批也调它——纯 OPD 模板 + research 开启原先第一步就崩);P0-3 research 分母乘回 `dp_size`(FSDP 对梯度做 mean-reduce,`local/T_global` 落在真值的 1/dp——上游 agg_loss 的同款补偿);P1-5 全 research 拆分(failure prompt 残组)回退不拆批;P1-7 `train_mini_batch` 新增 `advance_lr_scheduler` 门(原 flag 在末 mini-batch 无条件置位 ⇒ separate 每 step 推 scheduler 两次);P1-8 全零 advantage 的 research 拆分整体跳过 optimizer step(Adam 动量+weight decay 在零梯度下仍动参数);P2-11 组内可训候选 <2 整组零梯度;P2-12 log-ratio 先 clamp ±20 再 exp(防 inf*0 毒化) |
| 92f8ac9 | trace-opd:**自查加固(codex 轮之后)**。①vLLM 打分路径(1.7B 臂默认后端)此前对**每一行**做 prompt_logprobs 打分——16k 的 research 行会在 gpu_memory_utilization 预算之外分配 logprobs(R11a OOM 同型;expert 行的同款浪费也一直存在,loss 从不读它们的 teacher 列)⇒ sft/research 行改发对齐零 dummy(与 fsdp 后端跳过同形),行为等价、forward 不再烧;②separate 第二次 update 关 calculate_entropy(executor 侧曲线,对 research 长行是纯浪费的全词表 pass);③research 拆分行数不能被 dp 均分(failure prompt 残组)时响亮跳过该次 research update(短 rank 卡 collective 的廉价失败模式) |
| 5389497+111c0e9 | trace-opd:**分布观察包(蒸馏显微镜数据层)**。`TRACE_OBS_DIR` 门控:两个 topk loss 函数内采集"最小完备源"四件套——teacher top-K(切片、clamp 前)/ student 在 teacher 支撑上的 logprobs(KL 自身中间量)/ student **自身** top-K(ids 本就为 overlap 诊断算过,lps 是唯一新增 gather,chunk 安全)/ pre-outer-clamp per-token loss(审计真值锚);分发层把每行 response 窗从 THD 打包流切出(no_padding_2_padding 同窗口)、与 ulysses 段求交,per-row 记录交给缓冲写盘器(每 step 每 rank 一个压缩 npz,下一 step 出现时 flush;Unicode dtype 免 pickle;loss_mode/clamp 元数据自包含)。KL/overlap/mass/熵/rank 全部由包离线可导 ⇒ 分析层可进化而数据不重采。best-effort 契约:任何失败自禁一次警告,训练零感知 |
| 9172da4 | trace-opd:**观察包 codex 审阅修复(P0-1 + P1-9/10/11 + P2-4/9)**。P0-1:`global_steps` 在 TQ put 时逐行盖章 ⇒ 微批里是 NonTensorStack(list 语义),旧代码 `int()` 直接炸并在**首个微批永久自禁采集**(端到端测试手工构行、没走这条链)——现全形态解包;P1-9:`flush_all` 全仓零调用点 ⇒ 每个 run 的最后一个采样 step 必丢、单步冒烟零包——现 atexit 钩子 + join 在飞 writer;P1-10:`TRACE_OBS_*` 补进 Ray runtime-env 白名单(existing/remote 集群 worker 不继承 driver env,缺了就静默零包);P1-11:采样门提到一切 CPU 拷贝/组装之前(GPU 侧常量小开销如实注明);P2-4:SP 元数据随行;P2-9:非法 TRACE_OBS_EVERY 降级不外抛 |
| 0f8bde1 | trace-opd:**min_signal_rows 低信号闸**(intervention-OPD design 12)。`DistillationLossConfig.min_signal_rows`(默认 0=关):全批带蒸馏信号(response_mask 非零)的行数低于阈值时整步跳过 actor update——零 loss 批仍会经 AdamW decoupled decay 与 LR scheduler 推动权重。trainer `_update_actor` 开头经 TQ 取 response_mask 计数 |
| 224dd4e | trace-opd:min_signal 计数对 **NestedTensor** 掩码修复(CD-006 §3-X):`_count_signal_rows` shape-agnostic(nested 走 unbind、list/tuple 直用、张量按行迭代),干预机制的不等长训练行不再在首个 update 崩 `.tolist()` |
| 2ec0a65 | trace-opd:**seq-mean 聚合的全局序列分母**(CD-005 审阅 CD-008 的 loss_agg_mode 切换)。engine `forward_backward_batch` 在 batch_num_tokens 旁 all-reduce 注入**全局非空 response_mask 行数** `batch_num_seqs`(nested 安全);蒸馏调用点在 `loss_agg_mode` 为 seq-mean-* 时 fail-loud 读取并向 `agg_loss` 传 `global_batch_size`+`dp_size`(FSDP 梯度 mean-allreduce 补偿)——否则 seq-mean 分母退化为**本地微批行数**,均衡分区器把长行装进少行微批,长行独占时权重可达同批短行的数十倍,几乎复刻 token-mean 的按长加权(P0-A);且全零掩码微批 0/0→NaN(P0-B)。**token-mean 分支逐字节不变**(历史 run 全部建立于本地 clamp 分母契约);另在 sft_ce_loss docstring 声明 sft_loss_coef 的 mean-对-mean 对称性仅在 token-mean 下成立 |

## 修改/更新流程(保持单一真源)

1. **改代码只改 `../verl`(git 仓库)并 commit 到 trace-opd 分支**——本目录是导出物,
   不要直接编辑(会在下次导出时被覆盖);
2. 重新导出同步到此处:
   `rm -rf trace_opd/verl/* 除 VENDORED.md 外` 或直接
   `git -C ../verl archive HEAD | tar -x -C trace_opd/verl`(archive 不含本文件,追加导出安全);
3. 更新本文件的 HEAD 哈希与 patch 表。
