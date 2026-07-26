# Vendored verl(trace-opd 分支完整工作树)

> 本目录是 verl 仓库 **trace-opd 分支 HEAD(6c4e090)的 `git archive` 完整导出**
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

## 修改/更新流程(保持单一真源)

1. **改代码只改 `../verl`(git 仓库)并 commit 到 trace-opd 分支**——本目录是导出物,
   不要直接编辑(会在下次导出时被覆盖);
2. 重新导出同步到此处:
   `rm -rf trace_opd/verl/* 除 VENDORED.md 外` 或直接
   `git -C ../verl archive HEAD | tar -x -C trace_opd/verl`(archive 不含本文件,追加导出安全);
3. 更新本文件的 HEAD 哈希与 patch 表。
