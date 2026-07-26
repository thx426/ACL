# Vendored verl(trace-opd 分支完整工作树)

> 本目录是 verl 仓库 **trace-opd 分支 HEAD(1c6d457)的 `git archive` 完整导出**
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

## 修改/更新流程(保持单一真源)

1. **改代码只改 `../verl`(git 仓库)并 commit 到 trace-opd 分支**——本目录是导出物,
   不要直接编辑(会在下次导出时被覆盖);
2. 重新导出同步到此处:
   `rm -rf trace_opd/verl/* 除 VENDORED.md 外` 或直接
   `git -C ../verl archive HEAD | tar -x -C trace_opd/verl`(archive 不含本文件,追加导出安全);
3. 更新本文件的 HEAD 哈希与 patch 表。
