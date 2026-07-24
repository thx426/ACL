# Vendored verl(trace-opd 分支完整工作树)

> 本目录是 verl 仓库 **trace-opd 分支 HEAD(e8f697c)的 `git archive` 完整导出**
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

## 修改/更新流程(保持单一真源)

1. **改代码只改 `../verl`(git 仓库)并 commit 到 trace-opd 分支**——本目录是导出物,
   不要直接编辑(会在下次导出时被覆盖);
2. 重新导出同步到此处:
   `rm -rf trace_opd/verl/* 除 VENDORED.md 外` 或直接
   `git -C ../verl archive HEAD | tar -x -C trace_opd/verl`(archive 不含本文件,追加导出安全);
3. 更新本文件的 HEAD 哈希与 patch 表。
