# PGS S2 / SGLang · A800 dynamic researcher 正式运行卡

> 状态真值日期：2026-08-05。基础设施入口见
> `docs/infra/a800-sglang-training.md`；机制与实验反馈见
> `docs/collab/cd010-cd006/experiment-feedback.md`。

## 运行身份

| 项 | 值 |
|---|---|
| 机器 | `ssh -p 23653 u2021000902@10.10.252.11` |
| run | `pgs_s2_sgl_a800_researcher` |
| run dir | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/runs/pgs_s2_sgl_a800_researcher` |
| 主日志 | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/logs/pgs_s2_sgl_a800_researcher.log` |
| launcher PID / PGID | `41041 / 41041` |
| objective fingerprint | `976d84cfd3a3` |
| bundle | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/trace_opd_researcher_pgs` |
| 初始 policy | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/ckpt/sft_step6` |
| 逻辑终点 | 358（2875 条、batch 16、drop-last、2 epochs） |
| SwanLab | online；run ID `braki9u3`；`https://swanlab.cn/@Wushang/trace_opd/runs/braki9u3`；Api只读状态`RUNNING` |

启动manifest的`config_overrides=[]`、`overrides=[]`；特权文档源为
`researcher`，不是静态GT表。远端角色为`qwen3.6-35b-a3b`双endpoint pool，
`n_exp=16`、k=4..8、thinking off。每个`(global_step, task_id)`只生成一份文档，
组内n=8共享；student仍bare，文档只进入SPS teacher/verify前缀，validation bare。

## 正式训练配置

- 8×A800-SXM；SGLang TP1（8 replicas）；actor Ulysses SP2。
- train batch 16、rollout n=8、`ppo_mini_batch_size=16`。
- `grpo_loo`、`norm_adv_by_std=false`、`seq-mean-token-mean`、LR 1e-6 constant。
- `max_num_seqs=256`（每个SGLang replica）；agent workers=16。
- val/save每10步；`max_ckpt_to_keep=1`；`resume_mode=auto`。
- SwanLab正式命令为`trainer.logger=["console","swanlab"]`；真实启动若
  `SWANLAB_API_KEY`缺失或`SWANLAB_MODE`关闭online，会在建run目录前fail-closed。
  运行中的run禁止执行`swanlab sync`。

## 启动门证据

- 本地最终回归：180 passed、3 skipped；compileall和`git diff --check`通过。
- A800最终聚焦：preflight 4/4、版本审计3/3、compileall通过。
- 双endpoint均通过认证`/models`、精确model identity、non-thinking tool call、
  tool-result回传与精确`PREFLIGHT_OK`；正式启动要求2/2，不能1/2降级。
- 16任务完整预算门：16 archive、16非空doc、128 student；289次35B调用0错误；
  双端session 8/8；每task 8条student共享一份archive；secret实际值0命中。
- 修复后lineage门：archive精确`global_step=1, policy_version=0`；48条research
  episode均为版本0..0；8条student运行时版本断言通过；actor/weight update完成。
- 正式启动前八卡0MiB、无旧训练进程，`/fs/fast`可用约39.53TiB。

- 正式启动后新连接确认launcher未随SSH退出；manifest指纹与合同正确，双endpoint
  preflight 2/2，错误面全零。八卡实收约42.7–44.6GB/卡，GPU 3/4采样功耗
  260.5/228.1W，其余卡亦完成加载，证明真实GPU计算而非只占显存。
- SwanLab日志明确online登录并给出云端URL；`Api.run("Wushang/trace_opd/braki9u3")`
  只读返回name精确匹配、state=`RUNNING`、finished_at=null。未执行`swanlab sync`。
- 训前bare validation已274/274 JSON有效：seen 140、unseen 134；70/274=
  25.55% won，invalid事件9、truncated轨迹3。所有记录均`validate=true`、
  `global_step=0`、`research={}`、`steering={}`，此时ResearchArchive为0。
- Step 1随后16/16 researcher worker均发起真实调用；首轮观察24次35B请求0错误，
  双endpoint session精确8/8，无worker跨端。此时archive/student尚未落盘，符合
  “先research finalize，后n=8 student”的设计时序，不能当成完整step1结果。

## 长期只读监控

| 项 | 值 |
|---|---|
| watchdog PID / PGID | `47341 / 47341` |
| watchdog log | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/logs/watch_pgs_s2_sgl_a800_researcher.log` |
| 周期 | 600秒 |
| stale门 | 7200秒（dynamic researcher长step） |
| 完成门 | train step 358且128条非空JSON |

watchdog只读、只告警，不kill、不restart。首轮输出已确认launcher alive，并记录八卡
memory/utilization/power/temperature。故障处置只按产物、精确cmdline、Ray原始日志、
真实功耗和fatal证据进行；不凭进程名、瞬时利用率或console尾部盲目恢复。

## 实时显微镜

详细操作手册：[`docs/pgs-microscope-remote-access.md`](pgs-microscope-remote-access.md)。

当前run推荐远端`127.0.0.1:8794`，本地用SSH `-L`隧道访问。启动前必须重新探测端口；
服务只绑定loopback。该run使用per-step动态researcher文档，**不要传旧GT `--doc-table`**。
新版UI会沿每条rollout的`research.research_archive`安全载入`doc.md`和审计`meta.json`，
并从run实收配置显示`B∞=400`；student侧仍正确标记为bare。

## 待回填的正式端点

1. Step 1的16份archive、128条student、真实`policy_version=0`、SPS/W_beh、
   actor update和SGLang weight update。
2. 每10步val/checkpoint与`max_ckpt_to_keep=1`滚动保留实证。
