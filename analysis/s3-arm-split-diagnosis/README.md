# s3 分臂 maxopen 诊断脚本(CD-006,2026-08-07)

针对 `pgs_s3_4braw_n16_maxopen`(A800)的现象排查工具。**只读**,不改 run。

| 脚本 | 作用 | 用法 |
|---|---|---|
| `s3_analyze.py` | 从训练日志抽逐步全指标(分臂 won/gap、熵、grad_norm、W_beh、长度、D、倾斜/拒绝/花费、pt_window_miss) | `python3 s3_analyze.py <训练日志>` |
| `s3_rollout.py` | 逐 rollout 拆 bare/steered:赢输长度分布 + **逐 token W=π/b 在接受位与拒绝位的分布** + 拒绝位重采 token 的 π rank | `python3 s3_rollout.py <run_dir> <step> [...]` |
| `s3_adv.py` | 混合组 GRPO-LOO 下的**分臂 advantage 结构**(A_bare / A_steer / 赢输分解 / 退化组计数) | `python3 s3_adv.py <run_dir> <step> [...]` |

读日志要点:训练日志被进度条 `\r` 遮蔽,脚本内部已做 `\r`→`\n` 拆分;直接 grep 会
只看到第一行 step。

数据位置(A800):
- run: `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/runs/pgs_s3_4braw_n16_maxopen`
- 日志: `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-S3/logs/pgs_s3_4braw_n16_maxopen.log`
- 逐 token 记账在 rollout dump 的 `steering.steer_log_pi / steer_log_b / steer_rejected /
  steer_final_pi_rank`(maxopen 下全量携带,约 1.5MB/轨迹)
