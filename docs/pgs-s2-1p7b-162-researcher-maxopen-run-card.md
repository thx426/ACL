# 162 · Qwen3-1.7B · dynamic researcher · max-open正式运行卡

> 状态真值日期：2026-08-05。基础设施纪律见仓库根`docs/infra/README.md`；本卡只记录
> `pgs_s2_sgl_1p7b_162_researcher_maxopen`的可恢复运行身份与启动证据。

## 运行身份

| 项 | 值 |
|---|---|
| 机器 | `ssh -p 5102 haoyang@10.77.110.162` |
| run | `pgs_s2_sgl_1p7b_162_researcher_maxopen` |
| run dir | `/data4/ws/PGS/pgs_s2_sgl_1p7b_162_researcher_maxopen` |
| 独立bundle | `/data4/ws/PGS/trace_opd_1p7b_researcher_maxopen_bundle` |
| 统一环境 | `/data4/ws/PGS/env_sglang_train_1p7b` |
| 主日志 | `<bundle>/logs/pgs_s2_sgl_1p7b_162_researcher_maxopen.log` |
| launcher PID / PGID | `2294717 / 2294717` |
| 初始policy | `/data3/wushang/model/Qwen/Qwen3-1.7B`原始权重 |
| 逻辑终点 | 358（2875条、batch16、drop-last、2 epochs） |
| objective fingerprint | `c2a917a51eaf` |
| SwanLab | online；run ID `lg7bxx95`；`https://swanlab.cn/@Wushang/trace_opd/runs/lg7bxx95` |

新run不覆盖、不resume旧GT、标定或smoke目录。student bare、validation bare；每个
`(global_step, task_id)`由外部`qwen3.6-35b-a3b`动态产一份文档，同task n=8共享，文档只进入
SPS teacher/verify。researcher双endpoint为162 loopback 18888/18889，thinking显式关闭，
`n_exp=16`、k=4..8。

## max-open处理

用户裁定的运行配置为：

- `epsilon=0`；
- `gate_target_rate=0.20`；
- `eta_max=0.5`、`c=2.5`保持不变；
- `b_max=81920 = response_length(32768) × 2ηc(2.5)`；
- `budget_pacing=false`；
- `r_max=32769 > response_length`。

因此在32768-token有限响应时域内，总预算与拒绝cap严格不会先于硬时域绑定；每个有文档的
assistant token从首token起可进入D检测。20%是自适应主门目标，不表示每token都强制倾斜。

## dynamic P80标定

正式run没有复用旧GT/P90阈值。独立标定run：

`/data4/ws/PGS/pgs_s2_sgl_1p7b_162_researcher_maxopen_calib_p80`

标定配置永久冻结`theta_d=1e9`、`gate_adaptive=false`、`epsilon=0`，因此行为为bare但每个有文档
assistant token都做teacher verify并记录D。标定形成16/16非空archive、128/128 student、真实
`global_step=1 / policy_version=0`，并完成actor update和SGLang weight update；四个非绑定计数和
probe/tilt均为零。P80 materializer据完整D直方图得到：

`theta_d=0.00259473`

审计产物：`<bundle>/logs/maxopen_p80_calibration_audit.json`。materializer随后生成正式配置
`<bundle>/trace_opd/configs/experiments/162-pgs-s2-sglang-1p7b-researcher-maxopen.yaml`；正式run
仍从原始1.7B开始，未使用标定actor权重。

## 启动门与当前状态

- 本地全量回归：209 passed、3 skipped；compileall和`git diff --check`通过。
- 162统一环境聚焦：61 passed、2 skipped；关键文件同步前后SHA-256一致。
- 四卡启动前均14MiB、约41–47W，`/data4`可用5.1TiB；双endpoint鉴权门存在。
- 正式前逐endpoint真实验证模型identity、non-thinking fncall、tool-result回传与精确
  `PREFLIGHT_OK`，2/2通过。
- secret只从已授权进程内存经SSH stdin进入launcher子进程环境；不进argv、YAML、manifest、日志或
  文档。正式缺researcher/SwanLab key或offline模式均在建run目录前fail-closed。
- SwanLab日志确认online登录、run名称和URL；运行中禁止`swanlab sync`。
- 训前bare validation已274/274：14/274=5.11%，seen6/140、unseen8/134；此时无research/steering。
- 正式step1已进入dynamic researcher和student阶段。partial读数仅证实treatment执行，不作效果结论；
  完整128条后须用`audit_162_researcher_maxopen.py`证明每条有文档轨迹
  `steer_d_count == assistant token数`，且`budget_blocked/pacing_wait/bypassed/turns_capped/probe`全零，
  同时actor/SGLang更新完成。

- 正式step1已完整通过：16/16非空archive、128/128 student，`global_step=1/policy_version=0`；
  157,343个assistant token逐字对应157,343个D观测，`budget_blocked/pacing_wait/bypassed/
  turns_capped/probe_flipped`五项总计全零。P80初值下主门/倾斜9,988次（6.35%），红色替换972次，
  `reject/tilt=9.73%`；W_beh off-1=6.36%、grad norm=0.5145，actor/SGLang weight update
  23.95s/3.22s。498个配置/日志/run文本产物经受控secret扫描零命中；审计产物：
  `<bundle>/logs/maxopen_formal_step1_audit.json`。

## 只读watchdog与显微镜

| 项 | 值 |
|---|---|
| watchdog PID / PGID | `2295171 / 2295171` |
| watchdog日志 | `<bundle>/logs/watch_pgs_s2_sgl_1p7b_162_researcher_maxopen.log` |
| 周期 / stale门 | 600秒 / 7200秒 |
| 远端显微镜 | `127.0.0.1:8793`，PID `2427431` |
| 本地入口 | `http://127.0.0.1:18793/`，隧道PID `70021` |

watchdog只读、只告警，不kill、不restart。显微镜使用dynamic archive，不传静态GT表；实收显示
`B∞=81920`。旧GT显微镜8792/18792未触碰。首step状态曾因旧后端以“历史最大轨迹数”推断
完整批而在64/128时误标完成；已改为读取实收`train_batch_size×rollout_n=128`，新增回归证明
64条为partial、128条才完成，并只重启显微镜，训练进程未动。
