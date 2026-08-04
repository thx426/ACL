# CD-006 → CD-010 实验反馈(append-only,结论先行,带日期与优先级)

机制侧变更走 `mechanism-changelog.md`(CD-010 维护)。本文件只记实验侧发现的
现象、阻断与改进需求 —— 机制代码不在实验侧改。

---

## 2026-08-04 · [P1] launcher 把代码根硬编码为 `REPO/"trace_opd"`,按 run-guide 建议的部署路径会静默加载**别的树**的 verl

**结论**:`scripts/launch_train.py` 的 `REPO = Path(__file__).resolve().parents[2]`
之后所有路径都写作 `REPO / "trace_opd" / ...`(src / verl / configs / train)。这要求
worktree 在磁盘上**必须命名为 `trace_opd`**。run-guide §0 建议同步到
`/data3/.../TRACE-OPD/trace_opd-pgs/` —— 那样 `REPO/"trace_opd"` 会解析到 CD-008 的
`TRACE-OPD/trace_opd/`,PYTHONPATH 前置的是**那棵树的旧 verl(仍带蒸馏面)与旧 src**,
而 `configs/pgs.yaml` 在那里根本不存在(模板漂移检查会静默跳过)。

**失败形态**:不报错、跑起来、跑的是错的代码 —— 最坏的一类故障。

**实验侧规避(已用,零代码改动)**:部署为
`/data3/wushang/workspace/TRACE-OPD-PGS/trace_opd/`(父目录即 REPO,内层命名 trace_opd),
与 CD-008/CD-005 的 `TRACE-OPD` 树完全隔离。env 脚本
`/data3/wushang/workspace/TRACE-OPD-PGS/env_pgs.sh`,启动器 `run_pgs.sh`。
已验证:`[launch] verl from: .../TRACE-OPD-PGS/trace_opd/verl (vendored)`、
该树 `is_distillation_enabled` 引用数 = 0。

**建议(CD-010 定夺)**:launcher 改为自定位(`ROOT = parents[1]`,即脚本所在树的根),
路径全部相对 ROOT;这样 worktree 叫什么名字都对,也免去部署侧的隐性约定。
在此之前 run-guide §0 的建议路径应更正。

---

## 2026-08-04 · [P2] C3(PGC 线)崩溃根因 = /data3 写满;PGS 起跑前已清理 533G

**结论**:C3 `returncode 120`(2026-08-04 09:49,step 78/97)不是机制/代码故障 ——
`/data3` 当时 100% 满(可用 0)。现场:checkpoints 7 个共 623G,其中
`global_step_78` 只有 83G(其余均 90G)且 `latest_checkpointed_iteration.txt` 停在 68,
即保存到一半写爆磁盘。

**处置(用户授权)**:删除 global_step_{18,28,38,48,58} 与崩溃残骸 78,**保留
global_step_68**(最后一个完整 ckpt,PGC 线续跑/评测的唯一凭据)。释放 533G,
`/data3` 降至 97%。

**对 PGS 的影响与纪律**:PGS 侧 `max_actor_ckpt_to_keep=3` 且无 critic
(单 ckpt ≈45G vs C3 的 90G),100 step / save_freq=10 下峰值约 135G,余量够。
但 533G 不宽裕 —— 长跑期间需盯磁盘(建议纳入监控项:`df /data3` <100G 即告警)。
