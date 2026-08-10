# Engine-Resident SPS 验证运行手册(P0b,交 CD-006 执行)

**日期**:2026-08-10 · **面向**:CD-006 · **机制方**:CD-010
**设计真源**:`docs/plans/2026-08-10-engine-resident-sps-design.md`(§4 验收门、§6.5 实现状态)
**用户裁定**:验证必须在**真实生产形态**下看效果——八卡 A800、生产并发、SSB 大批量
rollout(v3.3×v4 全量参数),不是单卡玩具冒烟。GPU 窗口与在跑 run 的协调走用户。

**一句话背景**:steered 采样循环从训练进程(每周期 2 次 HTTP+巨型 JSON、拒绝丢整段)
搬进 rollout server actor 进程(每 turn 1 次 RPC + bounded K)。采样分布逐位恒等
(机器证明已过,fake 世界 bit-equal);真机要验的是①整链正确性(统计对拍)
②生产场景下的吞吐收益与归因。

---

## 0. 前置(顺序硬性)

1. **你们先把 pgs-v33 的未提交批次落提交**(enable_thinking / eval.n / selfres
   错误增强 / SSB build_dataset / 三 yaml pin——changelog 已催两轮)。
2. 通知 CD-010,我把 pgs-v33 merge 进 `pgs-engine-sps`,给出**同步点 sha**。
3. 按既有跨机同步纪律部署该 sha 到 A800(**新 worktree/新目录,勿动生产树**;
   先 sha 对拍分清远端状态,勿 stash——`sync-overwrites-prod-patches` 教训)。
   验证期间产生的一切配置/脚本按你们惯例进 `experiments/`。

> 在同步点确认之前,勿用本手册的任何配置起卡;`executor: engine` 在 V1 对拍
> PASS 之前**禁止**进任何正式 run。

## 1. V0 · 部署冒烟(单 replica,≈半小时)

任一 smoke config(ALFWorld 或 SSB 皆可)+ 两行差异:

```yaml
steering:
  # ...其余照抄对应 python 档 smoke...
  executor: engine
  draft_horizon: 64
```

**判据(全过才进 V1)**:
- server actor 无 `trace_opd` import 报错(有则先修 actor 进程 PYTHONPATH——
  报错信息里有提示);
- ≥1 条 steered 轨迹端到端完成,rollout dump 的 `steering` 段含
  `steer_n_draft_discarded`(新键,在场即 schema 通)且 `policy_version`
  的 min/max 非 null(=当前权重版本;None 说明原语适配层被绕过,停下报我);
- `preflight` 对该 config 打印 engine 档提示且全绿。

## 2. V1 · 正确性统计对拍(生产并发,1 个 rollout step ×2)

同一 ckpt、同一任务批(建议 SSB train 头 64 任务的固定 parquet 切片)、
**其余配置逐字节相同**,各跑 1 个 rollout step(不必训练,拿 dump 即可):

- run A:`executor: python`(draft_horizon 必须 0,校验会拦)
- run B:`executor: engine, draft_horizon: 64`

并发/replica 数用生产值(8 卡形态)。steered 行总量建议 ≥256 条。然后:

```bash
python scripts/engine_sps_parity_check.py \
  --a runs/<A>/rollouts/step_XXXXX --b runs/<B>/rollouts/step_XXXXX
```

**判据**:`PARITY PASS`。脚本语义(codex 审阅 P1-1/P1-2 后修订):
- **判定项**:账本完整性(坏文件/数组长度/非有限值/positions 单调,一律
  fail-loud 点名文件)、**逐行密度比界 |log_b−log_pi| ≤ 该行 steer_delta**
  (行级紧界,非轨迹级全局 cap)、两侧任务多重集与 global_step 一致、机制率
  (open/intervene 以 `steer_ent_count` 全执行位为公共分母——G-E bypass 触发
  面随 K 合法变化,旧的 n_candidates 分母会比较不同总体)、ent_hist L1;
- **参考项(打印不判)**:won(差值±2σ,低功效指标)、resp_len、d_hist、
  bypass/discarded/budget。
- **PASS ≠ 任务效果等价证明**——效果面由 V2 的 bare 同水位硬前提与任务级
  outcome 承担。FAIL 任一项:保留两侧 dump,报 CD-010,勿自修机制侧。

## 3. V2 · 吞吐效果与归因(用户重点;生产同构现场)

**现场 = 你们吞吐分析的同构场景**:A800 八卡、Qwen3.5-4B、`ssb_4b_selfres_a800`
同配置(v3.3×v4 全量参数,research/narration/训练全流程照跑),唯一差异是
executor 档。对照基线 = 你们已有的 step-1 分解(107 min:bare 7.3 / research
48.0 / steered 54.1 / 超时尾 ~27 / 训练 ~17)——若嫌陈旧,可先重跑 1 step
python 档刷新基线(顺便让新账本键就位)。

**基线刷新(codex 审阅 P2-6 后从"可选"改为必做)**:先在**同一部署 sha** 重跑
1–2 step 的 python 档 A0——旧 107 分钟基线的代码/timeout 语义已多轮变动,跨代
比较不可靠;两档各取第 2 步为主读数、两步差作粗方差参考。

**归因两步 + sweep(每档 2 个 step,第 2 步作读数主——第 1 步含缓存冷启)**:

| 档 | 配置 | 隔离的贡献 |
|---|---|---|
| E1 | `executor: engine, draft_horizon: 0` | **纯宿主搬迁**(HTTP/JSON 面消失,拒绝丢弃仍在) |
| E2 | `executor: engine, draft_horizon: 64` | + **bounded K**(近平方放大切断) |
| E3(E2 达标后抽验) | K ∈ {32, 128} | K 甜点 |

**每档记录**(方法沿你们的吞吐分析,§附录可复现):
1. 整步墙钟与三流分解(rollout json mtime 按 `sample_role`/`steer_arm`);
2. steered 吞吐(条/分)vs 基线 0.8;bare 吞吐 vs 8.8(不该动,是对照);
3. **`steer_n_draft_discarded`**(dump 直读,新计数):基线预期 ≈2.69×执行
   候选量,E1 应基本不变,E2 应降一个量级——这是 bounded K 生效的直接证据;
4. GPU util 与 sglang scheduler CPU(基线 8×440%——数据面消失的直接证据是
   它显著下降);
5. 超时/重试:`agent_loop_tq` 日志 `attempt 2/2` 与 `Purged ... residual` 行
   计数(**timeout 语义已变**:超时组 purge+同版本重试一次,二次失败
   ReplayBuffer fail-loud——长尾读数形态与旧基线不同,分解时把重试墙钟单列);
6. radix/前缀复用的代理读数(若 sglang 指标可得则记 cache hit;拿不到就记
   verify 段 TTFT 变化,标注口径);
7. **teacher 实际打分量降幅**(codex 审阅 P1-6:K=64 的段级 bypass 复活并无
   数学保证——(1−q)^64 在 q=35–46% 下仍≈0,除非真实序列有低熵长程聚类):
   每档记 `steer_n_gate_bypassed / steer_ent_count`;若 K 档相对基线无显著
   提升,说明第二根因(bypass 失效)未被 bounded K 解决——如实记录,后续
   走自适应窗口切分等设计层方案,不影响根因一(拒绝浪费)的收益判定。

**预注册判据**(2026-08-10):
- 主判据:E2 的 **steered 流墙钟 ≥3× 改善**(54.1 → ≤18 min);
- 整步墙钟显著缩短(预期 107 → 30–45 min 量级[推断],以实测为准);
- 硬前提:E1/E2 的对拍脚本对同 step dump 复跑仍 PASS,`won_bare` 与基线
  同水位(bare 臂不经新路径,动了=污染,立停);
- 若 E2 <2×:停 sweep,把 discarded/CPU/分解数据打包报 CD-010 归因。

## 4. 产物与反馈

- runs 目录按惯例;每档 step 的 rollout dump 完整保留(对拍与归因都吃它);
- 结论走 `experiment-feedback.md`:V0/V1/V2 逐项判据结果 + 数字表
  (含可溯路径)+ 撞到的坑;
- 任何 FAIL/异常:**回退就是 `executor: python`**(一键,状态与账本 schema
  完全兼容),保留现场再报。

## 5. 已知边界(读数时别误判)

- engine 档下 `steer_n_candidates` 口径不变(仍不含丢弃段)——GPU 功估算
  请改用 `steer_n_draft_discarded` 修正(你们 §2.3 指出的低估即此修正);
- 本验证不含 §7.5 SSB 预算标定与 continuation 并发(配套项,另行推进)——
  E2 的整步读数里 research 48 min 与超时尾的残余属预期,不计入本改造的
  成败判据;
- `draft_horizon` 与 executor 均不进 objective_fingerprint(纯执行层,分布
  恒等有机器证明+对拍门双保险)。
