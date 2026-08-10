# Engine-Resident SPS 代码审阅(codex / gpt-5.6-sol)

**日期**: 2026-08-10 · **审阅对象**: `f0848e9..HEAD`（4 commits: `5dc0e31` / `b2c5a06` / `96b461b` / `ad3af42`）  
· **审阅方式**: codex CLI，模型 gpt-5.6-sol，只读

> **范围核验说明**：任务所列四提交的实际终点是 `ad3af42`，对应 `18 files changed, 1656 insertions(+), 9 deletions(-)`。审阅开始时实际 `HEAD` 已是 `ed41496`，使 `f0848e9..HEAD` 扩大为 93 个文件；审阅过程中又被并发推进到 `18752cd`。为避免把合流批次和并发修补混入结论，下文 finding 与行号固定于 `ad3af42`。后续 `18752cd` 对 wire 注册表和 d_var/soft_or 测试的影响另行注明。

## 结论先行

- 总体判定为 **NO-GO**：发现 **3 条 P0、6 条 P1、6 条 P2**。当前不能仅凭现有 V1 `PARITY PASS` 上生产。
- `draft_horizon>0` 目前并非严格的纯执行参数：EOS/停止符恰落在 K 窗尾时会在终止符之后继续生成；固定采样 seed 时，每个 K 窗还会重置 proposal RNG。两者都否定了无条件的逐位/轨迹分布恒等声明。
- engine wire 会在“先前已花预算、当前 turn 没有正扣费”时把总账 `budget_spent` 静默覆盖为 0，违反账本完整性红线。
- 因分布恒等前提已破，`executor` / `draft_horizon` 不进 `objective_fingerprint` 的决定目前不能成立；修复并补齐机器证明前，应将相关配置纳入指纹或禁止跨 executor/horizon resume。
- 现有对拍脚本既能漏报真实错误，也会因 G-E bypass 改变观测总体而误报；其 `PASS` 不能作为生产正确性证据。
- 改造确实把拒绝浪费上界降到了 `K-1`，并移除了训练进程与 server 之间的宽 top-K 网络序列化；但 Python/Numpy 判定和 top-K Python 化被集中到了高并发共享 actor，且超时后可能留下未计入负载的 GPU 孤儿任务。
- 与 `359aa2c` 的正常异常分类链条总体自洽：engine turn 异常会成为 `ENV_ERROR`，单 episode 重试后继续失败会进入 prompt-group failure，而不是把残缺轨迹当成功终态采样。

## Findings

### P0 — 正确性（阻断上生产）

#### P0-1: EOS/停止符恰落在 K 窗尾时被误判为“分段边界”，会在终止符之后继续 draft

- 位置：`src/trace_opd/steering/hook.py:264-268, 287-315, 442-451`；`tests/steering/test_sps_bounded_k.py:209-217`
- 场景：设 `draft_horizon=64`、本 turn 剩余预算大于 64，engine 返回恰好 64 个 token，且第 64 个是 EOS。代码仅用 `len(gen_ids) < draft_budget` 判断自然终止，因此得到 `False`。只要 EOS 被接受，三条路径都会继续下一窗：
  - budget/cap 直通路径在 `hook.py:296-298` 继续；
  - G-E 段级 bypass 在 `hook.py:312-315` 继续；
  - 正常 verify 且接受 EOS 后在 `hook.py:451` 继续。
  
  只有“拒绝后重采出的 replacement 是停止符”在 `hook.py:442-443` 被显式处理。现有 EOS 测试把 EOS 放在第二个 K=4 窗的第 3 位，返回长度为 3 `<4`，没有覆盖窗尾反例。
- 建议修法：保留 engine 的真实结束类别，区分 `stop` 与 `length`，令 `draft_finished = ended_by_stop or len(gen_ids) < draft_budget`；对 token stop 至少增加 `gen_ids[-1] in stop_ids` 的防线。补齐 EOS/stop-string 恰在 K、2K 边界的测试，并分别覆盖无预算直通、G-E bypass、verify 接受、EOS 被拒成非停止符、replacement 变停止符。
- 置信度：确认（读代码可证）

#### P0-2: 固定采样 seed 会在每个 bounded-K 请求重新起流，使 K 进入 proposal 的联合分布

- 位置：`src/trace_opd/steering/hook.py:241-252`；`verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:718-723, 896-908`；`verl/docs/advance/determinism.md:84-87`；`tests/steering/test_sps_bounded_k.py:1-15`
- 场景：`full_determinism=true` 或 sampling params 显式携带 `seed=s` 时，每次 `_prim` 都建立新 UUID 请求，而 `generate()` 又为每个请求注入相同 `sampling_seed=s`。无拒绝地生成 128 token 时：
  - K=0：一个请求消费随机流 `U1…U128`；
  - K=64：两个请求分别从 seed `s` 起流，消费 `U1…U64, U1…U64`。
  
  后 64 位因此不是原本的连续 proposal 流，并引入跨窗相关性。fake 测试刻意把 draft token 做成“位置确定”，绕开了真实 request-scoped sampler RNG，不能证明该配置下的恒等性。
- 建议修法：在具备分段不变的 counter-based/token-position RNG 前，对 `engine + draft_horizon>0 + full_determinism/显式 seed` fail loud；若要支持，应让随机数由稳定轨迹 ID 与绝对 token 位置索引，而不是每请求从同一 seed 重启，并增加真实 SGLang seeded K=0/K>0 对拍。
- 置信度：强推断（仓库明确记录 per-request seed 重置；SGLang 实际 bitstream 仍应真机确认）

#### P0-3: fresh turn ledger 与 `budget_spent` latest-wins 组合会把累计花费静默清零

- 位置：`src/trace_opd/steering/server_loop.py:86-92, 107-129`；`src/trace_opd/steering/hook.py:387-388`；`src/trace_opd/steering/wire.py:73-84, 116`；`tests/steering/test_sps_state_wire.py:69-105`
- 场景：server 每个 turn 构造空 `SteeringLedger`，但 wire 只恢复 `TrajectoryBudget.spent`。`ledger.budget_spent` 只在当前 turn 发生正 `eta` 扣费时赋值。若 turn 1 已花 10，turn 2 因预算耗尽、pacing 未解锁或全程门关而没有正扣费，则 turn 2 delta ledger 的 `budget_spent` 仍为 0；客户端执行 `total.budget_spent = delta.budget_spent` 后，总账从 10 变成 0。以后某 turn 再扣费时可能“自愈”，因此错误经常只在轨迹最后一个无干预 turn 才落盘。
- 建议修法：恢复 state 后立即令增量 ledger 的 `budget_spent = state.budget.spent`，或直接以响应 state 中的累计 spent 为真源；merge 时校验它不小于旧值且与 state 一致。新增“turn 1 花费、turn 2 零花费”的 wire/engine 两路径测试。
- 置信度：确认（读代码可证）

### P1 — 鲁棒性

#### P1-1: 对拍脚本在输入损坏和逐 token 密度界上 fail-open

- 位置：`scripts/engine_sps_parity_check.py:27-40, 56-70, 96-111`；`src/trace_opd/steering/ledger.py:149-176`
- 场景：
  - JSON 解析失败被 `except Exception: continue` 静默跳过；
  - 不检查 A/B 的任务集合、轨迹数、checkpoint、fingerprint 或 policy version；
  - 直方图和 log 数组均用 `zip`，长度不一致时尾部被静默截掉；
  - 两边直方图都缺失时 L1 返回 0；
  - NaN 的比较结果为 false，可逃过密度检查；
  - 密度检查使用轨迹级 `2*eta_max*c`，而账本已有每行实际 `steer_delta=2*eta_actual*c`。例如预算缩小后实际界为 0.05、错误 log-ratio 为 0.5、全局 cap 为 2.5，脚本仍判合法。
- 建议修法：loader 对每个文件严格解析并报告文件名；校验两侧完全相同的任务/样本集合和 run 元数据；要求所有稀疏数组长度等于 `steer_record_rows`、值有限、positions 严增、hist 宽度固定；逐行检查 `abs(log_b-log_pi) <= steer_delta+eps`；缺字段或坏文件一律 FAIL。
- 置信度：确认（读代码可证）

#### P1-2: 当前统计判据不是等价性检验，并且 G-E bypass 使两侧比较的总体不同

- 位置：`scripts/engine_sps_parity_check.py:73-87, 112-135, 150-158`；`tests/steering/test_sps_bounded_k.py:182-195`
- 场景：
  - 推荐的每侧 256 条、胜率约 50% 时，合并 2σ 容差约为 8.8 个百分点；真实的 8pp win-rate 回归仍会 PASS。样本更少时更宽。
  - token 率把同轨迹内强相关 token 当独立 Bernoulli，噪声估计口径不成立。
  - bounded K 会合法地增加整窗 G-E bypass；被 bypass 的低强度位置不进入 `n_candidates` 和 D histogram。于是两路径即使生成分布相同，`open_rate=n_base_open/n_candidates` 和 D histogram 也在不同条件总体上计算，可能误报。
  - response length、gate bypass 和 discarded 只打印，不参加判定。
- 建议修法：使用预注册固定等价界和有功效分析的 TOST/置信区间，而不是“差异未超过观测 2σ 即等价”；按 task 做配对或分层 bootstrap。开门/干预率应以全部执行 token（如 `ent_count`）为公共分母，并把 bypass 位计为零；D 分布则只比较共同审计总体，或对 bypass 窗做不影响行为的 teacher audit 抽样。强制检查 response-length 分布和任务级 outcome。
- 置信度：确认（读代码可证）

#### P1-3: 客户端超时取消没有 server-side deadline，可能留下未计入负载的 GPU 孤儿 turn

- 位置：`verl/verl/trainer/ppo/v1/agent_loop_tq.py:142-165`；`src/trace_opd/steering/engine_client.py:78-89`；`verl/verl/workers/rollout/llm_server.py:289-293`；`src/trace_opd/steering/server_loop.py:95-129`
- 场景：prompt-group 的 `asyncio.wait_for` 超时后，客户端协程被取消，manager 的 `finally` 立即执行 `_release_server`。payload 没有 deadline/cancel token，server 的 `run_steered_turn` 也不检查请求是否已被放弃。若 Ray 取消未传播，server 可继续完成整个 assistant turn；SSB 单 turn 上限约 4096、K=64 时最多约 64 个 draft/verify 周期。外层随即开始同版本重试，而 load balancer 已把孤儿从 inflight 计数中移除，可能把重试继续压到同一 replica。
- 建议修法：payload 增加唯一 turn ID 与绝对 deadline，在每次 draft、verify 之前检查；提供 server-side active-turn/cancel 表并尽可能 abort 底层 request；负载统计应覆盖 server 真正结束，而不是客户端 await 结束。记录 orphan 数、剩余周期和被浪费 GPU 时间。
- 置信度：强推断（代码确认没有 deadline；实际 Ray 取消传播需部署版本运行时证据）

#### P1-4: 同步 top-K Python 化和 judge 循环会阻塞高并发 async actor 的共享 event loop

- 位置：`verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:64-141, 161-195`；`src/trace_opd/steering/hook.py:67-123, 283-445`；`verl/verl/workers/rollout/replica.py:257-260`
- 场景：actor 的 `max_concurrency` 至少为 1000。每个 K=64、obs_topk=256 周期在 engine await 返回后，同步构造 draft/verify 的 Python list，再逐 64 个位置创建两个 map、多份 Numpy 数组、kernel 与 ledger 行，中间没有 await。64 个并发 steered turn 会在同一 event loop 上轮流执行这些 CPU 段，也会拖延普通 `generate()` 与控制 RPC。网络 JSON 面虽然消失，但原先怀疑的 Python 热段没有消失，且被集中到共享 actor。
- 建议修法：先加入 event-loop lag、top-K extraction CPU、judge CPU、GC 和队列等待指标；在不改 RNG 次序的前提下每若干位置 cooperative yield；评估有界线程池或向量化 Torch/GPU 判定。生产门应包含 64 并发下普通 generate 的 p95/p99 隔离指标。
- 置信度：强推断（阻塞代码可证，影响量级需生产 profile）

#### P1-5: prompt-group 中一个 session 提前抛错时，清理路径可能无限等待另一个挂死 session

- 位置：`verl/verl/trainer/ppo/v1/agent_loop_tq.py:142-165`
- 场景：`asyncio.gather` 默认在第一个异常时向外抛出，但不会取消其他子 task。进入 `except` 后，代码直接执行一个没有 timeout 的 `await asyncio.gather(*tasks, return_exceptions=True)`。若 engine 路径中一个 session 快速报协议错、另一个 session 永久挂在 remote call，外层 2700 秒 timeout 已不再包住清理 await，prompt 永远无法 purge、重试或标记 failure。
- 建议修法：异常分支先取消全部未完成 task，再使用独立、有限的 cleanup timeout 等待；为了防超时后迟到 task 复活旧行，给每次 attempt 加 generation/epoch，写 TQ 时拒绝旧 epoch，而不是通过无限等待保证顺序。
- 置信度：确认（读代码及 asyncio gather 语义可证）

#### P1-6: K=64 并不保证恢复 G-E bypass，第二个性能根因可能基本未解决

- 位置：`src/trace_opd/steering/gate.py:117-125`；`src/trace_opd/steering/hook.py:301-315`；`docs/engine-sps-verification-runbook.md:75-88`
- 场景：代码注释记录 `w>=w_min` 的非零强度率约 35–46%。在独立参考模型下，64-token 窗全低于 `w_min` 的概率约为 `0.54^64` 到 `0.65^64`，仅约 `8e-18` 到 `1e-12`。真实位置可能有低熵长程聚类，因此不能直接断言为零，但“从 261 改为 64”本身不足以证明 teacher-scored token 数显著下降。d_var/soft_or 又明确 `needs_teacher=True`，完全没有该 bypass 红利。
- 建议修法：V2 前先按真实序列测量 all-low run-length 分布和各 K 的 `n_gate_bypassed/ent_count`；把“teacher 实际打分 token 降幅”列为显式判据。若 K=64 仍近零，应考虑只由学生熵决定的自适应窗口切分，或加入不改变行为分布的 force-verify/bypass 对照档。
- 置信度：强推断（需真实 SSB 序列相关性证据）

### P2 — 建议

#### P2-1: 当前 state wire 字段完整，但缺少真正的“新增可变态即 fail-loud”守卫

- 位置：`src/trace_opd/steering/wire.py:33-55, 65-70`；`src/trace_opd/steering/gate.py:150-158, 202-203`；`src/trace_opd/steering/budget.py:54-65, 101-102, 115-132`
- 场景：在 `ad3af42`，当前全部跨 turn 可变态确实只有 budget 的 `spent/turn`、cap 的 `count`、gate 的 `_decisions/_opens`，wire 没有现存遗漏。但以后新增属性而忘记修改 wire 会静默丢态。`ledger_from_wire` 对缺字段还会使用 dataclass 默认值，可能把版本漂移伪装成合法零值。
- 建议修法：增加协议版本和 exact-key 校验；用 `__slots__`/dataclass metadata 区分 immutable 与 wire-mutable，并断言所有实例属性都被分类。后续 `18752cd` 引入 `_WIRE_MUTABLE` 注册表，减少了重复清单，但新增属性同时忘记注册仍不会报错，尚未形成完备性证明。
- 置信度：确认

#### P2-2: `token_output_to_prim` 与 `EngineGenerate`/manager 的字段语义被手工复制

- 位置：`src/trace_opd/steering/server_loop.py:41-68`；`src/trace_opd/train/backends.py:37-72`；`verl/verl/workers/rollout/llm_server.py:263-275`；`tests/steering/test_engine_executor.py:175-205`
- 场景：当前字段与 min/max 语义是对齐的，但两套适配器没有共同真源；测试只覆盖了部分 response top-K 和 version 字段，未完整锁定 verify 的 prompt rows。`preflight.py:60-66` 也只是打印提示，没有真正探测 actor 是否提供 `steered_generate`。
- 建议修法：抽成共享的 typed adapter/schema；由底层 `generate()` 统一产生 version range，避免 manager 语义镜像。增加所有字段的 contract test，并让 preflight 做 capability/version probe。
- 置信度：确认

#### P2-3: 客户端响应落地不是完全事务式

- 位置：`src/trace_opd/steering/engine_client.py:87-97`
- 场景：RPC 抛异常时客户端 state 确实完全不动；但收到响应后先 apply state，再解析和 merge ledger。若 ledger schema、位置或 histogram 校验失败，state 已推进而账本未合并。当前 episode 重试会重建 state，因此通常不会入训，但该适配器本身不满足“完整提交或完全不提交”。
- 建议修法：先把响应解析、schema 校验和 ledger merge 应用于临时副本，全部成功后再一次性提交到真 state。
- 置信度：确认

#### P2-4: 每 turn 重传完整 teacher/student prefix 可优化，sticky 路由还应增加成本感知

- 位置：`src/trace_opd/steering/engine_client.py:79-85`；`verl/verl/workers/rollout/llm_server.py:80-110`
- 场景：8K token prefix 的 pickle 体量通常为数十 KiB，64 并发约低个位 MiB；即使接近 32K，也仍远小于旧路径每周期 30–196 MiB top-K JSON，因此不是当前主瓶颈。但多轮重复传输是纯冗余。负载均衡又把一个长 steered turn 与一个短 generate 都计为 1，sticky 后不会因 replica 长尾重新路由。
- 建议修法：首 turn 传全量，之后传内容 hash/长度；server 使用有界、按模型版本隔离的 prefix cache，miss 时 fail loud 请求补全。监控 replica 级在途 token/cycle，而不仅是请求数。
- 置信度：强推断（体量需以真实 Ray 序列化实测）

#### P2-5: 机器证明仍缺少决定性边界用例

- 位置：`tests/steering/test_sps_bounded_k.py:1-15, 168-227`；`tests/steering/test_sps_state_wire.py:69-105`；`tests/steering/test_engine_executor.py:116-205`
- 场景：四提交版本只覆盖 adaptive/entropy；entropy 不能代表 d_var/soft_or，因为后两者 `needs_teacher=True`，不会走段级 bypass。后续 `18752cd` 已补 d_var/soft_or 非平凡测试，但仍未覆盖 EOS 窗尾、真实 fixed-seed sampler、先花费后零花费的 ledger、坏响应事务性和 actor cancellation。
- 建议修法：补齐上述反例，并将五种 gate、预算恰耗尽、pacing 恰解锁、reject-cap 第 R 次拒绝、stop 的窗中/窗尾、verify abort 组合参数化。
- 置信度：确认

#### P2-6: V2 对 aggregate K 效应分离尚可，但不能分别归因于“拒绝浪费”和“bypass”，基线与重复数也不足

- 位置：`docs/engine-sps-verification-runbook.md:46-101`
- 场景：E1→E2 在同一 engine executor 下能隔离“启用 bounded K 的总体效果”，但 K 同时改变请求周期数、拒绝 suffix 浪费、G-E bypass 分组、scheduler batching、cache 压力和 actor CPU 切片，不能把差额单独归因于某一根因。旧 107 分钟基线还使用不同 timeout/代码状态，手册仅把刷新写成可选；每档只取第二个 step 也没有方差估计。
- 建议修法：同一同步 SHA 必须重跑 python A0；采用交错顺序和多次重复。增加 engine K64 + force-verify 的执行层对照，分开估算 suffix 与 bypass；对 ≥3× 指标报告置信区间，并统一任务集、并发和缓存预热口径。
- 置信度：确认

### 已核实为自洽的关键路径（非 finding）

- `SpsGenerate` 自身除外部 `TrajectorySteerState` 引用外没有跨 turn 可变态；draft/verify/prefix/config 属性构造后未修改。
- `TrajectoryBudget`、`TurnRejectCap`、`IntensityGate` 当前可变字段在四提交版 wire 中是完备的。
- d_var/soft_or 走 `IntensityGate.decide()` 的统一软门分支，但因 `needs_teacher=True` 不会进入 G-E 段级 bypass；因此原 entropy 测试不能代表其完整分支组合，不过未发现当前代码因此产生 RNG 错位。
- `budget_spent` 采用 latest-wins 的语义本身正确，因为 server 恢复的是轨迹累计 budget；缺陷在于零扣费 turn 没有把该累计值写入 fresh ledger。
- engine RPC 异常会由 `run_episode()` 分类为 `ENV_ERROR`，`_student_rollout()` 用新 state 重试一次，二次失败向上传到 prompt-group failure；`ReplayBuffer.sample()` 对 failure fail loud。没有发现把残缺 engine turn 当正常 truncated 轨迹入训的路径。
- 仓库内没有为 `steered_generate` 增加显式应用层重试包装或 Ray task-retry 配置。若运行时发生透明重放，无状态 payload 意味着客户端只合并最终返回的一份状态，不会逻辑双计账；主要副作用是重复 GPU 工作和 cache 污染。Ray 版本的具体 retry 行为因禁止访问仓库外文档，未作外部确认。
- 三个原性能问题中：拒绝 suffix 已被实质性限制到 `K-1`；宽 top-K 不再跨训练进程 RPC；段级 bypass 是否恢复则尚无代码层保证。

## 总体判定

**不能在现有 V1 对拍脚本 PASS 后上生产。**

必须先完成：

1. 修复 P0-1 的 K 窗尾停止语义并补边界测试。
2. 修复 P0-3 的跨 turn `budget_spent` 累计账。
3. 对 P0-2 实现分段不变的 seeded sampling，或在 engine bounded-K 下明确禁止 fixed seed/full determinism。
4. 修复 P1-1/P1-2，使 V1 成为严格输入校验、逐行密度界检查且具统计功效的等价性门。
5. 修复 P1-5；为 P1-3 增加 server-side deadline/cancel 护栏。
6. 在 64 并发真机上验证 P1-4/P1-6；若 event-loop lag 或 teacher-scored token 没有显著下降，先处理再灰度生产。

P0-1 与 P0-2 说明 bounded K 并非在所有受支持场景下分布恒等。因此，在它们修复且真实 backend 测试补齐前，必须暂停“`executor` / `draft_horizon` 不进 `objective_fingerprint`”这一设计决定：至少应禁止不同 executor/horizon 之间沿用同一 resume 指纹，或临时将二者纳入指纹。不能把它们只作为普通实现 bug 而继续维持“纯执行层”声明。

## 测试执行记录

- `python -m pytest tests/steering/ -q`
  - 退出码：1
  - 未进入测试收集。pytest 初始化捕获时无法创建临时文件，报 `FileNotFoundError: No usable temporary directory found`。
- `python -m pytest tests/ -q`
  - 退出码：1
  - 同样在测试收集前因全局只读沙箱无可用临时目录失败。
- pytest 执行时实际工作树为干净的 `18752cd`，不是任务声明的 `ad3af42`。因此没有可报告的 passed/skipped 数，也不能把本次失败解释为测试回归或环境依赖漂移。
- 全程未创建、编辑、移动或删除任何文件；结束时 `git status` 仍为干净工作树。