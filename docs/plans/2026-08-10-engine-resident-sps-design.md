# Engine-Resident SPS:引擎内双序列推测循环 —— 设计文档 v1

**日期**:2026-08-10 · **分支**:`pgs-engine-sps`(worktree `trace_opd-pgs-engine`,基点 `f0848e9`)
**用户裁定**(2026-08-10):在**保证方案机制有效性效果的前提下**,最大可能根治 SSB 吞吐问题;
支持引擎内改造(MENTOR 先例);四裁定点通过——server-resident 双序列形态 / PoC 资源由用户安排 /
配套两项(SSB 预算标定 + continuation 并发)同批 / 对拍验收红线(不过不切、开关可回退、不进指纹)。
**病因真源**:`docs/pgs-throughput-ssb-analysis.md`(CD-006,2026-08-09)。

---

## 0. 一句话设计

把 `SpsGenerate` 的 draft→verify→judge 循环从**训练进程(HTTP 客户端)搬进
SGLang server actor 进程**,以进程内 engine 调用为 draft/verify 原语,并给 draft
加 **bounded K 分段**;机制判定层(kernel/gate/budget/ledger)与循环体**原样复用**,
采样分布逐位恒等,每 turn 的网络往返从 ~2×(A+R) 次巨型 JSON 降为 **1 次**。

## 1. 病因与解法对应(为什么这样设计)

| 病因(SSB 实测) | 机理 | 本设计的解 |
|---|---|---|
| 拒绝重启近平方,GPU 无效功 3.69× | draft 整 turn 余段(~261 tok),任一拒绝丢弃余段 | **bounded K**:单次 draft ≤K tok,拒绝浪费上界 K−1;K 可取 32–128(见 §2 MENTOR 对照) |
| 每周期 2 次 HTTP+30–196 MiB JSON,8×440% CPU 解析 | 循环宿主在训练进程,256 宽明细逐周期过网络 | **循环下沉 server actor 进程**:明细以 numpy 直进判定,永不序列化;每 turn 仅 1 次 RPC(参数进、tokens+账本增量出) |
| 段级 bypass 失效((1−q)²⁶¹≈1.5e-6) | bypass 粒度 = 整 turn 段 | bypass 天然变 **K 窗级**;且 verify 变廉价后 bypass 从救命门降级为红利 |
| 27 min 超时长尾 | 轨迹长度无上界(非本设计范围) | 配套项:§7.5 SSB 语境预算标定(轨迹/拒绝数/墙钟三重封顶) |

关键修正:三条流(bare/research/steered)**共享同一批 GPU**(util 84%),steered 释放的
无效功回流全体请求——收益不是旧估的 1.23× 单流天花板,整步预期 107 → 30–45 min 量级
[推断,P0b 实测为准]。

## 2. MENTOR 对照结论(代码取证,`related_work/code/MENTOR/`)

它的"引擎内改造"= ①借 vLLM **stock** SpecDecodeWorker 框架(未改;`SpecGRPOSampler`
注册于 `vllm/spec_decode/spec_decode_worker.py:216`)②只新写接受采样器
(`vllm/model_executor/layers/rejection_sampler.py:162-217`:全词表 fp32 混合
`q=π+(p_T−π)·w`、`w=clamp(H_full/threshold)`、标准接受 + `(q−π)₊` 残差重采)
③rollout 起**第二个 LLM**(teacher 为主模型、student 为 draft model,
`verl/workers/rollout/vllm_rollout_spmd.py:90` speculative_config,K=4)。

**学**:循环不出引擎侧的执行思想;bounded K;GPU 侧批判定(我们的 P2)。
**不学**:①spec-decode 框架——单序列假设装不下 target-only doc 前缀(doc 入公共
prompt 污染 draft,不入则 target≠p_T(·|h⊕doc));②全词表算术混合——丢保尾核的
密度比界/尾质量恒等/预算证明;③其缺陷(live 权重整 gather 后丢弃、prompt
decode→retokenize、FLASHINFER 静默旁路)。
**结构优势**:它 draft/target 是两个模型(速度差 ⇒ K 只能 4);我们 teacher=student
**同权重**——零第二份显存,verify 只是另一条前缀的增量 prefill,K 可大一个量级。

## 3. 架构

### 3.1 三步改造原则(能复用的一行不重写)

`SpsGenerate` 现已是 draft/verify **注入式**、backend 无关的循环
(`steering/hook.py:149`:两个 `Callable[[ids, params], Awaitable]`)。改造 =

1. **bounded K**(P0a):循环体 draft 请求 `max_tokens = min(remaining, draft_horizon)`;
   `draft_horizon=0`(默认)= 现行为逐字节不变。恒等性见 §4。
2. **换注入**(P0b/P1):server actor 进程内构造 `_local_draft/_local_verify`
   (复用 `async_sglang_server.py` 既有取数路径:top-K 明细 :695、窗口化打分
   `logprob_start_len` :68–137、覆盖校验 :189——全部是已生产验证的自有 patch),
   直调 engine,不过 HTTP。
3. **搬宿主**(P1):`SpsGenerate` 实例化点从训练进程移到 server actor;训练进程
   的 hook 退化为一次 RPC 的薄客户端(`EngineSteeredGenerate`)。

### 3.2 时序(P1 形态,每 assistant turn)

```
训练进程 EngineSteeredGenerate ──1 次 RPC──▶ server actor `steered_generate`
  请求:prompt_ids(学生流)/ teacher_prefix_ids / sampling_params
        / 机制参数(topk,c,obs_topk,gate cfg,K)/ 轨迹状态(budget,cap,rng)
                                              │ loop(进程内,无网络):
                                              │  draft = engine(seq_S, ≤K tok, top-K 明细)
                                              │  [G-E K 窗全低熵 → 直通,teacher 不动]
                                              │  score = engine(seq_T+draft, 窗口 prompt-logprob)
                                              │  judge = 原 kernel/gate/budget/ledger 代码
                                              │  接受前缀 append 双序列;拒绝位重采 1 tok;续
  响应:token_ids / gen_extra(policy span,stop_reason)◀──
        / 轨迹状态回写(budget,cap,rng)/ 本 turn 账本增量(records+计数器+直方图)
训练进程:ledger.merge(增量) —— 落盘账本与现在逐字节同构
```

**状态往返(无状态 server,replica 无关)**:`TrajectoryBudget`/`TurnRejectCap` 是
计数器级小对象;rng 用 `Generator.bit_generator.state` 序列化——跨 turn rng 流与
单进程逐位一致(对拍前提)。账本**只回增量**(本 turn records + 计数器 delta +
直方图 delta),`SteeringLedger.merge()` 新增并单测;体量 ≈ 现落盘账本分摊
(~1.5MB/轨迹),对比现状每周期 30–196 MiB 明细小两个数量级。

### 3.3 KV / radix cache 账

seq_S(`h+流`)与 seq_T(`doc⊕h+流`)靠 radix cache 前缀复用——**与现状同一套
机制**(现在 verify 也走同 engine 同 cache),无新增显存类别;新增的只是命中率
诉求。逐出即重算前缀 prefill(正确性无损,吞吐退化)——P0b 实测 64 并发下的
命中率与逐出率,是 K 取值 sweep 的一部分。sticky 路由(同 episode 两前缀同
replica)沿用现有 hook 的 sticky key 机制。

## 4. 分布恒等论证与对拍验收门(用户红线)

**数学层**:SPS 每个位置的接受/重采只依赖(前缀,该位置 π 明细,p_T 明细,rng
draw)。draft 分几段只决定"哪些位置在同一批被 verify",不进入任何判定量 ⇒
bounded K 与整段 draft **逐位分布恒等**(恒等式而非近似)。engine 采样的 draft
token 流在给定前缀下同分布;判定层代码原样复用 ⇒ 核语义 bit 级一致。

**机器验证(切换前置,缺一不可)**:
1. `test_sps_bounded_k_equivalence`(P0a,无 GPU):fake 世界 = 确定性的
   位置→(π 明细, p_T 行, draft token) 映射 + 共享按序 rng;断言 K=∞ 与
   K∈{1,4,64} 的输出 token 流、账本记录**逐字节相等**。可行性依据:两路径的
   判定 rng draw 序列按执行位置对齐(draft 段边界不消耗判定 rng)。
2. P0b 脚本:同 ckpt 同任务批,HTTP 路径 vs 进程内路径,统计对拍(接受率/
   开门率/拒绝分布/账本字段分布/`e^{±2ηc}` 界核验)+ 可控 seed 下 draft 段
   逐位对拍。
3. preflight:FC 门同款,执行路径探针(engine 档要求 server 具备
   `steered_generate` 能力,fail-loud)。

**开关与指纹**:`steering.executor: python | engine`(默认 python,灰度可回退);
**不进 objective_fingerprint**(纯执行层,分布恒等);`draft_horizon` 同理为执行
参数。

## 5. 分期与出口判据

| 期 | 内容 | 出口判据 |
|---|---|---|
| **P0a**(无 GPU,即刻) | bounded K 参数落 hook + 恒等对拍测试 + 本文档 | 对拍测试绿;`draft_horizon=0` 下全量测试无一行为变化 |
| **P0b**(单卡窗口,用户安排) | server 进程内原语 + 单卡脚本对拍 HTTP vs 进程内 + 吞吐/radix 命中实测 + K sweep{32,64,128} | 统计对拍过;steered 流实测加速 ≥3× 才进 P1 |
| **P1**(生产化) | `steered_generate` endpoint + 薄客户端 + 状态往返/账本 merge + abort/降级语义 + 开关 + preflight 门 | 对拍门全过;SSB 短 run 整步实测;旧路径回退验证 |
| **P2**(可选) | GPU 侧批判定(候选集+核+judge torch 化驻 GPU)、`token_ids_logprob` 定向取数 | 仅当 P1 实测 CPU 判定成为新瓶颈 |

**配套项(同批,另行落地)**:①§7.5 SSB 语境预算标定(分布取数→用户裁定数值);
②research 侧 continuation 并发化(零语义)+ research 48 min 内部分解取数;
③已修的超时残组语义(`359aa2c`)与本设计正交叠加。

## 6. 风险与待核清单

| 风险 | 护栏 |
|---|---|
| engine 进程内并发调用的调度开销(周期 ~76/条) | P0b 首测项;ms 级 × 76 ≪ 现 HTTP 面,若超预期调大 K |
| radix 逐出(64 并发×2 序列×16k) | P0b 实测命中率;逐出只损吞吐不损正确性;预算墙封长度 |
| 多轮 turn 间状态漂移 | 状态序列化往返 + prefix 逐调校验(hook 现有 fail-loud 原样保留) |
| server actor import trace_opd 判定层 | 运行环境同一 Python env(verl→trace_opd import 先例待核;兜底:循环模块置于 verl 侧、from trace_opd.steering import) |
| sglang 升级耦合 | 只用自有 patch 面与公开 engine 调用,不 patch sglang 本体 |
| CD-006 pgs-v33 批次合流 | 本分支定期 merge pgs-v33;共享工作树不受本分支影响 |

**待核**(P0b 前):sglang 0.5.10 `token_ids_logprob` 请求参数可用性(P2 用,非阻塞);
server actor 内 handler 直调的最短路径(绕 HTTP 序列化层的接口点)。

## 7. 明确不做(与用户对齐过)

- 不深改 vLLM/sglang 推测解码框架(单序列假设,装不下双前缀);
- 不换全词表算术混合(保尾核的可审计性是机制身份);
- 不为吞吐降干预(θ/ε/η/w_min 不动——SSB 恰是教师增益最大处,bare=0 档 +31.1pp);
- 不在 HTTP 层做 bounded K(每周期几十 ms 固定成本 × 周期暴涨 = CD-006 已证的灾难)。
