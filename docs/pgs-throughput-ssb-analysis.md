# PGS 自进化在 SpreadsheetBench 上的吞吐分析

**日期**：2026-08-09 · **提出方**：CD-006 · **面向**：CD-010 深入探讨
**现场**：A800 八卡 · Qwen3.5-4B · `ssb_4b_selfres_a800`(指纹 271ef4e4cc02) · v3.3×v4 全量参数
**对照**：162 四卡 · 同模型同数据集的纯 GRPO 臂 `ssb_4b_grpo_162`(指纹 2c3cbff7c99e)

> 本文只陈述实测与分析,**不含任何已实施的机制改动**。所有改动建议均待与 CD-010 探讨后再定。
> 标注约定:**[实测]** 现场数据 · **[代码]** 本地代码可证 · **[推断]** 由前两者推出 · **[未验证]** 需要现场 profiler/cache 指标。

---

## 0. 结论先行

1. **SSB 上 steered 臂比 ALFWorld 慢约 30 倍,而 bare 只慢 12 倍** —— 额外劣化来自机制,不是任务本身。
2. **主因是「长段 draft → 长段 verify → 任一拒绝丢弃余段并重启」在长轨迹上的近平方级放大**,不是 CPU 空等、不是 LibreOffice、不是全局串行锁。
3. **一个已被推翻的判断**:开门率 35–46% **不能**解读为"熵门未标定"(详见 §3),CD-006 先前基于此的优先级排序作废。
4. **优化 steered 的收益有硬天花板 ≈1.23×**,因为 research 阶段(46.5 分钟)会立刻变成新的关键路径。
5. **MENTOR(ICLR 2026)的可迁移点是执行架构,不是它的混合分布**;且它自身生成墙钟也是 GRPO 的 3.04×,并非"免费"。
6. **有一个与吞吐无关但更紧急的正确性问题**:5400 秒超时后,残缺的 prompt 组仍被采样进训练(§6)。

---

## 1. 实测基线

### 1.1 两 benchmark 同机制对比(均 Qwen3.5-4B,v3.3×v4)

| | SpreadsheetBench | ALFWorld | 倍数 |
|---|---|---|---|
| bare 轨迹 respLen 中位 | 16390 | 682 | 24× |
| steered respLen 中位 | 4883 | 554 | 8.8× |
| `steer_n_candidates` | 2092～13572 | 156～2651 | ~5× |
| `steer_count`(干预) | 747 | 7 | 107× |
| `steer_reject_count` | 43 | 1 | 43× |
| bare 吞吐 | 8.8 条/分 | 107 条/分 | **12× 慢** |
| steered 吞吐 | 0.8 条/分 | 24 条/分 | **30× 慢** |
| steered/bare 比 | **11×** | **4.4×** | 额外劣化 2.5× |

[实测] 数据取自 `ssb_4b_selfres_a800/rollouts/step_00001` 与 `q35_4b_v33_selfres_162/rollouts/step_00020`。

### 1.2 单步时间线(SSB,step-1,每类 64 条)

```
0 ───────────── 62.9 ───────────── 90 ───────── 107 min
  191 条成功落盘      未完成 steered 长尾      训练约 17 min
                      被 5400s 超时取消

bare      0.0 → 7.3    research  1.5 → 48.0    steered  8.8 → 62.9
占比: 成功生成 58.8% / 未决长尾 25.3% / 训练 15.9%
```

[实测] **CD-006 先前报"整步约 80 分钟"是错的** —— 那是把最后一条成功落盘当成了步结束。真实为 107 分钟。

### 1.3 资源画像

[实测] GPU 利用率均值 **84%**(最低 69%,15 次采样**无一低于 30%**);生成阶段显存 73.1G/80G,训练阶段 12.7G/80G;8 个 sglang scheduler 各约 440% CPU。

**"GPU 忙"与"有效条数低"并不矛盾**:GPU 时间花在 (1) 长 student draft、(2) 长 teacher prompt-logprob extend、(3) 每位置 full-vocab logits + top-256、(4) **拒绝后注定被丢弃的 suffix**、(5) research/continuation 的并发生成。

---

## 2. 根因一:拒绝重启的近平方放大

### 2.1 代码事实

[代码] `SpsGenerate.__call__` 每轮让学生一次性 draft **当前 turn 的全部剩余长度**(`steering/hook.py:222`),整段只发一次 teacher verify(`hook.py:298`):

```python
draft_out = await self._draft(prompt_ids + executed,
    {"max_tokens": remaining, "logprobs": self._obs_topk, ...})
pt_rows = await self._verify_rows(..., gen_ids, n_positions=len(gen_ids))
```

[代码] 逐位 judge 命中拒绝立即 `break`,回外层 `while` 重新 draft+verify(`hook.py:419/425`)。

### 2.2 量化

设轨迹有 `A` 个 assistant turn、`R` 次拒绝、最终执行候选 `N`:

- 引擎周期数 `N_cycle = A + R`,实际 RPC 数 `≈ 2(A+R)`
  - SSB(A≈8, R=43):约 **51 周期 / 102 次 RPC**
  - ALFWorld(A≈8, R=1):约 **9 周期 / 18 次 RPC**
- 拒绝位置段内均匀时,每次拒绝平均浪费半段,重复 token-row 工作量:
  ```
  W ≈ N(1 + R/(2A))
  SSB:      1 + 43/16 = 3.69×
  ALFWorld: 1 +  1/16 = 1.06×
  ```
- 结构放大 `3.69/1.06 ≈ 3.47×`,再乘候选长度约 5× ⇒ **约 17×**,与实测 20.7×(成功窗口)/30.9×(含超时尾)同量级。

[推断] 当 `R` 随 `N` 线性增长而 `A` 受轮数上限约束时,`W = Θ(N²/A)` —— **不是"轨迹长 5 倍就慢 5 倍",而是近平方级**。SGLang radix cache 能减轻前缀重算,但**挽回不了已生成、已打分、随后被丢弃的 suffix**。

### 2.3 观测口径的坑

[代码] `steer_count` 只统计 `eta>0` 的最终执行位置(`steering/ledger.py:169`);`steer_n_candidates` 在逐位 judge 时递增,**拒绝后未 judge 的尾段已被 GPU 算出但不入账**。
⇒ **用这两个字段估 GPU 工作量会低估约 3.69×**。

[代码] `response_token_len` 含环境观察(以 `mask=0` 追加进同一 response stream,`core/episode.py:255`),**不是纯生成 token**。16390 vs 682 不能直接当"模型生成量"比较。

---

## 3. 根因二:段级 bypass 在长轨迹上失效 —— 兼一条重要更正

### 3.1 段级 bypass 的概率

[代码] 熵门的 bypass 是**段级**的:整段所有位置都低于门限才跳过(`hook.py:282`):

```python
if max(st.gate.base_w(entropy=h) for h in seg_entropy) < w_min:
    ... continue   # 整段跳过 teacher verify
```

跳过概率 `P = (1-q)^m`。SSB 候选 2092、约 8 turn,平均段长 `m ≈ 261`:

| q(非零干预率) | `0.65^261` / `0.95^261` |
|---|---|
| 35% | ≈ 1.5×10⁻⁴⁹ |
| **5%** | **≈ 1.5×10⁻⁶** |

⇒ **即使非零干预率降到 5%,整段跳过依然几乎不可能**。教师实际上为几乎所有候选位算了 top-256,而非只为 747 个"开门位"算。ALFWorld 段短(约 20–330)才碰巧能跳过 —— **同一份代码在两个 benchmark 上行为完全不同**。

### 3.2 ⚠ CD-006 先前判断的更正

CD-006 曾报"开门率 35–46% 远高于目标 5%,说明 `entropy_theta_init=1.28` 未在 SSB 标定",并据此把"标定熵门"列为第一优先。**该判断作废**,理由:

[代码] `entropy_target_rate = 0.05` 控制的是 `H ≥ Hq` 即 **`w=1` 的满强度位置比例**;而 `base_open` 的条件是 `H/Hq ≥ w_min = 0.05`,即 `H ≥ 0.05·Hq`(`steering/gate.py:165/188`)。

⇒ **5% 是「满强度干预」的目标,不是「非零干预」的目标。35–46% 的 `base_open` 未必异常。**

[对照] MENTOR 同样没有"只有 5% token 调教师"的性质:其 `w=min(1,H/γ_p)` 对几乎所有非零熵位置都 >0,**根本没有 hard bypass**(§5.2)。

**仍待 CD-010 判断的是**:当前 `w_min=0.05` 这个"非零干预下界"在 SSB 的长段场景下是否过低 —— 它直接决定段级 bypass 能否生效。这是机制参数,不是标定问题。

---

## 4. 根因三:宽 top-K 数据面

[代码] draft 请求 `logprobs = obs_topk = 256`(`hook.py:224`),verify 请求 `prompt_logprobs = 256`(`hook.py:459`);**`topk=64` 只在 Python 侧从两侧 256 宽观测窗构造候选集**(`hook.py:67`)。

⇒ **`topk=64` 不会把 GPU 输出、RPC payload、Python 解析宽度降为 64;两侧仍是 256。**

紧凑二进制下界 `2 × 256 × 8 = 4096 bytes/token`,按 `W/N ≈ 3.69`:

| 候选数 N | 双侧 top-K 数字下界 |
|---|---|
| 2092 | ≈ 30 MiB / 条 |
| 13572 | ≈ 196 MiB / 条 |

未计 Python list/dict 对象与 HTTP/Ray 开销。[推断] 这解释了 8 个 scheduler 各约 440% CPU —— **控制面并不闲,在处理海量宽 logprob 行**;但它不是主墙钟瓶颈(GPU 仍 84%)。

[代码] verify 返回后,hook 在**没有 `await` 的 Python 循环**中逐位置构造两个 256 项 map、NumPy 数组、核与 ledger(`hook.py:306`),会短暂阻塞同 worker 内其余 23 个副本的 event loop。[未验证] 无 py-spy 证据,标为次级放大器。

---

## 5. MENTOR(ICLR 2026)对比

**论文**:*Selective Expert Guidance For Effective and Diverse Exploration in RL of LLMs*,`related_work/papers/2510.04140_MENTOR_ICLR2026.pdf`;代码 `related_work/code/MENTOR/`。

### 5.1 它的做法

[代码] 把 **学生当 draft、专家当 target** 塞进 vLLM 推测解码,固定 `K=4`,自定义 `SpecGRPOSampler`(`vllm_rollout_spmd.py:90`):

```python
LLM(model=config.teacher_model_path,
    speculative_config={"model": model_path,
                        "num_speculative_tokens": 4,
                        "acceptance_method": "SpecGRPOSampler"})
```

[代码] GPU 内的混合(`vllm/model_executor/layers/rejection_sampler.py:189`):

```python
draft_entropy  = -Σ p_s log p_s
scaling_factor = clamp(draft_entropy / entropy_threshold, max=1.0)
target_probs   = draft_probs + (target_probs - draft_probs) * scaling_factor
```

即**全词表算术凸组合** `q = (1-w)π + w·p_T`,拒绝后从 `(q-π)₊` 归一化重采,**在同一引擎请求内继续下一 block,不退出到 Python**。

[代码] 阈值 `entropy_threshold = max(q95, 1.5)`(**首步 999**),`q95` 由 actor 每步从 on-policy 行的 token 熵算 `torch.quantile`(`dp_actor.py:146/275`)。注:各 rank 分位数再取 `np.mean`,**不严格等于全局 P95**(`metrics.py:23`)。

### 5.2 逐项对比

| 维度 | MENTOR | PGS 现状 | 判断 |
|---|---|---|---|
| 教师来源 | **外部更强模型**,同 prompt | 同模型 + 特权 doc | **本质不同,不可迁移** |
| 熵强度 | `w=min(1,H_full/γ_p)` | `w=min(1,H_top16/Hq)` | 同思想,信号口径不同 |
| hard bypass | **无** | 整段全 `w<w_min` 才跳过 | PGS 有,但长段下失效(§3) |
| 混合分布 | 全词表**算术**混合 | S 内 clipped log-ratio **几何**倾斜,S 外严格 = π | **不同分布族** |
| 密度比界 | 无统一上界 | `e^{-2ηc} ≤ b/π ≤ e^{2ηc}` | **PGS 更稳、可审计** |
| 执行层 | vLLM sampler 内(GPU) | Python `SpsGenerate` | **MENTOR 主要工程优势** |
| 粒度 | 固定 K=4 | 当前 turn 全部剩余 | PGS 浪费长 suffix |
| 拒绝处理 | 同请求内续写 | `break → 新 draft RPC + 新 verify RPC` | 同因果,成本不同 |
| 教师上下文 | 与 draft **同一 token 序列** | teacher 多一段 doc prefix,需第二条 KV 链 | **stock 实现不能直接套用** |
| 训练 advantage | on-policy mean/std;mixed `α·[R−mean]₊` + cosine 衰减 | bare strict LOO;steered positive excess × α | **v3.3 已吸收其核心训练设计** |
| 行为校正 | **无**(`ratio` 用当前 actor 重算 old_log_probs) | 支持 `π/b`、PDIS(当前 run 配 off) | 当前口径相同,PGS 可审计 |
| 环境 | 单轮数学文本(828→1751 tok) | 20 步工具环境,观察穿插 | **MENTOR 未解决工具状态** |

### 5.3 它自身的效率数据 —— 并不"免费"

[论文] Qwen2.5-7B 短数学轨迹(PDF p.21 Table 6):

```
GRPO     Gen 133s   总 244s   3474 tok/s
MENTOR   Gen 404s   总 627s   2860 tok/s
         生成 3.04×    总 2.57×    吞吐仍低 17.7%
```

[推断] MENTOR 证明了"引擎内 speculative 能把双模型干预控制在可用范围",**未证明它适合 16k–40k、20 轮工具环境**;它也没有 researcher 行。

### 5.4 其发布代码的三处缺陷(不可照抄)

1. [代码] **当前 actor 权重被完整 gather 到 CPU 后传入子进程,唯一用途是 `del model_state`**(`vllm_rollout_spmd.py:108`);draft 实际由静态 `model_path` 加载(`fsdp_workers.py:341`)。[推断] mixed rollout 很可能用的是 `π_init` 而非当前 policy,**而阈值却来自当前 actor,训练越久两者越错位**。
2. [代码] prompt 经历 decode→retokenize(`vllm_rollout_spmd.py:133`)。
3. [代码] 若 `VLLM_USE_FLASHINFER_SAMPLER=1`,直接把**未混合的** teacher 分布交给标准 sampler,绕过混合,且无 fail-loud(`rejection_sampler.py:124`)。

### 5.5 关于"无偏"的准确表述

[论文] 附录 A.2 只证明 speculative sampler 输出**精确服从 `π_mix`**。
[代码] 训练侧对所有行统一用当前 actor 重算 `old_log_probs`(`ray_trainer.py:669`),**没有 `π_old/π_mix` 行为校正**。

⇒ 与 PGS 当前 `w_beh_mode: off` 是同一口径。**两者都是有意采用的偏置 surrogate,均不应声称"无偏 on-policy 梯度"。**

同理,MENTOR 附录 A.3 的"错误教师自动过滤"只在**严格二元 reward** 下成立;其正式实验用 9:1 outcome/format 复合 reward,已不满足。**PGS 的 asym advantage 沿用同一思想,故也应表述为"抑制低于 bare baseline 的 guided 样本",而非"保证过滤错误指导"。**

### 5.6 PGS 做对而 MENTOR 没做的

1. PGS 的 policy 权重每步由 rollout engine 真实同步;MENTOR 发布代码丢弃了 live actor 权重。
2. PGS 用原始 token IDs,并对 student/teacher prefix 漂移 **fail-loud**(`hook.py:209`)。
3. PGS 有全词表密度比界、尾部质量恒等、可选轨迹预算。
4. PGS 保存逐 token 行为账本 + 随机探针(ε),即使 `w_beh=off` 也能审计教师改了什么;MENTOR mixed 行只剩一个 `is_expert` bit。
5. PGS 的 residual 是 lazy 且只在 S 上;MENTOR 对所有候选(含最终接受的)提前构造全词表 residual。
6. PGS 显式处理多轮、工具观察、replacement stop token;MENTOR 的正确性证明只覆盖普通自回归序列。
7. PGS 的 strict LOO baseline 不含样本自身,比普通 mean/std 干净。

---

## 6. 一个与吞吐无关但更紧急的正确性问题

[代码] `_run_prompt` 用 `asyncio.wait_for(asyncio.gather(*tasks), timeout=_timeout_s)`,超时后 pending task 取消、prompt 标 `failure`(`verl/trainer/ppo/v1/agent_loop_tq.py:124`);**但已完成的 session 早已独立写入 TransferQueue**(`agent_loop_tq.py:144`)。ReplayBuffer 把 `finished_keys | failure_keys` 都视为可采样终态(`replay_buffer.py:164`),并留有未实现的 `# TODO: failure samples dropping and refilling`(`replay_buffer.py:243`)。

⇒ **step_00001 的 191/192 不是少了一个归档文件,而是训练数据真实缺一条。**

[实测+推断] 本次缺任务 `42515` 的 1 条 steered。在 `seq-mean-token-mean` 下:

| 对象 | 权重变化 |
|---|---|
| 其他轨迹 | `1/192 → 1/191`,**+0.52%** |
| 任务 42515 executor 组 | `16/192 → 15/191`,**−5.8%** |
| 该任务 **steered 臂** | `8/192 → 7/191`,**−12.0%** |

**关键**:丢失的是**最慢的样本**,系统性关联长轨迹/难任务/失败结果 —— 属**非随机右删失**,不是随机缺失。

[代码] `objective_fingerprint()`(`common/config.py:1075`)**不含** `agent_task_timeout_s` / `ppo_max_token_len_per_gpu` / `log_prob_max_token_len_per_gpu`。
⇒ 把 timeout 提高到"不再截断正常任务"是**纯工程修复**;**但当 timeout 实际命中并允许残组入训时,它已在操作上改变了采样分布** —— 若有意用它截断难任务,则应作为实验语义记录,甚至纳入 fingerprint。

**建议的正确处置**(待 CD-010 定):`failure` prompt 不可采样 → 清理其已落地的部分轨迹 → 同一冻结 policy version 下完整重试或整步 fail-loud → 用**无进展 watchdog**(引擎 RPC 保险丝 + 无 session 进展超时 + 最终硬墙)替代单一总墙钟。

---

## 7. 优化的天花板

```
当前   bare 7.3 | research 48.0 | steered 54.1  → 可见段 62.9 分钟
若 steered 降到 39.2 分钟以下,research 48.0 变成关键路径
理论最大先省 62.9 − 48.0 = 14.9 分钟  ⇒ 约 1.23×
```

**即使 steered 提速 2–5 倍,整步也只能改善约 1.23 倍。** 之后必须转向 research 阶段:每 research 行最多 `n_exp=8 × k=2 = 16` 实验 episode(各带 narration)+ `5 × 4 = 20` 次续写 + 最多 20 个顺序 LLM turn;64 行合计 **2304 episode + 1024 次 narration LLM 调用**。

训练侧打包(`ppo_max_token_len_per_gpu: 8192`,SP2 ⇒ 有效 16384;最长完整序列 48358)只占真实整步的 **15.9%**,即使提速 3× 也只带来整步约 1.12×,**不是主瓶颈**。

---

## 8. 待与 CD-010 探讨的问题

### 8.1 方向性

1. **是否值得做引擎内改造**(同权重双前缀 + 持久双 KV + bounded K 的 exact speculative loop)?预估 2–4 周、高风险,但这是唯一能切断近平方放大的路径。考虑到天花板只有 1.23×(在 research 未优化前),投入产出比需要共同判断。
2. **research 阶段是否该同步优化**?它将立刻成为关键路径。可能方向:narration 是否必需、continuation 能否批量化、researcher 会话的顺序 turn 能否并行。**这些都会改变机制语义,须用户裁定。**
3. **`w_min = 0.05` 在 SSB 长段场景下是否过低**?它决定段级 bypass 能否生效(§3.1)。这是机制参数。

### 8.2 明确不该做的

- **不能只把 Python 层 draft 的 `max_tokens` 改成 4**:对 16k 轨迹会制造 4000+ 次 RPC,**比现在更慢**。bounded K 只有与"引擎内循环 + 持久双 KV"一起迁移才有意义。
- **不能照搬 MENTOR 的全词表算术混合**:会丢失尾部质量恒等、密度比界与预算证明,且 GPU 上需持有数百 MiB fp32 全词表张量。
- **不能直接复用 stock vLLM `SpecDecodeWorker`**:它只接收一套 `seq_group_metadata`,**不支持 target-only 的 doc prefix**。doc 塞进公共 prompt 会同时污染 draft,使 proposal 不再是 `π(a|h)`;不塞则 target 不是 `p_T(a|h⊕doc)`。

### 8.3 低成本先行项(不改行为,建议优先)

**增加真实工作量计数**:radix matched tokens、teacher prefill/decode tokens、**draft-discarded tokens**、block accept prefix、序列化字节。工作量 1–3 天、风险极低、不改任何行为。

价值:**当前无法区分"cache miss"与"拒绝丢弃 suffix"两种浪费**(§2.2 的 `steer_n_candidates` 不含丢弃段),而它们对应完全不同的优化方案。在拿到这些数字之前,§8.1 的收益估算都还是推断。

---

## 附:分析方法与可复现性

- 时间线由 rollout json 的文件 mtime 按 `sample_role` / `steering.steer_arm` 分组得出。
  **行类判别必须用 `sample_role`**:research 行的 `steering` 是空 dict,`steer_arm` 取出为 `None` 而非 `0`。
- 读训练日志须先规整:`tr '\r' '\n' | sed -E 's/\x1b\[[0-9;]*m//g; s/^\([^)]*\) //'`,否则 `grep "^step:"` 恒为 0 行。
- 独立分析由 codex(gpt-5.6-sol)执行三轮:① 本地代码根因定位 ② 注入现场证据后修订 ③ 独立研读 MENTOR。
  **两轮尝试 SSH 现场均被其执行沙箱以 `Operation not permitted` 阻断**,故所有远端相关结论均为"CD-006 提供数据 + 本地代码"推理,报告中已标 [未验证]。其运行的 steering/selfres/SGLang 相关单测:36 passed / 2 skipped,未修改任何文件、未触碰运行中的实验。
