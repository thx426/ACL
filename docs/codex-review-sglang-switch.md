# codex 深度审阅:SGLang 切换(gpt-5.6-sol,effort=max)

审阅对象:分支 pgs-sglang 提交 `36828c6`。2026-08-05,CD-010 发起;耗时 ~23min,
831,532 tokens。**判定 NO-GO(2×P0),全部发现已于同日修复(commit 见 git log,
修复对照见文末注记)。** 原始会话:scratchpad/sglang_review_out.md(会话目录)。

---

# 审阅结论

**NO-GO：提交 `36828c6` 目前不能作为安全的 SGLang 切换版本进入正式训练。**

发现两个发布阻断级问题：

1. **P0：SPS verify 的 `logprob_start_len` 行语义与 SGLang 0.5.8 源码相反。**正常 tail 请求会直接失败，当前实现并未获得预期的逐位置等价数据。
2. **P0：SGLang abort 返回 `"abort"`，消费端只识别 `"aborted"`。**在当前默认 sync trainer 路径会把基建中止静默归为普通失败轨迹。

此外还有一个会把 logprob 通道故障降级成普通空轨迹的 fail-soft 路径，以及若干测试、确定性和 launcher 配置缺口。

## 主要发现

- **P0 · [async_sglang_server.py:76](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:76)、[async_sglang_server.py:91](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:91)、[async_sglang_server.py:699](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:699) · 问题机理：**SGLang 0.5.8 对普通生成请求始终把返回窗口构造成 `[None] + input_token_logprobs[:-1]`，且 token 行覆盖 `origin_input_ids[start:]`；因此 `start>0` 时首项仍是 None，而不是本仓假设的真实预测行。官方源码明确如此：[scheduler_output_processor_mixin.py](https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/srt/managers/scheduler_output_processor_mixin.py)、[schedule_batch.py](https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/srt/managers/schedule_batch.py)。当前正常 SPS verify 会在 99–103 行主动抛错。· 修复建议：令 `start=max(0,L-T)`，所有普通请求无条件验证并丢弃首个 None，再追加 dummy；期望总行数应为 `L-start`，即 tail 情况下 `min(T,L)`。**

- **P0 · [async_sglang_server.py:723](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:723)、[backends.py:55](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/src/trace_opd/train/backends.py:55)、[hook.py:211](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/src/trace_opd/steering/hook.py:211)、[episode.py:181](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/src/trace_opd/core/episode.py:181) · 问题机理：**SGLang `FINISH_ABORT` 的实际类型是 `"abort"`，适配器原样透传；消费端只识别 `"aborted"`。官方 0.5.8 值域为 `"stop" | "length" | "abort"`：[schedule_batch.py](https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/srt/managers/schedule_batch.py)。当前默认 `trainer.v1.trainer_mode=sync` 使用普通 LLM client，不会替适配器重试/归一化；空输出随后被 episode 当普通 truncation，最终落成 `failed` 而非 `ENV_ERROR`，可静默入训。· 修复建议：在 SGLang adapter 边界严格复刻 vLLM：`abort→aborted`，`stop/length→completed`；同时保留完整 abort message/status_code/err_type，并为未知错误型 finish reason fail-loud。**

- **P1 · [async_sglang_server.py:726](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:726)、[async_sglang_server.py:740](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:740)、[async_sglang_server.py:773](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:773) · 问题机理：**任何 `output_token_logprobs` 缺失或长度不匹配都会清空合法的 `token_ids`，而不是报错；清空后又跳过 response top-K 校验。普通 stop/length 下，这会把数据通道损坏伪装成正常空生成。· 修复建议：只允许两个明确例外：`max_new_tokens==0` 的 prefill-only phantom，以及已经归一化的 abort；其他缺失/错长一律抛 `RuntimeError`，绝不能丢弃非空 token IDs。还应核对 tuple 中的 token ID 与 `output_ids[pos]` 一致。**

- **P1 · [async_sglang_server.py:115](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:115)、[async_sglang_server.py:124](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:124) · 问题机理：**prompt K 宽和行数的关键 fail-loud 防线使用 `assert`，在 `python -O` 下会被移除；空 payload、短行可能继续流入 hook。vLLM 参照实现已经使用 `RuntimeError`。· 修复建议：全部改为显式 `RuntimeError`，测试也不要期待 `AssertionError`。**

- **P2 · [async_sglang_server.py:654](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:654)、[vllm_async_server.py:609](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:609)、[rollout.py:170](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/config/rollout.py:170) · 问题机理：**`full_determinism` 和 rollout seed 在 SGLang 路径完全未实现。SGLang 0.5.8 参数名是 `sampling_seed`，直接透传 vLLM 的 `seed` 会在 `SamplingParams(**kwargs)` 处 TypeError；官方签名见 [sampling_params.py](https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/srt/sampling/sampling_params.py) 与构造路径 [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/srt/managers/tokenizer_manager.py)。· 修复建议：翻译 `seed→sampling_seed`，在 `full_determinism` 时注入 `replica_rank+config.seed`，并明确 SGLang kernel 是否能提供与 vLLM 同等级的确定性保证。**

- **P2 · [hook.py:423](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/src/trace_opd/steering/hook.py:423)、[async_sglang_server.py:680](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:680) · 问题机理：**hook 把 `top_k=0` 视为“禁用截断”，但 SGLang 0.5.8 只接受 `-1` 表示全词表，0 会被参数校验拒绝。字符串 `stop` 虽被 SamplingParams 接受，但当前 `skip_tokenizer_init=True` 下源码组合会令字符串停止检测依赖空 tokenizer，需 PoC 确认实际失败方式。· 修复建议：翻译 `top_k=0→-1`；字符串 stop 要么预先转 `stop_token_ids`，要么禁止与 skip-tokenizer 模式组合。**

- **P2 · [launch_train.py:147](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/scripts/launch_train.py:147)、[async_sglang_server.py:353](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:353) · 问题机理：**launcher 仍传 `max_num_batched_tokens` 和 `max_model_len`，但 SGLang server args 只映射了显存比例及 `max_num_seqs→max_running_requests`；前两者没有映射到 `chunked_prefill_size/max_prefill_tokens` 和 `context_length`。官方 0.5.8 参数见 [server_args.py](https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/srt/server_args.py)。当前 A800 的启发式 chunk size 很可能恰好也是 8192，但这是偶然，不是配置契约。· 修复建议：显式映射或在 SGLang 下拒绝无效配置，并修正 YAML 注释。**

- **P1 · [test_sglang_adapters.py:60](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/tests/test_sglang_adapters.py:60)、[test_sglang_adapters.py:84](/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs/tests/test_sglang_adapters.py:84) · 问题机理：**测试把“正 start 首行真实”写成正确 oracle，并明确要求真实的 0.5.8 payload 失败；因此 6 个测试全部通过反而掩盖了 P0。· 修复建议：用从 0.5.8 捕获的原始 payload fixture，按 token 位置而不是自造行形状断言。**

## 逐重点项结论

### 1. 行语义 off-by-one

**结论：发现问题，P0，已由 0.5.8 源码确认。**

设 prompt 为 `x₀…xₗ₋₁`，draft 长度 `n`，hook 请求 `T=n+1`：

| 路径 | 返回给 hook 的真实预测行 |
|---|---|
| vLLM tail=T | `x[L-n] … x[L-1]`，再加 dummy |
| 当前 SGLang 请求 | `s=L-n`；原始首行是 `None@x[s]`，只有 `x[s+1]…` 有效，适配器当场报错 |
| 正确 SGLang 请求 | `s=L-T=L-n-1`；丢弃 `None@x[s]` 后恰为 `x[L-n]…x[L-1]`，再加 dummy |

例：`L=10,n=3,T=4`。正确 `start=6`，原始行应为：

```text
None@x6,  p(x7|x<7),  p(x8|x<8),  p(x9|x<9)
```

丢首行、追加 dummy 后，与 vLLM 的 `x7,x8,x9,dummy` 完全同位。

短 prompt clamp 也应使用 `start=max(0,L-T)`；只要是普通请求，`start=0` 和 `start>0` 都丢首个 None。当前仅在 `T>=L+1` 时可能意外落入可用分支。

### 2. 参数翻译完备性

**结论：发现问题；当前 162/SPS 的中性采样键大部分已核无问题，但通用契约不完整。**

| 键 | SGLang 路径命运 | 结论 |
|---|---|---|
| `max_tokens` | 消化为 `max_new_tokens` 并 clamp | 已核无问题；同时出现两个键时另一个会漏入 SamplingParams，建议显式拒绝 |
| `logprobs=False/True/int K` | 消化为 `return_logprob`，int K 另设 `top_logprobs_num` | 当前 K=256 翻译正确；真实大 K 需 PoC |
| `prompt_logprobs=256` | 消化，开启 input LP 和 top-K | 翻译本身正确 |
| `prompt_logprobs_tail` | 消化为 `logprob_start_len` | **公式错误，见 P0** |
| `detokenize=False` | 消化为 no-op | 已核无问题；token-in/token-out 且本地 decode |
| `flat_logprobs=True` | 消化为 no-op | 已核无问题；只影响表示/性能 |
| `temperature=1` | 原样透传 | 已核无问题 |
| `top_p=1` | 原样透传 | 已核无问题 |
| `top_k=-1` | 原样透传，SGLang 内部转全词表 | 已核无问题 |
| `top_k=0` | 原样透传后被 SGLang 拒绝 | 发现兼容缺口；应转 `-1` |
| `repetition_penalty=1.0` | 原样透传 | 当前明确为 1.0，已核无问题；非 1 时实际采样 LP 口径需 PoC，hook 最好直接拒绝 |
| `stop_token_ids` | 原样透传 | 已核无问题 |
| 字符串 `stop` | SamplingParams 接受 | `skip_tokenizer_init=True` 下需 PoC；当前 TRACE episode 未发送 |
| `seed` | 未翻译 | SGLang 不接受，需转 `sampling_seed` |
| `sampling_seed` | 可原样透传 | 当前没有自动注入 |

当前正式训练参数 `temperature=1/top_p=1/top_k=-1/repetition_penalty=1` 不会因翻译而静默改变分布。

### 3. abort / stop_reason 值域

**结论：发现问题，P0。**

SGLang 值域是 `"stop" | "length" | "abort"`；vLLM 适配器输出 `"completed" | "aborted"`。当前 sync trainer 不经过能识别两种 abort 拼法的 FullyAsync 重试层，因此 `"abort"` 可以直达 episode，并被静默归为失败轨迹。

正常 `"stop"/"length"` 目前没有被消费端区分，故暂不影响数据；仍建议统一映射以锁定后端无关契约。

### 4. draft top-K

**结论：需实测确认。**

已核无问题部分：

- helper 强制 `rows == len(token_ids)`。
- helper 强制每行严格 K 宽，缺行/短行会 fail-loud。
- SGLang 普通 decode 源码按生成 token 同步追加 output ID、自身 LP 和 top-K；普通 EOS 提前结束应保持对齐。
- adapter 保留引擎返回顺序，没有自行重排。

仍需 PoC：

- `K=256` 是否在目标模型、目标 GPU 和该 wheel 上恒定返回 256 项。
- 实际 wire 顺序是否严格 best-first，尤其相同 logprob 的 tie。
- 每行是否 ID 唯一、logprob 有限且单调非增。
- sampled token 若位于 top-K，其 `output_token_logprobs` 是否与对应 top-K 项数值一致。
- EOS、stop token、MTP/spec decode 下输出裁剪是否仍严格逐行一致。

建议把单调性、唯一性、有限性和自身 token-ID 对齐加入运行时防线。

### 5. verify `max_tokens=1`

**结论：已核无问题；`max_new_tokens=0` 边界需 PoC。**

SGLang 0.5.8 把生成位置从 input logprob 数组中移除，输出 token 的 LP 单独放在 `output_token_logprobs`。本仓 `_verify_rows` 只消费 `prompt_ids/prompt_logprobs`，所以额外生成的一个 token不会污染 verify 行。

当前 `max_possible_tokens = max_model_len-L-1` 会比 vLLM 提前一 token clamp 到 0；对 verify 仍可做 prefill-only 打分，但须实测 phantom LP 防御。对普通 draft 会造成最多一 token 的容量差异，不过本 episode 保留了 64-token slack，当前配置下不可达。

### 6. 后端开关完备性

**结论：发现问题，但当前中性采样参数未见系统性分布偏置。**

已核无问题：

- `rollout.name` 已配置化。
- vLLM `max_logprobs` 只在 vLLM 后端设置，SGLang native API没有对应启动上限。
- `rollout_gpu_mem→mem_fraction_static`、`max_num_seqs→max_running_requests` 正确。
- `VLLM_RPC_TIMEOUT` 在 SGLang 下只是未消费环境变量，无副作用。
- `actor.calculate_entropy=True` 属训练 forward 指标，与 rollout 后端无关。
- `rollout.calculate_log_probs=False` 会被 SPS draft/verify 的请求级参数覆盖。

发现的缺口：

- `full_determinism/seed` 未实现。
- `max_num_batched_tokens` 是 SGLang 死参数。
- `max_model_len` 只在 adapter 做请求裁剪，没有传给 SGLang `context_length`。
- pinned 0.5.8 下代码默认强制 `attention_backend="fa3"`，而环境册按 flashinfer/triton 编写；目标 A800 wheel 的 FA3 可用性必须做启动 PoC。
- s2 与 s2_sgl 的 objective fingerprint 相同只说明研究配置相同，不能替代后端数据通道等价验证；修复 P0 前不可据此比较实验结果。

### 7. 测试盲区

**结论：发现问题。**

现有 6 个适配器测试通过，但它们只验证了自造数据与错误假设的自洽性。由于只读 sandbox 没有可用临时目录，标准 pytest runner 无法创建 temp；我以直接调用方式执行了本文件全部 6 个测试，结果 `6/6 passed`。

必须补充：

- 真实 0.5.8 `start=0/start>0/clamp=0` payload fixture。
- 逐 token 位置身份断言，而不只断言行数。
- 参数翻译到 `GenerateReqInput` 和 `SamplingParams` 的测试。
- `"abort"→"aborted"→ENV_ERROR/retry` 全链路。
- `max_tokens=1` 与 clamped 0。
- EOS/stop 提前结束、K=0/1/256、短行、缺行、重复 ID、NaN/Inf。
- sampled ID 与自身 LP tuple 的一致性。
- `seed→sampling_seed`、`top_k=0→-1`。
- `python -O` 下仍 fail-loud。
- prompt K 与 response K 同时出现且 K 不同的行为应显式拒绝。

### 8. 其他静默训练错误

**结论：发现问题。**

最主要的是 `output_token_logprobs` 错长后清空 token 的 fail-soft，以及 `assert` 在优化模式下消失。除此之外，没有发现 launcher backend 选择、EngineGenerate 字段搬运或正常 `max_tokens=1` 会主动错位数据的证据。

## 切换正确性总评

当前状态是：

- **SPS verify 主路径不可工作**：正 start 请求会因错误的 None 假设失败。
- **abort 契约存在真实静默污染窗口**。
- **logprob payload 损坏可能被降级为空轨迹**。
- draft top-K 的大 K、排序和目标硬件行为尚未由真实 0.5.8 证明。

因此建议在修复以下三项前禁止启动正式 s2_sgl 训练：

1. 修正 `start=L-T`，所有普通窗口统一丢首个 None。
2. 在 adapter 边界统一 finish reason。
3. 去除 LP fail-soft 和关键 `assert`。

## PoC 实测必须覆盖

1. **逐位置行语义**

   - 固定 `L=10,n=3,T=4,start=6`。
   - 原始返回必须是 `None@x6 + x7/x8/x9 三个真实行`。
   - adapter 输出必须是 `x7/x8/x9 + dummy`。
   - 与 vLLM tail=4 逐 ID、逐位置比较。

2. **边界窗口**

   - `start=0`。
   - `T=L`、`T>L`。
   - `n=1`。
   - 长教师前缀、短 draft，并分别测试冷/热 radix cache。

3. **draft K=256**

   - 每个生成 token 恰好 256 项。
   - ID 唯一、LP 有限、best-first。
   - sampled token 在 top-K 时 LP 一致。
   - EOS/stop 提前结束后行数仍等于 output IDs。
   - 记录显存峰值和 OOM 情况。

4. **verify 生成 token 隔离**

   - `max_tokens=1` 时 prompt 行与生成 token完全分离。
   - context 边界 clamp 到 0 时仍返回正确 prompt 行，phantom output 不进入消费端。

5. **abort 全链路**

   - 对在途请求执行 `PauseGenerationReqInput(mode="abort")`。
   - adapter 输出必须为 `"aborted"`。
   - episode 必须得到 `ENV_ERROR`，首次重试，二次失败则丢弃，不能形成 reward=0 轨迹。

6. **参数矩阵**

   - `temperature/top_p/top_k=-1/repetition_penalty=1`。
   - `top_k=0` 翻译。
   - `stop_token_ids` 与字符串 stop。
   - `seed/sampling_seed`，相同 seed 多次运行可重复。
   - `full_determinism` 开关行为。

7. **launcher/硬件**

   - 确认实际 `context_length`、`chunked_prefill_size`、`max_running_requests`。
   - A800 上验证 FA3；失败则显式钉 `flashinfer` 或 `triton`。
   - 验证第二次 tail 请求确实命中 radix cache，且命中前后 logprob 数值不漂移。

8. **vLLM/SGLang 对照**

   - 同一模型权重、同一 token prompt、全部中性采样参数。
   - 比较 sampled-token LP、top-256 ID/LP、verify 尾行位置。
   - 以同一 SGLang 请求冷/热重复误差作为数值容差基线，再判断跨引擎差异。
tokens used


---

## 修复注记(CD-010,2026-08-05,同日)

- **P0 行语义**:公式改 `logprob_start_len = max(0, L−T)`;extractor 统一
  "None 头必在且丢弃",行数=L−start;审阅的逐位置推演已固化为测试
  (`test_prompt_logprobs_tail_window_matches_vllm_contract`)。
- **P0 abort 值域**:新增 `_normalize_finish_reason`(abort→aborted、
  stop/length→completed、其余透传同 vLLM),adapter 边界归一化。
- **P1 fail-soft**:logprob 错长仅 abort / max_new_tokens==0 两个例外允许
  空账,其余 RuntimeError,永不清空非空 token_ids;另加逐位置 token id 对账。
- **P1 assert→RuntimeError**:extractor 全部防线换显式异常,`python -O`
  下测试通过。
- **P1 测试 oracle**:mock payload 重写为 0.5.8 真实形态(None 头恒在)+
  逐 token 位置身份断言。
- **P2**:`seed→sampling_seed` + full_determinism 注入、`top_k 0→-1`、
  双 max 键拒绝、`chunked_prefill_size`/`context_length` server args 映射、
  `rollout_attention_backend` 配置通道(A800 FA3 风险)、环境册门 3 断言
  修正(首行必须 null)。
- 未修(记录在案):字符串 `stop` × skip_tokenizer_init 组合(本仓只发
  stop_token_ids,不触达);MTP/spec-decode 路径(未启用)。
- PoC 实测清单已并入 `docs/sglang-migration-env.md` 门 2/3。
