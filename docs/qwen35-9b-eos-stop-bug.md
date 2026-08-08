# Qwen3.5-9B rollout 越界续写(DEGENERATE:format)诊断与修复

**日期**:2026-08-08 · **分支**:CD-010 · **状态**:根因定案,修复已落码(见 §6)
**数据源**:`TRACE-OPD-Q35/runs/q35_9b_formal.gpumem065.crash/rollouts/step_00001`(217 条,A800 实机分析)
**移交现场**:`docs/collab/cd010-cd006/experiment-feedback.md` 2026-08-08【★★★ P0】条目;
本地样本 `analysis/q35-9b/trajectories/`

## 0. 结论先行

**根因是引擎停止集缺 `<|im_end|>`,模型无辜**。Qwen3.5-9B 官方 repo(HF 与
ModelScope 均实测)**不带 generation_config.json**,config.json 只声明
`text_config.eos_token_id=248044`(`<|endoftext|>`);而 verl 语境以
`skip_tokenizer_init` 起 sglang,引擎拿不到 tokenizer 的 eos(`<|im_end|>`=248046),
请求侧又没显式传 `stop_token_ids` ⇒ 引擎停止集={248044}。模型每轮按训练分布
正确生成轮末 `<|im_end|>`,引擎不认、继续采样,模型顺势**续写伪多轮对话**
(伪造 user/assistant 轮),直到偶遇 `<|endoftext|>`(82.7% 的轮)或 2048 满额
(13.4%)。**全部 217 条轨迹、5265/5268 轮(99.9%)带越界尾巴;86.5% 的
response token 是垃圾**——DEGENERATE 145 条只是垃圾滚雪球到 parser 连续三次
取不到 action 的可见尖端。已报的截断率 68%、熵分布(中位 0.0039)、
gate_open_rate 30.3%、won 3.1% **全部是被垃圾 token 污染的假象,作废**。

修复 = 停止条件是**对话协议属性**,由唯一引擎通道 `EngineGenerate` 随每个请求
显式下发 `stop_token_ids`(单一真源 `turn_stop_token_ids`);对 Qwen3/4B/1.7B
既有线严格 no-op(实测其 generation_config 已含 im_end,集合并集不变)。

## 1. 现象与 token 级证据

样本轨迹 `…0694fc9d`(bare 臂,33 轮,respLen 25530,DEGENERATE:format):

| turn | len | acted | 末 token | 轮内首个 `<|im_end|>` 的位置 |
|---|---|---|---|---|
| 1 | 68 | True | 248044 EOT | offset 65(垃圾 2 token) |
| 4 | 1787 | True | 248044 EOT | offset **90**(垃圾 ~1700 token) |
| 15 | 1023 | False | 248044 EOT | offset 37(此后伪对话 5 轮,末个 action 块为空) |
| 19 | 2048 | False | 248068 `<think>` | offset 123(满额截断,无 EOT) |

- 每轮真实回复完整且连贯,以 `<|im_end|>` 正确收尾——**模型行为符合其
  chat template**(`chat_template.jinja` line 88/130:每轮闭合 `<|im_end|>\n`)。
- 越界段 decode 后是 `user`/`assistant`/`<think>` 复读:`<|im_start|>/<|im_end|>`
  是 special(decode 剥掉,只剩 role 词),`<think>`/`</think>`(248068/248069)
  是 special=False(可见)——与移交 txt 的"复读形态"逐字吻合。
- 伪 assistant 轮开头 `assistant\n<think>\n\n</think>` 恰是 gen_prefix
  (enable_thinking=False)在上下文里的形态——模型在复读它每轮看到的结构。

全量(217 条,`/tmp/full_stats.py` 于 A800):

```
轮总数 5268;轮内含 im_end 未停:5265(99.9%)
轮终止:EOT(248044)82.7% | 2048 满额 13.4% | 其它 3.9%
token 账:总 response 4,126,042;真实(≤首个 im_end)418,315;垃圾 3,569,605(86.5%)
per-traj 垃圾占比:P25=0.84 P50=0.86 P75=0.89 max=0.93
DEGENERATE:bare 81/128(63.3%) vs steered 64/89(71.9%)——两臂同病,非 steering 所致
```

## 2. 根因链(五环,均已实证)

1. **模型资产**:`Qwen/Qwen3.5-9B` 无 generation_config.json(HF 官方
   `resolve/main/generation_config.json` 返回 Entry not found;本地 ModelScope
   拷贝同缺)。config.json 仅 `text_config.eos_token_id=248044`。
   tokenizer_config 的 `eos_token="<|im_end|>"`(id 248046)。
2. **sglang 0.5.10 推导**:`ModelConfig._get_hf_eos_token_id()`
   (`srt/configs/model_config.py:1110`)只从 hf config/generation_config 取
   ⇒ 停止集 {248044}。
3. **verl 语境**:`skip_tokenizer_init`(token-in-token-out)⇒
   `Req.tokenizer=None`,`_check_token_based_finish`
   (`srt/managers/schedule_batch.py`)的 `tokenizer.eos_token_id` 兜底分支死路。
4. **请求侧**:`run_episode`/`EngineGenerate` 的 sampling_params 只有
   temperature/max_tokens,无 stop(`src/trace_opd/core/episode.py:175-181`)。
5. **模板**:每轮闭合=`<|im_end|>`,模型正确生成 248046 ⇒ 不在停止集 ⇒ 续写。

**为什么既有线三个月没暴露**:Qwen3-4B/1.7B 的 generation_config 自带
`eos_token_id=[151645,151643]`(含 im_end),掩盖了"依赖模型资产提供完整
EOS 列表"的隐式假设;9B 是第一个不满足该假设的模型。
**为什么门 1-4 探针全绿**:探针对普通 `launch_server` 起服跑,tokenizer 在场,
check_finished 的 tokenizer.eos 兜底会停在 im_end——**探针语境与 verl
skip_tokenizer_init 语境的停止行为不同构**(教训:探针必须与生产语境同构,
已在 sgl_probe_q35.py 新增 stop stage 修补)。

## 2.1 系列适用边界(2026-08-08 A800 全系实测;用户裁定记录为系列级规则)

| 模型 | generation_config.json | config_eos | tok_eos | 修复前是否踩坑 |
|---|---|---|---|---|
| **Qwen3.5-4B** | **缺失** | 248044 | im_end(248046) | **会踩,与 9B 完全同病** |
| **Qwen3.5-9B** | **缺失** | 248044 | im_end(248046) | 已实证踩坑 |
| Qwen3.6-27B | 有,[248046, 248044] | 248044 | im_end(248046) | 不踩(引擎集含 im_end) |
| Qwen3.6-35B-A3B | 有,[248046, 248044] | 248044 | im_end(248046) | 不踩(researcher 普通起服双保险) |
| Qwen3-4B-2507 / 1.7B | 有,[151645, 151643] | — | im_end(151645) | 不踩 |

坑恰好圈住 **Qwen3.5 系全体**(官方唯一不发 generation_config 的系列,
3.6 又带回)。三点推论:

1. **Qwen3.5-4B 未来上线零额外动作**:§5 的代码修复是模型无关通防
   (`turn_stop_token_ids` 从 tokenizer 推导,任何经 `EngineGenerate` 的
   rollout 自动带 stop)——只要跑在含 b2f62e1 的代码上即安全;per-model
   剩余动作只有首跑前的 `--stage stop` 探针验收 + 冒烟轮末体检。
2. **修复只覆盖本仓库训练 rollout 通道**:在 Qwen3.5 系权重上任何绕过
   tokenizer 的 token-in-token-out 部署(独立评测服务、其它推理框架)
   资产坑仍在,须各自显式配 stop;普通起服(tokenizer 在场)有 sglang
   兜底,风险低。
3. **新模型接入 checklist 固化**(任何家族):查 generation_config.json
   是否存在 + eos 三处(config/tokenizer/generation_config)是否一致;
   不一致或缺失即预期引擎停止集不完整,首跑必须过 stop 探针。

## 3. 对已报读数的重定性(全废)

| 读数 | 原判 | 重定性 |
|---|---|---|
| 截断率 68%/73% | 建议提 response_length | 预算被 ~7× 垃圾吃掉;38912 改动无必要,修复后回 32768 观察 |
| 熵四分位(P95=1.0586,中位 0.0039) | 熵门标定源 | 86.5% 观测位是伪对话复读位;**标定作废,修复后重 dump 重标** |
| gate_open_rate 中位 30.3% 越界 | 需调门参 | 垃圾位污染,修复前不裁定任何门参 |
| won 3.1%/2.2%、val(step_00000_val) | 9B 基线 | 上下文被垃圾污染,作废重测 |
| DEGENERATE 145/217 | 格式冷启动问题? | 145 条只是尖端;修复后再看 raw 模型真实格式失败率,再议 SFT 冷启动 |

## 4. 次生放大器(记录,不在本轮修)

- `protocol.py:90` parser 取 **LAST** action 块:越界伪对话里最后一个块常是
  空/破碎,把"首个 action 有效"的轮判成 acted=False,加速 DEGENERATE。
  修复停止后单轮只会有一个块,LAST/FIRST 无差异——**不动**(三线口径一致优先)。
- `after_generation`(template_delta)对末 token=EOT(∉stop_ids)走全量 glue:
  流结构仍正确(轮间 im_end 由 glue 提供),但垃圾+EOT 已入流。修复后引擎停于
  im_end(sglang `finished_len=matched_pos+1`,停止 token 保留在输出),
  `gen_ids[-1]∈stop_ids` 走 `glue_full[1:]` 去重分支——设计本就等着这个形态。

## 5. 修复(已落码,commit 见 git)

1. **`src/trace_opd/core/template_delta.py`**:新增模块函数
   `turn_stop_token_ids(tokenizer)`(tokenizer eos + `<|im_end|>`,去重排序;
   `TemplateDelta.stop_ids` 改用同源)。
2. **`src/trace_opd/train/backends.py` `EngineGenerate`**:构造时算好停止集,
   `__call__` 对每个请求 `setdefault("stop_token_ids", …)`——这是所有引擎生成
   (student bare/steered draft、researcher-engine、selfres、训中 val)的唯一
   通道,单点覆盖;调用方显式传入优先。
3. **单测** `tests/test_engine_stop_tokens.py`(5 例):注入/不覆盖显式值/
   Qwen3 去重 no-op/无 im_end 回退 eos/双 id 排序。全套 391 passed。
4. **真 tokenizer 实测**(A800,transformers 5.11.0):
   Qwen3.5-9B → `[248046]`;Qwen3-4B-2507 与 1.7B → `[151645]`
   ⊆ 其 generation_config `[151645,151643]` ⇒ 既有线行为逐字节不变。

**不做资产补丁**(不给模型目录自造 generation_config.json):官方无此文件,
自造=资产漂移,重下模型即复发;代码层显式下发才是不依赖资产的根修。

## 6. 验收判据(CD-006 执行)

1. **探针**(30 秒,重启训练前):
   `--skip-tokenizer-init` 起服 →
   `python scripts/sgl_probe_q35.py --stage stop --model-path $Q35_MODEL`。
   A 组复现越界(证语境同构),B 组硬门:末 token=248046、中段无
   im_start/im_end、im_end 恰一次 → PASS。
2. **冒烟后轮末体检**(新增入门 5 六查):对新 dump 跑
   逐轮统计——`轮内首个 im_end 位置 == 轮长-1` 的轮占比应为 100%
   (满额截断轮除外);`turn 内 248045 出现次数=0`。
3. 熵 dump 重标、val 基线重测随冒烟一并重做。

## 7. 处置

- crash run 与 **正在跑的 `q35_9b_formal`**(2026-08-08 09:40 起,写本文时仍在
  step_00000_val)数据同废——**建议立即停**(尚未进 step1,重启成本最低);
  由用户/CD-006 裁定执行。
- 同步链:本树 commit → bundle → A800(注意 CD-006 未提交的 `++` hydra 修复
  与 38912 配置改动在服务器侧,按 [[sync-overwrites-prod-patches]] 先 sha 对拍)。
