# Qwen3.5-9B 上线 v3.3 机制:最小环境变动试验方案(CD-010)

> 2026-08-07 调研起草。目标:在**尽量不动现有生产环境**的前提下,让 v3.3
> 方案机制(asym advantage + 熵门 + PGS 采样端干预)在 Qwen3.5-9B 上可跑。
> 状态:调研完成、方案待用户裁定;实施与冒烟在裁定后执行。

## 0. 结论先行

**推荐路线**:A800 新建平行环境 `env_sglang_train_q35`(现有 `env_sglang_train`
一个字节不动),升级面钉死为 **sglang 0.5.8 → 0.5.10** 一族(sglang-kernel
0.4.1 / flashinfer 0.6.7.post2 / quack-kernels),**torch 2.9.1 与 transformers
5.11.0 均不动**——0.5.10 是唯一 torch pin 仍为 2.9.1 的 Qwen3.5 支持版本
(0.5.11 起强制 torch 2.11.0,0.5.16 更要求 flashinfer[cu13],驱动 525 出局)。
训练面**零升级**:transformers 5.11.0 已内置 qwen3_5 全套实现(纯 torch
fallback 在位);且 vendored verl(0.9.0.dev 基点)**已自带 Qwen3.5 完整训练
支持**——Ulysses SP、packed varlen(含 DeltaNet 卷积跨分片前缀交换)、fused
CE 双后端全在 `verl/models/transformers/qwen3_5.py`,monkey_patch 注册齐全;
sglang 适配层是版本感知的(覆盖 0.5.8→0.5.12 命名/接口双轨)。**预期代码
零改动**,全部风险收敛到 PoC 冒烟门的实测验证。

## 1. 模型事实(全部实测/官方来源)

| 项 | 值 | 来源 |
|---|---|---|
| HF 名 | `Qwen/Qwen3.5-9B`(Instruct 语义)/ `Qwen/Qwen3.5-9B-Base` | hf-mirror resolve 实测(`-Instruct`/`-Thinking` 后缀不存在) |
| model_type | `qwen3_5` | config.json(已存 scratchpad `qwen35_9b_config.json`) |
| architectures | `Qwen3_5ForConditionalGeneration`(**VLM 形态**,native multimodal) | config.json |
| text 结构 | 32 层 = **24 linear_attention(Gated DeltaNet)+ 8 full_attention**(`full_attention_interval: 4`);hidden 4096;16 Q heads / 4 KV heads / head_dim 256 | config.json |
| linear attn 参数 | key_head_dim 128 × 16 heads;value_head_dim 128 × 32 heads;conv kernel 4 | config.json |
| 词表 | **248320**(Qwen3 系 151k 的 1.64×;eos=248044) | config.json |
| 上下文 | max_position_embeddings **262144** | config.json |
| vision tower | depth 27 / hidden 1152(≈0.6B,DeepStack ViT) | config.json |
| 权重 | bf16 **19.3 GB**(775 tensors;`model.visual.*` 前缀为视觉部分) | model.safetensors.index.json |
| tokenizer | `Qwen2Tokenizer` 类不变(词表文件变大),chat_template 在 tokenizer_config | tokenizer_config.json |
| 发布 | 系列 2026-02-16;小模型(0.8B/2B/4B/9B)2026-03-02 开源 | qwen.ai 博客/官方 X |

## 2. 支持矩阵与版本地图(升级决策依据)

### 2.1 A800 现状(2026-08-07 实测,`env_pgs_sglang_a800.sh` 入口)

- 驱动 **525.60.13**(硬边界:cu12x wheel 可用(minor-version compat 已有实测先例),**cu13 wheel 不可用**)
- sglang **0.5.8** / torch **2.9.1** / transformers **5.11.0** / flashinfer **0.6.1**
- sglang 0.5.8 模型注册表:有 `qwen3_next.py`(hybrid 基建已存在),**无 `qwen3_5.py`** ⇒ rollout 面必须升级
- transformers 5.11.0:**`qwen3_5` 全套已内置**(`Qwen3_5ForCausalLM/ForConditionalGeneration/Config/TextModel/VisionModel`)⇒ 训练面零升级
- 磁盘 /fs/fast 余 40T;八卡当前空闲

### 2.2 transformers 5.11.0 的 qwen3_5 实现要点(源码实读)

- DeltaNet 核心算子有**纯 torch fallback**:`causal_conv1d_update or torch_causal_conv1d_update`、
  `chunk_gated_delta_rule or torch_chunk_gated_delta_rule`(modeling_qwen3_5.py:419-421)——
  **不装 fla / causal-conv1d 也能训**;装了(`fla` 纯 triton 免编译)是性能优化,不是可行性前提。
- 实现原生支持 packed varlen(FA 路径用 cu_seqlens,modeling_qwen3_5.py:940 附近;
  chunked FLA kernel 单 `cu_seqlens` 参数)——为后续 rmpad/SP 优化留了通道。

### 2.3 sglang 版本地图(PyPI requires_dist 实查)

| sglang | torch pin | flashinfer | transformers pin | kernel 包 | qwen3_5 支持 | 判定 |
|---|---|---|---|---|---|---|
| 0.5.8(现) | 2.9.1 | 0.6.1 | — | sgl-kernel 0.3.21 | ✗ | 现状 |
| 0.5.9 | 2.9.1 | — | — | — | 有(带 bug #19393) | 跳过 |
| **0.5.10**(2026-04-05) | **2.9.1** ✓ | 0.6.7.post2(+cubin) | 5.3.0(可放宽) | **sglang-kernel 0.4.1** + quack-kernels | **✓ 含 qwen3_5.py**(GitHub tag 实证) | **推荐** |
| 0.5.11 | **2.11.0** ✗ | 0.6.8.post1 | 5.6.0 | — | ✓ | torch 连锁,弃 |
| 0.5.16(最新) | 2.11.0 ✗ | **[cu13]** ✗✗ | 5.12.1 | — | ✓ | 驱动 525 出局 |

- sglang-kernel 0.4.1 wheel:单一 `manylinux2014_x86_64 abi3`(352MB,无 per-cu 变体),
  cu12x 构建,sm80 预期覆盖——**PoC 门 1 首验项**。
- transformers pin 冲突处理:0.5.10 pin `transformers==5.3.0`,保持 5.11.0 安装时
  忽略该 pin(现环境本就运行在"transformers 新于 sglang pin"状态,先例成立)。
- Qwen3.5 追踪 issue #20069(3-07 建,已关):logprobs/memory-saver/权重更新相关**零条目**;
  a_log fp32 精度修复(PR #19961,3-18 合入)在 0.5.10 内;mamba 内存泄漏 issue #20010
  的场景是 **FP8 KV + 27B-FP8 + 高并发**(与我们 bf16/9B 不重合),且以 inactive 关闭
  ——降级为冒烟观察项(rollout 显存跨步回落监控),不否决版本。
- hybrid 专项:`--mamba-radix-cache-strategy` V1(默认,省显存)/V2(需 FLA backend);
  已知 issue 集中在 PD 分离/HiCache/PP——**均不在我们的用法里**(单机 colocated、无 PD、无 PP)。

### 2.4 vendored verl 现状(源码实读,2026-08-07)

- 版本基点 **0.9.0.dev**(晚于上游 v0.8.0/Qwen3.5 完整支持版),关键资产已在树上:
  - `verl/models/transformers/qwen3_5.py`:完整 per-arch patch——Ulysses SP
    (`ulysses_pad_and_slice_inputs` + `_ConvPrefixExchange` DeltaNet 卷积跨分片
    前缀交换)、packed varlen(cu_seqlens,fla `prepare_sequence_ids` 优先 + 纯
    torch fallback `_packed_causal_conv1d_fallback`)、fused CE 双后端
    (`forward_with_torch_backend`/`forward_with_triton_backend`);
  - `monkey_patch.py:270,497`:`model_type in ["qwen3_5","qwen3_5_moe"]` 注册齐
    (DecoderLayer/GatedDeltaNet/TextModel/VisionModel 全套)⇒ **rmpad+SP+fused
    kernels 对 qwen3_5 是现成能力**;
  - `sglang_rollout.py:72-95,121`:kernel 包名双轨(sgl-kernel / sglang_kernel,
    注释明示 "For sglang 0.5.12 and sglang_kernel > 0.4.2")+ packaging.version
    分支——适配层为 0.5.8→0.5.12 区间写成,**0.5.10 落在覆盖区间内**;
  - setup.py 钉 `sglang[srt,openai]==0.5.8` 仅是快照 pin,环境手动管理不受约束。
- 上游时间线佐证:verl v0.7.1(2026-03-16)起支持 Qwen3.5(sglang 0.5.9),
  v0.8.0 完整化(sglang 0.5.12)。

## 3. 方案(待裁定后实施)

### 3.1 环境形态:平行新环境,现有环境零改动

- 新建 `env_sglang_train_q35`(克隆现 conda/venv 底 → 升级四件:sglang 0.5.10、
  sglang-kernel 0.4.1、flashinfer_python/cubin 0.6.7.post2、quack-kernels);
  新入口脚本 `env_pgs_sglang_a800_q35.sh`(平行复制,改环境名与 TRACE_MODEL_PATH)。
- **不动**:`env_sglang_train`(v3.3 4B 主线随时可回)、env_vllm、162 全部环境
  (researcher 35B 服务照旧走隧道,9B 实验不上 162)。
- 模型下载:`Qwen/Qwen3.5-9B` 19.3GB → A800 模型资产目录(hf-mirror,磁盘余量充足)。
- fla / causal-conv1d:**SP=2 主档必须装 `flash-linear-attention`**——verl 的
  qwen3_5 patch 在 Ulysses SP>1 时硬性要求 FLA 的 cp_context 支持,缺失直接
  raise(verl/models/transformers/qwen3_5.py:169-173,agent 实测);而 transformers
  自身缺 FLA 只是静默降级到纯 torch。fla 是纯 triton 包(pip 装无 CUDA 编译),
  验收 = `from fla.ops.utils import prepare_sequence_ids; from
  fla.ops.gated_delta_rule import chunk_gated_delta_rule` 可 import。
  causal-conv1d 仍可缺(conv kernel=4 的小卷积有 torch fallback,首版不装,
  避免 CUDA 编译)。SP=1 fallback 档两者皆免。

### 3.2 verl 侧:预期零代码改动,四项冒烟验证点

vendored verl 0.9.0.dev 已内置全部所需能力(§2.4),不需要写适配 patch。
冒烟必须逐项证实的四个点:

1. sglang 0.5.10 + sglang-kernel 0.4.1 组合通过 `sglang_rollout.py` 的
   assert_pkg_version 双轨分支(0.4.1 处在新旧命名切换点附近,import 面首验);
2. FSDP worker 加载 `Qwen3_5ForConditionalGeneration`(VLM 形态、text-only 用法)
   ——vision tower 随 FSDP 分片,首版不冻结随走(bf16 参数 ~1.2G,浪费可控);
3. **训练档双预案**:主档 `ulysses_sp=2` + rmpad + fused CE(qwen3_5 patch 现成;
   fused CE 对 248k 词表是显存刚需);保守档 SP=1 + rmpad=off + token 预算减半
   (任一 patch 路径冒烟不过时的 fallback,数学等价、不进指纹);
4. 权重同步(FSDP→SGLang update_weights):`model.language_model.*`/`model.visual.*`
   命名映射在 2-step 冒烟中以"SGLang 侧生成行为跨 step 变化"实证。

### 3.3 PoC 门序列(复用 162 门 1-4 模式,9B 专项加项)

- **门 1** import/版本四行 + sglang-kernel sm80 真实 kernel 冒烟(单卡矩阵乘不算数,
  须 sglang 真实起服前置检查);
- **门 2** 单卡起服 Qwen3.5-9B(bf16、mamba-radix-cache V1)+ 真实往返;
- **门 3** PGS 机制三断言(**9B 专项**):top-256 宽度 / tail-only prompt_logprobs 行数与
  首行 null / tail 相对 full 的降时比(hybrid 的 mamba prefix cache 语义下重测——
  vLLM 侧有"短 prompt 不命中块粒度缓存"的已知行为,SGLang 侧收益多少以实测为准);
- **门 4** memory-saver 起服 + 真实往返(RL 权重更新依赖);
- **门 5** verl 全链 2-step 冒烟(8 卡、真实 ALFWorld、v3.3 全开关):actor update 真实
  发生、update_weights 往返、mamba pool 显存跨步回落、val 一轮 JSON-valid。

### 3.4 机制/实验语境适配项(环境之外、起跑前必做)

- **熵门重标定**:248k 词表 + 新基座 ⇒ `entropy_theta_init/floor` 必须在 9B 语境重标
  (ent_calib.py 流程已备;4B 的 1.28/0.6 禁止沿用);
- θ_D 同理(若跑 D 门对照臂);
- chat template / special tokens 变化(eos 248044):src/trace_opd 侧已确认为
  **符号化解析**(`template_delta.py:110` `convert_tokens_to_ids("<|im_end|>")`,
  无数字硬编码)——Qwen 系记号不变则自动适配,冒烟断言 im_end 解析非 None;
  thinking 行为语境:**enable_thinking 是跨 17+ 处的耦合面**(template_delta
  `_CHAT_KWARGS`、config 校验、14 个 yaml、launcher/脚本;agent 全量扫描),且
  SGLang 对未知 chat_template_kwargs 可能**静默忽略**——thinking 没关掉是最危险
  的静默失败形态(轨迹长度与 token 预算全面失真)⇒ 门 2 探针
  `scripts/sgl_probe_q35.py --stage thinking` 三组对照(缺省/off/on)一票判别;
- 9B raw 的 won_bare 基线探测(train val 各一轮)——决定臂设计的能力起点读数;
- FORK_LEXICON 是词面 regex,与 tokenizer 解耦,可沿用后按首批数据重标。

### 3.5 资源账(预估,冒烟修正)

- **rollout(SGLang,mem-fraction 0.5≈40G/卡)**:权重 19.3G + mamba state pool
  (state≈2MB/req/层 × 24 层 ≈ 50MB/req;256 并发 ≈ 12.5G——**需下调 max_num_seqs
  或用 max-mamba-cache-size 控池**)+ 8 个 full-attn 层的 KV(层少,KV 便宜:
  ≈16KB/token,32k 序列 ≈0.5G/条)⇒ 可行但 mamba pool 是新变量;
- **训练(FSDP2 8 卡,param_offload)**:参数+梯度+fp32 优化器 ≈20G/卡 静态;
  **248k 大词表 × 长序列的 logits/entropy 峰值**(32k×248k bf16 ≈ 16G/条)是
  首要 OOM 风险 ⇒ 主档开 fused CE(qwen3_5 专用实现在位,logits 不落地)+
  SP=2/rmpad(padding 浪费消除);保守档靠
  `ppo_max_token_len_per_gpu`/`log_prob_max_token_len_per_gpu` 减半托底;
  response_length 是否降 32k→16k 待裁定(9B 是新线,不必绑 4B 口径);
- 磁盘:ckpt(9B×FSDP 全量)≈ 4B 的 2.2×,save_freq/keep 沿 v3.3 档先行。

## 4. 风险清单(按影响排序)

1. **tail-only prompt_logprobs × hybrid mamba cache 组合无先例报告**(追踪 issue
   中 logprobs 零条目=没人踩过,不等于没坑)⇒ 门 3 一票制;radix 收益若归零,
   verify 回到全量 prefill,成本可估不致命,但 SPS 墙钟账要重算;
2. **sglang 0.5.10 的 qwen3_5 成熟度**(发布后一个月的版本;主修复已入):
   若门 2-5 踩到 0.5.10 已知 bug 且修复只在 0.5.11+(torch 2.11 连锁),则升级
   代价重估——这是本方案唯一可能被迫扩大变动面的分支;
3. **mamba pool 泄漏/尺寸**(issue #20010 场景不重合但同池):冒烟观察项
   (rollout 显存跨步回落)+`max-mamba-cache-size`/`max_num_seqs` 控池;
4. **大词表 logits 显存**:fused CE 主缓解 + token 预算保守档托底;
5. **verl fused CE/SP patch 对 qwen3_5 的实跑成熟度**(0.9.0.dev 代码在位≠
   在我们语境验证过):门 5 的 2-step 真实 update 实证;
6. 回摆通道:任何一门不过 ⇒ 现 4B 主线(env 未动)零成本回摆。

## 5. 与 v3.3 4B 主线的关系(排期语义)

9B 线是**新增并行线**,不替代 4B 正式臂(GO-V33 已交 CD-006):八卡窗口的分配
(先 4B 正式臂还是先 9B PoC)是用户排期裁定项;PoC 门 1-4 仅需单卡、与 4B 八卡
run 不互斥(可挤同机空隙),门 5(8 卡冒烟)需要整机窗口。

## 6. 附录:vendored verl 耦合面盘点(已确认项)

- sglang import 面共 6 文件:`workers/rollout/sglang_rollout/{sglang_rollout,
  http_server_engine,async_sglang_server}.py`、`utils/distributed.py`、
  `experimental/reward_loop/router/inner_sglang_router.py`、
  `checkpoint_engine/mooncake_checkpoint_engine.py`(后两者不在我们路径上);
- 架构 patch 注册表 `models/transformers/`:qwen2/qwen2_vl/qwen3_5/qwen3_vl/
  glm4v/kimi_vl/llama/apertus + dense_common fallback——**qwen3_5 在册**;
- 版本断言:`sglang_rollout.py:72-95` flashinfer 与 kernel 包双轨断言
  (0.5.10+kernel 0.4.1 组合落在切换点附近,门 1 首验);
- `monkey_patch.py` 五能力开关签名:`ulysses_sp_size / use_remove_padding /
  use_fused_kernels / fused_kernels_backend / use_prefix_grouper / use_tiled_mlp`
  ——9B 配置面即这些键的组合;
- **FSDP 加载分支(agent 实测,transformers 5.5.0 真跑)**:`Qwen3_5Config` 命中
  `AutoModelForImageTextToText._model_mapping`(verl/utils/model.py:704-705)⇒
  VLM 形态走 **AutoModelForImageTextToText**(非 CausalLM/Vision2Seq);
  attn_implementation 经 hf_config 携带(默认 flash_attention_2)。
  ⚠ `Qwen3_5Config` 顶层**没有** `num_attention_heads`(读取即 AttributeError);
  monkey_patch:332-338 有 text_config 回退,但任何别处直读该属性的代码在 VLM
  形态下会炸——冒烟报错指纹之一。
- **权重同步命名(agent 实测)**:全链路唯一重写点 `convert_weight_keys` 读 HF
  `_checkpoint_conversion_mapping`,Qwen3.5 两形态该表均为**空 dict ⇒ 恒等映射**,
  `model.language_model.*`/`model.visual.*`(VLM 89 键)**原样送进 SGLang**;
  正确性押在 SGLang qwen3_5 loader 认同一套 checkpoint 命名(两边加载同一份
  HF 权重,理应一致)——门 5 实测,失败模式大概率响亮但静默跳过不能排除,
  故验收用"update_weights 后同 prompt logprob 变化"行为证据而非仅无报错。
  vision 参数一并同步(带宽/bucket 显存把 ViT 算进去)。
- **src/trace_opd 硬编码(agent 四角度全扫)**:数字 token id / model_type 判断 /
  特殊 token 字面量三类**零命中**(唯一 `<|im_end|>` 在 template_delta.py:110,
  符号化解析安全);次级复核项:render_ids 无 verl 式"至少一条 user message"
  兜底(当前调用点均传 system+user,不触发)、ssb/math adapter 注释里的 Qwen3
  tool-role 渲染形态假设(对新模板复验)。
