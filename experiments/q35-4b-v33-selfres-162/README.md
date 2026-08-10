# 实验 · Qwen3.5-4B 完整自进化机制（v3.3 × v4 × FC）

**实验 ID**：`q35-4b-v33-selfres-162` · **机器**：162 四卡（2×A800 PCIe + 2×A100 PCIe）
**状态**：待起跑（阻塞项见 §5） · **建立日期**：2026-08-08

一个实验一个目录：`config.yaml` 是唯一配置真源，`run.sh` 自足可手工复现。

```bash
bash experiments/q35-4b-v33-selfres-162/run.sh dry      # 看门禁与将执行的命令
bash experiments/q35-4b-v33-selfres-162/run.sh smoke    # 1 步冒烟（独立 run 目录）
bash experiments/q35-4b-v33-selfres-162/run.sh formal   # 正式：2 epochs = 358 步
bash experiments/q35-4b-v33-selfres-162/run.sh resume   # 断点续跑
```

---

## 1. 这个实验是什么

**首次把三条线合到一起**（此前不存在同时具备三者的配置）：

| 层 | 内容 | 来源 |
|---|---|---|
| steering 机制 | **v3.3 六模块**：M1 不对称 advantage `pgs_asym_grpo` / M2 α cosine 有限期 / M3 `w_beh off` / M4 熵门 G-E / M5 不做截断 / M6 记账 | `a800-pgs-v33-4braw-asym.yaml` |
| researcher | **v4 自体研究员**：researcher 即 policy 本体，每任务 24 行 = 8 steered + 8 bare + 8 research，λ_r=1.0 | `162-pgs-s4-selfres-1p7b.yaml` 的 `research_rl` 段 |
| 协议 | **Function Call**：ALFWorld 单工具 `execute(command)`，researcher 走客户端解析 FC | commit `3be537f`→`c3942b8` |

**基座**：原始 Qwen3.5-4B，**不做 SFT 冷启动**。依据是 CD-006 2026-08-08 的 researcher 探针
（ALFWorld 32 任务分层抽样 × 2 会话 = 64 条，FC 协议）：finalize **98.4%**、工具调用错误 **0**、
doc 词数中位 676、含可执行命令 63/63。1.7B 当初必须先 SFT 是因为它不会用工具，4B 无此问题。

> ⚠ 探针走 external 后端 + 服务端解析，本 run 走 engine 后端 + 客户端解析。
> wire format 同一个、解析位置不同，**冒烟须复核 engine 侧 finalize 率**。

## 2. 关键参数与依据

| 项 | 值 | 依据 |
|---|---|---|
| `research_rl_weight`（λ_r） | **1.0** | 用户裁定 2026-08-08（原 0.1 太小）。`seq-mean-token-mean` 下行间等权 ⇒ 名义梯度份额 `8λ/(16+8λ)` = 33.3%。1.7B 实证：research 组内 reward 挤在 r_raw≈0.06，组内归一化后 advantage 仅 ±0.03，而 executor 侧 0.3–1.0 |
| `session_token_budget`（researcher） | **49152 (48K)** | 用户裁定。探针实测会话总量 p50 24672 / p90 42122 / max 58828，本值覆盖 p90 的 1.17× |
| `rollout_max_model_len`（保险丝） | **131072 (128K)** | 用户裁定："总上下文由一个参数控制，一般是不会达到的上限"。4B 的 `max_position_embeddings` = 262144，128K 在物理上限内 |
| `research.max_turns / researcher_max_tokens` | 20 / 4096 | 用户裁定。实测工作轮远低于 4096（`run_experiment` p90 2324、`finalize` p90 3063，全部 626 轮中 ≥4096 仅 5.6%） |
| `research.n_exp` | 8 | 用户裁定。探针 `experiments_used` 中位 7.0 |
| `episode.max_tokens_per_turn` | 1024 | 用户裁定。**该键曾是 dead config**（配 4096 实际生效 2048），CD-010 已修（`9c9568a`） |
| `total_epochs` / 总步数 | 2 / **358** | 用户裁定跑满 2 epochs。train 2875 条 ÷ batch 16 = 179 步/epoch |
| `steered_adv_decay_steps` | **240** | 设计文档建议 ≈ 总步数×2/3 = 238，取 240 与 A800 v3.3 线同值便于对照。α=0 后 steered 臂停产，末段 118 步纯 bare |
| `entropy_theta_init / floor` | 1.28 / 0.6 | 用户裁定"先用当前初始配置值（反正可以自适应调整）"。`gate_adaptive: true` 会按 `entropy_target_rate` 自校；该值源自 text 语境，**FC 语境前若干步门率可能落在 `gate_rate_bounds` 外，属预期不属故障** |
| `ulysses_sp` | 4 | 长轨迹要保留梯度（需求 §7.4）⇒ 长序列真进 backward，SP=4 把单卡激活降到 1/4。⚠ 162 四卡 ⇒ **dp=1**，该组合本机未验证 |
| `use_fused_kernels` | **true** | 默认是 false，**必须显式开**：Qwen3.5 词表 248320，不开则 logits 显存必爆 |

## 3. 预算链（§7.5 控制器/保险丝解耦）

```
① 保险丝    rollout_max_model_len = 131072      一般不触发
② 有效生成   researcher 20×4096 / executor 40×1024   派生,mask=1 的最坏上界
③ 流预算    researcher session_token_budget = 49152
            executor  train.response_length    = 49152  ← 见 §5,期望值是 16384
④ 单次调用   researcher 4096 / executor 1024 / narration 0(=3×max_words)
```

## 4. 起跑前必看

- **内存**：padding 内存 ∝ 行数 × `response_length` = 384 × 49152。按 1.7B 冒烟
  `cpu_memory_used_gb=177.2 @38912` 外推约 224 GB（+4B offload 约 28 GB），162 可用 414 GB。
  **冒烟第一步必须核对实际 `cpu_memory_used_gb`**。
- **SP=4/dp=1** 在 162 未验证过（A800 的 SP=4 是 8 卡 dp=2），冒烟须看显存峰值与单步墙钟。
- **`causal-conv1d` 未装**：Qwen3.5-4B 有 24/32 层是 `linear_attention`，transformers 侧
  回落 torch 实现（rollout 走 sglang 不受影响）。1.7B 实测 `update_actor` 仅占整步 11.5%、
  `gen` 占 84.5%，故暂不动手——装它要编译且有破坏现有 attention 栈的风险。

## 5. 阻塞项（起跑前须解决）

**用户裁定 executor 的 `session_token_budget` = 16K，当前代码不可表达。**

`train.response_length` 一身两职——既是 executor 流预算（`launch_train.py:356` →
`actor_rollout_ref.rollout.response_length`），又是 verl 唯一的 padding 维度
（`:299` → `data.max_response_length`）。两者是不同的 hydra 键但绑同一配置值，
所以它没法既等于 16384 又容下 researcher 的 49152。实测：

```
research.session_token_budget(49152) 超过 train.response_length(16384)
——research 行 stream 进该 padding 维度且 verl 只补不截
```

已提 **需求文档 §7.6** 给 CD-010（含实测证据与事实定位）。在该项落地前，
本配置的 `response_length` 暂填 49152，**executor 由轮预算 40960 主导，
与用户裁定的 16K 有差异**。

## 6. 观察重点

- **机制侧**：`steer_arm_gap`（δ）、`won_bare_rate`、`steer_gate_open_rate`（应在 `[0.05,0.25]`）、
  α_t 衰减曲线、steered 超额率
- **v4 自体**：`research_r_raw`（非恒零）、`research_experiments`（是否随训练上移，1.7B 基线 5.0）、
  `research_no_finalize`、`research_overlong`
- **崩塌前兆**：熵、`grad_norm`、`response_length` 膨胀、thought 逐字重复率
- **干涉监测**：executor 的 `invalid_format`（λ_r=1.0 的双目标干涉由它兜底，1.7B 冒烟基线 0.18）
