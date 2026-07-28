# AgentRL 深读:多任务联合训练的算法与工程

**对象**:arXiv 2510.04206(THUDM/Z.AI,2025-10-05)+ 官方代码 `github.com/THUDM/AgentRL` @ `6a73409`
**本地**:PDF `refs/pdfs/2510.04206-AgentRL.pdf`;代码 `refs/AgentRL/`
**一句话**:这篇论文的算法贡献只有两块,且都极小 —— `examples/training/agentrl_trainer.py` 相对基线 `examples/training/async_trainer.py` 的完整 diff 就是 **14 行的 `adv_norm` 函数 + 一组 stale rollout engine 的路由**,其余全是异步基建。所以逐行读完整套算法的成本很低,而结论很硬。

---

## 1. Task Advantage Normalization:精确形态

**论文**:§3.2 式 (1),p.6。
**代码**:定义 `examples/training/agentrl_trainer.py:60-71`,调用点 `:276-281`。

```python
def adv_norm(items):                                    # agentrl_trainer.py:60
    all_adv = []
    for item in items:
        adv = to_device(item["advantages"], "cpu")
        loss_mask = to_device(item["loss_mask"], "cpu")
        effective_adv = torch.masked_select(adv, loss_mask.bool())   # 只用可训 token 统计
        all_adv.append(effective_adv)
    all_adv = torch.cat(all_adv, dim=0)
    mean, std = all_adv.mean(), all_adv.std()
    for item in items:
        item["advantages"] = (item["advantages"] - mean) / (std + 1e-6)   # 作用于整条张量
```

调用侧按 `item["data_source"]` 分桶,而 `data_source` 就是任务名(`trainer/src/agentrl/trainer/agentic/data_provider.py:22`)。

**五个精确性要点**:

1. **粒度 = 当前 batch × 单任务**。不是滑动窗口、不是全局 EMA、不是任务内单组。每 step 重算,无状态。
2. **统计量只在 `loss_mask=1` 的 token 上算,但减/除作用于整条 advantage 张量**。mask=0 位置从 0 变成 `-μ/σ ≠ 0`,靠下游 `agg_loss` 的 `masked_mean` 丢弃(`algorithms/core_algos.py:320`)。功能上无害,复用时是坑(见 §7.1)。
3. **它是叠在 GRPO 组内 z-score 之上的第二层归一化**。`compute_advantage` 先跑完 GRPO(`algorithms/advantage.py:69-72` → `core_algos.py:134-135`,`(scores-mean)/(std+1e-6)` 后广播到 token),`adv_norm` 才动手。两层结构:**组内 z-score 定方向,任务内归一定尺度**。
4. **超参只有一个开关** `task_adv_norm: True`。无系数、无动量、无窗口、无温度。eps=1e-6。
5. **不会 NaN**:某任务全部组退化时 adv 恒 0、σ=0,`0/(0+1e-6)=0`,安全。

### 它到底在做什么(我做了定量核验)

论文只说"reduce inter-task variance"。我按其口径仿真了 32 组 × n=8、成功轨迹短(200 tok)/失败轨迹长(800 tok)的典型 agentic 分布,得到两个论文没写出来的性质。脚本可复跑:`docs/research-notes/agentrl-taskadvnorm-probe.py`(`python3` 直接运行,只依赖 torch)。

**(a) 增益 `1/σ_i` 关于任务难度呈 U 形。**

| p_success | 有效组占比 q | token 加权 μ_i | σ_i | 增益 1/σ_i |
|---|---|---|---|---|
| 0.10 | 0.66 | −0.179 | 0.446 | **2.24×** |
| 0.50 | 1.00 | −0.553 | 0.804 | 1.24× |
| 0.70 | 0.88 | −0.609 | 0.987 | **1.01×** |
| 0.90 | 0.50 | −0.408 | 1.022 | 0.98× |
| 0.99 | 0.06 | −0.057 | 0.437 | **2.29×** |

中等难度任务几乎不动;**过难或过易(全对/全错组多、有效组占比低)的任务被放大 2 倍以上**,把稀疏的残余信号顶回单位尺度。这才是论文 "mitigate rate disparities" 的机制实质 —— 它补偿的是**有效梯度信号的密度**,不是 reward 的绝对量级。

**(b) 减均值不是 no-op,而是按任务的长度偏置去偏。**
上表 μ_i 全为负,最大到 −0.61;我做了等长对照(成功/失败都 500 tok),μ_i 精确等于 0.0000。原因:失败轨迹跑满 `max_turns` 因而更长,而统计是 token 加权的,长的负 advantage 主导了均值。由于 loss 用**全局 token-mean** 聚合(`workers/fsdp_worker.py:295` + `:303`,配 `loss_reduce_mode: token-mean`),μ_i<0 等价于该任务净贡献一个"整体压低 log-prob"的梯度分量;减去 μ_i 把该任务在 token 空间的净梯度**精确置零**。副作用有方向性:**减轻长失败轨迹的惩罚、加重短成功轨迹的奖励**(见 §7.3)。

### 论文的消融证据

**Table 6**(p.8,Qwen2.5-14B,五任务联合):

| | AF | DB | KG | OS | WS | **AVG** |
|---|---|---|---|---|---|---|
| AgentRL-14B | 93.1 | 64.0 | 67.7 | 45.1 | 55.0 | **65.0** |
| − cross sampling | 91.9 | 61.6 | 55.7 | 39.7 | 54.5 | **60.7** (−4.3) |
| − task adv norm | 91.1 | 62.6 | 54.7 | 38.0 | 50.6 | **59.4** (−5.6) |

去掉 task adv norm 掉 5.6 分,五个任务全掉,KG 掉最多(−13.0)。Fig 7b(ALFWorld 训练 pass rate 曲线)显示去掉后收敛显著变慢并出现波动。论文口径:"the model tends to learn different tasks at different rates instead of learning jointly"(p.8)。

---

## 2. Cross-Policy Sampling:精确形态

**论文**:§3.1 p.5 + 附录 C 式 (20),p.16。
**代码**:`agentrl_trainer.py:149-156`(路由)、`:84-95`(资源切分)、`:112-118`(第二条 NCCL 通道)、`:322-324`(错步同步)。

**关键:实现里根本没有第二个模型。** 论文自己讲明:"During RL training, it is hard to incorporate models with different architectures in the pipeline. Instead, we let the model do cross-policy sampling with its **early version**"(p.6)。落地形态是:

- 划出 `rollout_stale_ratio = 0.25` 的 rollout GPU 跑一组独立 SGLang engine;
- 主 engine 每 step 同步权重,**stale engine 每 `stale_step = 25` 步才同步**;
- `cross_sampler` 在**每次 generate 调用**(即每个 env step / 每一轮)独立抛硬币:以 `1−0.25` 概率走 fresh,`0.25` 概率走 stale。同一条轨迹的不同轮次因此混着两个策略产生 —— 精确对应式 (20) 的 `a^{c,(t)} ~ random(M)(·|s^(t))`。

**最重要的正确性细节(论文没写,只有代码里有)**:`rollout_log_prob` 存的是**生成它的那个 engine 自己返回的 logprob**(`agentic/loops.py:73-78` 与 `:152`,来源 `workers/async_sglang_worker.py:65` 的 `meta_info["output_token_logprobs"]`),而 `ppo_loss` 直接拿它当 `old_log_prob`(`algorithms/loss_funcs.py:9`)。**所以 stale engine 生成的 token,PPO ratio 就是 π_θ/π_stale —— behavior policy 逐 token 对齐,重要性比值天然正确**,再由 clip(0.2/0.2)+ dual-clip(c=3.0,`core_algos.py:394-398`)截断。这不是"糊弄的 off-policy",是有正确 behavior logprob 的 off-policy。

**证据**:Table 6 去掉后 AVG 65.0→60.7,KG 掉 12.0。Fig 7a:无 cross sampling 的 KG 训练 pass rate **更早触顶然后停滞**(能力上界更低,不是收敛更慢)。Fig 8a(纯推理侧验证,不训练):低 k 时 cross 略逊单模型,k 增大后 pass@k **同时超过两个单模型和 mix 策略** —— 说明它确实探到了两个模型各自能力边界之外的状态。

---

## 3. 多任务混批:配比、攒批、失衡、退化组

**配比:固定 1:1:…:1,按 rollout 条数,不动态。**

```python
dataloader = iter(interleave(*[cycle(repeat(DataLoader(ds, batch_size=None, shuffle=True,
    generator=train_dataloader_generator), n)) for ds in train_datasets]))   # agentrl_trainer.py:244-249
```

`repeat`(`utils/utils.py:190-196`)给每个源样本发一个 `group_id` 并复制 n=8 份;`cycle` 让小数据集回绕(= 论文 p.6 "we replicate smaller datasets such that each task appears approximately the same number of times as the largest task… sequentially cycle through multiple datasets");`interleave`(`utils/utils.py:199-203`)逐条轮转。注意 `data_provider.py:26-37` 里那个 LCM 上采样的 `BalancedDataset` **训练脚本并没有用**。

**攒批:group 原子 + 动态 batch size。**
`Buffer`(`components/buffer.py:20-28`)只在一个 group 的 8 条全部完成时才整组放进队列 —— **batch 里永远是完整的组**。`get(minimum, multiple)`(`:33-43`)取 `len(queue)//multiple*multiple` 条且 ≥ minimum,即**把当时攒到的全拿走,可以多于 `real_bsz`**,对应论文 §3.1 "accepts a dynamic batch size that fluctuates within a certain range"。首步多取 1.7 倍(`agentrl_trainer.py:265`)。

**失衡:分发端严格均匀,落批端完全不控制。** 队列是 FIFO `deque`,谁先跑完谁先进批,短任务/快环境天然超配。他们**不纠正这个**,而是靠 task adv norm 在 advantage 尺度上事后补偿。

**必须看清的一层**:梯度权重是**全局 token-mean**(`loss_reduce_mode: token-mean` → `loss_weight = loss_tokens`,`agentrl_trainer.py:294-300` → `fsdp_worker.py:295, 303`)。所以 task adv norm 均衡的是**每 token 的 advantage 尺度**,**不均衡 token 数量** —— 长轨迹任务的梯度质量依然更大。AgentRL 五个任务序列长度同量级(都 ≤8192、≤20 轮),这个残余失衡可以忽略;换到长度差数倍的场景就不能忽略(见 §6a)。

**全对/全错组:不过滤,只监控。** 没有 DAPO 式 dynamic sampling。GRPO 里组内全同 → `(s−mean)/(0+1e-6) = 0` → 该组 advantage 恒 0,只占分母不贡献梯度。`algorithms/metrics.py:26-28` 打三个指标:`effective_uids`(组内 reward 非常数的比例)、`all_1_uids`、`all_leq0_uids`。这些退化组正是把 σ_i 拉小、从而触发 task adv norm 放大的直接来源 —— 两个机制是耦合的。

---

## 4. 多任务联合 vs 单任务专训

**Table 4**(p.7,Qwen2.5-14B-Instruct):

| | AF | DB | KG | OS | WS | AVG |
|---|---|---|---|---|---|---|
| 单任务 AgentRL-ALFWorld | **89.7** | 49.7 | 22.3 | 33.7 | 15.9 | 42.3 |
| 单任务 AgentRL-DB | 0.2 | **73.9** | 26.2 | 43.1 | 16.0 | 31.9 |
| 单任务 AgentRL-Webshop | 0.0 | 57.9 | 25.3 | 40.1 | **60.3** | 37.8 |
| Best-of-Five(五个专家各取最好) | 89.7 | 73.9 | 72.2 | 43.1 | 60.3 | **67.8** |
| **AgentRL 多任务单模型** | **91.5** | 72.2 | 72.8 | 43.6 | 58.5 | **67.7** |

结论:**单任务专训几乎零跨任务迁移**(DB 专家在 ALFWorld 上 0.2,Webshop 专家 0.0);**多任务联合基本无损地拿到五个专家的水平**(67.7 vs 67.8)。分任务看,ALFWorld +1.8、KG +0.6、OS +0.5 受益,DB −1.7、WS −1.8 略受损 —— 受损的恰是单任务分数最高的两个。泛化:BFCL-v3(Table 5)multi-turn 16.2→19.2(+3.0),overall 59.9→61.4,单轮 nonlive 反而略降 86.0→85.8。

**训练稳定化手段全清单**(逐条给出处):

1. task adv norm(§1);2. cross-policy / stale engine(§2);
3. **KL 到 ref 保留但很小**:`use_kl_loss: True`、`kl_loss_coef: 1e-4`、`low_var_kl`(`loss_funcs.py:64-79`);
4. **熵奖励完全关闭**:`entropy_coef: 0.0` —— 探索全靠 temperature 0.8 + cross-policy;
5. **clip 对称 0.2/0.2**(**不是** DAPO 的 clip-higher)+ dual-clip `clip_ratio_c=3.0`;
6. loss 全局 token-mean;actor fp32(`torch_dtype: float32`),ref bf16;
7. `grad_clip=1.0`(`fsdp_worker.py:324` 默认);
8. reward 统一到 [0,1] 二值,异常终止罚 −0.2(`incomplete_punishment`,`loops.py:139-141`;论文 E.1);
9. off-policy 上界:限制数据队列最大长度 + 每 step 强制把在途轨迹推给训练端(论文 §3.1)。

论文 G.1 承认 cross-policy 会引入 "minor distributional shifts… mild, transient instabilities in training dynamics",定性为可接受权衡,**未给曲线**。

---

## 5. 超参全套(configs/qw14b_waodk.yaml 五任务主实验 + 论文 E.2)

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-{3,7,14,32}B-Instruct / GLM-4-9B;**Qwen 系无 warm-up SFT**,GLM-4-9B 需 cold-start SFT |
| lr / wd | `1.0e-6` / `0`,**配置里没有 scheduler** |
| batch | `batch_size: 256` 组 × `n: 8` = 2048 条轨迹/step(3B 单任务为 32×8=256) |
| group size | train `n=8`,val `n=4` |
| steps | `max_steps: 1000`(论文 E.2 "over 1000 steps";Fig 1b 横轴到 3.0M samples) |
| 采样 | temperature `0.8`(rollout 与 loss 的 logits/T 同为 0.8),`max_new_tokens: 1024`/轮 |
| 序列 | `max_total_len: 8192`,`max_turns: 20` |
| loss | KL `1e-4` low_var_kl;entropy `0.0`;clip `0.2/0.2`;dual-clip `3.0`;token-mean |
| adv | `adv_estimator: grpo`(`norm_adv_by_std_in_grpo` 默认 True) |
| 异步 | `rollout_ratio: 0.5`、`rollout_stale_ratio: 0.25`、`stale_step: 25`、`concurrency: 2048` |
| 并行 | `fsdp_size: 16`、`max_tokens_per_micro_batch: 16384` |
| 硬件 | H800,14B 最小 16 卡 |

---

## 6. 适用性判决(对 TRACE-OPD 双角色)

### (a) 两通道各自内部做,**不要跨通道做** —— 但必须补一刀 AgentRL 没做的

**判决:executor 通道内部做一次 adv_norm,research 通道内部做一次,互不相干。** 三条理由:

1. AgentRL 的分组键是 `data_source`,语义是"**同一 reward 定义 + 同一支撑集**的一簇样本"。你们 executor 的 reward 是二值任务成功,researcher 的 reward 是 8 份文档各自 K=8 次 doc-conditioned 成功率 `p_i∈[0,1]` 的组间比较 —— 两者 reward 语义和分布形状都不同,合并算 μ/σ 会让样本多的一侧(executor)独占统计量的定义权,直接抹平 researcher 的相对信号。
2. 你们的痛点②(**executor z-score 后 O(1) vs research 去均值不除 std 后 ≤1**)**正是 task adv norm 的目标场景**。各自内部做完后两通道 token advantage 都是零均值单位方差,**跨通道尺度差自动消失**,`research_rl_weight` 从此只承担"我想给 research 多少权重"的策略含义,不再兼职"补尺度差"。这比手调 weight 干净得多,也让 weight 的消融变得可解释。
3. 顺带解决痛点②的一个隐藏项:research 组内**可以继续保留"不除 std"**,由通道级 σ 统一处理。这正是 AgentRL 的两层结构 —— **组内定方向(谁比谁好),通道内定尺度(整体多大力)**。

**必须补的一刀:token 数量均衡。** AgentRL 五任务长度同量级所以能省;你们 researcher 会话(产文档,长)和 executor 轨迹(ALFWorld,可能短)长度可能差数倍,而全局 token-mean 下梯度质量 ∝ token 数,adv norm 管不到。建议**两个通道各自 token-mean 聚合成一个标量 loss,再按 `1 : research_rl_weight` 加权相加**(通道级聚合而非全局 token-mean)。你们已经是"两次 update / 独立打包"的结构,这一步几乎免费。

### (b) 配比启示:固定,不要动态

AgentRL 的经验是**按 group 数严格均匀 + replicate 小数据集,完全不做动态配比**,然后把失衡全部交给 task adv norm。三点推论:

- **痛点①(researcher 冷启动)不该靠调配比解决。** AgentRL 五任务难度差极大(ALFWorld base 8.7% vs DB 48.4%,Table 3)仍用固定均匀配比,靠 adv norm 兜住。你们同理。
- **均匀 ≠ 1:1。** AgentRL 的"均匀"前提是五任务地位对等;你们两个角色不对等(executor 是主任务)。建议**固定一个 group 数比例(executor:research 取 4:1 ~ 8:1),用 `research_rl_weight` 表达重要性,且不随训练动态调** —— 动态配比会和 adv norm 的自动放大叠成正反馈。
- **一个反直觉后果要预期到**:adv norm 会**自动放大冷启动期的 researcher**(其组内 `p_i` 方差小 → σ 小 → 增益大,见 §1 表 p=0.99 一行的 2.29×)。方向是对的(给信号稀疏的通道更多权重),但要盯住增益上界(§7.2)。

### (c) 明确不适用的部分

1. **cross-policy sampling 不要照搬。** 三条理由:(i) 你们 OPD 崩塌的病根已锁定为 moving self-teacher 正反馈,修复方向是**冻结 teacher + 锚**,而 stale engine 本质上是**再引入一个滞后的自己**去产数据,与修复方向冲突且徒增分析维度;(ii) 它常驻吃掉 25% 的 rollout GPU 显存,而你们 A800 4B 已经贴在显存天花板上;(iii) **它的收益恰好在你们的任务上最小** —— Table 6 里 KG 掉 12.0、OS 掉 5.4,而 **ALFWorld 只掉 1.2**(93.1 vs 91.9)。若一定要试,唯一廉价的形态是让 researcher 会话用滞后权重、executor 用当前权重,让"多样性"来自角色分工而不是额外引擎。
2. **不要照搬"不过滤全对/全错组"。** AgentRL 敢不过滤,是因为 batch 极大(2048 条/step)且 adv norm 把稀释掉的信号又放大了回来。你们 research 通道每任务只有 N=8 份文档,8 个 `p_i` 若全相同则整组零信号却占满该 step 的 research 配额。建议至少**先把他们那三个监控指标搬过来**(`effective_uids` / `all_1_uids` / `all_leq0_uids`,`metrics.py:26-28`),按通道分别打点,research 通道 effective 比例持续偏低时再决定过滤还是换任务。
3. **异步 stale/queue-bound 那套基建不适用。** 你们是同一 global step 内两次 update 的同步结构;AgentRL 的动态 batch、group 原子 buffer、队列长度上界全是为全异步管线设计的,搬过来只会平白引入 off-policy 偏差。
4. **clip / entropy 配置不要直接抄。** 对称 clip 0.2 + `entropy_coef=0` 成立的前提是"cross-policy 已经供给了探索"。你们不上 cross-policy 就少了这个探索源,照抄 `entropy_coef=0` 可能加速熵坍缩。

---

## 7. 警告(实现级,踩过就疼)

1. **`adv_norm` 把偏移写进了 mask=0 的位置**(`agentrl_trainer.py:71`),完全依赖下游 `masked_mean` 丢弃。你们 loss 里若有任何**不做 mask 的求和**复用同一 advantage 张量(比如 SFT 项),会静默注入一个常数偏置。
2. **统计量是 token 加权的,长序列主导 μ_i/σ_i。** 若 research 文档/会话长度方差大,通道级 σ 会被最长的几份会话定义。**建议改成按轨迹加权**(每条先 token-mean 再进统计)—— AgentRL 没这么做,但对长度方差大的场景更稳。同时给增益 `1/σ` 加一个上界(如 clamp 到 3×),防止某 step 只剩一两个有效组时把噪声放大 5 倍以上。
3. **减 μ 有方向性副作用,已在 §1(b) 量化**:失败长于成功时 μ<0,归一化会**减轻长失败轨迹的惩罚、加重短成功轨迹的奖励**。ALFWorld 上失败恒为跑满 `max_turns` 的长轨迹,这个效应会持续存在。你们若观察到"失败轨迹惩罚不足 / 长轨迹不收敛",这是头号候选原因。
4. **Buffer 组死锁 + 配额泄漏(真 bug)**:`retry_openai_chat_agent_loop` 失败返回 `None` 时该条不进 buffer(`components/task_manager.py:116-119`),该 group 永远凑不齐 8 条,**永久滞留在 `self.groups` 字典里**(`buffer.py:22-28`),既漏内存又漏该 step 的配额。你们若照搬"group 原子释放",必须加超时补齐或残缺组降级放行。
5. **口径陷阱**:`pass_rate` 指标是在 `incomplete_punishment` 覆盖 reward **之前**算的(`loops.py:137-141`),所以论文曲线里的 pass rate 和 advantage 实际用的 reward **不是同一个量**。复现对数时会对不上。

---

## 附:一句话摘要

AgentRL 的多任务处方是 **"配比死板 + 归一化兜底"**:分发端按任务严格均匀轮转、不过滤退化组、不动态调配比,把全部失衡交给一个 14 行的 per-task batch 内 token 级 z-score 去补偿;而这个归一化的真实作用是**按"有效梯度信号密度"给任务加权**(U 形增益,过难过易的任务放大 2 倍以上)+ **按任务消除长度偏置**。对我们双角色场景,这套的可迁移部分是 §6a 的"通道内各做一次 + 补 token 数均衡",不可迁移部分是 cross-policy sampling 和整套异步基建。
