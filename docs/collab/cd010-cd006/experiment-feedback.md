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

---

## 2026-08-04 · [P1] 162 部署硬前提:必须 `LD_PRELOAD` GCC15 的 libstdc++,否则 vLLM EngineCore 起不来

**结论**:首次冒烟在 vLLM 引擎初始化即崩:
`ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found
(required by venv_train/.../flash_attn_2_cuda.cpython-312-*.so)`。
venv_train 的 flash-attn 来自 conda-forge 预编译包,需 GLIBCXX_3.4.29 / CXXABI_1.3.15,
162 系统库只到 3.4.28。**PGS 自身走 sdpa 用不到 flash-attn,但 vLLM worker 会导入它**,
所以"PGS 不用 flash-attn"并不能免疫。

**处置**:env 脚本加
`export LD_PRELOAD=/data3/wushang/libcxx_x/un/lib/libstdc++.so.6.0.34`
(PGC 线 `run_experiment_1p7b.sh` 的成熟绕法,库已在机)。重启后恢复正常。

**建议**:run-guide §0 环境清单补这一条 —— 它是 162 上的**启动必要条件**,
不是可选优化;新人按现文档照做必踩。

---

## 2026-08-04 · [P1] P4 头号风险有定论:vLLM 0.17.1 对 `prompt_logprobs` 请求**自动禁用 prefix-cache 读**

**结论(源码级)**:`vllm/sampling_params.py:414-418`
```python
if self.skip_reading_prefix_cache is None:
    # If prefix caching is enabled, the output of prompt logprobs may less
    # than n_prompt_tokens, we need to skip reading cache at this request.
    self.skip_reading_prefix_cache = self.prompt_logprobs is not None
```
即引擎自己认识到"prefix cache 会让 prompt_logprobs 行数少于 prompt token 数",
并对这类请求跳过读缓存。

**两个后果,一好一坏**:
- ✓ **正确性风险消除**:SPS verify 的 `prompt_logprobs` 行永远完整,不会出现
  缓存命中导致的空行/缺行 —— 原预案(教师路由禁 prefix cache)**不需要了**。
- ⚠ **NF09 算力账的前提失效**:spec §2.5 假设"文档在 system prompt 静态最前部
  ⇒ 教师侧 KV 跨轮可复用"。在本 vLLM 版本下,**verify 请求永远不读缓存**,
  每次都全量重算 prefill(教师前缀 + 已生成响应流 + draft 段),开销随轮次增长
  (对 30 轮 episode 近似二次)。run-guide 步骤② 第 5 项(墙钟 <2×)的实测
  由此成为**关键读数**,而非例行检查。

**待 CD-010 定夺**:若实测开销超预期,可选方向 ①减小 verify 频率(按轮而非按段)、
②缩短教师前缀、③接受成本。实验侧先如实测量,不自行改机制。

---

## 2026-08-04 · [P2] launcher 硬编码 `max_actor_ckpt_to_keep=3`——参数真值不在配置层

**结论**:`build_command` 里这一项是字面量,配置与 manifest 都看不到它。按既有教训
(参数真值三层:配置 / manifest / verl 实收,**硬编码排最后会赢**),这类值必须在
配置层可见。已提为 `train.max_ckpt_to_keep`(默认 4,用户裁定)。同类项建议 CD-010
复查:凡是 `build_command` 里的字面量常数,只要会影响资源/行为,都应上提配置。

## 2026-08-04 · [P2] 首轮正式实验的执行侧配置裁定(用户 2026-08-04)

与机制无关、不进 objective fingerprint,但两臂必须一致,记录备查:

| 项 | 值 | 理由 |
|---|---|---|
| `run.runs_root` | `/data4/ws/PGS` | /data3 曾 100% 写满并炸掉 C3;/data4 余 5.6T |
| `train.rollout_tp` | **1**(原 2) | 4 卡 = 4 个独立 vLLM replica,无跨卡通信 |
| `test_freq`/`save_freq` | 10 / 10 | 100 步内 10 个 val 点 |
| `train.max_ckpt_to_keep` | 4 | 单 ckpt ≈45G ⇒ 峰值 ~180G |

## 2026-08-04 · [P1] 训练侧显存实测:峰值 allocated 72.1G / reserved 76.7G(80G 卡,96%)

**数据**(base 冒烟 2 步,sp=1、rmpad=False+sdpa、ppo_max_token_len_per_gpu=8192):
`actor/perf/max_memory_allocated_gb` 69.4 → **72.1**;`max_memory_reserved_gb` 74.1 → **76.7**。
训练侧序列最长 **8094**(已贴着 8192 预算);val 侧出现 32704(但 val 不进训练反向)。

**与历史对照**:同模型在 ppo=8192 下曾实测 38.98G —— **现在几乎翻倍**。差异在
`use_remove_padding=False`:该口径下微批内序列要 pad 到最长条,`ppo_max_token_len_per_gpu`
只约束**未 padding 的 token 总数**,padding 后张量可放大若干倍
(例:[2500, 1000×5] 总 7500 ≤ 8192,padding 到 6×2500 = 15000,放大 1.9×)。

**风险**:训练序列长度继续增长(本项目有 step-9 长度暴涨的既有记录)时,
padding 放大 + 单条独占微批会把峰值推过 80G。

**可选缓解(待裁定,均为数学等价、不进指纹的执行档)**:开 `ulysses_sp=2`
——自动联动 `use_remove_padding=True`(packed THD,消除 padding 浪费)+ 序列切两卡。
前提已实测:LD_PRELOAD 后 flash-attn 2.8.3 **真实前向可用**;Qwen3-4B 32 heads / 8 kv_heads,
4 卡与 2 均可整除。**首轮暂按 sp=1 跑(维持预注册),显存纳入监控**;
若峰值逼近 78G 或出现 OOM,再切 sp=2 重跑。

---

## 2026-08-04 · [P0] SPS 真机墙钟 ~20×(而非设计预期 <2×);根因=接口层序列化,不是算力

**发现**:标定 run(steering on,门恒关)rollout 落盘 **2.8 条/分钟** ⇒ 128 条约需
46 分钟;base(steering off)整步仅 3.58 分钟。按此 s1 正式 100 步需 **77 小时**。

**剖析定位(实测,非推断)**:
```
vLLMHttpServer.generate ×4 : CPU 85.7 / 85.6 / 85.5 / 85.2 %   ← 打满
AgentLoopWorkerTQ       ×16: CPU 5.1 / 3.8 / 5.1 %             ← 几乎闲置
GPU                        : 多数采样 0%,偶发 300W             ← 算力闲置
```
⇒ 瓶颈在 **vLLM 服务进程内 `extract_prompt_logprobs` 的 Python 对象构造**
(单线程 GIL),不是 GPU、不是并发度。

**为什么设计时没算到**:NF09 的算力账只算 FLOPs(draft 与普通 rollout 同价、
verify 是一次增量 prefill),**没有"返回数据构造成本"这个维度**。实际每次 verify
要为序列每个位置构造 obs_topk=256 宽的明细(约 1600 位置 × 256 = 41 万个 Python
对象),而 SPS 只读最后 len(draft)+1 行 ⇒ **约 95% 算了、打包了、传了然后丢弃**。
(叠加此前发现的 prefix-cache 对 prompt_logprobs 自动失效,但那一项不致命:
1600 token prefill 对 A800 是亚秒级。)

**已实施修复(实验侧,纯序列化裁剪、零语义变化)**:commit `ff817e6`
- `extract_prompt_logprobs(..., tail=N)`:只序列化尾部 N 行(仍带 dummy 行)
- `vllm_async_server`:从 sampling_params 取 `prompt_logprobs_tail` 透传
- `hook._verify_rows`:请求 `tail=len(draft)+1`,行定位改为**相对返回数组长度**
  ⇒ 全量返回与裁剪返回都正确(全量时退化为原式)

**正确性守卫(同时补上审阅测试缺口 #2)**:新增位置相关 verify mock(第 r 行把
token r%V 排 rank-1),两条测试:①裁剪 vs 全量返回必须产出逐 token 相同的执行
序列与 D 观测;②反证 —— 人为整体挪一行必须被检出(否则①为空测)。

**给 CD-010 的建议**:①规范的算力账应补"接口/序列化成本"一维;②`obs_topk=256`
是另一层 4× 放大(S 每侧只用 topk=64),若裁剪后仍不够快可再降,代价是窗外
代入率上升(实测现为 10.4%)。

---

## 2026-08-04 · [P0·机制变更请求] 门控从「固定 θ_D」改为「固定目标门率 + 阈值自适应追踪」(用户裁定)

**背景(标定 v1 实测,384 条轨迹 / 84 万 D 观测)**:
1. D 分位数跨 step 高度稳定(P85 = 0.0866 / 0.0860 / 0.0859,三步任务构成完全不同),
   但 **P85 与 P95 只差 0.01**(0.086 vs 0.096)—— D 的 CDF 在门槛区间**极陡**,
   固定绝对阈值下,分布平移 0.01 就会让门率从 15% 跳到 5% 或 40%。
2. 100 步训练后 π 的熵/长度/行为会漂移,冻结阈值必然过时(spec §2.3 原本的
   robust-EMA 设计动机;此前审阅裁定 IF-11 定为"v1 冻结"——**本次用户裁定推翻
   该冻结,推翻正确**)。

**用户裁定的方案**(2026-08-04,原文确认):
```
θ_D(t+1) = (1−β)·θ_D(t) + β·P_{1−q}( 第 t 步实测的 D 分布 )
q = 目标门率(如 10%),β = 慢速率(如 0.1)
```
控制的量从「值」变成「率」:门率构造性稳定在 q,policy 演化自动跟随。
同构先例:PPO adaptive KL penalty(固定系数不可调 → 目标 KL + 系数自适应)。

**实现要点(供 CD-010 参考,实验侧不动机制代码)**:
- `gate.robust_quantile_update()` 预留件即为此设计,只差接线;
- 数据源:上一 step 已完成轨迹的 D 直方图(ledger.d_hist,2048 桶已够分辨率);
  滞后一步,实测分布 step 间稳定 ⇒ 滞后无害;因果性无损(判定仍不看当前 token);
- 落点建议:rollout worker 级各自 EMA(每 worker 每步 ~8 条轨迹 × 数百观测,足够
  估分位;worker 间独立追踪同一分布,差异二阶)——训练端零涉及;
- 初值:标定 run 的 P(1−q)(标定 run 角色从「一锤定音」降为「初值」,仍必要);
- 兜底:门率带告警保留(EMA 追飞时报警);θ_D 逐步值进记账(可审计)。
- 新配置键建议:`steering.gate_target_rate: 0.10`、`steering.gate_ema_beta: 0.1`;
  `theta_d` 语义改为 EMA 初值(标定回填)。
- spec §2.3 注记需同步(替换「v1 冻结」裁定)。

**验收**:mock 下分布平移 ⇒ 门率回到 q±2%(N step 内);θ_D 轨迹平滑无跳变;
门率带告警在 EMA 正常工作时不触发。

## 2026-08-04 · [P1·机制变更请求] B_∞ 调大(用户裁定方向)+ 关键参数消融矩阵(论文需求)

**用户裁定**:每轨迹 4 个干预位置(B=10, δ=2.5)太少,「可以调大一些」;且这些
关键参数需在论文中出现并以实验/消融验证调整效果。

**语义确认(已向用户确认)**:「干预 N 次」= N 个 token 位置的采样分布被倾斜
(从 b_S 采样,可能换 token 也可能接受),其余位置纯学生分布。实测轨迹 600~900
个决策位置,4 个 = 0.5% 作用面。

**实验侧建议(CD-010 定夺)**:
- 首轮主实验 **B_∞ = 20**(≈8 次,1~2% 作用面);
- 安全性:训练端 per-token W 界 e^{±2ηc}=e^{±2.5} 与 B 无关;B 只放宽轨迹级
  总偏移(机制目的本身),8~16 次离「处处覆写」(G0 病因)有量级距离;
- 第二轮消融矩阵(预注册):B_∞∈{10,20,40}(剂量-响应曲线,论文核心图)、
  目标门率 q∈{5%,10%,20%}、c∈{1.5,2.5}(选做)、w_beh_mode 三档(代码已备)。

## 2026-08-04 · [P1·配置收编清单] 实验侧已落地的模板级配置变更(请 CD-010 确认收编 + 同步 run-guide)

以下均已改入 worktree 的 `configs/pgs.yaml` 模板与三份 162 实验配置(commit 链
27c3407 / c010e89 后的实验侧 commit),请机制侧确认收编为正式默认:

| 键 | 旧 → 新 | 依据 |
|---|---|---|
| `train.rollout_tp` | 2 → **1** | 用户裁定:4 卡 = 4 个独立 replica |
| `train.agent_num_workers` | 8 → **16 (= batch)** | 用户长期裁定:必须等于 train_batch_size |
| `alfworld.spawn_concurrency` | 64 → **128** | 用户裁定:并发拉满(80 核) |
| `train.rollout_max_num_seqs` | (未显式) → **128**(新键) | 用户裁定:单 replica 并发上限显式钉死 |
| `train.ulysses_sp` | 1 → **2** | OOM 实锤(32704 轨迹 × rmpad=False 单条 logits 18.5G);sp=2 联动 rmpad+FA2,162 历史成熟档 |
| `train.max_ckpt_to_keep` | launcher 硬编码 3 → **配置键,默认 4** | 用户裁定;参数真值必须在配置层 |
| `train.test_freq`/`save_freq` | 20/20 → **10/10** | 用户裁定 |
| `run.runs_root` | 相对 runs → **/data4/ws/PGS** | 用户裁定(/data3 曾写满炸 C3) |
| `steering.obs_topk` | (hook 内隐 4×topk) → **显式键 256** | 审阅 P0(IF-01)修复 |
| `ledger.D_HIST_BINS` | 64 → **2048** | D 主质量 <0.03,64/256 桶插值不可信;桶数已随账落盘(schema 事故修复) |
| 部署硬前提 | — | `LD_PRELOAD=/data3/wushang/libcxx_x/un/lib/libstdc++.so.6.0.34`(必要条件)+ worktree 服务器目录必须命名 `trace_opd`(launcher 路径假设)——run-guide §0 需补 |

另:SPS 性能三层修复(prompt_logprobs_tail / detokenize=False / flat_logprobs=True,
commit ff817e6 + 76a9a6f)已使 SPS 墙钟 ≈1.0× base(修复前 20×),建议机制侧
将「接口/序列化成本」补入 NF09 算力账一维。

---

## 2026-08-04 · [P1] rollout 期显存实测:SPS 打分请求触发 vLLM 预算外分配(#5907),峰值超预算 11-22G

**实测**(标定 v2,TP=1,gpu_mem=0.7):vLLM worker 67↔78G 动态起伏(预算 56G),
FSDP 侧 rollout 期仅 0.8G(offload 生效)。超额部分 = logprobs/prompt_logprobs
路径的 fp32 log-softmax 全词表副本(~5G/份)× 并发 prefill 叠加 + topk 工作区
—— vLLM issue #5907 的已知无保护路径,**与 R11a 历史事故同源**(旧机制 self-teacher
打分被同一机理炸掉,当年的解法是挪去 FSDP;PGS 的在线判定不能挪,只能留 headroom)。

**长序列研判**:单条长序列不线性爆(chunked prefill 常数化 chunk logits;
prompt_logprobs 输出累积 ~100MB/34k 请求);风险 = 并发尖峰时序。

**已实施防御**(实验侧配置,已入模板):`rollout_gpu_mem` 0.7→**0.6**(结构性
让 8G headroom,KV 池缩、preempt 增,degrade 不崩)+
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(减碎片假性 OOM)。

**给 CD-010**:NF09 算力账之外,规范的资源账应加「logprobs 请求的预算外显存」
一维;若未来 vLLM 版本支持 logprobs 预算化(#5907 跟进),0.6 可回调。

## 2026-08-04 · [P1] 标定 v2 定稿:θ_D 初值 0.0203(P85)/0.0338(P90);v1 "D 分布极陡" 结论作废(64 桶分辨率伪影)

**结论**:2048 桶重标定(pgs_calib,sp2,384 轨迹/829,300 个 D 观测)推翻 v1 读数:
真实 P50=0.0027 / P85=0.0203 / P90=0.0338 / P95=0.0721 / P99=0.3033。
v1(64 桶)的 P85≈0.0866 与 "P85→P95 只差 0.01、CDF 极陡" **全部是桶分辨率伪影**
——桶宽 6.25/64≈0.098 比真实 P95 还大,P95 以下质量全部挤在首桶,分位数=首桶内
线性插值,高估 ~4×。若机制侧任何设计论证引用过 v1 的 0.086/"极陡",请以本条为准修正。

- **θ_D 取值**:固定门方案 → 0.0203(P85);自适应门控(q=10%)→ θ_D(0)=0.0338(P90)。
- **可辨识性重估**:P85→P99 拉开 15×,固定阈值的可辨识性其实良好;自适应门控的
  必要性论据应放在**训练中策略漂移使 D 分布整体移动**上,而非"阈值不可辨"。
- **c=2.5 维持**:P(|ℓ|>c) 实测 0.9%(设计预期 10-25%,低一个量级);c=1.5 也仅
  2.8%。截断几乎不激活,δ=2ηc 预算刻度的保守性成立。
- **p_T 窗外代入率 16.1%**(133,549/829,300;v1 约 10%):draft token 不在教师
  top-256 的占比,走单侧鲁棒代入。非小量,建议机制侧评估该路径对 D/ℓ̃ 的偏差方向
  (若系统性偏 suppress,会虚增 D)。
- 全量数据:`/data4/ws/PGS/pgs_calib`(dump 含 2048 桶 d_hist + 96 桶 ell_hist);
  分析脚本 `scripts/calibrate_theta_d.py --rollouts <run>/rollouts [--q]`。
- 小修建议(P3):`sps_checklist.py` [1] 项 grep 会命中 [launch] 命令行回显
  (行内含 `truncation=error` 字样)导致假阳性 FAIL,建议排除 `^\[launch\]` 行。

## 2026-08-04 · [P2] 运行进度同步:base 对照臂已起跑(100 步);s1 臂等待自适应门控交付

pgs_base(纯 GRPO 对照,steering off)已于 14:49 起跑:100 步、sp2、TP=1、
gpu_mem 0.6、16 worker/128 seqs、save/test_freq=10、keep=4,产物
`/data4/ws/PGS/pgs_base`。预计 8-13h。**s1 机制臂具备开跑的全部标定参数
(θ_D 初值见上条),唯一阻塞项 = 自适应门控实现(MCR 见前条,用户已裁定公式)。
交付后 CD-006 即回填 s1 配置起跑。**若自适应门控交付时间 >1 天,备选:s1 先以
固定 θ_D=0.0203 起跑(标定显示固定阈值可辨识性良好,短程 100 步漂移风险可控),
自适应版留第二轮——此备选需用户裁定,仅供机制侧排期参考。

## 2026-08-04 · [P2] 撤回前条建议:PYTORCH_CUDA_ALLOC_CONF=expandable_segments 与 vLLM 不兼容,勿采纳

前述"预算外显存分配"条目中提到的 expandable_segments 缓解项**实测不可用**:
vLLM CuMemAllocator(sleep/wake 内存池)启动时显式
`assert "expandable_segments:True" not in conf`(pytorch#147851),worker 直接
拒绝启动——pgs_base 首启 10 分钟崩溃。launcher 已删除该 env 并留注释防回归。
显存防御定案:仅 `rollout_gpu_mem: 0.6`(48G 预算 + 实测预算外峰值 ~22G = 70G,
A800 80G 下余量 10G)。机制侧模板配置若已抄该 env,请同步删除。

## 2026-08-04 · [P0] 用户裁定:s1 机制臂优先、base 对照臂延后;自适应门控交付即是 s1 起跑闸门

**结论**:用户当面裁定(2026-08-04 下午):**先跑方案机制实验(s1),对照实验
(base)放后面;s1 等自适应 θ_D 实现交付后用新机制跑**。此前起跑的 pgs_base 已
停止并清理(仅损失 ~20 min,step-1 未完成)。**自适应门控现在是唯一关键路径**:
交付确认(changelog 条目)→ CD-006 同步代码 → s1 起跑,当日可完成。

实验侧就绪状态(交付后零等待):
- `162-pgs-s1.yaml` 已预填:theta_d=0.0338(P90 初值,q=10% 对应;若机制侧定
  q=15% 则换 0.0203)、b_max=20.0(用户裁定二次调大,8 次整额干预/轨迹)、
  全部并发/存储/sp2 参数与 calib 实测定版一致。
- 162 四卡已空(pgs_base 已清,GPU 4×14MB),随时可起。
- 接口对齐请求:自适应门控落地后请在 changelog 注明 (a) 配置字段名与语义
  (theta_d 是否即 θ_D(0)、q/β 的键名),(b) ledger 新增记账字段(EMA 轨迹、
  每步实测门率),(c) gate_rate_bounds 语义是否从"告警带"变为"目标带"。

## 2026-08-04 · [P1] 用户需求:逐轨迹·逐 token 的机制观察面(可视化系统数据 schema 增量;建议与自适应门控同批落地)

**结论**:用户要求可视化系统能逐轨迹、逐 token 呈现 SPS 决策细节(每个位置的
D 值/门判定/阈值、干预位置的学生-教师两侧分布、"本来 token vs 实际 token")。
盘点现状:`TokenSteerRecord` 稀疏行已覆盖大部分(position/d/eta/delta/
propensity/rejected/budget_after/**draft_token/final_token**),**增量只有三项**,
全部 CPU 侧落盘、零 GPU 代价:

1. **轻量全 token 通道**:候选位置的 D 值数组(现在只进直方图,逐位置值丢弃)。
   `steer_d_all: [round4]`(按候选顺序)+ `steer_cand_positions: [int]`(与
   response token 坐标对齐;若候选=全部生成位置则可省,用 turn_spans 重建)。
   体积 ≈15KB/轨迹(2MB/step),可接受。未开门位置的"为什么没触发"由此可视。
2. **干预位置两侧分布明细**:仅 **η>0 的行**(实际执行倾斜的位置,含被拒绝行)
   附 `pi_top16: [[id,logp]×16]` + `pt_top16: [[id,logp]×16]`(倾斜后分布 viz
   端由 η/c/ℓ̃ 确定性重算,不必落盘)。挂在 η>0 行上体积 ≈1KB/行 × ≤~8-30 行
   /轨迹,可忽略。开门但预算阻断的行不需要(未执行干预)。
3. **每轨迹 `theta_d_used` 标量**(判定时生效的 θ_D;自适应门控下逐训练步变,
   轨迹内恒定)——viz 标尺与门率审计都要它。自适应实现的步级记账(θ_D(t) 轨迹、
   实测门率进 swanlab metrics)前条已请求,此处指轨迹 dump 内的快照。

另请确认:dump 是否已含 response 的 token_ids(viz 需 token 级文本对齐;若无,
建议加 `response_token_ids`,~10KB/轨迹,比 viz 端重新 tokenize 对齐更稳)。

背景:CD-006 将扩展现有轨迹渲染工具(traj-viz)消费以上字段:轨迹文本按
三态着色(未开门/开门-接受/开门-拒绝重采)、点击 token 展开 D/θ_D/ℓ̃/两侧
top-k 对比条形图/预算余额、步级面板(门率曲线、θ_D EMA 轨迹、预算耗尽分布)。
消费端全在实验侧,机制侧只需落盘上述字段。s1 首跑即想带上——若能与自适应
门控同一批交付最好;若时间紧,1(D 数组)+3(theta_d_used)优先,2 可后补。

## 2026-08-04 · [P1] 分工修订(用户裁定):逐 token 可视化系统由机制侧(CD-010)全责实施

**结论**:用户裁定,上一条(逐轨迹·逐 token 机制观察面)中的**可视化消费端也由
CD-010 实施**——修订前条"消费端在实验侧"的分工:数据落盘 + 可视化系统一体,
机制侧全责完善交付;CD-006 只做验收与使用反馈。

实施规格沿用前条:
- 数据面三项增量(全 token D 数组 / η>0 行两侧 top-16 / theta_d_used 快照)
  + response_token_ids 对齐通道;
- 呈现面:轨迹文本逐 token 三态着色(未开门/开门-接受/开门-拒绝重采,拒绝位
  内联 x→y)、点击 token 下钻(D vs θ_D 标尺、ℓ̃、π/p_T top-16 对比条形图、
  倾斜后分布重算、预算余额)、步级面板(实测门率曲线、θ_D EMA 轨迹、预算耗尽
  分布、D 直方图漂移)、step→轨迹→token 三级导航;
- 可参考/复用实验侧现有轨迹渲染工具 traj-viz(本机 `~/trace-opd-viz/`,一条
  命令渲染 rollout dump 的会话视图),在其上叠 SPS 层或重建均可,以离线读
  run 目录 dump 为输入形态。

节奏:viz 是离线消费 dump 的,**不阻塞 s1 起跑**——数据落盘字段随自适应门控
同批进 s1 首跑,viz 系统在 s1 运行期间交付即可(用户预期在 s1 前几个 ckpt 时
能逐 token 查看第一批干预实例)。

## 2026-08-04 · [P0] s1 首跑 step-1 实测:首查项全过,但发现"预算前端速烧"结构行为——8 次干预全部落在每条轨迹前 ~16 token

**结论**:自适应门控交付质量好,W_beh 通路终验 PASS(`rollout_is_weight_off1_frac
= 0.335% > 0`,量级与干预面自洽;steer_count 7.5/轨迹落在 4-8 预期带;
θ_D(1)=0.0338 初值正确)。但发现一个**机制层结构行为**,需机制侧评估:

**现象**(pgs_s1/step_00001,128 轨迹):
- candidates 18.1/轨迹 vs 标定 run 的 ~2160/轨迹;bypassed 1962.6/轨迹(99%);
- 样例轨迹(resp 1408 tok)干预位置 = [4,9,10,11,13,14,15,16] —— **全部在前 16 token**;
- 实测开门率 40.3%(目标 q=10%):前端位置的 D 天然高(样例 d 0.037-1.94,
  全过阈),θ_D(0)=0.0338 是全轨迹混合分位,对前端形同虚设;
- budget_spent 18.75/20 —— 预算在前 ~18 个候选内烧光,此后全轨迹直通。

**机理**:预算从头往后线性消费 + 前端 D 分布性偏高(首轮 reasoning 开头教师
影响最强)⇒ B_∞ 无论多大、q 无论多准,干预永远集中在"轨迹最前 8×(1/q) 个
token"内;EMA 自适应会把门率收敛到 q,但**位置结构不变**(它只会把前端的门
开得更挑剔,预算耗尽点后移一些,不改变前端优先本质)。这与"干预关键决策点"
的设计意图相悖——1900+ token 的后续轨迹(含全部后续 action 决策)零干预。

**给机制侧的三个方向**(供评估,CD-006 不动机制代码):
1. **per-turn 预算**:B_∞ 均分到轮(如 20/40 轮=0.5/轮,轮内不用不结转或半结转)
   ——干预贴着决策节奏分布;
2. **流式预算回充**:按 token 位置线性放预算(budget_available = B_∞ × pos/len
   的在线近似,len 未知可用 EMA 轨迹长度);
3. **判定域限定**:D 门只在 action 段/每轮前 N token 判定(候选域本身对齐决策点,
   预算花在刀刃上)。
另附标定侧修正建议:若采用 1/2,θ_D 标定应分位置段(per-turn-index)出分位数
——全轨迹混合分位在位置异质性下对任何局部都不准(step-1 实测前端 D 中位远超
全局 P90)。

**实验侧处置**:run 继续(val 曲线是效果的最终裁判——前端引导若已足够超 base,
简单形态反而是卖点);机制侧改进落地后可作第二臂/第二轮对照。**是否接受当前
形态待用户裁定**。附:miss 率前端 32.6%(标定全轨迹 16%);step 墙钟 3.6 min
(预算尽后 bypass 使 SPS 开销趋零,358 步全程 ≈21.5h);actor 峰值显存 31.2G(sp2)。

## 2026-08-04 · [P1] 用户对可视化系统的四项改进要求(用户实际使用后逐条提出;含前后端实时化的架构性需求)

**结论**:用户在 pgs_s1 真实数据上使用了 viz 系统(截图逐页看过),肯定了功能
方向,但对**说明充分性、实时性、可达性、界面精致度**四方面提出明确改进要求。
以下按页面逐条汇总,含具体设计建议供实施参考。

### 1. index 页四张统计图:解释说明不足,"不能直接明白意思"

- 每张图只有英文短标题,缺:指标定义(怎么算的)、期望/目标参考线的含义、
  当前读数的解读指引。具体:
  - `gate open rate` 图中**绿色带没有任何标注**(它是 gate_rate_bounds
    [0.05,0.25] 合法带)——必须图内标注"目标带"与 q=10% 目标线;
  - `theta_D (EMA trajectory)`:需注明"EMA 自适应轨迹,预期从初值向目标门率
    对应分位爬升;平线=尚未更新(θ_D 每 step 更新一次,当前仅 2 点)";
  - `reject rate` 图 y 轴自动缩放到 [0.0688, 0.0698],**0.1% 的噪声被拉成
    满幅暴涨线**,视觉误导——y 轴应从 0 起或显式标注"放大视图";
  - x 轴 step 数出现 1.0/1.2/1.4 小数刻度——step 是整数,刻度必须整数化。
- 底部汇总表各列(门率/倾斜/拒绝/预算)无表头说明、无预期值对照。要求:
  每列 tooltip 或表下注释行,注明健康区间(如"倾斜/轨迹预期 4-8")。
- 通用要求:**每张图配一句中文"怎么读"副标**(指标是什么、往哪个方向好、
  异常长什么样),不懂机制的人也能看懂。

### 2. 架构性需求:改造为前后端实时系统(用户明确要求)

用户原话意图:后端部署在服务器上、**实时获取运行中实验的最新数据**;前端在
本地浏览器、实时观察。建议架构(供参考,实现自定):
- 服务端:轻量 HTTP 服务(FastAPI 级)驻 162/A800,监听 localhost 端口;
  增量扫描 run 目录(新 step 目录/新 dump)+ 增量聚合(现有 _cache 机制可复用),
  提供 JSON API(run 概览/step 列表/step 聚合/单轨迹逐 token 数据);
- 本地访问:ssh -L 端口转发(项目已有成熟隧道模式),浏览器开 localhost;
- 前端:轮询或 SSE 自动刷新(新 step 出现自动进列表,曲线自动延长);
- 约束:①服务读盘为主,勿抢训练 CPU(A800 64 核余量注意);②端口遵循
  docs/infra 端口约定,避开已占用;③nohup/守护化,崩了不影响训练;
  ④单轨迹 API 按需渲染,顺带解决第 3 条的全量可达问题。

### 3. step 页轨迹列表:多数轨迹不可点击(用户以为是 bug)

- 现状定性:--max-trajs 12 只预渲染拒绝数 top-12 的轨迹页,其余 116 条在列表
  中无链接,**且界面无任何说明**——用户直接判为 bug。这是可达性+沟通双缺陷。
- 要求:**任意轨迹都可点进**。静态模式下方案:逐 token 数据打包为每步一个
  JSON,轨迹页做成单模板+JS 动态装载(避免 128×36 个静态页的体积爆炸);
  前后端模式下天然解决(按需 API)。过渡期至少:列表内注明"仅 top-N 预渲染
  (--max-trajs)"并给全量渲染命令。

### 4. 轨迹详情页:信息充分度/说明性/理解友好度/视觉效果全面升级(用户明确要求"精致的前端界面设计")

现状问题(用户截图逐项):
- 头部统计行全是紧凑术语("倾斜 8 次·拒绝 2·预算 20.00/20.0·候选 14·开门率
  50.0%·旁路 1299"),无解释、无健康区间对照,读不懂;
- 正文纯 monospace 文本流,视觉层级弱;着色图例一行小字;J/K 跳转极客向;
- 拒绝重采内联(draft 划线→final)方向对但视觉拥挤;
- token 下钻信息量与呈现均不足。

具体设计要求/建议:
- **头部 → 指标卡片组**:每指标一张小卡(数值+中文标签+悬浮解释+相对预期带
  的红绿着色),won/outcome 用醒目徽章;
- **轨迹迷你地图(强烈建议)**:页顶一条全轨迹长度的横向时间轴,干预/拒绝/
  预算耗尽点(⛔)打点标注,点击跳转——预算前端速烧现象在此一眼可见,亦是
  分析工具;
- **token 流**:优化字号/行距/配色(区分 reasoning/action/observation 段的
  底色层次);干预 token 用角标+描边强化;hover 即出迷你浮层(D vs θ_D、
  ℓ̃),click 展开完整面板;
- **下钻面板 → 右侧固定抽屉**(非遮罩):π vs p_T top-16 双色对比条形图
  (token 文本+概率),倾斜后分布,决策路径图示(draft→D 门→倾斜→接受/拒绝
  →final 流程高亮当前分支),预算余额进度条;
- **全局**:干预导航(上一处/下一处按钮+计数)、图例常驻侧栏、术语 (?) 帮助、
  整体设计系统(统一配色/间距/圆角/阴影,专业工具感)。
- 实现形态不限(原生 JS 单页/轻框架均可),与第 2 条前后端化合并设计最优。

**优先级建议**:2+3 合并做(前后端化顺带解决可达性)为主线;1 与 4 的说明性
改进(图表注解/指标卡片/图例)可先以静态版快速迭代,界面精致化随前后端版落地。
pgs_s1 正在产出数据(现 step≥2,全程 358 步),可作为实时开发的活数据源。

## 2026-08-05 · [P1] SGLang 环境配置进展:162 走通方案 B(venv_sglang),门 1 全过;两处执行册勘误;门 2-4 待空卡

**结论**:162 环境侧不占卡部分完成。**路线 = 方案 B**(独立 venv):方案 A dry-run
判死——sglang 0.5.8 强制 torch 2.10.0→2.9.1 降级(连动 torchvision/audio、
flashinfer 0.6.4→0.6.1),venv_train 不可动。`/data3/wushang/workspace/TRACE-OPD/
venv_sglang`(uv 建,py3.12)已装齐。

**版本四行**(uv pip list 实测):
- sglang 0.5.8 / torch 2.9.1 / flashinfer-python 0.6.1 / sgl-kernel 0.3.21
- flash-attn:venv_sglang 未装(推理面不需要;训练面补齐待 verl-sglang 适配联调时
  按 torch 2.9.1 匹配重装,版本届时回报)。

**门 1 逐项**:sglang import ✓ 0.5.8;flashinfer import ✓ 0.6.1(**无需 triton
兜底**);**LD_PRELOAD 不需要**(manylinux wheel 自带 libstdc++ 兼容,与执行册
§0.4 的预期不同——162 的 GLIBCXX 坑不迁移到 sglang 栈)。

**执行册两处勘误**:
1. `sglang[srt,openai]` 的 extras 在 0.5.8 不存在(uv warning:no extra named
   srt/openai)——裸 `sglang==0.5.8` 已含 srt 全栈,文档命令请更新;
2. 安装源:PyPI 官方直连在 162/A800 均为几百 KB/s(4GB 栈 >1h);TUNA 镜像
   (`UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`)分钟级完成。
   执行册建议补默认镜像源。

**门 2-4(起服/往返/机制三断言/memory-saver)待 1 张空 GPU**:162 四卡在
pgs_s1、A800 八卡在 pgs_base_a800,无空窗;等 run 停点或用户裁定腾卡后立即执行,
门 3 五个 latency + 三断言数字随后回报。

**A800 前瞻(dry-run 进行中)**:结构性风险预判——sglang 依赖链是 PyPI CUDA
二进制(torch 2.9.1/sgl-kernel/flashinfer 预编译 wheel),本机驱动 525 上
PyPI CUDA 栈有"import 通但真实运算报驱动过旧"实测前科,且 conda-forge 无
sglang/sgl-kernel。若 dry-run 后实装验证失败,A800 选项:①保持 vLLM 后端
(回摆通道,base/对照臂本来不吃 verify 收益);②驱动升级(管理员级,需裁定)。

## 2026-08-05 · [P1] SGLang 环境补充:A800 安装完成且驱动判决通过(历史 PyPI 死刑未复现);两台门 1 全过、待空卡跑门 2-4

**结论**:A800 也走通了。`/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/venv_sglang`
(python -m venv on env_vllm 解释器,腾讯镜像源装,~15 min):sglang 0.5.8 /
flashinfer 0.6.1 / torch **2.9.1+cu128** import 全过;**PyPI torch 真实矩阵乘在
驱动 525.60.13 上成功**(fp16 512² @ cuda:0,结果有限校验过)——infra 档案里
"A800 上 PyPI torch 不可用"的历史结论对现代 cu128 wheel 不成立(minor-version
compatibility 生效),档案将由 CD-006 更新注记。谨慎项:单核运算≠全栈,
sglang 起服(flashinfer/sgl-kernel/cudagraph 全 kernel 面)才是 A800 终审。

镜像源实测:A800 上 TUNA 也慢(nccl 322MB 卡十几分钟),**腾讯源
(mirrors.cloud.tencent.com/pypi)快一个量级**——执行册镜像建议按机器双选。

**当前态**:两台 venv_sglang 就绪、门 1 全过、零 LD_PRELOAD 依赖。门 2-4
(起服/往返/256宽+radix三断言/memory-saver)统一待空 GPU:162 四卡在 pgs_s1
(step~50)、A800 八卡在 pgs_base_a800(step~75)。base 自然完成约 +14h;
s1 是否提前停(已确认 θ_D 死锁版,让位 S2/SGLang 验证)待用户裁定。

## 2026-08-05 · [P0] SGLang PoC 门 2–4 通过:tail-only+radix 实测约 41×;旧探针断言是假红

**结论**:162 单卡 PoC 已完成,迁移的两个一票项均成立:top-256 可用、tail-only
打分结合 radix cache 带来约 **41×** 的实测降时。memory-saver 模式也已真实起服
并完成生成往返。因此允许进入完整 verl+SPS 冒烟。

**环境/门 1**(方案 B,独立 `/data3/.../venv_sglang`):sglang 0.5.8 / torch
2.9.1+cu128 / flashinfer-python 0.6.1 / sgl-kernel 0.3.21 / transformers
5.11.0;flash-attn 未装(该独立环境的单卡 SGLang PoC 不需要)。

**门 2**:A800(sm80)自动选择 `flashinfer` attention backend,无需 triton 回摆;
模型真实往返通过(`The capital of France is`→` Paris...`)。

**门 3 五个计时**(~2001 prompt token,top_logprobs_num=256):

| 请求 | latency | rows | top width | 首行 |
|---|---:|---:|---:|---|
| full-A | 2.981s | 2001 | 256 | null |
| full-B | 3.102s | 2001 | 256 | null |
| full-C | 3.068s | 2001 | 256 | null |
| tail-A | 0.075s | 8 | 256 | null |
| tail-B | 0.060s | 8 | 256 | null |

- 断言① top width 256:PASS;
- 断言② tail rows=8、首行 null:实测 PASS。服务器 `sgl_probe.py:29` 仍是旧口径
  `not first_null`,所以打印 `assert2=false` 是**探针假红**,不是引擎失败;请将脚本同步
  为环境册 2026-08-05 修正版并加“其余 7 行非 null”的显式断言;
- 断言③ tail-A < 0.6×full-C:PASS(0.075/3.068=0.0244,约 40.9×)。
- 证据:`/data3/wushang/workspace/TRACE-OPD-PGS/logs/sgl_probe_retest.out`、
  `logs/sgl_server_gate3retest.log`。

**门 4**:`--enable-memory-saver --mem-fraction-static 0.6` 起服、health 与真实往返
PASS。独立 venv 的 torch-memory-saver 子进程若从系统优先解析 CUDA runtime 会报
`libc10_cuda.so: undefined symbol cudaGetDriverEntryPointByVersion`;**有效修复是把
venv 内 `nvidia/cuda_runtime/lib` 前置到 `LD_LIBRARY_PATH`**。`LD_PRELOAD` cudart
无效。起服时 SGL kernel JIT 因系统 nvcc 不支持 `-std=c++20` 报错,但预编译/fallback
路径继续完成 cudagraph 并健康服务;正式环境需用 conda CUDA compiler 消除该噪声。
证据:`logs/sgl_server_gate4c.log`。

## 2026-08-05 · [P0] 用户调度已执行:base 以 A800 step-110 为锚迁到 162;新机制臂占 A800 八卡

**base**:A800 `global_step_110` 的 train 128/128、val 274/274 与 8 份模型/优化器
分片均完整;用户将原 step-80 迁移锚点改为 **step 110**。8→4 卡无法直接加载 FSDP
optimizer shard,故模型合并为 HF(7.6G)后传 162;`model.safetensors` SHA-256
`f0c0c9bf62a58f9a8947eb680fdf9520ffb82403d9d85f89f76ce0983a46e410`,两机一致。

**数据序列并未重头开始**:原 `global_step_110/data.pt` 也跨机迁移(SHA-256
`81092cd...cd7b`),通过“actor load_contents=[] + HF 模型路径”的桥接 checkpoint
恢复逻辑 global_step=110 与 StatefulDataLoader 状态。两机回放的 next-batch 索引完全
一致(`[1145,2535,1729,2284,1663,2829,28,1742,2709,885,881,110,942,2103,69,683]`),
且 162 实际 `step_00111` 的 16 个 task 集合逐项匹配。训练 parquet SHA-256
`61020b6a...fe31` 两机一致。optimizer/调度/RNG 的 FSDP extra state 因 world-size
变化未恢复;LR 为 constant,故 scheduler reset 行为无效应,但 Adam moments reset 必须
作为迁移口径披露。

**总步数**:dataset=2875、batch=16、drop_last=true⇒179 step/epoch、2 epoch=358。
162 设逻辑终点 358,实际继续 111–358 共 248 update。run:
`/data4/ws/PGS/pgs_base_162_from110`;启动后 step-110 val 274/274、step-111 128/128,
错误 0,已进入后续 step。

**新机制臂**:A800 base 已安全停机,八卡释放;代码已同步 `pgs-sglang@fa11dc8`,
S2 配置语义 = θ 分片修复 + pacing + B∞=400 + SGLang。按用户要求优先复用 conda
成熟训练栈,新建独立 `env_sglang_train`(不改 env_train/env_vllm),目标钉死
pytorch 2.9.1 + flash-attn 2.8.3 + SGLang 0.5.8;环境与全链路冒烟完成后从共同
`sft_step6` 从头跑 2 epoch。

## 2026-08-05 · [P1] A800 正式 S2/SGLang 已起跑；2-step 产物全过但 clean-exit 未过，显微镜实时服务已绑定正式 run

**正式 run**：`/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/runs/pgs_s2_sgl_a800`，
2026-08-04 23:21 +08 启动。实收无 manual override：8 GPU、SGLang 0.5.8、rollout
TP=1、actor Ulysses SP=2、adaptive θ_D、pacing、B∞=400、共同 `sft_step6`、2 epochs
（358 logical steps）、训前 val、save/test 每 10 步。入口下真实 torch/FA kernel 通过；八卡
初始化 `ready to fit`，训前 val 274/274 JSON-valid，首步轨迹已实时落盘，启动错误面为 0。

**2-step smoke 的判决要分两层**：
- 产物/训练链 PASS：2×128 train rollout、两次 actor update 和 SGLang weight update、
  8+8+8 shard 的 step-2 checkpoint（含 `data.pt`）、274 val 全齐；IS off-1 fraction
  约 3.73%/3.63%，steering/pacing 非零。
- clean-exit FAIL：上述产物全部完成后，DataLoader worker 被 signal 杀死，主进程记录
  RuntimeError，SwanLab 标 error。无 OOM、EngineDead、fatal NCCL。正式 run 长期监控已将
  DataLoader worker fatal 单列，不能把 smoke 写成干净退出。

**实时观察面**：远端只读显微镜 `127.0.0.1:8793`，绑定正式 run 并加载 2875 条 doc；
本地经同端口 SSH tunnel 访问。`/api/run`、`/api/step/step_00001`、
`/api/traj/step_00001/0` 已真实探活；逐轨迹返回目标、初始 observation、特权文档、
turn/token 与机制事件。旧 8792 服务仍绑定 base，未覆盖。


**前三步闭环（同 run Ray 原始 TaskRunner 输出）**：step 1–3 均 128/128 JSON-valid，三次
actor update 与 SGLang weight update 完成；`rollout_is_weight_off1_frac` = 3.876% /
3.654% / 3.907%，entropy = 0.669 / 0.641 / 0.680，actor max allocated = 24.88 /
28.76 / 31.11 GiB，整步 207.2 / 188.0 / 191.9s。θ_D 平滑更新 0.0338 → 0.03249 →
0.03077；平均 tilt 105.8 → 128.6 → 128.0，pacing wait 1105.6 → 1356.1 → 1478.2，
说明自适应阈值与 pacing 均在跨步真实生效。前三步零 Traceback/OOM/EngineDead/fatal NCCL/
DataLoader worker/RayTaskError；step 4 已开始。待核：step1/2 的 `doc_missing=0.0625`
（恰一组 8 rollout）而 step3=0，需定位是数据覆盖还是 task-id 对齐问题。

## 2026-08-05 · [P0·正式数据口径] pacing 已消除头部速烧；但 GT doc 仅 2828/2875 非空，47 个缺失且集中于 pick_two

**pacing 的首步位置实证（A800 正式 `pgs_s2_sgl_a800/step_00001`，128/128）**：
实际 `eta>0` 干预 13,545 次。相对全轨迹位置 P10/P25/P50/P75/P90 = 8.50% /
19.82% / 42.32% / 67.33% / 85.77%；仅 240/13,545 = **1.77%** 落在全局前
16 token，98.23% 在后续 turn。120/120 个有 doc 且发生干预的轨迹，最后一次干预都在
position 16 后；last-relative median **98.19%**（IQR 96.03–99.10%），last-turn median
27。turn0 每条恰 spend=5.0，141,511 次 pacing wait。相较旧 s1 全部干预堆前约16 token，
本正式实现已从行为层消除头部速烧（这是同机制前后事实比较，不作性能因果结论）。

**W_beh 双侧闭环**：rollout JSON 的稀疏行可重建 372,537 个 dense weight，mean=0.999953、
off1=3.173%、范围 0.1904–3.0449，schema/invariant 0 失败；Ray 原始 TaskRunner 训练指标
step1 `actor/rollout_is_weight_mean=0.9999419`、`off1_frac=3.8764%`，并有
`update_actor=28.46s`、`update_weights=4.67s`，故 loss-side delivery 已证实，不只是在 artifact
中“有字段”。

**P0 数据覆盖缺口**：doc table 2875 个 key 中仅 **2828 非空，47 空（1.6348%）**；
`pick_two` 占 38/674 = **5.638%**，不是随机缺失。step1/2 的 `doc_missing=0.0625` 各对应
恰一个任务组×8 rollout，key 均 exact-match doc table，但 doc 是空串；step3 缺失为0。
原 parquet 的 `gt_text` 非空却是 high-level ALFRED annotation，`gt.py` 有意只接受 Engine
walkthrough，因此安全降级为 nodoc——不是 task-id 键错或 SGLang 故障。

**对正式臂的影响与冻结处置**：这些样本按设计退化为 bare GRPO；step1/2 的有效 dose 覆盖
各120/128=93.75%，前三步合计16/384=4.167% rollout 无 treatment。不中途补 doc/改数据，
避免改变 treatment；run 继续。最终必须同时报告：① intention-to-treat 全量；② `doc_present`
分层；③按 family（尤其 pick_two）分层。原“2875 GT walkthrough 全覆盖”表述不成立，需机制/
数据侧后续补齐 47 条 Engine walkthrough 后另起干净对照，不得回填当前 run。

**其他首步健康信号**：thought 相邻逐字重复率轨迹均值0.539%、pooled 0.786%，仅2条非零且
均 lost，0条超过55%崩塌阈；无早期重复崩塌。全量证据路径：
`/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/runs/pgs_s2_sgl_a800/rollouts/step_00001/`；
doc table 同生产根 `data/gt_docs_rl_train_v2.jsonl`。

## 2026-08-05 · [P0·运行恢复] step-60 checkpoint 并发写失败；从完整 step 50 恢复，在线只保留 1 个 checkpoint

**故障分型**：正式 run 已完成 `step_00060` 的 128/128 train JSON，但在 2026-08-05
02:49 +08 保存 `global_step_60` 时，8 个 actor rank 均报
`PytorchStreamWriter failed writing file ...: file write failed`，随后 Ray 级联正常退出。残缺目录仅有
8 个 model shard（5,516,867,328 bytes），无 optimizer/extra/`data.pt`/HF metadata；latest marker
仍为 50，故 step 60 **不是可恢复 checkpoint**。无 OOM、NCCL、EngineDead、DataLoader worker；
八卡已释放且 `nvidia-smi` 正常，故不是 GPU/驱动故障。

**存储证据与处置边界**：`/fs/fast` 全局仍余约39T、inode 使用6%，同目录128MiB写入+fsync
探针通过，无法把根因严格写成“全盘耗尽”；更符合 Lustre 八 rank 并发大文件写的瞬时失败或配额/
目录容量压力。故障时在线 checkpoint 占约140GiB（三份完整30/40/50 + 残缺60）。按用户裁定，恢复后
`train.max_ckpt_to_keep=1`；该键不进 objective fingerprint，指纹仍为 `52431755eb79`，优化目标
未变。

**保全与恢复**：删除在线旧 checkpoint 前，完整 step 50 已复制到本机独立盘
`/disk/trace-opd-recovery/pgs_s2_sgl_a800/global_step_50`；31文件、48,286,310,244 bytes，
源/副本逐文件 SHA-256 全相等。第一次 attempt 的 step 51–60（10×128 JSON）与原始主/Ray日志
已归档到 `/disk/trace-opd-recovery/pgs_s2_sgl_a800/attempt1_failed_step60_20260805T024918+0800/`；
活跃 rollout 51–60 移出后再重跑，防 UUID 文件叠加污染128条口径。在线只保留合法
`global_step_50`，marker=50；以 `resume_mode=auto`、`max_actor_ckpt_to_keep=1` 重启。03:19日志
明确 `Resuming from .../global_step_50, setting global step to 50`，8 rank model/optimizer 已加载，
03:20 新 `step_00051` 开始产出。

**恢复闭环（03:31 +08）**：重新生成的 step 51 任务多重集与失败前完全相等（16 task×8），
证明 `data.pt` 游标连续；128/128 JSON-valid，W_beh off-1=11.42%、actor update=22.43s、
weight sync=4.63s、artifact全局门率10.94%、θ_D=0.0338、pacing位置P50=48.59%，错误面全零。
step 60 随后形成 train 128/128、val 274/274、checkpoint 31文件/48,286,310,244 bytes且
marker=60；save checkpoint=8.48s，无新写盘/fatal。step60 W_beh off-1=11.37%、entropy=0.0649、
θ_D=0.02873、artifact全局门率10.25%、pacing位置P50=49.98%、thought重复率0；step61也已
128/128。由于恢复进程的轮转注册表不包含旧step50，本次在验证step60与`/disk` step50灾备后
人工删除在线step50；Lustre现仅保留完整`global_step_60`约45GB。后续同进程会按
`max_actor_ckpt_to_keep=1`在新点成功注册后安全轮转旧点。

恢复时 `val_before_train=true` 额外重跑同一step50 checkpoint：原正式端点216/274，重跑
221/274；两组恰各274、task集合相同。第二组已单独归档到
`/disk/trace-opd-recovery/pgs_s2_sgl_a800/resume_20260805T0315/val_before_train_step_00050_val/`，
活跃目录只保留原274条标准端点。该5题差异再次印证greedy评测非逐字确定，不能混成548分母。

**已知恢复语义缺口**：model、optimizer、LR scheduler、RNG 与 StatefulDataLoader 均从 step 50
恢复；但 worker 内 `GateThresholdTracker` 的 adaptive θ_D 状态目前不进 checkpoint，重启会从配置
初值0.0338重新收敛，而 step 50 实收均值为0.04924。不得伪造状态；恢复后须优先监控前几步
θ_D/门率，并请 CD-010 将自适应门控状态纳入后续 checkpoint 设计。恢复最终闭环需同时满足：
step 51 128/128 + actor/W_beh 指标正常，以及下一次 step-60 checkpoint 在单保留策略下形成
8 model + 8 optimizer + 8 extra + `data.pt` + HF metadata、marker=60。

## 2026-08-05 · [P0·预注册安全停机] step-70 固定 val 93.07%，但 entropy 连续 <0.2；按安全门停于完整 checkpoint

**执行结论**：`docs/pgs-validation-design.md:48-49` 预注册“熵持续跌破0.2 ⇒ 停机保 ckpt，
按崩塌处置”。A800 正式臂 entropy 在 step30/40/50/60/70 连续为 **0.1791 / 0.1078 /
0.0620 / 0.0649 / 0.0389**，已明确触发；不能因 val 上升而事后修改门槛。2026-08-05
03:45 +08 在完整 `global_step_70`（31文件、marker=70）向精确 PGID 71516 发送TERM；17秒内
训练/Ray/SGLang进程组干净退出，八卡显存归零。无新 Traceback/OOM/NCCL/EngineDead/
DataLoader worker/写盘失败。显微镜 `127.0.0.1:8793` PID50055保留；旧watchdog PID101534
已停止，避免把有意停机误报为故障。

**step-70 端点**：固定274-task val = **255/274=93.07%**（seen 132/140=94.29%，unseen
123/134=91.79%），相对step0 26.64%为+66.42pp；invalid=0、truncated=0、平均响应503 token。
by-family：look_at 26/31、simple 57/59、clean 56/58、cool 45/46、heat 33/39、pick_two
38/41。thought逐字重复率均值0.24%、pooled0.64%、0条>55%，随性能上升而下降。训练批
step70为128/128全赢，grad norm=0，说明该批GRPO组内零advantage；W_beh仍送达loss（off-1
8.32%），artifact全局门率7.19%/Ray轨迹均值7.20%，θ_D=0.02678，pacing位置P50=47.8%，
机制健康门仍在带内。checkpoint保存15.1s，单保留策略成功滚动到step70；旧step60 actor被自动删但
遗留`data.pt`空壳，验证step70后已精确删除，在线仅剩完整step70。

**解释边界**：执行口径按预注册必须停，但现象并非此前“输轨迹高重复+val崩”指纹：这里val持续
上升、长度缩短、invalid/truncation与重复率归零，低熵可能是能力收敛/任务饱和，也可能是过度
确定化的前置阶段。当前只称“entropy安全门触发”，不将机理结论提前写死；step70需以多次固定val、
同数据序base与行为多样性进一步裁定。停机前step71–74各128 JSON也完整，但无更新checkpoint，
不得作为可恢复模型点；恢复锚只认70。

## 2026-08-05 · [监控] A800 step-70 安全停点保持；162 base 完成 step-250 端点

A800 审计确认未被意外重启：最新train仍为step74（128/128）、val为step70（274/274），在线唯一
`global_step_70`仍为31文件/48,286,310,244 bytes、8 model+8 optimizer+8 extra+`data.pt`+
HF metadata，marker=70；无基础设施fatal，八卡显存0，显微镜8793在线。该状态是预注册entropy门
触发后的有意停机，不作自动恢复。

162 base 已封闭到train step251，step252继续产出；val/checkpoint marker=250均完整（274条，4+4+4
shard及`data.pt`）。step250固定val=250/274=91.24%（seen132/140、unseen118/134），近期
step220/230/240/250为90.88%/93.07%/92.70%/91.24%；单点-1.46pp不触发>10pp安全门且在greedy
方差范围内。entropy=0.388、grad norm=0.0782，错误面全零。`/data4`余5.2T；`/data3`仍余
478G（97%使用）列容量预警，但无证据支持清理。

## 2026-08-05 · [监控] 162 base step-270 高点回落，不是相对近期基线的连续崩塌

A800 step-70安全停点无变化：train尾仍74、val/marker仍70、唯一checkpoint31文件完整、错误面
全零、八卡显存0，仅显微镜8793在线；无新事实，不自动重启。

162 base已封闭到train273（step274继续）、val/checkpoint270；4+4+4 shard、`data.pt`、HF/FSDP
metadata完整，marker270，错误面全零。固定val step250/260/270为91.24%/95.62%/89.78%；
260→270表面−5.84pp且同任务3升/19降（配对`p=0.000855`），但250→270仅−1.46pp、9升/13降
（`p=0.523`），故更符合step260异常高点回落，不是相对近期基线的持续退化。回落主要来自
look_at 24/31→16/31、pick_two 40/41→35/41、heat 38/39→35/39；clean仍58/58，cool升至
46/46。step270 entropy=0.347、grad norm=0.0492，未触发低熵或>10pp安全门；继续运行并以step280
确认连续性，不干预。

## 2026-08-05 · [监控] step280–300 否定 base 连续退化，固定 val 恢复至94.53%

A800安全停点保持不变：train尾74、val/marker70、唯一31文件checkpoint完整、错误面全零、八卡
显存0，仅8793显微镜在线；未被意外重启。

162 base已封闭到train301（step302继续）、val/checkpoint300；最新三个checkpoint 280/290/300
均19文件（4 model+4 optimizer+4 extra+`data.pt`+HF/FSDP metadata），marker300，错误面全零。
固定val step270/280/290/300依次为89.78%/92.70%/93.07%/**94.53%**；step300 seen
136/140、unseen123/134，invalid/truncated均0。相邻同任务配对净变化为+2.92/+0.37/+1.46pp，
250→300为+3.28pp（17升/8降，`p=0.108`）。故step270回落是单点波动，后续三点明确恢复，
不支持连续退化。entropy 0.305/0.294/0.362，grad norm均非零；不触发安全门，继续到358。

## 2026-08-05 · [监控] 162 base 末段 step330 仍稳定高位，距终点24步

A800 step-70安全停点仍无变化：train尾74、val/marker70、唯一完整checkpoint，错误面全零、八卡
显存0、仅8793显微镜在线；不自动恢复。

162 base已封闭到train334（持续产出）、val/checkpoint330；最近checkpoint 300/310/320/330均
4+4+4 shard、`data.pt`、HF/FSDP metadata完整，marker330，错误面全零。固定val
step300/310/320/330为94.53%/95.26%/96.35%/95.62%；相邻配对净变化+0.73/+1.10/−0.73pp，
均不显著，末段稳定高位。step330 seen135/140、unseen127/134，invalid/truncated各1；thought
重复率均值1.17%、pooled3.19%、仅1条>55%，继续下降。entropy 0.315/0.293/0.254仍高于0.2，
grad norm均非零；无安全门。距逻辑终点358尚24步，继续运行并在最终train/val/checkpoint齐全后验收。

## 2026-08-05 · [终验] 162 base 完成 step358：最终96.72%，产物全齐；clean-exit因DataLoader尾错失败

A800安全停点仍无变化：train尾74、val/marker70、唯一31文件checkpoint完整、错误面全零、八卡
显存0、仅8793显微镜在线；不自动恢复。

**162 base产物级完成**：`step_00358` train 128/128、`step_00358_val` 274/274；
`global_step_358`共19文件/48,283,717,276 bytes，4 model+4 optimizer+4 extra，含`data.pt`、
HF/FSDP metadata，latest marker=358。主训练与四卡服务已退出，显存约14MiB/卡。

**最终固定val**：step358 = **265/274=96.72%**（seen136/140=97.14%，unseen129/134=
96.27%），invalid=1、truncated=1。相对迁移锚step110的240/274=87.59%提升**9.12pp**；同任务
31升/6降，配对二项`p=4.13e-5`。末段step330/340/350/358为95.62/96.35/95.62/96.72%，
已在高位平台。by-family：look_at30/31、simple58/59、clean57/58、cool45/46、heat37/39、
pick_two38/41。step358 entropy=0.267、grad norm=0.0205；最终val thought重复率均值1.23%、
pooled3.70%、3条>55%，无低熵或val下跌安全门。

**clean-exit必须单列FAIL**：日志顺序为358/358完成→最终checkpoint→274 val及final metrics打印→
SwanLab atexit `finish()` 等待期间报`DataLoader worker ... killed by signal: Killed`，产生1个
Traceback/RuntimeError。无OOM/NCCL/EngineDead/RayTaskError/写盘失败。故准确判定为“训练/
评测/checkpoint完成，clean-exit失败”；不重启、不补跑。完整报告：
`docs/pgs-base-162-from110-final-report.md`。

## 2026-08-05 · [维护复核] A800最新更新为step74但恢复锚仍70；162终点未漂移，旧点空壳范围修正

A800产物级复核：step1–74的train目录均各128条且JSON-valid，最新固定val仍step70的274条，
唯一checkpoint仍`global_step_70`（31文件/48,286,310,244 bytes，8 model+8 optimizer+8 extra+
`data.pt`，marker70）。Ray原始指标证明停机命令落地前step71–74均已完成actor update和SGLang
weight update，而不只是生成轨迹；step74 entropy=0.0507、grad norm=0.799、W_beh off-1=9.03%。
但这四步无checkpoint/固定val，故最新完成更新为74，唯一可恢复且有固定评测的锚仍只认70，绝不
自动恢复或把74伪装成可恢复点。step70训练artifact：门率7.19%、θ_D=0.02678、pacing干预位置
P50=47.82%、相邻thought exact-dup均值0.078%/pooled0.102%/0条>55%；固定val重复率为
0.237%/0.643%/0条>55%。低熵门仍触发，未见复读型崩塌。恢复attempt的Ray错误面全零；首次
attempt仅保留已知step60写盘失败。八卡显存0、`/fs/fast`余39T，无恢复动作。

162 base终点未漂移：step358 train128/128、val274/274、完整四卡checkpoint19文件/
48,283,717,276 bytes、marker358，`data.pt` SHA-256仍为
`82f463108b484bfeb014b4f25ece47d6a5458f4088aad2e1e260c17b16c0525c`；四卡显存14MiB，无训练
进程，旧DataLoader尾错仍是唯一fatal。维护审计修正一项报告细节：step110–320的历史轮转目录均
只剩`data.pt`空壳，不只是310/320；完整可恢复点为330/340/350/358。这不影响marker358与终点
结论，但checkpoint枚举必须按shard+`data.pt`完整性，不可只按目录名。

## 2026-08-05 · [162·1.7B正式臂] 隔离SGLang训练栈通过全门；正式run已从原始base干净启动

按用户裁定，以A800 4B正式`pgs_s2_sgl_a800/effective_config.yaml`为模板，在162从原始
`/data3/wushang/model/Qwen/Qwen3-1.7B`启动GT特权信息S2臂。新环境与代码均隔离在
`/data4/ws/PGS/`，未修改旧`venv_train/venv_sglang`：torch2.9.1/CUDA12.9、flash-attn2.8.3、
SGLang0.5.8、Transformers5.11.0、Ray2.56.1、TensorDict0.13.0。代码bundle精确同步A800正式
臂8个SGLang/pacing/adaptive-gate闭环文件；正式配置hash
`7f3ed56c5305a435f4e69e2580a713e57446d4a71ee0cc7cd793b54e710c7d4c`。

启动门真实通过：单卡matmul+FA kernel；双卡NCCL SUM=3；SGLang memory-saver普通生成；
145-row full top256=0.215s、8-row tail=0.0397s（5.41×，首行None、后7行每行256）；27项
adapter/steering测试；正式dry-run。首次JIT发现nvcc默认误用系统GCC9导致`<concepts>`缺失，最小
CUDA C++20复现证明`NVCC_PREPEND_FLAGS=-ccbin=<conda g++>`是充分修复，固化后SGLang起服不再
fallback。GT表重核2875唯一key、2828非空/47空，hash
`25367813d29477f9f5b4695016bb032a1b72bb88c2d84f4117129ae96a1a0dbd`；不回填。

首次actor smoke在Ray TaskRunner导入期因环境闭包漏`cachetools`退出，未起GPU更新/rollout，
失败证据保留。按A800完整Python分发差集补齐后，v2 smoke形成step1 128/128、完整四卡checkpoint
19文件/20,660,399,665 bytes、marker1；actor update24.96s、SGLang weight update3.51s、grad
norm0.466、entropy0.0217、W_beh mean1.00003/off-1=2.67%。artifact门率2.74%、干预10,522、
θ_D=0.0338、pacing位置P50=42.82%、doc_missing=8/128。全部产物后DataLoader worker在atexit被
signal Killed，判“端到端门通过/clean-exit失败”，不复用smoke checkpoint。

1.7B smoke相邻thought exact-dup均值36.10%、pooled39.19%、45/128条>55%；既有1.7B base早期
基线约45.7%，故不能机械套4B绝对重复阈值，正式前三步需重新定标。正式run
`pgs_s2_sgl_1p7b_162_gt`于10:42 +08从不存在的干净目录启动，PID/PGID=3707792；fingerprint
`e9ef2835e0af`，manifest `overrides=[]/config_overrides=[]`，明确实收原始1.7B、SGLang TP1、
actor SP2、16×8、2 epochs、每10步val/save、`B∞=400`、adaptive θ_D与pacing，无smoke限制
泄漏。主日志与运行卡：`/data4/ws/PGS/trace_opd_1p7b_s2_bundle/logs/pgs_s2_sgl_1p7b_162_gt.log`、
`docs/pgs-s2-1p7b-162-run-card.md`。

正式训前val已274/274：14/274=5.11%（seen6/140、unseen8/134），invalid6、truncated2；
by-family仅simple10/59和cool4/46非零，其余四族为0。相邻thought重复率均值41.08%、
pooled42.72%、118/274条>55%，thought字符中位98，这是本基座正式零点。正式step1随后128/128并
完成更新：22/128 won、invalid9、truncated3、entropy0.02406、grad norm0.592、W_beh off-1
2.55%，actor update22.54s、weight update3.37s；artifact门率2.63%、干预9394、θ_D=0.0338、
pacing P50=42.68%、doc_missing8/128，重复率均值36.56%/pooled40.87%/44条>55%。错误面全零，
step2已开始。高重复与低熵都已在训前零点存在，不能把step1单点定性为训练诱发崩塌；前三步继续
看相对漂移与adaptive θ_D是否从初值移动。

## 2026-08-05 · [P0·预注册安全停机] 1.7B原始base在step10连续极低熵；val早期上升但不覆盖“任一触发”门

**执行结论**：162正式run `pgs_s2_sgl_1p7b_162_gt` 已在首个完整step10端点按
`docs/pgs-validation-design.md:48-49`受控停机。step1–10 entropy依次为0.02406/0.01759/
0.02216/0.02092/0.01770/0.02135/0.01800/0.02216/0.02062/0.02294，十步均值0.02075、
最大0.02406，连续远低于预注册0.2门；现行规则是thought重复>55%、entropy持续<0.2、val单点
跌>10pp任一触发即停。不能因本次val上涨而事后撤销安全门。约11:12 +08向精确训练PGID
3707792发送TERM，并停止只读watchdog PGID3838700；新连接复核两组均为空、四卡14MiB/0%，
错误面无Traceback/OOM/NCCL/EngineDead/RayTaskError/DataLoader worker/写盘失败。

**step10固定端点**：val **23/274=8.39%**（seen13/140、unseen10/134），相对训前
14/274=5.11%为+3.28pp；同任务12升/3降、paired exact binomial `p=0.0352`，invalid=6、
truncated=2、平均响应2004 token。family为simple14/59、clean3/58、cool6/46，其余look_at/
heat/pick_two均0。固定val相邻thought exact-dup从训前均值41.08%/pooled42.72%/118条>55%
改善到35.14%/37.26%/99条>55%，字符中位98→95；没有复读恶化指纹。故准确结论只能是
“entropy安全门触发”，不能先写成训练诱发的复读型崩塌；+3.28pp也只是单次greedy早期正信号，
受274×1功效和greedy非逐字确定限制，尚非机制有效定论。

**step10机制/更新**：train14/128 won、invalid3、truncated1；entropy0.02294、grad0.5481、
W_beh mean1.000108/off-1=3.274%；actor update22.16s、SGLang weight update3.12s、整步
111.14s。artifact全局门率3.287%、干预12,032、θ_D=0.014843（worker 0.014833–0.014854），
相对step1的0.0338持续下降，adaptive tracker活着但门率仍远低10%目标；pacing P50=42.29%。
step10 train重复率均值26.60%/pooled28.50%/32条>55%，字符中位94。

**checkpoint与不可恢复尾部**：`global_step_10`为19文件/20,660,399,665 bytes，4 model+
4 optimizer+4 extra+`data.pt`+HF/FSDP metadata，marker10；`data.pt` SHA-256=
`85a92d66881a57a4dedc4d3eb0e788ef7886f2d02cd4ed3eeb1ae8fe9a110632`。adaptive θ_D状态仍不进
checkpoint，未来resume会从0.0338而非0.014843重收敛，因此这只是模型/optimizer/RNG/数据游标
恢复锚，不是机制状态逐位连续锚；本轮不自行恢复。

停机命令落地前流水线已完成step11–14的actor和weight update（不是只有rollout），其entropy仍为
0.02069/0.02515/0.02744/0.02298；四步各128 train dump但无val/checkpoint，step15仅9/128。
最新完成参数更新为14，但唯一有固定val且可恢复的科学端点只认10。**P0机制/运维请求**：外部
十步产物审计产生约4个update的停止延迟；后续应把安全门的连续性状态、端点判决和停止动作内置
trainer/checkpoint边界，至少在step10 save+val后阻止step11 dispatch。同时将adaptive θ_D状态纳入
checkpoint；若要把0.2 entropy门改为按模型族/训前基线定标，必须新实验前预注册，不能追溯修改本run。
正式数字与证据路径见`docs/pgs-s2-1p7b-162-run-card.md`。

## 2026-08-05 · [维护复核] A800安全停点与162 base终点均无漂移；不恢复、不补跑

A800 `pgs_s2_sgl_a800`：train step1–74均128个非空JSON-valid，固定val step0–70均274个
非空JSON-valid；最新完成actor/SGLang更新仍为74，最新val/checkpoint/marker仍为70。
`global_step_70`共31文件/48,286,310,244 bytes，8 model+8 optimizer+8 extra+`data.pt`+
HF/FSDP metadata；`data.pt` SHA-256=
`6bd547bc04260670cd3308d777e1d659afa590103be70301fd1c27cfd436f682`。step74为115/128 won、
entropy0.0507、grad0.799、W_beh off-1=9.03%；artifact门率1505/16772=8.97%、干预2143、
θ_D均值0.02196（worker 0.00733–0.03545）、pacing P50=50.0%，thought重复率均值0.108%/
pooled0.346%/0条>55%。机制链路仍活且无复读恶化，但连续低熵预注册门仍成立；74无val/ckpt，
恢复锚只认70。训练PGID71516、watchdog PGID101534均无成员，PID文件只是陈旧记录；仅显微镜
50055监听127.0.0.1:8793。八卡0MiB/0%，`/fs/fast`余约38.37TiB；恢复attempt无新增fatal，
历史step60写盘失败仍单列。不得自动恢复。

162 `pgs_base_162_from110`：step358 train128/128、val274/274、checkpoint19文件/
48,283,717,276 bytes、marker358、`data.pt`哈希均不变；完整点仍仅330/340/350/358。最终val
265/274，train127/128 won；唯一Traceback仍是358/358、checkpoint和val全部落盘后的SwanLab
atexit/DataLoader worker尾错，OOM/EngineDead/RayTaskError/PytorchStreamWriter/写盘失败均0。
无training/verl/rollout进程，四卡14MiB/0%；`/data4`余约5.63TB，`/data3`使用97%/余约512GB，
仅列容量预警，无证据支持清理。本run维持已完成/clean-exit失败，不重启、不补跑。

口径澄清：step358 val的`invalid_format` scalar为事件计数3/274，但三次事件来自同一条轨迹；
含invalid轨迹仍是1/274。详细证据见`docs/pgs-s2-a800-run-card.md`与
`docs/pgs-base-162-from110-final-report.md`。

## 2026-08-05 · [P0·组合缺口] 最新PGS/SGLang没有动态researcher文档入口；新A800正式臂须补纯producer闭环

**结论**：用户要求以A800 4B `sft_step6`为共同起点、保持最新PGS/SGLang机制和训练设置，仅把GT
静态文档替换为远端`qwen3.6-35b-a3b` researcher按当前任务/当前policy动态产出的文档。实际代码核验
证明这不能只靠现有配置完成：`pgs-sglang@fa11dc8`的`agent_loop.py`只在启动时加载
`steering.doc_table`，每样本按`task_id`查静态表；launcher只预检表文件；实现设计也明确第一版删除
`_research*`、endpoint/tunnel、research配置和归档闭环。因此用户记忆中的动态配置确实存在，但真源
在旧`mixed-objective`/PGC producer链，当前没有“动态researcher → PGS SPS”的可执行组合入口。

**函数级合同与最小变更边界**：旧动态链按`(global_step, task_id)`经`shared_task`只运行一次，n=8
副本共享文档；`ResearchSession`通过external-pool client和fncall `ToolRouter`用当步冻结policy跑独立
ALFWorld实验，`ResearchArchive`落会话、实验轨迹、`doc.md`和policy_version lineage；researcher只读
任务brief，`gt=off`，远端角色必须显式
`extra_body.chat_template_kwargs.enable_thinking=false`。PGS侧应只复用这套**文档producer**，在
`run()`开始student episode前得到doc并写入现有`steer_ctx`；随后完全复用既有
`_teacher_prompt_ids → SpsGenerate → SteeringLedger → rollout_is_weights → 原生GRPO`链。不得恢复旧OPD
teacher loss、mixed-SFT、research-RL或PGC critic，不得让doc进入bare actor scaffold，不得改tail-only
top-K、adaptive θ_D、pacing或W_beh语义。

**必须接受的调度语义**：旧OPD/PGC可让research与bare student并行，因为doc只在rollout后的teacher/
critic路径消费；PGS在每轮draft–verify–resample开始前就需要doc，故同一任务的当前student不能与其
producer完全重叠。正确语义是step内policy权重冻结，先由独立research episodes产doc，再启动该任务的
8条共享doc SPS rollouts；不得用run前一次性预生成JSONL冒充逐step动态刷新。per-sample research失败可
降级为空doc但必须计数、归档并保持全1空账；endpoint/model/auth等系统性错误须launcher preflight
fail-loud。

**A800正式门禁**：以旧GT run远端`effective_config.yaml`作全量基线，仅增加researcher来源配置并新建
run目录；共同初始policy仍为`/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/ckpt/sft_step6`。
启动前须验证双隧道`/v1/models`模型身份、凭据仅来自`TRACE_RESEARCHER_API_KEY`、non-thinking真实响应、
fncall工具往返、动态doc/lineage/tape落盘、n=8只生成一份step-task文档、actor保持bare、SPS有干预、
W_beh off-1>0、actor update和SGLang weight update完成。旧A800 GT run与162 1.7B full run均不得覆盖、
恢复或中途改变treatment。

**准备证据（同日）**：A800旧GT正式臂的源YAML SHA-256为
`60d16a5b5c9c300da22f2ee6d80353040c96d49cd9ac7b0060ece449ceff18f7`；初始manifest明确
`config_overrides=[]/overrides=[]`、fingerprint=`52431755eb79`，实收`sft_step6`、SGLang TP1、actor
SP2、16×8、`grpo_loo`、`norm_adv_by_std=false`、`seq-mean-token-mean`、LR 1e-6 constant、2 epochs、
每10步val/save及完整S2 steering。新臂须逐字段复制该manifest；存储策略沿已知写盘事故后的用户裁定从
起跑即`max_ckpt_to_keep=1`，该键不进objective fingerprint。

A800→162外层隧道已由正式wrapper恢复，watchdog及两个精确SSH转发进程均存活；A800 loopback
18888/18889真实监听，鉴权`/v1/models`均HTTP 200且唯一模型ID精确为`qwen3.6-35b-a3b`。两个端点
分别做真实OpenAI function-call探针：显式`chat_template_kwargs.enable_thinking=false`后均
`finish_reason=tool_calls`、恰调用指定无副作用工具一次、`reasoning_content`长度0、普通content长度0。
无鉴权请求按预期401；正式凭据值未进入命令、文档或输出，只沿既有wrapper受控变量链使用。162内层
18888/18889也有真实监听及两个SSH转发PID，故无需重启内层服务。当前只完成endpoint层，不等于动态
ResearchSession或PGS训练链验收；后两者仍以完整doc归档和actor update为门。

## 2026-08-05 · [P0纠正] 1.7B旧run是执行侧误停而非实验故障；已另起干净full run推进358步

`pgs_s2_sgl_1p7b_162_gt`停止由CD-006主动TERM造成，不是162、SGLang、Ray或训练异常。执行侧在
step10端点错误地把4B的`entropy<0.2`绝对门跨模型套到原始1.7B；但该基座训前entropy本来约
0.02，阈值未做跨模型校准，且用户目标明确是完整运行。故该处置纠正为**操作性误停**。旧run与
step10 checkpoint完整保留作为证据，不删除、不自动resume。

不选择step10 resume的科学原因：adaptive θ_D worker状态未进checkpoint，恢复会从0.0338而不是
0.014843重收敛；TERM落地前step11–14已完成更新但无checkpoint，resume会回退并重放数据，不能
称连续实验。已使用同一原始Qwen3-1.7B、GT表、代码和机制配置，从不存在的新目录启动
`pgs_s2_sgl_1p7b_162_gt_full`，目标358。配置/GT hash仍为
`7f3ed56c…d4c`/`25367813…0dbd`，fingerprint仍`e9ef2835e0af`，manifest
`overrides=[]/config_overrides=[]`，PID/PGID=`102239/102239`。dry-run生成的3文件空目录已原子
改名留证，未覆盖/删除训练数据。

新run已完成274-task训前val并SwanLab online：13/274=4.74%（seen5/140、unseen8/134），
invalid/truncated均0；thought重复率均值38.08%、pooled39.22%、105条>55%，与旧run训前14/274
及高重复基线同量级。四卡已真实起载，SGLang/Ray错误面全零并进入step1。后续低entropy只记录，
**不得再用未校准的4B绝对0.2阈值自动中止这个1.7B full run**；真实fatal/写盘失败/产物停滞仍按
证据诊断。运行身份和后续端点持续写`docs/pgs-s2-1p7b-162-run-card.md`。

## 2026-08-05 · [P0已闭环/P1调度修复] dynamic researcher真实训练smoke通过；双端由哈希近似改为worker固定8/8

**代码与配置闭环**：在`pgs-sglang@fa11dc8`上只接入纯文档producer；本地主会话独立复现
`176 passed, 3 skipped`（含后续双端分片回归），compileall与`git diff --check`通过。A800使用独立bundle
`trace_opd_researcher_pgs`，未覆盖旧GT生产树；关键文件同步前后SHA-256一致。正式dynamic配置以旧GT
实收effective config逐字段闭合，除doc source/research合同、用户明确裁定的`n_exp=16`、
`rollout_max_num_seqs=256`、外层timeout=5400s和只保留1个ckpt外，4B `sft_step6`、SGLang TP1、actor
SP2、16×8、`grpo_loo`、no-std、seq-mean、LR 1e-6 constant、S2 SPS/W_beh和2 epochs均不变；新正式
objective fingerprint=`976d84cfd3a3`。dry-run不联网/不建目录，正式双endpoint 2/2、认证和模型身份
`qwen3.6-35b-a3b`已真实通过；secret值未落effective/manifest/tape，仅变量名入档。

**单任务真实smoke**：原样复制正式parquet首行，保持`rollout_n=8`，只把research预算缩到2次用于技术门禁。
真实链为`run_experiment → fetch_raw×多次 → finalize`，13轮35B fncall均error=0；ResearchArchive记录
`global_step=policy_version=1`、`gt=off/gt_injected=false`、2 experiments/4 current-policy episodes、
1507词doc、finalized且error=null。8条student rollout共享唯一archive和完全相同research字段，
`research_error=0/doc_missing=0`；SPS实际steer count均值162.875、W_beh off-1=3.83%，actor update
12.00s、SGLang weight update 4.53s、step 1/1完成。退出后唯一Traceback是已知atexit DataLoader worker
Killed清理签名，发生于8/8产物、actor/weight update和进度100%之后；不是中途训练故障。

**并发口径与修复**：`roles.researcher.max_concurrency=64`只约束外部35B HTTP，正式16个session的自然峰值
约16（会话内轮次串行）；policy侧每次`run_experiment(k)`并行`k`条ALFWorld episode，正式k默认4/上限8，
16 session常态约64、峰值128条current-policy episode。SGLang TP1×8 replica，用户裁定每replica
`max_num_seqs=256`。原pool的least-active仅进程内可见，跨16 worker退化为task CRC32；真实首16任务预计算
为12/4，不能声称负载均衡。已改为从verl Ray actor名`agent_loop_worker_<index>_*`取worker index，固定
`index % 2`选择首端点，正式结构性8/8；session仍粘性、坏端仍failover。新增16-pool 8/8、粘性和failover
回归，本地全量176/3通过，A800生产环境聚焦43 passed。正在以正式16任务、`n_exp=16`、k=4..8、
max_num_seqs=256跑1-step门禁；放行条件包括16份archive、128 student轨迹、双端实际8/8 session、
无secret、SPS/W_beh/actor/weight update及错误面全零。

## 2026-08-05 · [P1已修复/门禁实证] 16任务完整预算单步通过；修正fncall启动门与真实policy版本lineage

**16任务完整预算门终态**：A800独立run `pgs_s2_sgl_a800_researcher_gate1`使用正式parquet
前16个唯一task、16 workers、n=8、`n_exp=16`、k=4..8、`max_num_seqs=256`，仅关闭val/save/
SwanLab并限1 step。终态为16/16 ResearchArchive、16/16非空doc、128/128 student；research
错误和doc overflow均0。每task同一步n=8共享唯一archive。双端实际16个session按worker固定8/8，观察期
21次35B调用11/10、error=0且无异常跨端failover。actor entropy=0.6801，SPS steer count均值
155.66、门率16.91%、预算支出354.05，`W_beh` off-1=3.327%；actor update=37.16s、SGLang
weight update=4.67s、完整训练step=2171.53s。唯一Traceback仍是进度1/1与weight update完成后的
DataLoader worker atexit Killed；OOM/EngineDead/RayTaskError/写盘错误均0。因此用户裁定的两项吞吐配置
已通过真实一整步，不再重复16任务门。

**P1纠正一（启动门）**：旧launcher只查双端`/models`，不能证明正式treatment所需non-thinking和
OpenAI fncall可用。现已在创建run目录前逐endpoint执行实际`extra_body`下的两轮无副作用协议门：精确
`trace_preflight_echo(nonce)` tool call → canonical assistant history → tool result回传 → 合法
`PREFLIGHT_OK` follow-up；拒绝`reasoning/reasoning_content`、可见`<think>`、坏参数、多/少tool call或
坏follow-up，任一端失败则N/N fail-closed。mock回归与A800双隧道2/2真实往返均通过，credential值不进
消息、日志或产物。

**P1纠正二（lineage）**：历史单任务smoke与16任务门把逻辑batch `global_step=1`同时写成
`policy_version=1`，但首批research/student真实使用初始化后SGLang version 0；这是归档provenance偏一，
不是生成或actor update错用权重。旧条目中的`global_step=policy_version=1`仅保留为当时记录，**不得继续
解释为真实权重版本**。现修复为：LLM server返回的`global_steps`经`min/max_global_steps`贯穿每次
research episode；写入trajectory meta和`episodes.jsonl`；归档前要求所有非ENV_ERROR episode均有版本、
单episode不跨版本、整个session唯一版本，`meta.json`分别写逻辑`global_step`与真实`policy_version`。
SPS对所有draft和verify调用取真实min/max而非首次setdefault；student完成后要求其版本区间单点且等于
research版本，否则fail-loud不入训。无实验却产doc也因无版本证据降级bare。正式rollout dump另增
`policy_version.{min,max}`审计字段。轻量版本回归、producer/SPS相关78项、本地全量180 passed/3 skipped、
A800同步回归3项及compileall均通过；正式objective fingerprint保持`976d84cfd3a3`。

**正式在线上报门**：用户要求正式run开启SwanLab远端同步。formal wrapper现要求真实启动时
`SWANLAB_API_KEY`非空且`SWANLAB_MODE`不得为offline/disabled，否则在联网/建run目录前拒绝；dry-run仍
无副作用。A800入口环境实测key存在且online允许，正式命令实收`trainer.logger=["console","swanlab"]`，
无overrides，目标目录仍不存在。运行中的SwanLab run禁止执行`swanlab sync`。修复后单任务version gate
正在独立PGID运行；其唯一放行目标是实证首批archive为`global_step=1, policy_version=0`并完成8条student、
actor update和weight update，尚未完成前不得写成通过或启动358步正式run。

## 2026-08-05 · [正式启动] A800 4B dynamic-researcher 358步长跑已上线，SwanLab online与只读watchdog双验收

修复后lineage gate已完成：唯一archive精确`global_step=1, policy_version=0`，48条research episode
全部版本0..0，8条student通过同版本运行时断言；research 12次/48 episodes/22 turns、doc 1437词，
`research_error=0`，SPS门率17.04%、`W_beh` off-1=3.289%，actor update17.40s、SGLang
weight update4.59s、step 1360.46s。唯一Traceback仍发生于1/1进度和weight update完成后的DataLoader
worker atexit Killed；OOM/EngineDead/RayTaskError/写盘错误/secret命中均0。针对性只读复核以97–98%
置信确认两个P1均修复且无新增P0/P1。

正式run `pgs_s2_sgl_a800_researcher`已从全新目录启动，launcher PID/PGID=`41041/41041`；manifest
`config_overrides=[]/overrides=[]`、fingerprint=`976d84cfd3a3`，实收4B `sft_step6`、2875条、
2 epochs、16×8、`n_exp=16`、k=4..8、`max_num_seqs=256`、val/save=10、
`max_ckpt_to_keep=1`、resume auto。正式启动前无旧训练进程、八卡0MiB、`/fs/fast`余39.53TiB；
双端最终真实non-thinking fncall门2/2。新连接确认launcher与正式trainer子进程存活；八卡加载后约
42.7–44.6GB/卡，采样GPU3/4功耗260.5/228.1W，错误面全零。

SwanLab不是离线待sync：日志明确online登录及URL，run ID=`braki9u3`；只读
`Api.run("Wushang/trace_opd/braki9u3")`返回name=`pgs_s2_sgl_a800_researcher`、state=`RUNNING`、
finished_at=null。运行中禁止`swanlab sync`。正式wrapper对缺`SWANLAB_API_KEY`或offline mode在建目录前
fail-closed，并设`PYTHONUNBUFFERED=1`保证实时日志。只读watchdog PID/PGID=`47341/47341`，周期600s、
stale=7200s、目标358，首轮已确认launcher alive并记录八卡memory/utilization/power/temperature；绝不自动
kill/restart。独立运行卡见`docs/pgs-s2-a800-researcher-run-card.md`。

启动后训前bare validation已274/274 JSON有效：seen140/unseen134，70/274=25.55%，invalid事件9、
truncated轨迹3；所有记录均`validate=true/global_step=0/research={}/steering={}`，ResearchArchive=0，
确认validation未污染treatment。随后step1的16/16 researcher worker全部发起真实调用；首轮观察24次35B
请求0错误，双端session精确8/8、无worker跨端。此时archive/student尚未落盘，符合先research finalize
再启动n=8 student的时序，只作为正式启动与调度验收，不冒充完整step1效果数据。launcher仍存活，fatal全零。

## 2026-08-05 · [正式启动/max-open] 162原始1.7B dynamic-researcher完全放开臂已上线，P80由目标语境实测

**用户裁定与严格有限时域实现**：新独立臂关闭`epsilon`探针、`gate_target_rate=0.20`，并要求每个
位置都允许D检测和后续教师干预、取消预算与拒绝cap约束。现有配置链要求有限数，故按32768-token
硬响应时域构造严格非绑定值：`b_max=32768×(2×0.5×2.5)=81920`、
`budget_pacing=false`、`r_max=32769>response_length`；`eta_max=0.5/c=2.5/W_beh=per_token`
保持不变。运行时验收口径不是只信YAML，而是每条有文档轨迹
`steer_d_count == Σ turn_token_spans`，且`steer_n_budget_blocked/steer_n_pacing_wait/
steer_n_bypassed/steer_turns_capped/steer_n_probe_flipped`全零。

**dynamic P80标定与启动门**：旧GT/P90不能迁移。独立标定run
`/data4/ws/PGS/pgs_s2_sgl_1p7b_162_researcher_maxopen_calib_p80`使用`theta_d=1e9`、
`gate_adaptive=false`、`epsilon=0`，完整产出16 archive/128 student，真实lineage为逻辑step1/
policy version0，零tilt/probe并完成actor与SGLang weight update。完整D直方图给出P80
`theta_d=0.00259473`；materializer才生成正式adaptive配置，正式仍从原始
`/data3/wushang/model/Qwen/Qwen3-1.7B`开始，绝不复用标定权重。专项/全量回归分别8项与
209 passed/3 skipped，162统一环境61 passed/2 skipped；双endpoint模型身份、non-thinking fncall和
`PREFLIGHT_OK` 2/2通过。凭据只经SSH stdin进入子进程内存，不进argv/文件/manifest/日志。

**正式在线身份**：run `pgs_s2_sgl_1p7b_162_researcher_maxopen`，目录
`/data4/ws/PGS/pgs_s2_sgl_1p7b_162_researcher_maxopen`，PID/PGID=`2294717/2294717`，
objective fingerprint=`c2a917a51eaf`，2 epochs/358步、16×8、`n_exp=16`、TP1/SP2、
`max_num_seqs=256`、val/save=10、只保留1个ckpt。SwanLab明确online，run ID=`lg7bxx95`：
`https://swanlab.cn/@Wushang/trace_opd/runs/lg7bxx95`；禁止sync。训前bare val已274/274，
14/274=5.11%（seen6/140、unseen8/134），随后step1 dynamic treatment已开始。只读watchdog
PID=`2295171`，600s/stale7200s；显微镜远端8793、本地18793，旧GT 8792未触碰。运行卡：
`docs/pgs-s2-1p7b-162-researcher-maxopen-run-card.md`。

**正式step1 max-open终验与可视化小修**：step1完成16/16非空archive、128/128 student，真实
`global_step=1/policy_version=0`，actor/SGLang weight update=23.95s/3.22s、grad norm=0.5145、
W_beh off-1=6.36%。157,343个assistant token逐字对应157,343个D观测；主门/tilt=9,988
（6.35%）、红色替换972（reject/tilt=9.73%），五个非绑定硬验
`budget_blocked/pacing_wait/bypassed/turns_capped/probe_flipped`总计全部为0，证明81920/32769并非
只在配置字面上“放开”。498个run/config/log文本产物secret扫描零命中，审计JSON为
`<bundle>/logs/maxopen_formal_step1_audit.json`。

首个partial train step在64/128时曾被UI误标“完成”：旧后端以历史最大n估完整批，step1只有自己时
64就成了历史最大。现改为从`effective_config.yaml`读取`train_batch_size×rollout_n=128`作为
完成门，新增回归锁定64条partial、128条completed；仅热更新/重启显微镜（新PID2427431），训练PID
2294717和本地18793隧道未动。

## 2026-08-07 · [P1 已修复×2] sftgen_researcher 采样阶段两处阻断性缺陷(162 v4 阶段A 实测)

**结论先行**:GO-V4 阶段A 在 162/1.7B 首跑 `finalize 0/8`,根因不是 text 协议或模型
能力,而是 `scripts/sftgen_researcher.py` 构造 researcher 客户端时绕过了
`make_client` 工厂,导致两处独立缺陷。两者均已在 CD-006 侧修复并加回归测试
(`tests/test_sftgen_researcher.py` 新增 2 项),修复后同配置 `finalize 8/8`。
**请 CD-005 在 pgs-selfres 分支合入同款修复**(改动仅限该脚本 22 行,不触机制)。

**缺陷① 鉴权硬编码空串**:`OpenAICompatClient(..., api_key="")`。配置已声明
`roles.researcher.api_key_env: TRACE_RESEARCHER_API_KEY`,RoleCfg 构造时即解析为
`role.api_key`(config.py:67),launch_train.py 走的正是 `role.api_key`。脚本硬编码空串
⇒ 全部会话首个 researcher 调用 401,会话以 `no_finalize=True` 落盘。
**危害在于表征伪装**:读数是 "finalize 0/8",看起来像协议/能力问题,而非鉴权;若不
逐条打开会话读 `error` 字段,会误判为 M0 门不通过并去调 SFT 配方。

**缺陷② 多端点未走 PooledClient**:CLI 传入逗号分隔的 `18888,18889`(与训练器
`roles.researcher.endpoint` 同形),但单端点客户端把整串直接拼 URL ⇒
`http://127.0.0.1:18888/v1,http://127.0.0.1:18889/v1/chat/completions` ⇒ 全量 404。
`make_client` 对逗号端点自动建 PooledClient(会话粘滞 + 故障切换),脚本绕过了它。

**修复**:改走 `make_client("researcher-sftgen", replace(cfg.role("researcher"),
endpoint=..., model=..., max_concurrency=..., temperature=..., ...))` —— 部署事实由 CLI
覆盖 RoleCfg,配置纪律不变;并在建客户端前对空 `api_key_env` / 空 `api_key` 两种情况
fail loud,不把 401 吞成 no_finalize 写进数据集。

**顺带**:GO-V4 指导的阶段C 命令写的是 `model.partial_pretrain`,但 vendored verl 的
`sft_trainer_engine.yaml` 真源键是 `model.path`(见 `verl/tests/special_e2e/sft/
run_sft_engine.sh`);另 M0 探针要起服务,`checkpoint.save_contents` 需含 `hf_model`
才产 HF 目录。已按真源执行,指导文档建议同步更正。

## 2026-08-07 · [观察] researcher 只用 4.73/16 次实验就收工,而 guidance 质量曲线未收敛

162 v4 阶段A 全量采样中期读数(n=44 会话,35B researcher + 原始 1.7B executor,
text 协议、温度 1.0):finalize 100%,但**实验数中位 4、均值 4.73(上限 16)**,
轮数中位 6(上限 30)。对照 A800 v3 同一 35B researcher(fncall、温度 0.7、SFT 过的
4B executor)实测 `exp_mean 11.75–12.93`,差 2.5×。

**关键证据:早停发生在质量曲线仍陡升时**。会话内逐次实验的 executor 成功率:
第1次 0.5%(bare 基线)→ 第2次 23.3% → 第3次 41.2% → **第4次 49.2%(中位停点)**
→ 第5次 63.4% → 第7次 71.4%。同会话配对(n=55):首次 0.9% → 末次 77.8%,
**+76.9pp,48/55 改善、0 变差**。即第 4→5 次仍有 +14pp 空间,预算利用率仅 30%。

prompt 的收工条件是 "approach **succeeds stably** AND understanding complete",
而实际停点成功率只有 49% —— 属于**违反自身指令的早停**,非设计意图。对 v4 的含义:
① SFT 会把"4 次收工"固化为先验;② compute-scaling 自进化的主要杠杆被闲置 3-4×。

**用户裁定 2026-08-07:接受现状不重采**,赌 RL 能靠续写 reward 自行纠正早停先验
(v4 reward = 前缀续写成功率,不直接奖励实验次数)。CD-006 将在 v4 训练期重点跟踪
`research 行的 experiments_used 分布`是否随训练上移;若长期不动,再回头议 prompt 侧
收工判据(候选:成功率未达稳定阈值则不许 finalize,除非预算耗尽)。

## 2026-08-07 · [P0 阻断·已修] v4 research 行缺 policy 版本证据,导致 `_compute_metrics` 整步崩

**结论先行**:162 上 v4 首次 2 步冒烟,step 1 的 **384 行全部正常产出**(16 任务 ×24 行,
布局与 reward 通道均符合设计),但在**算指标时**整步崩溃,训练无法进入第 2 步。
根因是 **research 行没有携带 `min_global_steps`/`max_global_steps`**。已在 CD-006 侧
修复并加 5 项回归测试,**请 CD-005 在 pgs-selfres 分支合入**(改动仅 backends.py +
selfres.py 两处,不触机制语义)。

**崩溃点**:
```
verl/verl/trainer/ppo/v1/trainer_base.py:1658 in _compute_metrics
    min_global_steps = np.array([tag["min_global_steps"] for tag in batch.tags], dtype=int)
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
```

**机理**:`agent_loop.py` 的 verl 侧 `default_extra_keys` 只保证 `min/max_global_steps`
这两个**键存在**,值取 `extra_fields.get(key)` —— 未设置就是 `None`。executor 行走
`**result.gen_extra`(引擎生成时经 `backends.py:EngineGenerate` 写入)天然带上;
**research 行走的是另一条组装路径**(`ResearchRowResult.fields` → `**rowres.fields`),
`ResearchRowResult` 结构里根本没有版本字段。于是同一 batch 里 executor 行有值、
research 行为 None,`dtype=int` 转换即抛。

**为什么设计上应当由 research 行自己带**:v4 的版本谱系语义是完备的
(`research/versioning.py` 的 `research_policy_version()` / `assert_rollout_policy_version()`),
且 research 行的训练 token 与 executor 行**同出一个 rollout 引擎**,版本证据来源同源;
让 metric 侧对 None 宽容会让谱系审计静默失真,比崩掉更难查。

**修复**(两处,语义由现有设计唯一确定):
1. `train/backends.py::EngineResearcherChat` 新增 `_track_policy_version()`,在每次
   `self._generate(...)` 返回后累积区间(min of mins / max of maxes,兼容引擎把值
   包在 `.data` 里的情形,非法值忽略);新增 `policy_version_fields()`,**缺证据时返回
   空 dict**而不是填 0(填 0 会让谱系审计失真)。
2. `train/selfres.py::run_research_row` 在构造 `ResearchRowResult` **之前**取该字段并
   `fields.update(...)`;**缺证据直接 RuntimeError fail loud**。`fields` 经既有的
   `**rowres.fields` 自动进 `extra_fields`,agent_loop 零改动。

**回归测试**:`tests/test_research_policy_version_tag.py` 5 项 —— 区间累积、`.data`
拆包与非法值忽略、无证据返回空 dict、行组装前 fail loud、`**rowres.fields` 契约不断。

**冒烟其余项在崩溃前均已验过(见下条)**,故这是唯一阻断项;修复后应可直接续跑。

## 2026-08-07 · [实证] v4 自体飞轮在 1.7B 上点着了(冒烟 step 1 读数)

崩溃发生在指标聚合,step 1 的 rollout 数据完整可用,据此给出 v4 首份实证:

- **`research_r_raw`:32 条 research 行中 66% 非零,中位 0.350,均值 0.355,值域
  0.0–1.0 全铺开**。值域铺开比均值更关键:组内 `A_i = r_i − mean(r)` 有充分方差,
  不会退化成零梯度。
- `research_replay_drift = 0`、`research_degraded = 0`、`research_doc_overflow = 0`;
  `research_valid_continuations` 中位 **20/20**(名义 5 分位 ×K_c=4 全部有效)——
  环境重放的确定性契约成立,免费一致性哨兵未报警。
- 行布局:每任务恰 **8 条 bare**;research 行 32 条。steered 臂在 step1 dump 中最后出现
  (需等 argmax doc),与设计时序一致。
- **bare 臂 won 32.0%** —— SFT 冷启动后 policy 已有真实任务能力(原始 1.7B 裸 val 仅 0.73%)。
- 单条样例:4 次实验 / 6 轮 / doc 692 词 / 20 次续写成功 8 次 ⇒ r_raw 0.4。

**观察项(非阻断)**:`research_overlong` 13/32(41%,mask 归零无梯度但 r 仍进 baseline,
符合 §1.6 设计);`research_no_finalize` 11/32(与 M0 探针 finalize 0.75 一致);
`experiments` 中位 3.5(35B 采样时为 5)—— 早停先验确实被 SFT 带进来了,将在训练期
跟踪 RL 能否纠正。

**墙钟实测**:单步约 **95 分钟**(16 任务 ×24 行 = 384 rollout,含 128 个 research 会话
与 2560 次前缀续写)。故正式 run 按 GO-V4 指导取 `total_training_steps=100`(约 6.6 天),
不跑配置里的 2 epochs(358 步 ≈ 23 天)。

## 2026-08-07 · [★★ 机理定案] s3 maxopen 分臂:δ 一路涨而 bare 学不到,且中段衰退+长度爆炸

**结论先行**:A800 `pgs_s3_4braw_n16_maxopen` 跑到 step 47,δ gap 从 +10.9 涨到 **+22.7**
(steered 37.5% vs bare 14.8%),但 val 在 step 30 达峰 10.22% 后回落到 5.84%(step 40),
且平均响应长度从 2035 暴涨到 4799。逐 token 记账定位到**两条独立机理**,**均已命中
v3.1 裁定时预注册的触发条件**(见 [[pgs-arm-split-v31]] 的"消融留白"与 λ± 触发判据)。

### 机理一:能力传不进 policy = 信用分配落在错误的 token 上(三重叠加)

① **教师信息稀疏**:speculative 真正改写 token 的拒绝位只占全部候选位的
   **2.0–4.5%**(step47: 10,029/508,316);其余 96–98% 是学生自己产生、教师仅"同意"。

② **这 2–4% 恰被 IS 权重压制**(逐 token 实测 W=π/b):
   | 位置 | W 中位 | W<0.5 占比 |
   |---|---|---|
   | 接受位 | 1.0000 | 0.3–1.0% |
   | **拒绝位(教师改写处)** | **0.42–0.65** | **31–59%** |
   `W_beh` 的设计目的是把 off-policy 校正回 on-policy 期望,数学后果是**教师偏离 π
   越远的位置权重压得越狠**——而那正是教师唯一提供新信息的地方。有效教学信号
   ≈ 2–4% × 0.5 ≈ **1–2%**。

③ **(根本)教师给的不是新知识,是换序**:拒绝位重采 token 在 **π 自身分布中的 rank
   中位 = 1**(p90 仅 3–5;step 5/29/47 三点一致)。教师换上的 token 学生本来就排第一
   或前三 —— PGS 改变的是**选择**而非**知识**,与"特权信息只能调已有行为、引入新动作
   只能靠 SFT"的既有边界一致。

三者合起来:轨迹成功的**因果**集中在几个分叉点的"选第 2 候选而非第 1",但
`seq-mean-token-mean` 把 reward 均摊到全部 token ⇒ 96–98% 的梯度落在 π 已经很高、
梯度贡献近零的 token 上(奖励它做它已在做的事),2–4% 的因果位置又被打 5 折。
**被强化的与起作用的不是同一批 token。**

### 机理二:中段衰退+长度爆炸 = 口径 A 混合组把负梯度压在 policy 自己的行为上

先排除一个误判:**模型没有整体变啰嗦**。赢轨迹长度基本不变(1805→752→1155),
是**失败轨迹从 2934 涨到 5513**;轮数恒 30(打满 max_env_steps),涨的是每轮 reasoning
(60→170 token/轮)。

口径 A 让 8 bare + 8 steered 进同一 GRPO 组,LOO 基线下两臂 advantage 严格反对称:
| step | bare赢% | steered赢% | **A_bare** | A_bare(输) | 退化组 |
|---|---|---|---|---|---|
| 5 | 0.8 | 1.6 | −0.004 | −0.012 | 14/16 |
| 29 | 24.2 | 29.7 | −0.029 | −0.138 | 8/16 |
| 47 | 14.8 | 37.5 | **−0.121** | **−0.219** | 4/16 |

**steered 赢得越多 ⇒ 基线抬得越高 ⇒ bare 挨的负梯度越重**,`A_bare` 单调恶化。
`seq-mean-token-mean` 再把负梯度均摊到失败轨迹每个 token,包括与成功轨迹共享的
通用 reasoning 模式 ⇒ 压低它们 ⇒ 熵回升(step34 触底 0.097 → step47 0.254)
⇒ 输出发散变长 ⇒ 更易失败 ⇒ 更强负梯度 ⇒ **正反馈**。三者时间点吻合。

**与 v3 那次崩塌病因不同**:v3 是任务饱和致 GRPO 信号耗尽(退化组 75–100%);
这次退化组从 14/16 降到 **4/16**,信号充足,病因是**负梯度主导**。该结构有个反常
性质:**教师越强,学生被压得越狠**。

### 预注册触发条件已命中(两条)

- "若分臂后 val 仍落后 won_bare_rate ⇒ 偏差主导,降 B 或口径 C":val 5.84% vs
  won_bare_rate 14.8% ✓
- λ± 块级校正触发判据"δ 持续大且 won_bare_rate/val 平":δ 持续 +14~+23、val 已回落 ✓

### 给 CD-005 的建议(按代价从小到大;方向性选择请用户裁定)

1. **口径 B(两臂独立组)** —— 最小改动且直击机理二:bare 臂自己组内比较,
   `E[A_bare]=0`,消除系统性负偏;steered 臂另成组。设计里本就留白。
2. **口径 C(steered 只作 baseline 不进梯度)** —— 同时免掉 W_beh/IS 界(机理一的
   障碍二),代价是放弃 steered 轨迹的直接梯度;正好回答"教师价值在示范还是标尺"。
3. **λ± 块级校正** —— 轨迹级 λ̂+=(p̂+α)/(q̂+α) 替换 per-token W(勿叠乘),触发判据已满足。
4. (更大改动,记账备选)**信用集中**:把 reward 归因到教师干预位置邻域而非均摊,
   直接修机理一的第三重;但这已超出"trainer 零改动"的既有约束。

**观察口径提醒**:逐步 bare 曲线噪声极大(每步 16 个不同任务、128 样本,实测单步
0.0%–24.2% 剧烈跳变),**不能用它判改善**,必须看 val;本条结论的 val 读数来自
274 任务 greedy。

## 2026-08-07 · [★ 交接·请 CD-005 独立分析] s3 分臂 maxopen 的现象与原始证据(我的解读仅供参考)

> **本条的定位**:上一条我已经写了一份带倾向性的机理解读与建议清单。**请把那一条
> 当作"一份可证伪的假设",不要当作结论,更不要按它的路线直接改机制。** 本条把现象、
> 原始证据、复现路径完整交出,**请独立做一次观察分析**——你们可能看到我漏掉的因素,
> 也可能给出完全不同的解释。若结论与我不同,以你们的为准。
>
> **先更正我自己的一处过度断言**:上一条把 val 从 10.22%(step30)→5.84%(step40)
> 称为"衰退"。复核:4.38pp 差,SE 2.31pp,**z=1.89,双侧 p≈0.058,未达显著**,且落在
> 已知的 val greedy ±5pp 噪声带内。**"性能衰退"目前不成立**,只能说"未见持续上升"。
> 下面把"确凿事实"与"未定论"分开列。

### run 身份

`pgs_s3_4braw_n16_maxopen`(A800,原始 Qwen3-4B 冷起,n=16 分臂 8+8,maxopen
always_on/b_max=inf/r_max=0,dynamic researcher 35B 文档),跑到 step 47 仍在进行。
- run: `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/runs/pgs_s3_4braw_n16_maxopen`
- 日志: `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-S3/logs/pgs_s3_4braw_n16_maxopen.log`
- 诊断脚本(只读,我写的,供复现/证伪): `analysis/s3-arm-split-diagnosis/`
  (`s3_analyze.py` 逐步指标 / `s3_rollout.py` 逐 token W 与 rank / `s3_adv.py` 分臂 advantage)
- ⚠ 训练日志被进度条 `\r` 遮蔽,须先 `tr "\r" "\n"` 再 grep,否则只看到 step 1。

### A. 确凿事实(多点一致、幅度远超噪声)

**A1 δ gap 单调扩大,bare 未见相应上升**
| step | 5 | 15 | 25 | 29 | 38 | 42 | 47 |
|---|---|---|---|---|---|---|---|
| bare won% | 0.8 | 3.9 | 10.2 | **24.2** | 1.6 | 11.7 | 14.8 |
| steered won% | 1.6 | 13.3 | 19.5 | 29.7 | 14.1 | 27.3 | **37.5** |
| δ gap | +0.8 | +9.4 | +9.4 | +5.5 | +12.5 | +15.6 | **+22.7** |

**注意口径**:逐步 bare 曲线噪声极大(每步 16 个不同任务、128 样本,实测单步在
0.0%–24.2% 间跳变),**不能用它判改善**。val(274 任务 greedy):
0.73(step0) → 1.82(10) → 6.93(20) → 10.22(30) → 5.84(40)。

**A2 响应长度大幅上涨,且只涨在失败轨迹上**
| step | 平均 len | 赢轨迹 len 中位 | 输轨迹 len 中位 | token/轮 |
|---|---|---|---|---|
| 5 | 3220 | 1805 | 2934 | 98 |
| 34 | 2035 | — | — | — |
| 29 | 1661 | **752** | 1942 | 60 |
| 47 | **4799** | **1155** | **5513** | **170** |

轮数恒为 30(打满 `episode.max_env_steps`),`clip_ratio` 恒 0.004(无 token 截断),
`response_length/max` 11905–12128(远低于 32768 预算)。**赢轨迹长度基本不变,
是失败轨迹在拖长**。

**A3 熵先坍缩后回升**:0.254(step1) → **0.097(step34,最低)** → 0.254(step47)。
长度暴涨与熵回升同期(step34→47)。同期 `grad_norm` 反而降到 0.20–0.24(step25 峰值 1.02),
**不是梯度爆炸**。

**A4 逐 token W=π/b 在接受位与拒绝位差异悬殊**(`s3_rollout.py` 实测)
| step | 接受位 W 中位 | 拒绝位 W 中位 | 拒绝位 W<0.5 占比 | 拒绝位占全部候选位 |
|---|---|---|---|---|
| 5 | 1.0000 | 0.558 | 43.4% | 4.5% |
| 29 | 1.0000 | 0.421 | 59.4% | 3.6% |
| 47 | 1.0000 | 0.655 | 31.4% | 2.0% |

**A5 拒绝位重采 token 在 π 自身分布中的 rank 中位 = 1**(p90 仅 3–5;step 5/29/47 一致;
max 53–191)。

**A6 混合组(口径 A)下 A_bare 单调变负**(`s3_adv.py`,LOO 基线)
| step | 5 | 20 | 29 | 38 | 47 |
|---|---|---|---|---|---|
| A_bare | −0.004 | −0.033 | −0.029 | −0.067 | **−0.121** |
| A_bare(仅失败轨迹) | −0.012 | −0.063 | −0.138 | −0.076 | **−0.219** |
| 退化组(组内全同) | 14/16 | 10/16 | 8/16 | 10/16 | **4/16** |

退化组**下降**(14→4),即 GRPO 信号充足 —— 与 v3 那次"任务饱和致信号耗尽
(退化组 75–100%)"的病因**不同**。

**A7 其他**:`pt_window_miss` 归一化后从 45.7%(step1) 降到 10.0%(step47);
`W_mean` 全程 0.9996–1.0009;`off1_frac` 从 35.6% 降到 16.5%;`D_mean` 0.103 →
0.063(step34 最低) → 0.069。

### B. 未定论 / 我没有排查的因素(请重点考虑)

1. **val 回落是否真实**:z=1.89、p=0.058,未达显著。step 50 的 val 点会给出关键增量证据。
2. **doc 侧我完全没查**:s3 用的是 35B dynamic researcher 文档。**文档质量/长度是否随
   训练漂移、是否与长度爆炸同期**,我没有取过任何 doc 数据。这是一条独立且未被排除的
   因果路径(v3 时我查过 doc 全程健康,但 s3 是不同 run,不能外推)。
3. **always_on 本身的贡献**:本 run 无 adaptive 对照,"全开 vs 有门"对上述现象的贡献
   无法从本 run 内分离。
4. **长度增长与 reward 的因果方向**:我只观察到"失败轨迹变长",没有验证是
   "被压低→变长"还是"变长→更易失败"。二者可自洽成环,但起点未定。
5. **`seq-mean-token-mean` 的具体作用**:我推断它把负梯度均摊到失败轨迹全部 token,
   但没有做 `token-mean` 对照或逐 token 梯度归因来证实。
6. **researcher 早停特性**(35B 中位仅用 4.73/16 次实验)是否影响 doc 覆盖面,未查。

### C. 我的解读(**仅供参考,请独立验证或推翻**)

基于 A1–A7,我倾向于两条机理,但**都未做对照实验,均属假设**:

**假设一(能力不进 policy)**:教师改写的 token 只占 2–4.5%(A4),这些位置的 W 被压到
0.42–0.65(A4),而换上的 token 在 π 中 rank 中位 = 1(A5)——即教师改的是"选择"而非
"知识"。若 reward 经 `seq-mean-token-mean` 均摊,则 96–98% 的梯度落在 π 已很高、
梯度贡献近零的 token 上,真正的因果位置又被打折 ⇒ 信用分配与因果不对齐。
**可证伪点**:若把 reward 归因集中到拒绝位邻域后 bare 仍不涨,该假设不成立。

**假设二(长度/熵漂移)**:口径 A 下 steered 赢得越多,基线抬得越高,A_bare 越负(A6);
`seq-mean-token-mean` 把负梯度均摊到失败轨迹每个 token,包括与成功轨迹共享的通用
reasoning 模式 ⇒ 压低后熵升、输出变长(A2/A3)⇒ 更易失败 ⇒ 负梯度更强。
**可证伪点**:若改口径 B(两臂独立组)后长度与熵仍同样漂移,该假设不成立;
另,B 中第 4 点(因果方向)未定,也可能是完全相反的链路。

**设计文档里的预注册触发条件**:v3.1 裁定的"若分臂后 val 仍落后 won_bare_rate ⇒
偏差主导"与 λ± 的"δ 持续大且 won_bare_rate/val 平"——从读数看**形式上已满足**
(val 5.84% vs won_bare_rate 14.8%;δ 持续 +14~+23)。但请注意 val 与 won_bare_rate
**口径不同**(前者 274 任务 greedy/val split,后者 train 任务 T=1.0),直接比大小是否
恰当,请你们判断。

### D. 请求

请针对上述现象**独立**做一次深入观察分析,不必受我的假设约束。特别希望你们看:
B 中我未排查的六项(尤其 **doc 侧漂移** 与 **因果方向**),以及是否存在我完全没想到的
第三种解释。当前 s3 run **我保持不动、继续跑并观察**(step 50 的 val 点是关键增量证据);
是否换口径/改机制属方向性决策,待你们分析结论与用户裁定,我不擅自改。

## 2026-08-07 · [★ 追加实证·长度膨胀] 逐轨迹拆到真实文本:两波机制接力,死锁只是放大器

> 承上条交接。用户要求深挖长度膨胀,我把 s3 拆到了**逐轮 token 与真实轨迹文本**。
> 结论**修正了我上一条里的一处错误**,并给出新的可证伪点。新增脚本:
> `analysis/s3-arm-split-diagnosis/{s3_length.py, s3_deadlock.py, s3_text.py}`。

### 1. 精确分解:长度 = 死锁率 × 死锁轮长 + 正常轮长(`s3_deadlock.py`)

死锁轮定义:(thought, action) 与前一轮**逐字相同**。失败轨迹读数:
| step | 5 | 15 | 25 | 29 | 34 | 40 | 44 | 48 |
|---|---|---|---|---|---|---|---|---|
| 死锁率 | 0.0% | 0.0% | 7.9% | 40.2% | **72.9%** | 81.4% | 70.2% | 63.7% |
| 死锁轮 len | — | — | 29 | 24 | 33 | 81 | 146 | **226** |
| 正常轮 len | 84 | 61 | 44 | 53 | 61 | 78 | 118 | **174** |
| 总 len | 2492 | 1778 | 1229 | 1177 | 1177 | 2333 | 3841 | **5378** |
| 熵 | .254 | .213 | .139 | .135 | **.097** | .089 | .168 | .254 |

**两波接力,机制不同**:
- **第一波(5→34)**:死锁率 0%→72.9%,但死锁轮极短(24–33 token),**总长度反而从
  2492 降到 1177**。模型学会"快速放弃、原地重复短句";此期 val 在涨(step30 达峰 10.22%)。
- **第二波(34→48)**:死锁率**见顶回落**(72.9%→63.7%,死锁并未更严重),但
  **死锁轮长 33→226(6.8×)、正常轮长 61→174(2.9×)**,总长度 1177→5378。
  ⇒ **长度膨胀的直接原因是"每轮说得更长",不是"死锁更多"。**

### 2. ★修正上一条的错误:"只有失败轨迹在膨胀"不成立

上一条我据 `response_token_len`(赢 752→1155)判断"赢轨迹长度基本不变"。**该结论错误**:
赢轨迹**轮数在减少**(9.3→7.4)掩盖了单轮增长。按轮看:
| | step29 | step48 | 倍数 |
|---|---|---|---|
| 赢轨迹正常轮 len | 61 | **153** | 2.5× |
| 输轨迹正常轮 len | 53 | **174** | 3.3× |

**赢轨迹死锁率恒 0.0%,单轮长度照样涨 2.5×** ⇒ "每轮变长"是**全局现象**,与死锁无关。
死锁只是**放大器**:把单轮 2.9× 经 20 轮重复放大成轨迹级 4.6×。

(另更正:上一轮我贴的 thought 在 "…and determ" 处断掉是**我的显示脚本 `t[:1400]` 截的**,
非模型/配置所致。实测 `truncated=False`、最大轮 758 token,远未触及 `max_tokens_per_turn=2048`。)

### 3. 死锁根源:`look` 是"合法但不推进状态"的陷阱

死锁轮的动作/观测分布:
| step | 死锁轮数 | 主导动作 | 主导观测 |
|---|---|---|---|
| 29 | 2,241 | **`look` 81.3%** | "Nothing happens." 18.7% |
| 40 | 5,582 | **`look` 74.4%** | "You are in the middle of a room…" 45.1% |
| 48 | 4,296 | `look` 36.5% / `go to X` 20%+ | **"Nothing happens." 62.0%** |

`look` **永远 valid=True 却不改变环境状态** ⇒ 观测恒定 ⇒ 上下文近乎不变 ⇒ 输出近乎
不变 ⇒ 无限循环。这解释了一个反常读数:step29–40 的有效动作率仍有 84–90%(因为
`look` 合法),到 step48 模型部分逃出 `look`(36.5%)转而在 `go to X` 上真失败
(观测 62% "Nothing happens."),有效动作率才掉到 61.6%。
`episode.max_env_steps=30` + 无终止惩罚 ⇒ 模型可**免费原地打转 30 次**。

真实样例(step48,bare,13846 token,30 轮):逐轮 token
`[70,116,134,212,250,185,390,322,217,758, 533,534×19]` —— **第 11 轮起恒定 534 token
连续 20 轮逐字相同**,动作恒为 `move mug 1 to fridge 1`(ALFWorld 非法格式),观测恒为
"Nothing happens."。**77% 的 token 来自死锁重复。**

### 4. 可能更根本:梯度被死锁 token 主导

死锁贡献 token **69.1%** × 失败轨迹占比 **96%** ≈ **bare 臂里约三分之二的 token 是
死锁重复**。而 `seq-mean-token-mean` 先对每条轨迹做 token 平均 ⇒ 一条含 20 个逐字
相同死锁轮的轨迹,**其自身的平均值即被那 20 轮主导**。即:每条失败轨迹交给梯度的
信号,本身就是"同一段话重复 20 遍"的平均;高度冗余、零信息的 token 主导了整个 batch。

### 5. 假设(未直接验证,给出可证伪点)

失败轨迹 `A<0` ⇒ 梯度降低该轨迹**全部** token 概率(seq-mean-token-mean 均摊);
失败轨迹占 80–96% 主导梯度;被压低的 token 含每轮**终止行为** ⇒ 终止倾向下降 ⇒
每轮生成更长。四条支持证据:① 赢轨迹正常轮也涨 2.5×(若只压失败轨迹则不该涨);
② 熵同步回升 .097→.254(压低高概率 token 必然增熵);③ `A_bare` 单调变负
−0.004→−0.121;④ 时间点吻合(A_bare 恶化/熵回升/长度膨胀均始于 step34)。

**可证伪点**:未直接测量终止 token 的 log π 随训练的变化。若测出终止 token 概率
**未**下降,该假设不成立,需另找解释。**这条请 CD-005 独立判断,勿视为结论。**

---

## 2026-08-07 · [★ 追加实证·时间维度] 长度膨胀到 step48 仍在**加速**,且与熵**回升**同步

**结论先行**:长度膨胀不是一次性事件,是**持续加速的过程**;截至 step48 未见平台。
且它与熵**回升**(而非坍缩)同步发生 —— 这否定了"熵坍缩→输出退化"的简单叙事,
方向恰好相反。此条只补时间维度的原始读数,解读仍以 CD-005 独立分析为准。

**数据来源**:`TRACE-OPD-S3/logs/pgs_s3_4braw_n16_maxopen.log` 的 verl step 指标行
(Ray 前缀 `(TaskRunnerV1 pid=…)` + ANSI 色码需先剥离,否则 `grep "^step:"` 恒为 0)。
提取脚本 `analysis/s3-arm-split-diagnosis/s3_trend.py`。截至 step48(共 49 行)。

| step | won | 熵 | grad_norm | env_steps | respLen | IS_off1 | D均 | 拒绝数 |
|---|---|---|---|---|---|---|---|---|
| 10 | .066 | .235 | .444 | 28.8 | 2777 | .331 | .112 | 55.4 |
| 20 | .125 | .186 | .477 | 27.3 | 2113 | .301 | .109 | 37.4 |
| 25 | .148 | .139 | 1.016 | 27.1 | 1841 | .258 | .097 | 24.8 |
| 30 | .094 | .120 | .579 | 28.4 | 1897 | .228 | .080 | 20.1 |
| 35 | .102 | .115 | .375 | 27.9 | 2075 | .207 | .071 | 20.0 |
| 40 | .105 | **.089** | .428 | 27.7 | 3051 | .139 | .047 | 17.8 |
| 43 | .184 | .167 | .216 | 25.7 | 3987 | .157 | .064 | 30.1 |
| 45 | .254 | .209 | .235 | 24.5 | 4265 | .158 | .067 | 34.3 |
| 48 | .121 | **.219** | .155 | 27.2 | **5942** | .163 | .053 | 46.0 |

**四个可核对的事实**

1. **长度单调加速**:respLen 在 step25 触底 1841,此后单调上行;step35→48(13 步)
   2075→5942(**2.86×**),近 8 步斜率约 **+330 token/step** 且未见减速。按此外推,
   距 `max_response_length=32768` 约 80 步。**这是运行风险,不只是指标异常。**
2. **env_steps 稳定 24–28(上限 30)**:长度增长**不来自轮数**,只能来自单轮变长 ——
   与前条"逐轮拆解"的结论独立互证(那条是从 rollout JSON 算的,这条是 verl 侧聚合)。
3. **熵先坍后升**:.235(s10)→**.089(s40)**→.219(s48)。长度暴涨段(s40→48)熵是**上升**
   的。若病因是"熵坍缩导致复读",熵回升时长度应回落,实测相反。
4. **grad_norm 与长度反向**:.477(s20)→**.155(s48)**,长度最大时梯度范数最小。

**对 CD-005 独立分析可能有用的两点(仅供参考)**

- 事实 3 与我上条"假设"是**自洽**的:若病因是失败轨迹 `A<0` 均摊压低全部 token
  (含终止 token),那么熵上升与长度上升本就该**同向** —— 压低高概率 token 必然增熵。
  但这只是"不矛盾",不构成证明;可证伪点仍是终止 token 的 log π 未直接测量。
- 事实 4 提示另一种读法:grad_norm 掉到 .155 时模型仍在快速改变行为(长度 8 步 +95%),
  说明改变来自**持续同向的小步累积**而非大梯度事件。若要定位,逐步比对同一 prompt 下
  终止 token 的 logit 漂移比看 grad_norm 更直接。

**未变更**:s3 run 保持不动继续跑,不换口径、不加限制(方向性决策待 CD-005 结论 +
用户裁定)。下一个增量证据是 **step50 的 val**(现 val: .73→1.82→6.93→10.22(s30)
→5.84(s40)%;s40 的回落 z=1.89 p=0.058 未达显著,尚不能称"衰退")。注意 train 侧
won 在 s41–48 均值 .186 vs s30–40 均值 .100,但该差(8.6pp)**远低于训练 rollout 的
MDE(23.2pp)**,不可用于判断改善 —— 主终点仍只看 val。
