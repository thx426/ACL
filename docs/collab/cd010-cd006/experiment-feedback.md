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
