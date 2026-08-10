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

---

## 2026-08-07 · [★★ 更正] 前条"v4 飞轮点着了"的数字是抽样偏差;全量复算后读数差很多

**先说结论**:上面 `[实证] v4 自体飞轮在 1.7B 上点着了` 一条里的
`research_r_raw 66% 非零 / 中位 0.350 / valid_continuations 中位 20/20 /
research_overlong 13/32 / no_finalize 11/32` **全部作废**。那批数字取自
128 条 research 行中按文件名排序的**前 32 条**,而该子集系统性偏乐观。
**全量 128 条同口径复算**(脚本 `v4_smoke_accept.py`,两次冒烟各跑一遍):

| 指标 | 原记(前 32 条) | 全量 128 条 · smoke1 | 全量 128 条 · smoke2 |
|---|---|---|---|
| r_raw 非零率 | 66% | **30%** | **28%** |
| r_raw 中位 | 0.350 | **0.000** | **0.000** |
| valid_continuations 中位 | 20/20 | **0/20** | **0/20** |
| research_overlong | 41% | **84%** | **82%** |
| research_no_finalize | 34% | **69%** | **71%** |
| research_error | — | 61% | **65%** |

两次独立冒烟的全量读数**互相吻合**(84/69/30 vs 82/71/28),所以这是稳定状态而非
波动。**"飞轮点着了"这个判断在全量口径下不成立**,请勿据前条推进任何决策。

未变的部分(这些原记是对的):`replay_drift/degraded/doc_overflow` 全零;
每任务恰 8 条 bare;bare 臂 won 32.0%(smoke1)/39.1%(smoke2)。

**教训(给自己也给 CD-010)**:rollout 目录按文件名排序不是随机序 ——
session_id 与任务/时序相关,取前 N 条会系统性偏。抽样统计一律加 `random.sample`
或直接全量。

---

## 2026-08-07 · [★★ P0 阻断·已修] `reward_extra_info` 锯齿字典致 `won[keep]` IndexError 崩整步

**症状**:162 v4 冒烟 step1 的 rollout(384 行)与 actor update **全部正常完成**,
崩在最后的指标聚合:

```
File ".../verl/trainer/ppo/metric_utils.py", line 465, in compute_rollout_extra_metrics
    arm, won = arm[keep], won[keep]
IndexError: boolean index did not match indexed array along axis 0;
            size of axis is 256 but size of corresponding boolean axis is 384
```

**根因(两层,第二层更值得修)**

层 1 · 字段类型不对称。`agent_loop.py` 两条路径构造 `reward_extra_info`:
- executor 行显式 `"won": int(traj.won)`、`"truncated": int(result.truncated)` —— int;
- research 行走 `{k: v for k, v in fields.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}`,
  而 fields 里是 `"won": False`、`"truncated": bool(...)` —— **bool 被过滤器吃掉**。

层 2 · 组装方式无长度契约。`trainer_base.py:1696-1700` 是 **per-key append**:
```python
for key, value in extra_field.get("reward_extra_info", {}).items():
    reward_extra_infos_dict[key].append(value)
```
某 key 在部分行缺席不报错,只让该 key 的列表短一截 ⇒ `steer_arm` 384 个值而
`won` 只 256 个。`compute_rollout_extra_metrics` 的守卫只查了
`keep.shape == arm.shape`((384,)==(384,) 通过),没查 `won` ⇒ 崩在 `won[keep]`。

**注意这个 bug 的隐蔽面**:即使不崩,锯齿字典也会让 `rollout_extra/<key>/mean`
**静默变成"部分行的均值"**。任何 key 部分缺席都会这样,且无任何提示。

**已修(三处,均为最小可逆改动)**

1. `agent_loop.py` research 路径改为与 executor **结构对称**:四个公共标量
   (`won`/`env_steps`/`invalid_format`/`truncated`)显式 int 化后进 dict,
   再展开其余数值键。不动 `fields` 本身,故 rollout JSON 与显微镜的消费方不受影响。
2. `metric_utils.py` 守卫补齐:`keep.shape == arm.shape == won.shape`。
   宁可少一条分臂曲线也不能崩整步。
3. `metric_utils.py` 新增**锯齿检测**:各 key 长度不一致时打印
   `[rollout_extra] WARNING ragged reward_extra_info: ...`,把静默降级变成可见事件。

**测试**:新增 `tests/test_rollout_extra_metrics_ragged.py` 3 项 —— ① research 行
不稀释 `won_bare_rate`(混入会让 2/4 变成 2/8);② 锯齿字典不崩且打 warning;
③ 无 `role_research` 键的 v3.1 两段 batch 行为不变。162 上 12 项全过
(本地缺 tensordict,该文件按路径加载绕开 `verl/__init__`);本地全量 313 passed。

**建议(仅供参考)**:层 2 才是复发源。若在 `trainer_base.py` 组装处对
`reward_extra_info` 做一次 key 集合一致性校验(首行的 key 集合作为契约,
后续行缺键即 fail loud),这类问题就不会再走到下游。我没动那里 —— 它在
CD-005/CD-010 的其他路径上也被使用,改动面超出实验分支的"小修"边界。

---

## 2026-08-07 · [★★ P0 设计] `research_max_sample_tokens=24576` 把 82% 的 research 行 mask 成零梯度

**结论先行**:v4 的 research 目标在 1.7B/162 上**基本没生效**。λ_r=0.1 的实际
有效权重 ≈ `0.1 × 18% = 0.018`。病因是阈值恰好切在长度分布的左尾外侧,
不是模型不会做研究 —— **被砍掉的是有效信号而不是噪声**(见下方第 3 点)。

**1. 阈值与实测分布错位**

`selfres.py:197` `overlong = len(prompt_ids) + len(stream) > research_max_sample_tokens`,
命中即 `mask = [0]*len(mask)`(零梯度,r 仍留组内做 baseline)。而 128 条 research
行的实测 `response_token_len` 分布:

| 分位 | p10 | **p50** | p90 | max |
|---|---|---|---|---|
| token | 17110 | **37185** | 39838 | 44507 |

阈值 24576 落在 p10 与 p50 之间 ⇒ 只有 20% 的行存活。抬阈的救回曲线是**悬崖式**的:

| cap | 未超限 | |
|---|---|---|
| 24576(原) | 25/128 (20%) | |
| 32768 | 40/128 (31%) | |
| **40960** | **123/128 (96%)** | ← 悬崖 |
| 49152 | 128/128 (100%) | |

**2. 因果链已坐实**(交叉表 + 分组对比)

`(overlong, no_finalize, error) = (True,True,True)` 占 **64%**(82/128),是最大单元格。
分组对比更干净:

| | n | respLen 中位 | turns 中位 | doc_words 中位 | overlong |
|---|---|---|---|---|---|
| r_raw>0 | 36 | 21587 | 10.0 | **516** | 44% |
| r_raw=0 | 92 | **37216** | 16.5 | **0** | **97%** |

`valid_continuations` 是**严格双峰**:0(71%)或 20(29%),无中间值 —— 要么文档没产出
(续写无从谈起),要么 20 次全有效。

**3. 关键反证:被砍掉的不是噪声**

只看**未 overlong 的 23 行**:`r_raw` 非零 **87%**、中位 **0.450**、
`valid_continuations` 中位 **20**、`no_finalize` 仅 **13%**。
也就是说 —— **只要会话没被长度切断,v4 飞轮的读数是好的**。
这同时解释了上面那条"更正"里的抽样偏差:前 32 条恰好富集了未超长的行。

**4. 抬阈的代价比直觉小**

overlong 行**本就完整进 batch 参与 forward**(只是 `response_mask` 置 0),
抬阈只是让**已经付出的 forward 产生梯度**,增量主要在 backward,不是"多跑一遍长序列"。
旁证:本次冒烟里最长序列 **45176 token** 已经在训练(日志 `max_token_len=16384
below longest sequence 45176 — 60 oversized sample(s) ride in their own micro-batch`),
四卡 100% 利用、显存 62–63GB/80GB,没有 OOM。

**5. 已按用户裁定调整(2026-08-07)**

用户裁定:**模型最大上下文 256K;researcher 单轮输出 8K,任务执行 agent 单轮输出 4K**。
落到配置(162 版,A800 版未动):

| 参数 | 原 | 新 | 依据 |
|---|---|---|---|
| `train.rollout_max_model_len` | 40960 | **262144** | 用户裁定(256K) |
| `episode.max_tokens_per_turn` | 2048 | **4096** | 用户裁定(executor 4K) |
| `research.researcher_max_tokens` | 8192 | 8192 | 用户裁定(researcher 8K),已符合 |
| `train.response_length` | 32768 | **65536** | 连带:executor 单轮翻倍 + research p90 已达 39838 |
| `eval.response_budget` | 32768 | **65536** | 连带:训评同构口径必须一致 |
| `train.max_token_len_per_gpu` | 34816 | **40960** | 连带:×sp2=81920 ≥ 2048+65536 |
| `research_rl.research_max_sample_tokens` | 24576 | **65536** | 连带:本条的直接目标 |

**两机对照性说明**:这次改动让 162 版与 A800 版的 objective_fingerprint **不再相同**
(`research_max_sample_tokens` 在 research_rl 段内、进指纹)。测试
`test_162_selfres_config.py` 已相应改为"把长度参数对齐回 A800 后指纹必须相等",
即**除这一处长度裁定外机制仍零漂移**,并新增 `test_length_ruling_2026_08_07`
锁住新值防回退。若 CD-010 认为 A800 版也该同步调整,请在 changelog 说明,
我这边跟随。

**验证中**:冒烟 `pgs_s4_smoke3`(PID 2832152)带新配置 + 上一条的 P0 修复重跑,
盯三件事 —— ① 显存(response 上限翻倍后是否仍 <80GB);② `research_overlong`
是否从 82% 降下来;③ 单步墙钟(rollout 变长会直接决定正式 run 的总天数)。

---

## 2026-08-07 · [★★ 硬约束更正 + 病因定位] 256K 上下文不可执行;40960 的 ctx 墙才是 research 失败的物理根因

**先更正上一条**:上一条表格里写的 `rollout_max_model_len → 262144`、
`response_length → 65536`、`response_budget → 65536`、
`research_max_sample_tokens → 65536` **均未落地**,实际落地值见下。原因是
用户裁定的 256K 在 Qwen3-1.7B 上物理不可执行 —— 起 SGLang 时直接:

```
ValueError: max_model_len (262144) should be less than or equal to
            max_position_embeddings (40960)
   verl/workers/rollout/sglang_rollout/async_sglang_server.py:245
```

模型 `config.json`:`max_position_embeddings=40960`、`rope_scaling=None`;
verl 也未把 `json_model_override_args` 暴露到配置层(只在 fp8 量化路径内部用)。
**要突破 40960 只能改模型 config 开 YaRN —— 那是改基座不是改配置**,
会影响占绝大多数 token 的 executor 短轨迹,已升级用户裁定,我没有擅自实施。

**更重要的收获:40960 这堵墙就是 research 会话失败的物理根因**

按 `response_token_len` 分档看 finalize 成功率(smoke2 的 128 条 research 行):

| respLen 区间 | n | finalize 成功率 | r_raw>0 |
|---|---|---|---|
| (8192, 16384] | 11 | 90.9% | 90.9% |
| (16384, 24576] | 14 | 85.7% | 85.7% |
| (24576, 32768] | 15 | **93.3%** | 86.7% |
| (32768, 36864] | 5 | 20.0% | 20.0% |
| **(36864, 40960]** | **78 (61%)** | **0.0%** | **0.0%** |
| (40960, 49152] | 5 | 0.0% | 0.0% |

**61% 的会话堆积在紧贴 ctx 上限的最后一格,且无一成功**。这是典型的
**撞墙质量堆积**:会话不是自然收敛的,是被 context 截断的。分组中位数同向:

| 组 | n | respLen 中位 | turns 中位 | msg 数中位 |
|---|---|---|---|---|
| 成功(error=F, no_finalize=F) | 37 | 21803 | 10.0 | 20 |
| 撞轮数上限(error=F, no_finalize=T) | 8 | 35182 | **30.0**(=max_turns) | 63 |
| 撞 ctx 墙(error=T, no_finalize=T) | 83 | **37221** | 16.0 | 30 |

**成功的会话 10 轮内收敛;失败的会话在第 16 轮撞墙**。所以"给更多上下文"
确实是对症的 —— 那 83 条会话不是不会做研究,是话没说完就被掐断。

**实际落地(在物理上限内顶满)**

| 参数 | 原 | 实际落地 | 说明 |
|---|---|---|---|
| `train.rollout_max_model_len` | 40960 | **40960** | = max_position_embeddings,已是上限 |
| `episode.max_tokens_per_turn` | 2048 | **4096** | 用户裁定(executor 4K),已落地 |
| `research.researcher_max_tokens` | 8192 | 8192 | 用户裁定(researcher 8K),已符合 |
| `train.response_length` | 32768 | **38912** | = 40960 − 2048,物理上限内顶满 |
| `eval.response_budget` | 32768 | **38912** | 训评同构 |
| `train.max_token_len_per_gpu` | 34816 | **40960** | ×sp2=81920,2× 余量 |
| `research_rl.research_max_sample_tokens` | 24576 | **40960** | = ctx 上限 ⇒ 本阈不再是绑定约束 |

抬阈后仍会有 ~4% 的行超限(实测 max 44507 > 40960),那些正是撞墙的行 ——
**只有扩 ctx 能救,本阈已退出成为限制因素**。

**给 CD-010 的三个问题(仅供参考,不预设答案)**

1. research 会话有没有**历史压缩/滑窗**的设计空间?成功组 10 轮收敛而失败组
   16 轮撞墙,若能把前几轮的工具原文压成摘要,同样的 ctx 可能多撑 10 轮。
   这比扩 ctx 更根本,也不会污染 executor。
2. `max_turns=30` 与 ctx 40960 的组合本身不自洽:30 轮 × 实测 ~2.3K/轮 = 69K
   远超 ctx。是否应让 `max_turns` 由 ctx 反推,而不是独立取值?
3. 若确实要扩 ctx,YaRN 的静态缩放会作用到 **executor 轨迹**(中位仅 2–3K)。
   有没有办法只给 research 会话用长 ctx(例如 research 单独起一个 rope_scaling
   不同的副本)?若没有,扩 ctx 就必须权衡主实验被污染的风险。

**验证中**:冒烟 `pgs_s4_smoke4`(PID 2880914)带上述落地值 + 两条 P0 修复重跑。

---

## 2026-08-07 · [环境事实 + 我的操作事故] 162 的 v4 树曾落后本地 HEAD 23 个文件;已整体对齐

**给 CD-010 的事实**:162 上的 v4 工作树(`/data4/ws/PGS/trace_opd_selfres_v4/trace_opd`)
是 **2026-08-06 22:22** 同步的,此后本地 HEAD 又前进了 —— 对齐前的差距:

- **仅本地有 9 个文件**:`envs/registry.py`、`envs/_template/*`、
  `envs/spreadsheetbench/*`(= CD-010 的环境抽象层与 SSB 接入)
- **14 个文件内容不同**:`core/adapter.py`(2602 → 8934 字节,`EnvFactory` 协议是新增的)、
  `core/{episode,protocol,template_delta,types}.py`、`common/config.py`、
  `envs/alfworld/adapter.py`、`research/{continuation,prompts}.py`、
  `narrate/*`、`evalm/run_eval.py`、`train/build_datasets.py`
- **162 无独有文件**(纯落后,不存在被覆盖的生产改造)
- `verl/` 树 347 个文件**逐字节一致**;`configs/` 只差 3 个无关文件

也就是说:**之前四次冒烟(smoke1–4)跑的是环境抽象层落地之前的代码**。
若 CD-010 以为 162 在验证最新链路,这个前提当时不成立。现已整体对齐,
备份留在 `trace_opd/src.bak.20260807`(可回滚)。

**我的操作事故(记录以免重演)**:我只同步了改动的 3 个文件
(`agent_loop.py` / `metric_utils.py` / 配置),而本地 HEAD 的 `agent_loop.py`
`from ..core.adapter import EnvFactory` —— 162 的旧 `adapter.py` 没有这个符号,
于是 smoke4 起来后 **每个 AgentLoopWorker 都 ImportError**:

```
ImportError: cannot import name 'EnvFactory' from 'trace_opd.core.adapter'
hydra.errors.InstantiationException: Error locating target
    'trace_opd.train.agent_loop.TraceAgentLoop'
```

教训:**部分同步比全量同步更危险** —— 全量同步的风险(覆盖远端改造)有
`comm`/`sha256` 对拍可以事先查清;部分同步的风险(依赖链断裂)在同步时完全
不可见,只会在运行时炸。正确顺序是:先全树对拍分清"远端落后"还是"远端有独有
改动",再决定全量同步还是逐文件打补丁。

**162 测试环境的两个已知缺口(不阻断训练,但会让"跑测试"这个动作失真)**

1. **缺 `pytest-asyncio`**:`asyncio_mode = "auto"` 被报 `Unknown config option`,
   31 个 async 测试直接 FAILED(`async def functions are not natively supported`)。
   在 162 上跑全量测试会看到一片红,与代码质量无关。
2. **`from tests.fakes import ...` 收集失败**(pytest 9.1.1 的 namespace package
   行为与本地版本不同),影响 `test_{selfres,continuation,engine_researcher_chat}.py`
   与 `test_adapter_and_evalm.py` 的一个用例。在 `pyproject.toml` 的
   `[tool.pytest.ini_options]` 加 `pythonpath = ["."]` 应可解;我没改 —— 那是
   共用文件,且不阻断实验。

排除这两类后,162 上 **20/20 通过**(含配置合同 4 项、锯齿字典 3 项、
policy 版本证据 5 项、maxopen 合同 5 项)。

---

## 2026-08-07 · [★★ 重要更正·请优先读] step50 val 创新高 13.14%,"bare 学不到"与"中段衰退"两个前提都不成立

**给 CD-005 的紧急更正**:我此前那条交接
`[★ 交接·请 CD-005 独立分析] s3 分臂 maxopen 的现象与原始证据` 的标题与前提里写了
**"δ 一路涨而 bare 学不到"**和**"中段衰退"**。step50 的 val 落地后,这两个说法
**都要撤回**。若已按该前提开始分析,请以本条为准重新起点。

**step50 val 实测**(`valid_seen 0.171429`×140 + `valid_unseen 0.089552`×134):

| step | 0 | 10 | 20 | 30 | 40 | **50** |
|---|---|---|---|---|---|---|
| val(274 题) | 0.73% | 1.82% | 6.93% | 10.22% | 5.84% | **13.14%** |
| 命中数 | 2 | 5 | 19 | 28 | 16 | **36** |

**① "中段衰退"不成立 —— 是噪声**

- step50 vs step40:13.14% vs 5.84%,z=**2.92**,p=**0.0035**(显著回升)
- step50 vs step30:13.14% vs 10.22%,z=1.06,p=0.29(在噪声内,即 step30→50 是平稳上行)

step40 的 5.84% 是单点下探,与 [[val-greedy-nondeterminism]] 记录的 ±5pp greedy
噪声带一致。我此前写"回落 4.38pp 但 z=1.89 p=0.058 未达显著,'衰退'尚不成立"
是对的,现在可以下定论:**没有衰退**。

**② "bare 学不到"不成立 —— 学到了,而且极显著**

val 就是 **bare policy 的 greedy 读数**(无教师、无文档)。0.73% → 13.14%,
**涨 18 倍**;step50 vs step0:z=**5.72**,p<1e-7。

交叉验证:step50 的训练侧混合 `rollout_extra/won/mean = 0.1484`,与 val 13.14%
**高度一致**(两个独立口径互证,不是 val 的偶然)。

**③ 那我此前那些观察还算数吗 —— 算,但结论强度要下调**

这些仍是真实观察,未被推翻:教师改写 token 仅占 2.0–4.5%、其 W 中位 0.42–0.65、
拒绝位重采 token 在 π 中 rank 中位=1(教师只换序不给新知识)、`A_bare` 单调变负
−0.004→−0.121、δ gap 稳定 +0.20~0.23。

但它们支持的结论应该是 **"教师增益的吸收效率不高"**,而不是我原来写的
**"能力进不了 policy"**。两者的区别很实在:

- gap 不缩小 ≠ bare 不涨。若 steered 臂也在同步上移,gap 完全可以保持在
  +0.20 而两臂**同时**上行 —— 实测正是如此(bare 从 ~0 到 13–15%,steered ~35%)。
- 我把"gap 不收敛"读成了"bare 不动",这是**把相对量当成了绝对量**。

**④ 长度膨胀那条仍然成立,且与本条不矛盾**

respLen 2075(s35)→5942(s48) 仍在加速、熵回升、死锁重复占 bare 臂约 2/3 token ——
这些都还是真的。但现在要重新定位它:**长度膨胀与能力提升在同时发生**,
所以它不是"崩塌的前兆",至少到 step50 还不是。它更像是一个**并行的效率问题**
(大量 token 花在死锁重复上),而不是能力的病因。

**⑤ 我的教训**

只用 train rollout 的分臂 gap 判断"能力有没有进 policy",而主终点 val 的下一个
数据点还没到,就写了"机理定案"。[[training-rollout-cannot-judge-improvement]]
早就记过"训练 rollout 曲线判不了改善,改善看 val",我自己违反了它。
**下次:主终点没落地前,措辞一律停在"观察"级,不写"定案"。**

**当前**:s3 continues(`total_epochs=2`,未设步数上限),step50 已过,watcher 盯 step60。
若 step60 继续在 13% 附近或更高,则"maxopen 分臂在 4B 上有效"可以进入正式结论;
若掉回 6% 以下,则说明方差比我估计的还大,需要加密 val 频率。

---

## 2026-08-07 · [★ 更正 + 新机制风险] 放宽 mask 阈值只解决了"有没有梯度",没解决"会话撞墙";且可能引入错误归因的惩罚

**先更正我自己**:我在上一轮基于 smoke5 的**部分数据**(前 41 条 research 行)
写过"YaRN 大概率不需要"。**全量 128 条不支持这个结论,撤回。**
幸存者偏差的量级比我预估的大得多:

| 指标 | 部分(前41) | **全量(128)** | smoke2 全量 |
|---|---|---|---|
| overlong | 2.4% | **8%** | 82% |
| error | 22.0% | **58%** | 65% |
| no_finalize | 41.5% | **65%** | 71% |
| r_raw>0 | 53.7% | **32%** | 28% |
| turns 中位 | 11 | **14.5** | 15 |
| respLen 中位 | 21149 | **37076** | 37185 |

先完成的任务系统性偏短 —— 这跟"别按文件名排序取前 N 条"是同一个坑的另一面
(**按时间排序的同偏窗口也救不了它**,因为两个 run 的进度不同步时,
"前 41 条"在一侧是 32% 的完成度、在另一侧是 100%)。
**教训:跨 run 比较必须两侧都跑完,没有捷径。**

**放宽阈值的真实效果:只有一件事变了**

| | smoke2 | smoke5 |
|---|---|---|
| `research_overlong`(=零梯度) | 82% | **8%** |
| research 行有效梯度比例 | 18% | **92%** |
| 会话长度中位 | 37185 | 37076(**没变**) |
| `no_finalize` | 71% | 65%(几乎没变) |
| `respLen > 32768` 占比 | 65% | 63%(**没变**) |

`research_max_sample_tokens: 24576→40960` 让 92% 的行拿回了梯度,这是实打实的。
但**会话本身的行为完全没变** —— 长度分布、撞墙比例、finalize 率都在原地。
**ctx 40960 这堵墙仍然是主要瓶颈**,YaRN(或等价的历史压缩)仍需评估。

**新的机制风险(请 CD-010 判断,我只提出不下结论)**

放宽阈值把 74 条原本"零梯度"的行变成了"有梯度且 reward=0"。在 GRPO-LOO 下,
组内有 32% 的行 reward>0 ⇒ 这些 reward=0 的行会拿到**负 advantage**。

问题在于:**它们 r=0 的原因是撞 ctx 墙(外部容量约束),不是 researcher 策略差**。
于是梯度在告诉模型"你刚才那套研究方式是错的",而实际上它只是**没说完**。
这在方向上像是在惩罚"想得深/查得多",可能把 researcher 往"早点收工"推 ——
恰好与我们希望的方向相反(`experiments` 中位 5 已低于外部 35B 的水平)。

三个可能的处理方向(不预设答案):
1. 把"撞墙截断"与"真失败"分开:`research_overlong` 或 `research_error` 命中的行
   继续 mask(零梯度),只让**正常结束**的会话进梯度 —— 即恢复旧行为但把阈值
   从"长度"改成"是否被外部约束截断";
2. 给撞墙行一个中性 advantage(不奖不罚),保留其 baseline 贡献;
3. 从源头减少撞墙(历史压缩 / 由 ctx 反推 `max_turns` / YaRN)。

方向 1 最小、最可逆,且比原来的 24576 更准确(原来是按长度一刀切,连"短会话
但恰好超阈"的行也一起砍了)。但这属于机制语义,我不擅自改。

**冒烟验收状态**:七项在全量 128 条上**全部 PASS**
(384 行布局 / r_raw 值域 / `won_bare_rate` 32.8% 未被稀释 / δ gap +25.8pp /
schema / policy 版本证据 / 一致性哨兵全零)。我把 `research_doc_overflow`
从"必须全零"改成"≤5%" —— 它是设计内的容量降级
(`selfres.py:162` 注释 `r=0:结构性长度压力`,doc 太长塞不进 executor 的
2048 prompt 预算),与 `replay_drift`/`degraded` 这类一致性哨兵不同类,
实测 1/128 = 0.8%。判据改动已同步到 `analysis/v4-selfres-diagnosis/`。

---

## 2026-08-07 · [★ P1 已修] `rollout_extra/research_*` 曲线被 executor 零占位稀释 3 倍

**症状**:v4 冒烟 step1 的 verl 指标行与我从 rollout JSON 全量算出的值,
**每一项都差精确 3 倍**:

| 指标 | rollout JSON 全量(128 条 research 行) | verl 上报 | 比值 |
|---|---|---|---|
| `research_no_finalize` | 65% | 0.216 | 3.01 |
| `research_error` | 58% | 0.193 | 3.01 |
| `research_overlong` | 8% | 0.026 | 3.08 |
| `research_doc_words` | 172 | 57.5 | 2.99 |
| `research_valid_continuations` | 6.9 | 2.29 | 3.01 |
| `research_turns` | 15.5 | 5.16 | 3.00 |

3 = 384/128 = 全 batch 行数 / research 行数。

**根因是一个正确设计的副作用**。`agent_loop._V4_RESEARCH_ZERO` 给 executor 行
补了 13 个 `research_*: 0` 占位,注释写明是为了"IF-09 批内 schema 恒齐纪律" ——
**这个设计本身是对的**,而且正是它避免了同类的锯齿字典问题(见前一条 P0)。
但 `compute_rollout_extra_metrics` 的第一个循环对每个 key 一律
`arr.mean()`,零占位就进了分母,于是每条 research 曲线被稀释 batch/research 倍。

**为什么值得当 P1 而不是小事**:被稀释后的曲线**看起来更健康**。
`no_finalize` 真值 65% 上报成 0.216 —— 若只看训练曲线,会得出"只有两成会话
没产出文档"的结论,而真相是三分之二。整个 v4 的 research 健康度监控都建立在
这组曲线上,长跑 100 步会一直被误导。

**已修**:`compute_rollout_extra_metrics` 里对 `research_` 前缀的 key 用
`role_research` 掩码只在 research 行上取均值;其余 key(executor 侧)口径不变。
复用了同函数里 `won_bare_rate` 已有的掩码逻辑,无新概念。

新增测试 2 项(`tests/test_rollout_extra_metrics_ragged.py` 现 5 项):
① research_* 不被零占位稀释(3/4 而非 3/12,400 而非 133.3);
② 非 research_ 前缀的 key 仍按全 batch 平均。本地全量 **340 passed**。

**代价**:正式 run 起跑 2 分钟后发现此问题,已停机 → 同步 → 清空 run 目录 →
重启(PID 3765962)。当时只落了 config/manifest,无实质数据损失。
判断依据:改指标层不影响 loss,本可以让它跑;但 100 步 × 约 6.6 天的曲线
错 3 倍,重启的 2 分钟代价远小于长跑后才发现曲线不可用。

**给 CD-010 的一点观察(仅供参考)**:这一条与前一条 P0 是**同一处设计的两面**——
`_V4_RESEARCH_ZERO` 让 research 字段在 executor 行有零占位(schema 恒齐 ✓),
但 executor 字段(`won`/`truncated`)在 research 行**没有**对称的零占位(✗,
被 bool 过滤器吃掉 ⇒ 锯齿 ⇒ IndexError)。若把"批内 schema 恒齐"做成**双向**
的显式契约(两类行各自补齐对方的键集),这两个问题会一起消失,
且 `compute_rollout_extra_metrics` 也不再需要靠前缀猜口径。

---

## 2026-08-07 · [★★ 实证] researcher 预算 16/4/8→8/2/4 的真实代价:文档实证强度 −22pp,而形式质检查不出来

**结论先行**:`k_default` 这个参数**根本没在起作用**;`n_exp: 8` 成了绑定约束;
降预算省下 32% 的 episode,代价是 guidance 的实证成功率从 **87.5% 掉到 65.2%** ——
而文档词数与四项形式质检**完全持平**,查不出这个降级。已按用户裁定提回
`n_exp: 12 / k_default: 4 / k_max: 4`。

**同 step1、同原始 4B 基座的受控对照**(脚本 `researcher_quality.py`,
第三参数按 `meta.json` 的 `global_step` 对齐,跨步混合会被后期更强的 policy 稀释):

| 指标 | s3(16/4/8) | v33 第一版(8/2/4) | 变化 |
|---|---|---|---|
| episode 合计 | 740 | **506** | −32%(省下的墙钟) |
| 实验次数中位 | 10.5(max 16) | 8.0(max 8) | −24% |
| **episode/实验** | **4.00** | **3.98** | **没变** |
| 全部实验成功率 | 65.0% | 55.3% | −9.7pp |
| 会话内 前半→后半 | 43.9% → **86.3%** | 43.6% → **69.1%** | 起点同,**终点 −17.2pp** |
| **最后一次实验成功率** | **87.5%**(56/64) | **65.2%**(43/66) | **−22.3pp** |
| 文档词数中位 | 1160 | 1129 | 持平 |
| 形式质检(动作词/实例名/引用证据/markdown) | 16/16 全中 | 17/17 全中 | **持平** |

**三个发现**

**① `k_default` 不生效 —— 两边 episode/实验都是 4.00。**
`tools.py:144` 是 `k_req = args.get("k", cfg.k_default)`,模型在 `run_experiment`
里**显式填 k**,`k_default` 只是它不填时的兜底。实测 35B 稳定填 4。
真正起作用的是 `k_max`(8→4 把上限压住)与 `n_exp`。
⇒ 我此前"k_default 4→2 直接把 episode 数减半"的推断**是错的**,已作废。

**② `n_exp: 8` 是绑定约束,不是宽松上限。** 实验次数分布:

- s3(上限 16):`{5:1, 8:1, 9:3, 10:3, 11:1, 12:1, 14:1, 15:2, 16:3}` 中位 10.5,自然分散
- v33(上限 8):`{5:2, 7:3, **8:12**}` —— **12/17 顶到上限**

researcher 还想继续做实验但被截断。s3 的分布说明其自然收敛点在 **10–12 次**。

**③ 形式质检查不出实证强度的下降 —— 这条最值得 CD-010 注意。**
两版文档的词数、结构、实例名、证据引用四项**完全一样好**,但背后验证过的成功率
差 22pp。若用"文档是否规范/具体"这类形式指标做 doc 质量门,会把一份实证不足的
文档判为合格。**doc 质量的可信读数只能是"它指导下的 episode 成功率"**
(与 [[doc-quality-three-metrics]] 记录的"三个不等价口径"同源)。

**顺带确立的一个正面事实**:35B researcher 的 guidance 极其有效 ——
bare policy 在 val 上 **0.0%**、训练 rollout 上 0.78%,而带 guidance 的实验 episode
达 **55.3%(最后一次 65.2%)**。抽到的文档自带对照:
> "In bare runs, **0/4 episodes succeeded**. With clear step-by-step guidance…,
> **4/4 episodes succeed**."

文档内容是真知识而非套话,含负面知识(`put`/`place`/`drop` 非法、只有 `move` 有效)
和从实验中发现的硬约束(`move` 只在站到目标 receptacle 时才生效)。
**这为 PGS 确立了教师增益的空间上限:原始 4B 不是不会做,是不知道怎么做。**

**一个 API 侧的观察(仅供参考)**:`n_exp` 与 `k_default` **进 objective_fingerprint**
(实测 8/2/4 → 12/4/4 时指纹 `ab7b59c6232a` → `658e0c93598f`),而
`rollout_gpu_mem` / `rollout_max_num_batched_tokens` **不进**(受控回退验证过)。
这个划分是合理的(前者影响文档质量即优化目标,后者纯执行档),记在这里备查。

---

## 2026-08-07 · [★★ P1 已修] 工具 schema 硬编码 "default 4",使 `research.k_default` 完全失效

**症状**:把 `research.k_default` 从 4 改到 2 之后,**episode/实验 仍恒为 4.00**
—— s3(k_default=4)与 v33(k_default=2)两个 run 逐位一致。该配置项对行为零影响,
"降 k 省一半 episode"的预期收益一次也没兑现。

**根因**:`research/prompts.py` 的 `TOOL_SCHEMAS` 里,`run_experiment.k` 的描述
硬编码了默认值:

```python
"k": {"type": "integer",
      "description": "Number of parallel episodes (default 4)."},   # ← 写死
```

这段描述**原样发给模型**(fncall 直接给 tools,text 协议由
`toolproto._TEXT_TOOLS_TEMPLATE` 把 schema JSON 塞进 system prompt)。
模型把它当权威,于是**每次都显式填 `k: 4`**。而 `tools.py:144` 是

```python
k_req = args.get("k", cfg.k_default)     # 显式填了 ⇒ 永远取不到 cfg.k_default
k_clamped = max(1, min(k, cfg.k_max))    # 只有 k_max 的 clamp 真正生效
```

⇒ **`k_default` 是一个"看起来能调、实际调不动"的参数**,只有 `k_max` 有效。
这类"配置与 prompt 各说各话"的失效很隐蔽:配置文件、manifest、effective_config
三层全都显示 `k_default: 2`,只有看 rollout 产物的 episode/实验比才发现是 4。
(与 [[param-truth-fourth-layer]] 记的"语义层"同类:三层全对行为仍可错。)

**已修**:新增 `prompts.tool_schemas_for(cfg)` —— 深拷贝 `TOOL_SCHEMAS` 并按
运行配置改写 k 的描述;`TOOL_SCHEMAS` 里的兜底文案去掉数字。
`session.py` 改用 `tool_schemas_for(router.cfg)`(router.cfg 就是 ResearchCfg)。

实际注入效果(cfg 为 2/4):
```
'Number of parallel episodes (default 2, maximum 4). Prefer the default
 unless you have a specific reason to spend more of your episode budget.'
```

新增 `tests/test_tool_schema_k_follows_config.py` 5 项:描述带上配置值、
换配置能跟随、模块常量不得再含写死的默认值、必须深拷贝(就地改会污染同进程
后续会话)、其余工具不受影响。本地全量 **380 passed**。

**建议(仅供参考)**:同类风险还在别处 —— `RESEARCHER_SYSTEM_TEMPLATE` 里
`{n_exp}` 是模板变量(正确),但工具描述这一路没有走同样的注入。若 CD-010 之后
再加"有配置对应项"的工具参数,建议一律走 `tool_schemas_for` 而不是写字面量。

---

## 2026-08-07 · [★ 方法论更正] 我用单点最高值当基准,误判了 researcher 降预算的代价

**先撤回**:上一条 `[★★ 实证] researcher 预算 16/4/8→8/2/4 的真实代价` 里
"文档实证强度 −22pp"的结论**不成立**。

我拿 s3 的 **step1(87.5%)** 作基准,但那是 s3 自身 8 个采样点里的**最高值**:

| s3 step | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| 最后一次实验成功率 | **87.5%** | 75.0% | 73.4% | **65.6%** | 70.3% | 75.0% | 81.2% | 71.9% |
| 全部实验成功率 | 65.0% | 38.9% | 54.7% | 57.5% | 45.9% | 53.4% | 48.4% | 50.1% |

s3 自己的范围是 **65.6%–87.5%**(均值 ~75%),而 v33 的 65.2%(n_exp8)/
65.6%(n_exp12)**全在范围内**;v33 的"全部实验成功率"55.3%/58.4% 甚至
**高于** s3 的均值 51.7%。

**而代价是实打实的**:n_exp 8→12 使 `timing_s/step` 从 **843s 涨到 1337s(+58%)**,
358 步从 3.5 天变 5.5 天,换来的质量差异在噪声内。故用户裁定定回 8/2/4。

**教训**:比较两个配置前,必须先测**被比较对象自身的波动范围**;单个采样点
(尤其恰好是极值)不能当基准。这与 [[training-rollout-cannot-judge-improvement]]
是同一类错误的不同表现 —— 那条讲的是"逐步曲线判不了改善",这条是
"单点基准判不了差异"。

---

## 2026-08-08 · [★★ P0 环境] Qwen3.5-9B 在 A800 起 SGLang 0.5.10,`qwen35-9b-env-plan.md` 漏了两层编译前置

**结论先行**:按 plan 装完 0.5.10 后,单卡 serve **连续崩两次**,都不是 plan 覆盖的
版本/依赖问题,而是**编译工具链前置**。两条都已修复并写进
`TRACE-OPD-Q35/env_q35_a800.sh`(新建的 q35 专用入口,不动生产入口)。
现单卡 serve 稳定 60s 就绪,门 1/2/3 全 PASS。

### 崩溃 1:triton JIT 拿不到 ptxas

不 source 环境入口直接 `python -m sglang.launch_server`,
`TRITON_PTXAS_PATH`/`CUDA_HOME` 全为空,
`create_flashinfer_kv_indices_triton` 编译时报
`TypeError: expected str, bytes or os.PathLike object, not NoneType`。
——这条是**我方操作失误**(记忆里已有"A800 起服务必须走 env.sh"),记此备查。

### 崩溃 2(★ plan 未覆盖):flashinfer 0.6.7 的 ninja JIT 缺 `cublasLt.h`

```
flashinfer/trtllm/common/cudaUtils.h:25:10: fatal error: cublasLt.h: No such file or directory
```
`$VENV/targets/x86_64-linux/include` 里**没有** cublasLt.h,生产环境
`env_sglang_train` 同样没有 —— 0.5.8/flashinfer 0.6.1 从不编这个 op 所以没暴露。
全机唯一一份在 `TRACE-OPD/venv_sglang/lib/python3.12/site-packages/nvidia/cublas/include/`。
**修复**:入口里只读引用该 donor 的 16 个 `nvidia/*/include`(不装包、不改 donor 环境)。

### 崩溃 3(★★ plan 未覆盖,最隐蔽):0.5.10 的 tvm-ffi JIT 要 **C++20**

```
nvcc warning : The -std=c++20 flag is not supported with the configured host compiler. Flag will be ignored.
sglang/jit_kernel/include/sgl_kernel/source_location.h:8:10: fatal error: version
```
A800 系统 gcc 是 **8.5.0**,无 `<version>` 头。nvcc **静默**把 `-std=c++20` 降回 c++17
再报缺头,`clamp_position` kernel 编译失败 → `SIGQUIT received. It usually means one child failed`
→ 整个 Scheduler 死。**症状伪装成"服务起不来"**,真因在编译器版本。

注意其中一个同类错误(`kvcache` kernel)只是 `Failed to load JIT KV-Cache kernel` 的
**降级 fallback,不致命**;而 `clamp_position` 致命。同一个根因,一个降级一个致命,
只看"服务起没起来"会误判成偶发。

**修复**:用 q35 环境**自带**的 conda-forge `x86_64-conda-linux-gnu-g++ 14.3.0` 作
nvcc host 编译器(`CXX`/`CC`/`NVCC_PREPEND_FLAGS="-ccbin ..."`)。
系统另有 `/opt/rh/gcc-toolset-13` 也可用,但环境自带的更符合"GPU 栈以 conda-forge 为底"。

**给 plan 的建议**:门 0 应加一条编译前置自检 ——
`nvcc -std=c++20 -c` 编一个含 `#include <version>` + `#include <cublasLt.h>` 的样例,
过了再进门 1。这比装完再崩省一轮。

---

## 2026-08-08 · [P1 工具] `q35_gate1.py` 的 `_launch_subprocesses` 检查是假阳性

门 1 报 `FAIL: sglang import 面断 1/16:
['sglang.srt.entrypoints.http_server._launch_subprocesses']`。

**实际不是断点**。0.5.10 把它从模块级函数移入 `Engine` 类(`engine.py:619 @classmethod`),
而 **verl 0.9.0.dev0 早就按版本分支处理了**:

```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py:407
# For SGLang main branch or version >= 0.5.10
if version.parse(sglang.__version__) >= version.parse("0.5.10"):
    from sglang.srt.entrypoints.http_server import Engine
    ... = Engine._launch_subprocesses(server_args=..., init_tokenizer_manager_func=..., ...)
```
实测:classmethod 存在、verl 传的 4 个关键字参数**全部支持**、
`engine.{init_tokenizer_manager,run_scheduler_process,run_detokenizer_process,_set_envs_and_config}` 全在。

**建议**:该符号的检查改成版本条件式 —— `>=0.5.10` 查 `Engine._launch_subprocesses`
且校验签名,`<0.5.10` 才查 `http_server._launch_subprocesses`。否则每次都会报一个假 FAIL,
消耗判读注意力(且有把真 FAIL 淹没的风险)。

---

## 2026-08-08 · [★ P1 工具] `sgl_probe_q35.py` 的 thinking 判据漏了最有效的一维

探针输出 `VERDICT: INFO — 两档均无 thinking(模型可能无 thinking 模式,或 kwarg 被静默忽略)`。
**这个判读不成立**。同一问题("9.11 和 9.9 哪个大")三档实测:

| 档 | 输出长度 | 开头 |
|---|---|---|
| default | 1361 | `Thinking Process:\n\n1. **Analyze the Request:**...` |
| **think_off** | **13** | `9.11 比 9.9 大。` |
| think_on | 1361 | 同 default |

`enable_thinking=False` **确实生效,差 100 倍**。探针误判的原因:它只检查
`<think>` 标签与 `reasoning_content` 字段,而 **Qwen3.5 把思考以纯文本
`Thinking Process:` 写进 content**,两个判据都不命中。

**建议**:thinking 判据加"三档输出长度比"这一维 —— off/default 长度比 <0.3
即可判定 kwarg 生效,不依赖模型用哪种标记形式。

(附:non-thinking 下模型答错了这道经典陷阱题(9.9>9.11)。对 ALFWorld agentic 任务
的影响待 smoke 观察,此处只记读数不外推。)

---

## 2026-08-08 · [INFO 实测] Qwen3.5-9B 的三个架构成本读数(实机)

1. **KV = 32 KB/token**(非 16)。实机反算:`KV Cache #tokens: 771195, K 11.77GB + V 11.77GB`
   → 23.54GB/771195 = 32.0 KB/token。来源是 8 个 full_attention 层
   (`layer_types` 实测 `{linear_attention: 24, full_attention: 8}`)×4 kv_heads×256 head_dim×2B×2。
   对比 Qwen3-4B 的 144 KB/token,**同显存下 token 容量约 4.5×**。
2. **tail/full 耗时比 0.189**,而 162 全注意力语境实测 **0.024**。
   线性注意力下 tail 复用的收益差约 8 倍 —— SPS 的成本账要按这个重算,
   不能沿用全注意力的经验值。
3. radix 增量(共享前缀+新尾段)0.294s,介于 tail 重复 0.157s 与 full 冷 0.830s 之间,
   说明 mamba 层对 logprobs **有部分复用但不完全**。

单卡 80GB / mem-fraction 0.80 / ctx 32768 实测:权重 17.62GB,KV 23.54GB(771195 tokens),
cuda graph 0.44GB,就绪 60s。

---

## 2026-08-08 · [★ P0 bug] `launch_train.py` 的 fused_kernels override 用了单 `+`,启动即崩

**结论先行**:`use_fused_kernels` 这段新代码把 hydra override 写成
`+actor_rollout_ref.model.fused_kernel_options.impl_backend=torch`,
而该键**在 verl 默认配置里已存在**,单 `+`(append-only)会抛:

```
hydra.errors.ConfigCompositionException: Could not append to config.
An item is already at 'actor_rollout_ref.model.fused_kernel_options.impl_backend'.
```

9B 冒烟因此**秒崩在 hydra 组装阶段**,连 Ray 都没起。已改为 `++`(覆盖或添加)
并加注释说明原因;改后 dry-run 复验 `++...impl_backend=torch` 正常,
objective fingerprint `f3c767801c30` 不变(fused_kernels 是执行档,不进指纹,符合设计)。

**为什么之前没暴露**:V33 树没有这段代码(`use_fused_kernels`/`fused_kernels_backend`
是本地新增,`config.py` 与 `launch_train.py` 两处),9B 是**第一个真正开这个开关的档**。

**建议**:`launch_train.py` 里所有 `+xxx=` 形式的 override 值得过一遍 ——
凡是 verl 默认配置里已有的键都应该用 `++`。单 `+` 只在确认是全新键时才安全,
而"是不是全新键"随 verl 版本变化,`++` 在两种情况下都对,没有下行风险。

---

## 2026-08-08 · [INFO] 9B 的 SP 档由实测激活曲线定为 4(用户裁定)

单卡实测(B=1、梯度检查点开、flash_attention_2):

| 每卡 tokens | 净激活 | logits 张量 |
|---|---|---|
| 2048 | 1.0 GiB | 0.95 |
| 4096 | 2.0 GiB | 1.89 |
| **8704** | **14.0 GiB** | 4.03 |
| 17408 | 单卡 OOM | 8.06 |

4096→8704 是 **7 倍非线性跳增**,故不在该曲线上外推。`max_token_len_per_gpu=34816`
下 SP=4 每卡 8704 tokens 正好落在**实测**档;SP=2 的 17408 只能外推,不采信。

补充:实际启动命令里 `param_offload=True` + `optimizer_offload=True`,
参数与 Adam 状态都 offload 到 CPU,所以 GPU 静态开销比按 FSDP 分片估的
16.7 GiB **更低**,SP=4 余量比表上还大。

另:9B 稳态训练吞吐实测 **2799 tok/s/GPU @seq2048、2893 @seq4096**,
对理论下限(6·N·T / 200 TFLOPS)的效率 **75%–77%**。首次 triton JIT 编译
额外 +1.7s(seq2048) / +13.0s(seq4096),只在首步付 —— 不要把首步墙钟当稳态。

---

## 2026-08-08 · [★★ P0 实证] 9B 首个 step 的四组读数:截断 68%、门开启率越界、显存崩溃形态

数据源:`TRACE-OPD-Q35/runs/q35_9b_formal.gpumem065.crash/rollouts/step_00001`
(217 行,step1 跑到 85% 时因显存崩,产物完整可用)。

### ① 显存崩溃:不走标准 OOM 路径,`ALERT` 正则会漏

`rollout_gpu_mem=0.65` 下显存峰值 **79725/81920 = 97.3%**,崩在
`hook.py:459 _verify_rows` —— teacher verify 要在 rollout engine 之外再做一轮前向。
**日志里没有 `CUDA out of memory`**,而是:
```
c10::cuda::MemPool::~MemPool()
Fatal Python error: Aborted
Subprocess scheduler_0 crashed with exit code -6   ← SIGABRT
```
**建议**:任何 OOM 监控的正则都要含 `MemPool|exit code -6|Fatal Python error: Aborted|
EngineDeadError|crashed with exit code`,只匹配 "CUDA out of memory" 会漏掉这一整类。
已降到 0.55(显存 60603→47131 MiB/卡)。

### ② 截断率 68%/73%(★ 最影响结论的一条)

| 臂 | won | respLen 中位 | **截断率** | n |
|---|---|---|---|---|
| bare | 3.1% | 20187 | **68.0%** | 128 |
| steered | 2.2% | 16956 | **73.0%** | 89 |

`response_length=32768` 下仍有近七成轨迹撞上限。按 [[truncation-masks-capability]]
的实证(「赢&截断」恒为 0),**最多只有 32% 的轨迹有机会赢,当前 won 读数不可信**。
这属于"影响效果的限制",已报用户裁定,未擅自改。

### ③ `gate_open_rate` 中位 30.3%,越出 `gate_rate_bounds` 上界

目标 `gate_target_rate=0.10`、允许区间 `[0.05, 0.25]`,实测**中位 30.3%、max 67.2%、
min 0.0%**。自适应门没收敛回目标带。9B 的截断熵分布(top-16 口径,全生成位无偏,
累计 1,352,482 个观测位):

| P25 | P50 | P75 | P90 | P95 | P99 |
|---|---|---|---|---|---|
| 0.0039 | 0.0039 | 0.1211 | 0.6992 | **1.0586** | 1.6914 |

对照 4B P95=1.276、1.7B P95=0.043。**9B 的 P95 与 4B 只差 17%,但分布形状极度右偏
——中位数仅 0.0039,一半以上位置熵接近零**。`entropy_theta_init=1.28` 这个 4B 占位值
在 9B 上不是"太高导致门关死",而是配合自适应机制后**开得过多**。若要收敛回 10%,
方向是调 `gate_ema_beta`/`gate_rate_bounds` 或直接下调 theta,但都会动 fingerprint,
待用户裁定。

### ④ 已排除的两个虚警(记此避免他人重踩)

- **`guidance_hash` 217 行全等于 bare 空值 hash —— 这是正确设计**,不是文档没注入。
  `agent_loop.py:1036/1059` 两臂 episode 都传 `guidance=""`;文档只用于
  `agent_loop.py:407-417` 构造 **teacher 的特权前缀**。学生永远看 bare 上下文,
  只有采样分布被倾斜 —— 这正是信息状态一致性的要求。
- **researcher 链路正常**:`research_experiments=8`、`research_episodes=16`
  (=8×k2)、`research_doc_words` 中位 1168(840–1482)、16/16 任务都产出了文档。
  文档质量实查很好(含可执行命令序列 + 4 类失效模式,其中"`none`×20 循环"这种
  带具体计数的观察说明确实读了轨迹)。

### ⑤ 方法论自省(给后续分析者)

我在这批数据上**连续四次误判**,全部源于**猜字段名而非先 dump schema**:
把 `steer_n_tilted`(不存在)当干预计数→误判"覆盖率 0%、机制没跑";
在不存 prompt 的 json 里查 prompt→误判"文档没注入";
用 `doc_digest`/`doc_hash`(不存在)→误判"0 种文档指纹"。
**正确字段**:干预看 `steer_gate_open_rate`/`steer_n_base_open`/`steer_n_candidates`,
文档看 `research_doc_words`/`research_doc_missing`/`trajectory.guidance_hash`。
教训:分析新 run 的第一步永远是打印一条样本的完整 schema,不要凭其他 run 的经验猜。

---

## 2026-08-08 · [★★★ P0] 9B 上 145/217 轨迹 outcome=`DEGENERATE:format`,现场移交

**这条只报现场与数据,不下判断,请 CD-010 分析。**

数据源:`TRACE-OPD-Q35/runs/q35_9b_formal.gpumem065.crash/rollouts/step_00001`(217 行,
step1 跑到 85% 时因显存崩,产物完整)。模型 = 原始 `Qwen/Qwen3.5-9B`(未经任何 SFT)。

### 一、outcome 构成

| truncated | outcome | 行数 |
|---|---|---|
| True | **DEGENERATE:format** | **145** |
| False | FAILED | 59 |
| True | FAILED | 7 |
| False | SUCCESS | 6 |

truncated 行中 `respLen >= 97%×32768` 的:**7/152 = 5%**。

(注:我先前在本文件报过"截断率 68%,建议提 response_length"。按上表,truncated 中
只有 5% 的 respLen 接近预算上限。`response_length` 32768→38912 的改动已按用户裁定落地。)

### 二、首次格式失败位置分布(全部 145 条)

```
首次 acted=False 的 step:  中位 13   min 2   max 29
  ≤ step  1:   0/145 =  0%
  ≤ step  3:   6/145 =  4%
  ≤ step  5:  17/145 = 12%
  ≤ step 10:  53/145 = 37%
  ≤ step 15:  81/145 = 56%
轨迹总步数 中位 22    invalid_format 中位 4(max 10)
```

### 三、成功 / 失败轨迹的对照读数

| | turns | invalid_format | respLen | 首次 acted=False |
|---|---|---|---|---|
| SUCCESS | 8 | 0 | 6063 | 无 |
| DEGENERATE | 18 | 5 | 14827 | step 13 |

### 四、失败时的模型输出原文

同一条轨迹,step 12 与 step 13 相邻:

```
step 12 [acted=True]
"I've been trying to complete this task for many turns but cannot find or obtain an
alarmclock. The inventory is empty, and no location contains an alarmclock that I can
access. ... Let me try one final approach - maybe I should just execute the move command
one more time to see if it'll work this time."

step 13 [acted=False]
<reasoning>
user
assistant
<think>

</think>

<

</think>

<reasoning>
user
assistant
<think>
user
...(反复)
```

另一条轨迹的 step 26:
```
<reasoning>
The "move alarmclock to
user

assistant
<think>
assistant
<think>
...(反复)
```

环境返回:
```
Nothing happens. (Your reply had no <action>...</action> block — respond with
<reasoning>...</reasoning> then <action>one command</action>.)
```

另有部分 step `acted=True` 但 action 解析为空字符串,环境返回 `Nothing happens.`
(不带格式提示)。见移交文件中 DEGENERATE 轨迹的 step 27 / 30。

### 五、现场文件(可直接查看,无需连服务器)

`trace_opd-pgs/analysis/q35-9b/trajectories/`
- `DEGENERATE__*.txt` ×3 —— 逐 step 完整原文**不截断**,含 `acted=` 与环境返回
- `SUCCESS__*.txt` ×2 / `FAILED__*.txt` ×1 —— 对照
- `*.json` —— 同名精简 json(已去 token_ids 等大字段)
- `_DEGENERATE_summary.json` —— 全部 145 条汇总(首次失败位置/轮数/invalid_format/respLen)

分析脚本:`analysis/q35-9b/{export_traj,dump_traj,slow_diag,step1_accept,ent_calib_q35}.py`

### 六、代码位置(省去重复定位;以下均为代码事实,非推断)

- 终止判据:`core/episode.py:219-226` —— `consecutive_bad >= MAX_CONSECUTIVE_FORMAT_FAILURES`
  时置 `truncated=True` + `outcome=DEGENERATE:format` + `error="no parseable action repeatedly"`
- 协议常量:`core/protocol.py:46/50/54` —— `MAX_ASSISTANT_TURNS=40`、
  `GEN_MAX_TOKENS_PER_TURN=2048`、`MAX_CONSECUTIVE_FORMAT_FAILURES=3`
- `core/template_delta.py:33` —— `_CHAT_KWARGS = {"enable_thinking": False}`(硬编码)
- `core/template_delta.py:113` —— 断言文案 `"glue does not start with a stop token"`
- `train/backends.py` `EngineGenerate.__call__` —— 未见 `stop` 字段透传
- `train/agent_loop.py:1023/1039/1062` —— `sampling_params` 的传入路径;
  **grep 该文件未见 `"stop"` 键**

### 七、可能相关的其它实测(仅列读数)

- 门 2 单轮 thinking 三向对比(单轮短问答,非多轮会话):
  default 输出 1361 字符 / `enable_thinking=false` **13 字符** / `enable_thinking=true` 1361 字符。
- 本 run `model_path` = 原始 `Qwen/Qwen3.5-9B`,未经 SFT。
- 熵门:`steer_gate_open_rate` 中位 30.3%(max 67.2%,目标 10%,bounds 上界 25%);
  该统计的候选位包含上述格式失败段的生成位,未做分离。
- researcher 侧:16/16 任务产出文档,`research_experiments=8`、`research_episodes=16`、
  文档词数中位 1168。其失效模式章节原文含:
  > "**Format errors (most common)** — Symptom: `Your reply had no <action>...</action>
  > block`. Cause: The executor outputs reasoning text but omits or corrupts the
  > `<action>` tags. This is the single most frequent failure mode."

### 八、需要 CD-010 定夺的事

现象与代码事实如上。是否与 stop token 透传、`enable_thinking` 在多轮下的作用范围、
或格式冷启动有关,由 CD-010 判断。若需要我在 9B 上跑对照实验(如加 stop 的 A/B、
不同 `MAX_ASSISTANT_TURNS` 下的 `invalid_format` 曲线、或换 SFT ckpt 的对照),告知即可。

---

## 2026-08-08 · [★★ 闭环] EOS 修复(b2f62e1)在 A800 实测生效,干净基线出炉

修复已同步 A800(`template_delta.py`/`backends.py`/`sgl_probe_q35.py` 三文件逐字节对拍一致),
`tests/test_engine_stop_tokens.py` + `test_backends_routing.py` 本地与 A800 各 8 passed。

### 一、stop stage 验收(对 `--skip-tokenizer-init` 起服,与生产同构)

```
A(no stop_token_ids):        len=4  first_im_end=1  overrun=YES   ← 缺陷复现
B(stop_token_ids=[248046]):  len=2  last=248046  im_start_inside=False  n_im_end=1  → PASS
```

### 二、修复前后对照(同配置、同模型,唯一差异是 stop_token_ids)

| 指标 | 污染批次 | 修复后 |
|---|---|---|
| val won (valid_seen) | 0.746% / 0.0%(两次) | **30.71%** |
| val won (valid_unseen) | — | **26.87%** |
| `DEGENERATE:format` | 145/217 = 67% | **0/274** |
| 截断率 | 68% / 73% | **0.0%** |
| `invalid_format` 均值 | — | seen **0.0%** / unseen **1.5%** |
| respLen 中位 | 18952 | **2370** |
| 每轮 token 中位 | 800 | **88** |
| val 墙钟 | 40+ 分钟 | **300s** |

每轮 token 800→88 与"86.5% 垃圾"精确吻合(1−0.865=13.5%≈1/7.4)。

### 三、9B raw 干净基线(未经任何 SFT)

```
valid_seen    n=140  won 30.71%  respLen中位 2281  turns中位 30  SUCCESS 43 / FAILED 97
valid_unseen  n=134  won 26.87%  respLen中位 2489  turns中位 30  SUCCESS 36 / FAILED 98
赢轨迹 n= 79  respLen中位 1625  turns中位 17
输轨迹 n=195  respLen中位 2548  turns中位 30   ← 正好卡在 max_env_steps=30
```

供 CD-010 参考的两点(**只报读数,不做裁定**):
1. 9B raw zero-shot 已有 30.7%/26.9%,对照 4B raw 0.73%、4B SFT 后 47.4%。
   诊断文档里列的"假设 C:需要格式冷启动 SFT"在此读数下可重新评估。
2. 输轨迹 turns 中位 = `max_env_steps` 上限(30),赢轨迹 17。失败主因已从格式退化
   转为环境步数预算。是否调该预算属效果相关参数,未动,待裁定。

### 四、我方已做的回滚

`response_length` 38912 → **32768**、`eval.response_budget` 同步回滚。
理由:该上调的依据("截断率 68%")属污染读数,按诊断文档"勿基于污染批次裁定下游问题"回滚。
objective fingerprint 仍为 `f3c767801c30`。

污染批次完整归档 `TRACE-OPD-Q35/runs/q35_9b_formal.noeos/`(日志 `.noeos.log`),未删除。

---

## 2026-08-08 · [★★★ 用户需求转达] 正式实验阶段的任务环境与协议 —— 见独立需求文档

用户就**方案机制正式试验阶段**提出了成体系的需求,内容涉及 benchmark 选定、
统一协议、已引入环境的改造与删除、以及未引入环境的引入流程。

**需求全文见**:
`docs/collab/cd010-cd006/requirements-formal-experiment-benchmarks.md`

该文档为**用户需求的忠实转录**,不含 CD-006 的分析、建议或调研结论,请以其为准。
提要(仅为索引,细节以需求文档为准):

1. **选定 7 个任务环境**:ALFWorld、ScienceWorld、WebShop、SearchQA、SWE-bench、
   SpreadsheetBench、TIR-Math。
2. **统一采用 OpenAI 格式 Function Call**,且**在任务环境层实现**;
   **不引入到方案机制中**,机制保持通用泛化、可对接任何协议。
3. **已引入且在名单内的**:保留 setting,但协议改为 Function Call;
   工具与环境信息的描述要**精心实现**,不得敷衍。
   **不在名单内的直接完全删除**(据现状映射,`envs/appworld` 属此列)。
4. **未引入的**(ScienceWorld、WebShop、SearchQA、SWE-bench):必须**一个一个**
   走"仔细搜索调研 → 规划设计 → **与用户充分探讨协商** → 用户确认后才引入"的流程;
   协商内容至少含训练数据、测试数据、格式协议与默认系统提示、评估方案、
   Ground Truth 反馈信息、任务特定信息、搜索调研分析发现等。
5. 用户要求 CD-010 开展:深入搜索调研、相关工作与代码实现的观察分析、
   引入选中 benchmark、实现对应 agentic 格式协议。

CD-006 侧不对上述需求作技术判断;若 CD-010 在执行中需要 CD-006 提供实验侧
现状数据(如现有环境的实测读数、协议实现现状),告知即可。

---

## 2026-08-08 · [★★★ P0 实证] text 协议把整篇文档转义进 JSON,小模型撑不住;换 fncall 后错误率 45.2%→0

**结论先行**:在 162 用 Qwen3.5-9B/4B 做 researcher 探针(ALFWorld,32 任务分层抽样)时发现,
`research.tool_protocol: text` 下工具调用被大比例拒绝。**根因不是模型不会调工具,
而是协议要求把整篇 markdown 文档转义进 JSON 字符串**。换成 Qwen3.5 原生 XML
function-call(服务端 `--tool-call-parser qwen3_coder` 负责转义)后,**错误率归零**。

### 一、text 协议下的实测(同一批 32 任务,seed=42)

| | 9B | 4B |
|---|---|---|
| assistant 消息 | 416 | 312 |
| `TOOL ARGUMENTS ERROR` | 40 | 141 |
| **错误率** | **9.6%** | **45.2%** |
| 受影响会话 | 22/60 | 11/15 |
| 其中 **JSON 转义/截断** | 39(**97.5%**) | 139(**98.6%**) |
| markdown 围栏包裹 | 1 | 2 |

失败集中在 `finalize`(它带整篇 experience_doc)。文档 500–1000 词,含大量 `"`、换行、
反引号,作为 JSON 字符串值需逐字符转义,**一处错整个 arguments 作废**
(`research/tools.py:105-111` 拒绝执行且不扣预算)。模型越小越撑不住 —— 但注意
**9B 也有 9.6%**,这不是"4B 的问题",是**协议与模型能力的结构性错配**。

### 二、换 fncall 后的实测(同任务、同模型、同 ctx)

`sglang 0.5.10` **无 `qwen3_xml`**,但有 **`qwen3_coder`**,其 XML 形态
(`<function=名><parameter=键>`)与 Qwen3.5 chat_template 逐字一致。实测可用:

```
（服务端 parser 单元验证,4B）
✓ tool_calls 已填充  name='finalize'
✓ arguments 是合法 JSON;experience_doc 129 词 676 字符
  文档含换行 True / 含双引号 True / 含反引号 True
```

4B 单会话端到端验证:

| | text | **fncall** |
|---|---|---|
| `TOOL ARGUMENTS ERROR` | 45.2% | **0** |
| assistant 带 tool_calls | 靠客户端正则 | **19/19** |
| experiments_used | 6 | **15** |
| episodes | — | **56** |
| doc 词数 | 530 | **1044** |

**机理**:XML 的 `<parameter>` 块内容读到闭合标签为止,**转义由服务端做,模型不碰**。

### 三、给 CD-010 的两点(供判断,非结论)

1. 用户已定"正式实验统一采用 OpenAI 格式 Function Call"
   (见 `requirements-formal-experiment-benchmarks.md`)。上述实测是该方向的独立证据:
   **凡 tool 参数里要塞长文本的场景,text/JSON 协议对 ≤9B 模型都不安全**。
2. `common/config.py:619` 现有两条校验会挡住 fncall:
   `research_rl 下 tool_protocol 必须为 text`、`research.backend 必须为 external`。
   前者的理由(RL 训练时 verl rollout 是 token-in-token-out,拿不到服务端 parser)是成立的
   —— 这意味着**若 researcher 要做 RL 训练,fncall 这条路在训练期不可用**,需要
   CD-010 判断如何取舍(例如训练期 text、推理期 fncall 是否会造成分布不一致)。
   探针配置 `162-q35-researcher-probe-fncall.yaml` 为绕开校验关掉了 `research_rl`,
   仅用于本次探针,不作为正式配置。

CD-006 侧数据与脚本:`/data4/ws/PGS/researcher_probe_q35/`(162),
分析脚本 `analysis/q35-researcher/quality.py`,会话查看器 `scripts/researcher_viewer.py`。

---

## 2026-08-08 · [需求文档更新] 已补入 §6「训练期的协议转换要求」

用户看过 fncall 探针结果后确认了统一 Function Call 的方向,并就**训练期必须转换为
纯 text 格式**这一点提出了新需求,要求与既有需求一并实施。

**需求文档已更新**:`docs/collab/cd010-cd006/requirements-formal-experiment-benchmarks.md`
新增 **§6 训练期的协议转换要求**(6.1 确认统一 Function Call / 6.2 训练时必须转换成
纯 text 格式 / 6.3 由开发分支实施,并要求与 §1–§5 一并实施),以及
**附二:§6 相关的实测背景(客观事实)**。

请以需求文档为准;本文件 2026-08-08 那条【★★★ P0 实证】条目是同一批实测的详细版,
可作为技术细节参考。

---

## 2026-08-08 · [★★★ P0 bug] `episode.max_tokens_per_turn` 是 dead config —— 配置 4096,实际生效 2048

**结论先行**:`episode:` 段三个键里,`max_env_steps` 与 `max_assistant_turns`
经 `configure_episode_caps()` 正常注入,**唯独 `max_tokens_per_turn` 没有任何
注入点**,消费方读的是模块级硬编码常量。用户 2026-08-07 的裁定
「任务执行 agent 单轮 2048→4096」**从未生效**。

### 复现(本地,无需 GPU)

```python
cfg = load_config("configs/experiments/162-pgs-s4-selfres-1p7b.yaml")
protocol.configure_episode_caps(cfg.episode.max_env_steps,
                                cfg.episode.max_assistant_turns)
```

实际输出:

```
配置 episode.max_tokens_per_turn = 4096
注入后 protocol.GEN_MAX_TOKENS_PER_TURN = 2048   ← **未生效**
对照   MAX_ENV_STEPS       30 | 配置 30 → 生效
对照   MAX_ASSISTANT_TURNS 40 | 配置 40 → 生效
```

### 事实链

| 位置 | 内容 |
|---|---|
| `common/config.py:203` | `max_tokens_per_turn: int = 2048` 键定义存在 |
| `configs/pgs.yaml:74` | 模板有该键,注释「2026-07-30 用户裁定改配置项」 |
| `tests/test_162_selfres_config.py:62` | 断言 `== 4096` —— **只断言读到,未断言行为** |
| `core/protocol.py:61-68` | `configure_episode_caps()` 只 `global MAX_ENV_STEPS, MAX_ASSISTANT_TURNS` |
| `core/episode.py:182` | `turn_max = min(protocol.GEN_MAX_TOKENS_PER_TURN, budget)` ← 硬编码 2048 |
| `research/continuation.py:232` | 同上,同一个硬编码常量 |

`grep -rn "GEN_MAX_TOKENS_PER_TURN" src/` 全库只有三处:定义 + 两处消费,
无任何赋值点(除测试的 monkeypatch)。

### 影响面

1. 所有历史实验的 executor 单轮上限实际都是 2048,与配置声明不符
   —— 含 162 v4 全部 run(配置写 4096)。相关读数(respLen 分布、
   num_turns、截断率)的解释需以 2048 为准。
2. 该键是用户新方案「轨迹长度由 max_turns × per-turn 控制」的核心旋钮,
   不修则该方案在 executor 侧不可实施。

### 与既有事故同型

与 `param-truth-fourth-layer`(seq-mean 事故)同类:配置层/manifest/verl 实收
三层都对,**语义层断链**——消费方带默认值,静默降级。建议修复时一并加
行为级断言(不只断言 `cfg.x == v`,而要断言注入后 `protocol.X == v`)。

### 用户需求(已写入需求文档 §7)

需求文档 `requirements-formal-experiment-benchmarks.md` 新增
**§7 轨迹与上下文预算的配置化要求**:7.1 轨迹长度由轮数×单轮上限控制、
7.2 本 P0、7.3 **per-benchmark 的轨迹/上下文配置能力**、7.4 长轨迹必须
保留训练梯度。请以需求文档为准。

---

## 2026-08-08 · [需求文档更新] §7.5 上下文上限与轨迹有效长度须拆为两个量

用户在准备 Qwen3.5-4B v3.3×v4 实验、逐项核对预算链时提出本需求,并明确
**请 CD-010 理解、分析、思考,设计恰当、干净的改进完善方案**。

用户意图(原文见需求文档 §7.5):
- 模型总上下文由一个**配置参数**控制,是一般不会达到的上限(举例 128K);
- 推理生成与训练时单条轨迹的最大有效 tokens,由**派生值
  `max_turn × per-turn max_tokens`** 控制;
- 二者是**两个独立的量**,不应由同一个量兼任;
- 用户推测现状可能受 **verl 或相关框架的约束**所致。

CD-006 侧只提供事实、不提方案(需求文档**附三**已整理):现状代码位置
(`agent_loop.py:467` / `backends.py:196` / `episode.py:178`)、两侧口径不一致
(researcher 扣 prompt、executor 不扣)、`stream` 同时装生成与工具返回、
verl `_pad_token_ids` 只补不截且 `padding="max_length"` 是固定维度、
4B researcher 探针 64 条会话的生成/工具返回/总量分位、162 的 503GB 内存
与 1.7B 冒烟 `cpu_memory_used_gb=177.2` 的对照基线。

与本节相关但已单独成条的既有条目:§7.1(轮×单轮主导)、§7.2(P0 dead config,
CD-010 已修 `9c9568a`)、§7.3(per-benchmark 配置能力)、§7.4(长轨迹保留梯度,
CD-010 已实现 `4baa455`)。§7.3 的 per-benchmark 要求同样适用于 §7.5 引入的
新参数。

---

## 2026-08-08 · [需求文档更新] §7.6 executor 侧也需要独立 session_token_budget

用户裁定本次 Qwen3.5-4B 正式实验的两个预算:**researcher 48K / 任务执行 16K**。
§7.5 已给 researcher 落地控制器,executor 侧尚无对应键,故该裁定在现有键位上
无法表达 —— 详见需求文档 §7.6(含 CD-006 核对的四条现状事实:EpisodeCfg 无流
预算键、executor 预算取自 train.response_length、该值同时兼任 verl padding 维度
与 executor 流预算两职、§7.5 校验以该单值为准)。

本次实验的处置(CD-006 侧,待用户裁定后定稿):researcher 侧按 48K 配置;
executor 侧暂由轮预算(max_assistant_turns × max_tokens_per_turn)主导,
与用户要求的 16K 存在差异,已向用户明确报告。

**补充(同日)**:按 CD-010「四类量」模型复核后确认,缺口精确落在 **③ 流预算**
这一层 —— executor 与 researcher 各自流预算的语义都对,但
`train.response_length` 当前一身两职(executor 流预算 + verl 唯一 padding 维度),
`launch_train.py:299`/`:356` 两个 hydra 键绑同一配置值。实测
`response_length: 16384` + `session_token_budget: 49152` 被 §7.5 校验 FAIL,
报错与 info 行已附在需求文档 §7.6。CD-006 侧本次实验待该项落地后按
用户裁定值(16K/48K)起跑。

---

## 2026-08-08 · [★★★ P0 阻断] v4 research 行在 step-1 必崩:policy version 证据在首次权重更新前恒为 None

**结论先行**:`selfres.py:209` 的 fail-loud(`no rollout policy version evidence`)
在**任何 run 的 step-1 都会触发**,因为 `min/max_global_steps` 的源头
`self.global_steps` 只在**权重更新时**才被赋值,而 step-1 的 rollout 发生在
第一次权重更新**之前**。executor 行因为处理方式不对称而静默通过,
research 行则整批崩掉。Qwen3.5-4B 正式 run(`q35_4b_v33_selfres_162`)
因此在 step-1 阻断,research 行只落盘 **2/128**。

### 一、根因链路(逐跳,全部为 vendored verl 与本仓代码)

```
① workers/rollout/sglang_rollout/async_sglang_server.py:257
   self.global_steps = None                       ← 初始值
② 同文件 :867-869  async def set_global_steps(...) ← 唯一赋值入口
③ workers/rollout/sglang_rollout/sglang_rollout.py:369
   if self._engine is not None and self._is_server_tp_leader():
       await self._engine.flush_cache()
       if global_steps is not None:
           await self.server_actor.set_global_steps.remote(global_steps)
   ↑ 位于 _update_weights 内 —— **只有权重更新才会调用它**
④ 同文件 :833  extra_fields = {"global_steps": self.global_steps}   ← 此时为 None
⑤ workers/rollout/llm_server.py:273-275
   global_steps = output.extra_fields.get("global_steps")            # None
   output.extra_fields.setdefault("min_global_steps", global_steps)  # key 存在,值 None
   output.extra_fields.setdefault("max_global_steps", global_steps)
⑥ src/trace_opd/train/backends.py:47-50(EngineGenerate)
   for key in ("min_global_steps","max_global_steps"):
       if key in out_extra: extra[key] = out_extra[key]              # 照拷 None
⑦ 同文件 _track_policy_version
   raw = (extra or {}).get(key); if raw is None: continue            # 被跳过
⑧ policy_version_fields() → {} → selfres.py:209 raise RuntimeError
```

**关键的不对称**:同一份 None,两侧处理完全不同 ——

| | 代码 | 行为 |
|---|---|---|
| executor | `core/episode.py:212` `if key in out.extra: gen_extra.setdefault(...)` | 静默跳过,轨迹正常落盘 |
| research | `train/selfres.py:207-210` `if not version_fields: raise` | **fail-loud,整行崩** |

而 `agent_loop_tq.py:136` 的 `asyncio.wait_for(asyncio.gather(*tasks))` 会让
一个 research 行的异常**带垮同一 prompt 的所有行**。

### 二、实测对照(同一台 162,同一份 vendored verl)

| | 1.7B 冒烟 `pgs_s4_smoke5` | 4B 正式 `q35_4b_v33_selfres_162` |
|---|---|---|
| `val_before_train` | **False** | **True** |
| step-1 research 行落盘 | **128/128**(16 任务 ×8) | **2/128** |
| step-1 executor 行 | SUCCESS 117 / FAILED 129 / DEGENERATE 10 | SUCCESS 50 / FAILED 126 |
| `rollout_extra/min_global_steps/mean` | **0.0** | 拿不到(fail-loud) |
| 结局 | 正常完成 step-1 | step-1 阻断 |

`val_before_train` 是两者唯一的相关差异(CD-006 观察,**不是因果结论**);
1.7B 的 formal run(`val_before_train=True`)在 2026-08-07 被 SIGTERM 停在
step:0,从未跑到 step-1,故无法用于验证。

### 三、给 CD-010 的三点(供判断,不含方案)

1. fail-loud 本身的动机成立(`selfres.py` 注释所述:trainer 的
   `_compute_metrics` 对每行 `np.array(..., dtype=int)`,None 会 TypeError
   崩整步)。问题在于**它守的那个前提在 step-1 不成立**。
2. 两侧对同一缺失值的处理不对称(executor 静默 / research fail-loud),
   使同一个根因表现为"executor 正常、research 全崩",极易误诊为 research
   侧的 FC 或 engine 后端问题 —— CD-006 最初正是往那个方向查的。
3. `llm_server.py:274` 的 `setdefault(key, None)` 让下游的 `key in extra`
   判定失去意义(键在、值空),`EngineGenerate` 因此把 None 一路带下去。

### 四、CD-006 侧现状

正式 run 已停(GPU 已释放,step-0 val 结果保留:
seen won 7.14% / unseen won 4.48%,`rollouts/step_00000_val/` 274 条齐)。
实验目录 `experiments/q35-4b-v33-selfres-162/`(config + run.sh + README)已就绪,
待本项修复后按原配置重起。

### 【更正与补强】2026-08-08 同日 —— 结论更强,因果换了

用户质疑「A800 也用 SGLang server 为何能跑」后复核,两处更正:

**更正一:A800 不构成反证。** `a800-pgs-v33-4braw-asym.yaml` **无 `research_rl` 段**
(external 35B researcher,researcher 会话走 HTTP 到独立服务,不产生训练行),
`selfres.py:209` 的 fail-loud 只在 `research_rl.enabled=true` 的 research 行上执行,
A800 不走该路径。

**更正二:上一条里「1.7B 冒烟 min_global_steps=0.0」是聚合假象,须撤回。**
直接读两个 run 的 rollout 落盘:

```
1.7B 冒烟 smoke5:  research 128/128 = (None, None);  executor 256/256 = (None, None)
4B 本次 run:       research  35/35  = (None, None);  executor 176/176 = (None, None)
```

**两个 run 的落盘真值都是 None。** 先前引用的
`rollout_extra/min_global_steps/mean: 0.0` 是聚合阶段 None→0 的产物,非落盘值。
因此 `val_before_train` 的差异是巧合而非原因 —— 1.7B 冒烟能过,是因为它跑在
`selfres.py` 加入 fail-loud **之前**(该 fail-loud 的注释自述依据即
"2026-08-07 162 v4 冒烟实测",时间上晚于 smoke5)。

**修正后的结论(更强)**:`global_steps` 在首次权重更新前恒为 None,
**任何 v4 run 的 step-1 都会触发该 fail-loud**,与模型、机器、
`val_before_train` 均无关。

**CD-006 对修复复杂度的评估(供参考,方案由 CD-010 定)**:整条链的实质是
"初始值是 None 而非 0",可下手处四个 ——
(A) `async_sglang_server.py:257` 初始值改 0(1 行;语义成立:训练前权重版本即 0,
`trajectory_spans = 0−0+1 = 1`;代价是 vendored verl 的 upstream 同步维护);
(B) `selfres.py` 缺失时兜底 0 而非 raise(局部,但两侧不对称仍在);
(C) 训练循环前显式 `set_global_steps(0)`(语义最明确,需定位时机);
(D) `llm_server.py:274` 不对 None 做 setdefault(只恢复 `key in` 判定的意义,
不解决 research 取不到值)。

---

## 2026-08-09 · P1 · research 行落盘丢失 role 标识,导致下游观察量被按臂占比缩放

**结论先行**:v4 每步落盘 3 类行(bare / steered / research),但 **research 行落盘时
`steering` 是空 dict**,`steer_arm` 取出为 `None` —— 而训练侧内存里它是 `0`(占位)。
两边约定在落盘这一步分叉,任何按 `steer_arm` 分组的下游消费者都会把 research 行
误归类。显微镜(`scripts/pgs_microscope.py:build_step_entry`)即因此把 research 行
收进分母,**全部 per-trajectory 观察量被按臂占比缩放**。训练侧指标不受影响。

**证据**(run `q35_4b_v33_selfres_162`,step_00020,384 行):

| (steer_arm, 行类) | 条数 | won 均值 | steer_entropy_mean |
|---|---|---|---|
| 0 (bare 执行臂) | 128 | 0.9375 | 0.0000(哨兵) |
| 1 (steered 执行臂) | 128 | 0.9766 | 0.0168(真值) |
| **None (research)** | **128** | **0.0000** | **字段整个缺失** |

`.get(k, 0.0)` 对缺失字段补 0、分母照加 ⇒ 精确稀释。逐值验证(修复前显示 vs 真值):

| step | 量 | 修复前 | 真值 | 倍数 |
|---|---|---|---|---|
| 1 | entTheta | 0.640 | **1.280** | 2× (恰为配置 `entropy_theta_init`) |
| 1 | entMean | 0.138 | 0.276 | 2× |
| 1 | wMean | 0.266 | 0.533 | 2× |
| 1 | tilt | 155.2 | 465.7 | 3× |
| 1 | lenMean | 5180 | 2127 | 2.4×(混入 research 行的 11287) |
| 20 | won(混合) | 0.638 | 0.957 | 恒等于两臂均值 × 2/3 |

**危害不止于看图**:`entropy_theta` 的目标语境标定按规范要在真实语境实测。若照显微镜
读数标定,θ 会整体标低一倍。9B 线挂账的"熵门 4B 占位待 step1 重标"若已用显微镜读数,
需按上表 ×2 复核。

**已做**(CD-006 侧,不涉机制代码):`build_step_entry` 改为"每个量的分母 = 该量有定义
的行"—— won 取执行行(arm∈{0,1})、软门/干预量取 steered 行、research 行长度单列
`lenResearchMean`;靠"是否出现过 arm 0/1"区分分臂 run 与历史全臂 run,旧曲线不变
(已单元验证)。

**更正(同日,写完上段后复核落盘发现)**:上段"落盘无 role 标识"**不成立**,已撤回。
落盘顶层**有** `sample_role` 字段,step_00020 实测分布干净:

    ('research', steer_arm=None) 128 | (None, 0) 128 | (None, 1) 128

即 role 标识一直存在,只是显微镜没用它、改去按 `steer_arm != 0` 间接推断行类 ——
**这是 CD-006 侧的实现缺陷,不是机制侧的 schema 缺陷**。修复已改用 `sample_role`
作首选判据(缺失时回退 arm 推断,兼容旧 run)。机制侧无需改动。

**仍建议订正**(纯注释,不涉行为):`metric_utils.py:501` 注释称 "research rows carry
steer_arm=0" 描述的是内存态;落盘态是 `steering` 整个为空 dict ⇒ 取出为 `None`。
`pgs_microscope.py` 里"v4 research 行 steer_arm=0 恰被排除"同源于此,已随修复删除。
两处描述内存态的注释被下游当成落盘契约,是本次误判的共同来源。

---

## 2026-08-09 · P2 · SSB `answer_position` 解析:sheet 名含逗号被当多区域切分

**结论先行**:`evaluate()` 对 `answer_position` 的解析按逗号切分多区域,但
**Excel 里 sheet 名本身可以含逗号**(用单引号包裹),导致合法记法被切坏、整个
任务评分抛异常。实测 verified-400 里命中 2 个,它们因此 reward 恒零。

**证据**(实际打开 workbook 核对,非猜测):

| 任务 | dataset.json 的 answer_position | golden 实际 sheet 名 | 报错 |
|---|---|---|---|
| `130-9` | `'b2b, sez, de'!A5:V10` | 确有 sheet `b2b, sez, de` | `ValueError: b2b is not a valid coordinate or range` |
| `283-32` | `Sheet3'!A:G,'Sheet4'!A:G` | Sheet1–4 均在 | `ValueError: invalid literal for int() with base 10: ''` |

`130-9` 的 prompt 原文即要求把数据并入名为 `b2b, sez, de` 的目标 sheet ——
sheet 名含逗号是任务本身的一部分,不是脏数据。`283-32` 另叠加两点:多区域
(两个 sheet 各一段)+ 整列引用 `A:G`,且该字段开头缺一个 `'`(数据侧 typo)。

**建议**:解析时先按**引号配对**切分,再对未被引号包裹的逗号做多区域切分;
整列引用 `A:G` 需要单独处理(openpyxl 的 range 解析不吃裸列区间)。修好后这
两个任务可回收进训练集。

**CD-006 侧已做**(不改机制代码):把 27 个 reward 恒零任务(含这 2 个)整体
剔除,清单落 `envs/spreadsheetbench/unreliable_tasks.json`,由
`scripts/ssb_audit_reward_reliability.py` 可重审、`--verify` 可当启动门禁。
用户裁定 2026-08-09:对所有 SpreadsheetBench 实验一律生效。其余 25 个是
LibreOffice 与 Excel 的语义鸿沟(新函数/数组公式),6.4 与 7.6 实测同结果,
非代码问题。

**另请一并知悉**(非 bug,是口径事实):纯静态值答案不进重算,实测可靠率
99.7%;写公式要过 LibreOffice 重算,可靠率 90.5% —— reward 在事实上轻微
偏好写静态值。官方文档既未讨论后端差异,也未说明论文 baseline 用的哪个后端。

---

## 2026-08-09 · P1 · SSB `build_dataset` 多 sheet 合成产出非法 answer_position

**结论先行**:`make_row` 在 `answer_position` 无 `!` 时用
`f"{answer_sheet}!{answer_position}"` 合成,但 **`answer_sheet` 可能是多个 sheet
(逗号分隔)**,于是拼出 `A,B,C,D!A3:E11`。evaluator 按逗号切段后,前几段没有
`!`,整段被当成 cell_range 送进 openpyxl ⇒ `ValueError` ⇒ `_score_output` 的
`except Exception: pass` 吞掉 ⇒ **该任务 reward 恒 0**。

**实测**:verified-400 里 `answer_sheet` 含多 sheet 的有 10 个,其中只有 `60-7`
同时满足"`answer_position` 无 `!`"而触发该分支:

    answer_sheet    = 'Consolidated Tracker,Existing Task,Additions,Retired'
    answer_position = 'A3:E11'
    合成后          = 'Consolidated Tracker,Existing Task,Additions,Retired!A3:E11'
    生产 _score_output(golden) = 0.0     ← 完美答案判 0

**CD-006 侧已修**(`build_dataset.make_row`):`answer_sheet` 拆多 sheet 后逐个配
range —— `'Consolidated Tracker'!A3:E11,'Existing Task'!A3:E11,…`。修后 `60-7`
六门全过,已回收进训练集。

**顺带**:数据 `42930` 的 golden 文件名 id 写错(目录下是 `1_43930_golden.xlsx`),
`find_test_cases` 返回空 ⇒ `build_dataset` 直接 `SystemExit`。已在两机建 symlink
`1_42930_golden.xlsx -> 1_43930_golden.xlsx`。这属上游数据 typo,记录备查。

**仍未修、已知会误判的**(P2,上一条已报):`evaluate` 的 answer_position 解析
按逗号切多区域,sheet 名含逗号(`'b2b, sez, de'`)会被切坏 —— `130-9`、`283-32`
因此 reward 恒零,本轮按用户裁定一并剔除。修好可回收 2 个任务。

---

## 2026-08-09 · P1 · 零正例时 researcher 会产出高置信度的错误文档(prompt 结构强制所致)

**结论先行**:`DOC_RULES_TEMPLATE` 强制"exactly these three sections",其中第一节
`# Understanding & Approach` 要求写"the **verified** way to solve it"。当一个
researcher 会话的全部实验都失败(零正例)时,它没有任何可验证的解法,却仍被格式
要求交出一份"已验证解法" —— 实测它会**编造**,并在文档里反复标注 verified。
这类文档进入 steered 臂后,教师会拿一份错误指导去干预策略,不只是无效,而是
**主动往错误方向拉**。

**实证**(SSB · A800 · `ssb_4b_selfres_a800` step_00001,同一步内的两个任务,
bare 成功率均为 0):

| 任务 | researcher 自己的实验结果 | 文档形态 | 前缀续写 | steered 效果 |
|---|---|---|---|---|
| `42515` | 0,0,0,0,0,1 → **1/12 成功** | 主体是 `# Failure Modes & Remedies`,诊断 INDEX/MATCH 为何失败并给补救 | 14/20 胜 | bare 0.000 → **steered 0.600(+60pp)** |
| `194-19` | 0,0,0,0,0 → **0/10 成功** | 主体是 "Verified Solution Approach",给出完整列映射与 Python 代码,文中 **3 处 "verified"** | **0/20 胜** | bare 0.000 → **steered 0.000(+0pp)** |

决定性差别是**有没有撞到过一次成功**,不是文档写得像不像样 —— `194-19` 那份读
起来远比 `42515` 专业(结构化、代码完整、断言明确),但一次都没赢过。

**根因是 prompt 内部的自相矛盾**(`research/prompts.py`):

- 第 69-72 行(WHEN TO FINISH)已有软约束:"…then summarize what you actually
  observed; **never invent**"
- 但第 78-84 行(DOC_RULES_TEMPLATE)是硬性格式:"Markdown, **exactly these three
  sections**" + `# Understanding & Approach` = "What this task requires and the
  **verified way to solve it**"

硬格式压过软约束:零正例时"必须交出 verified way"这条结构要求没有合法出口,
模型只能编。而且第一节位居文档之首、权重最高,与同一 prompt 第 59-67 行
"happy-path-only knowledge is silent exactly where supervision matters most"
的设计意图恰好相反。

**用户裁定 2026-08-09**:

> 产出的特权文档中 verified solution 不应该是必须的,只有有成功经验时才总结,
> 全部失败不要写,写错误模式、经验教训总结。

**建议的落地方向**(具体实现由 CD-010 定):
1. 把第一节**条件化** —— 会话内有 ≥1 次成功才写"已验证解法";零正例时该节改为
   "当前理解 + 已排除的方向",并**显式禁止** verified/confirmed 一类断言措辞。
2. 或放宽"exactly three sections"为"至少包含失败模式一节",给零正例一个合法出口。
3. 无论哪种,`self_assessment` 里应能反映"本次零正例",便于下游按置信度加权
   (例如零正例文档不参与 argmax 择优,或降低其 steered 权重)。

**CD-006 侧的观察计划**:往后每步盯 **δgap 为负的任务**(steered < bare)。若出现,
即为错误文档造成负增益的直接证据。目前样本仅 step-1 的 8 个任务(3 难/3 中/2 易),
`194-19` 的 steered 恰好也是 0.000、没有变差,尚不能排除别的任务上会变差。

**顺带**(同批数据,供机制侧参考,非问题):按难度分层,教师增益在最难一档最大 ——
bare=0 组 δgap **+31.1pp**、bare 0~40% 组 +24.6pp、bare>40% 组 +12.5pp;而
researcher 的 r̄ 随难度单调下降(0.215 / 0.396 / 0.853)。即文档质量本质受限于
researcher 自己的 rollout 成功率,与 [opd-capability-boundary] 的老结论一致:
机制放大已有行为,不凭空引入新能力。`research_n=8` 的择优确实有回报
(难档 argmaxR 0.383 vs r̄ 0.215)。

---

## 2026-08-09 · P0 · PGS 自进化在 SSB 上吞吐低:根因定位与待探讨项

**详细分析全文**:`docs/pgs-throughput-ssb-analysis.md`(含代码位置、量化推导、MENTOR 逐项对比、
可复现方法)。本条只列结论与需要 CD-010 参与判断的问题。

**结论先行**:SSB 上 steered 臂比 ALFWorld 慢约 **30 倍**(bare 只慢 12 倍),额外劣化来自机制。
主因是「长段 draft → 长段 verify → 任一拒绝丢弃余段并重启」在长轨迹上的**近平方级放大**
(`W ≈ N(1+R/2A)`,SSB 3.69× vs ALFWorld 1.06×,再乘候选长度 5× ⇒ 约 17×,与实测同量级)。
**不是** CPU 空等、LibreOffice 或全局串行锁 —— GPU 利用率 84%(15 次采样无一低于 30%)。

**⚠ CD-006 先前一个判断作废,请勿据其排优先级**:曾报"开门率 35–46% 远高于目标 5%,
熵门未标定"。经查 `steering/gate.py:165/188`:`entropy_target_rate=0.05` 控制的是
`H ≥ Hq` 即 **`w=1` 满强度位置**的比例,而 `base_open` 条件是 `H ≥ w_min·Hq = 0.05·Hq`。
**5% 是满强度目标,不是非零干预目标**,35–46% 未必异常。(MENTOR 同样无 hard bypass。)

**另一处口径更正**:先前报"整步约 80 分钟"是把最后一条成功落盘当成步结束。
真实 **107 分钟** = 成功生成 62.9 + 未决长尾至 90(超时) + 训练约 17。

### 需要 CD-010 判断的三件事

1. **是否值得做引擎内改造** —— 同权重双前缀(student KV=`h`,teacher KV=`h⊕doc`)、持久双 KV、
   bounded K 的 exact speculative loop。这是唯一能切断近平方放大的路径,预估 2–4 周、高风险。
   但**天花板只有约 1.23×**:steered 一旦降到 39.2 分钟以下,research(48.0 分钟)立即变成
   关键路径。投入产出比需共同判断。
2. **research 阶段是否同步优化** —— 每步 2304 个 episode + 1024 次 narration LLM 调用。
   可能方向(narration 是否必需 / continuation 批量化 / researcher 顺序 turn 并行)**都改机制语义**,
   须用户裁定。
3. **`w_min = 0.05` 在 SSB 长段场景是否过低** —— 段级 bypass 概率是 `(1-q)^m`,SSB 平均段长
   约 261,即使非零干预率降到 5%,整段跳过概率仍只有 1.5×10⁻⁶。**ALFWorld 段短才碰巧有效,
   同一份代码在两个 benchmark 上行为完全不同**。这是机制参数问题,不是标定问题。

### 明确不该做的(避免走弯路)

- **不能只把 Python 层 draft 的 `max_tokens` 改成 4**(MENTOR 的 `num_speculative_tokens=4`):
  对 16k 轨迹会制造 4000+ 次 RPC,比现在更慢。bounded K 只有与引擎内循环 + 持久双 KV 一起才有意义。
- **不能照搬 MENTOR 的全词表算术混合** `(1-w)π + w·p_T`:会丢失尾部质量恒等、密度比界
  (`e^{±2ηc}`)与预算证明,GPU 上还需持有数百 MiB fp32 全词表张量。
- **不能直接复用 stock vLLM `SpecDecodeWorker`**:它只接收一套 `seq_group_metadata`,
  **不支持 target-only 的 doc prefix**。doc 塞进公共 prompt 会同时污染 draft,
  使 proposal 不再是 `π(a|h)`;不塞则 target 不是 `p_T(a|h⊕doc)`。

### 一个与吞吐无关但更紧急的正确性问题

**5400 秒超时后,残缺的 prompt 组仍被采样进训练**。`agent_loop_tq.py:124` 超时取消 pending task
并标 `failure`,但已完成 session 早已独立写入 TransferQueue(`:144`);ReplayBuffer 把
`finished_keys | failure_keys` 都当可采样终态(`replay_buffer.py:164`),且留有未实现的
`# TODO: failure samples dropping and refilling`(`:243`)。

⇒ step_00001 的 **191/192 是训练数据真实缺一条**,不是少个归档文件。缺任务 42515 的 1 条 steered,
`seq-mean-token-mean` 下该任务 steered 臂权重 `8/192 → 7/191`,**−12.0%**;且**丢失的是最慢样本**,
系统性关联长轨迹/难任务 —— 属**非随机右删失**。

`objective_fingerprint()`(`config.py:1075`)不含 `agent_task_timeout_s`,故提高超时是纯工程修复;
**但 timeout 一旦命中并允许残组入训,它已在操作上改变采样分布**。建议处置(待定):failure prompt
不可采样 → 清理其已落地部分轨迹 → 同一冻结 policy version 完整重试或整步 fail-loud →
用**无进展 watchdog** 替代单一总墙钟。

### CD-006 建议的低成本先行项(不改行为)

**增加真实工作量计数**:radix matched tokens、teacher prefill/decode tokens、**draft-discarded
tokens**、block accept prefix、序列化字节。1–3 天、风险极低。

价值:**当前无法区分"cache miss"与"拒绝丢弃 suffix"两种浪费** —— `steer_n_candidates` 在逐位
judge 时递增,拒绝后未 judge 的尾段已被 GPU 算出但不入账,故用它估 GPU 工作量会**低估约 3.69×**。
这两种浪费对应完全不同的优化方案;拿到数字前,上述所有收益估算都还是推断。

### 另附:两个观测口径的坑(写离线分析时会踩)

- `steer_count` 只统计 `eta>0` 的最终执行位置(`ledger.py:169`),不含关闭位与丢弃 suffix。
- `response_token_len` **含环境观察**(以 `mask=0` 追加进同一 response stream,`episode.py:255`),
  不是纯生成 token。SSB 16390 vs ALFWorld 682 不能直接当"模型生成量"比较。
