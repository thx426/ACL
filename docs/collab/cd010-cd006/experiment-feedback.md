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
