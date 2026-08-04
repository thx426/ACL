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
