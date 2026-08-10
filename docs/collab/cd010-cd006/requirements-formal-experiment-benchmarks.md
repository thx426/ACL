# 需求文档:正式实验阶段的任务环境与协议

**提出人**:用户 · **日期**:2026-08-08 · **面向**:CD-010(机制开发分支)
**性质**:用户对方案机制**正式试验阶段**的想法与需求。
**本文档只记录用户需求本身**,不含 CD-006 的分析、建议或调研结论。

---

## 1. 选定的任务环境 benchmark

正式方案机制最终选定以下 **7 个**任务环境 benchmark:

1. **ALFWorld**
2. **ScienceWorld**
3. **WebShop**
4. **SearchQA**
5. **SWE-bench**
6. **SpreadsheetBench**
7. **TIR-Math**(当前代码中已经引入)

---

## 2. 统一的 agentic 协议要求

- 上述所有任务环境下的 agentic setting,**统一采用 OpenAI 格式的 Function Call**。
- 该协议**仍然在任务环境(benchmark)这一层实现**。
- **不要把该协议引入到新方案机制中**。
- **方案机制仍然保持通用、泛化,可对接任何协议。**

---

## 3. 对"当前代码中已引入"的任务环境的处理要求

### 3.1 在选定名单内的

- **保留**其已有的任务实验 setting 实现。
- 但**格式协议这一块要修改为 Function Call 协议**。
- 对于**工具的描述、环境信息的描述等,要精心实现**。
  - **不要随便写一下就行。**
  - **要认真对待实现每一个任务引入,不要敷衍了事。**

### 3.2 不在选定名单内的

- **直接完全删除掉。**

---

## 4. 对"尚未引入"的任务环境的引入流程要求

对于还没有引入的任务环境,**必须一个一个地**按以下流程进行:

1. **仔细搜索调研**;
2. **规划设计**;
3. **必须与用户充分探讨协商**——协商内容**至少包括**:
   - 训练数据情况
   - 测试数据情况
   - 格式协议和默认系统提示
   - 评估方案
   - Ground Truth 反馈信息
   - 任务特定的信息
   - 搜索调研分析发现等等
4. **由用户确认后,才能开始一个一个引入。**

---

## 5. 用户对开发分支工作方式的要求

用户希望 CD-010 依据上述需求开展以下工作:

- **深入搜索调研**;
- **相关工作与代码实现的观察分析**;
- **引入用户选中的任务 benchmark**;
- **实现对应的 agentic 格式协议**。

---

## 6. 训练期的协议转换要求(2026-08-08 补充)

用户在 162 上用 Qwen3.5-9B / 4B 做 researcher 探针后,确认了以下事项:

### 6.1 确认统一采用 Function Call 协议

- 实际观察到:**换成 Function Call 协议后,极大改善了 4B 模型的 researcher 效果**。
- 因此确认:**确实应该都换到 Function Call 协议上来**(与 §2 的要求一致)。

### 6.2 训练时必须转换成纯 text 格式

- **训练时必须转换成纯 text 格式。**
- 用户认可该问题的存在:Function Call 协议依赖服务端的工具调用解析器,
  而训练场景下的 rollout 通道拿不到该解析器。
- 因此,**采样/推理期使用 Function Call 协议、训练期转换为纯 text 格式**,
  这两者之间的转换是必须实现的。

### 6.3 由开发分支实施

- 该问题**涉及更细节的、与训练机制相关的细节**。
- 用户决定:**使用开发分支(CD-010)来实施**。
- 用户要求:**把本节需求与 §1–§5 的需求(新的数据集引入、Function Call 格式统一)
  放在一起,由开发分支一并实施。**

---

## 7. 轨迹与上下文预算的配置化要求(2026-08-08 补充)

用户在准备 Qwen3.5-4B 完整自进化实验时提出以下需求:

### 7.1 轨迹长度应由「轮数 × 单轮上限」控制,而非总长度阈

- **实际的轨迹限制应当由 `max_turns` 与 per-turn `max_tokens` 两个参数控制。**
- verl 侧的 `data.max_response_length` **可以保留**,作为供 verl 使用的一个上限声明。
- 用户要求把这一点**反馈给开发分支修改**。

### 7.2 【P0】`episode.max_tokens_per_turn` 当前是 dead config

CD-006 实测(2026-08-08)确认该配置项**读得到但不生效**:

```
配置 episode.max_tokens_per_turn = 4096
注入后 protocol.GEN_MAX_TOKENS_PER_TURN = 2048   ← 未生效
同段对照: MAX_ENV_STEPS 30 → 生效 / MAX_ASSISTANT_TURNS 40 → 生效
```

事实链:`common/config.py:203` 有键定义、`configs/pgs.yaml:74` 有键、
`tests/test_162_selfres_config.py:62` 断言其等于 4096(只断言读到,未断言行为);
但 `core/protocol.py:61` 的 `configure_episode_caps()` 只注入
`MAX_ENV_STEPS` 与 `MAX_ASSISTANT_TURNS`,**没有注入
`GEN_MAX_TOKENS_PER_TURN`**;`core/episode.py:182` 与
`research/continuation.py:232` 消费的是模块级硬编码常量 2048。

影响:用户 2026-08-07 的裁定「任务执行 agent 单轮 2048→4096」从未生效;
在此之前所有以该配置跑出的实验,单轮实际上限均为 2048。

### 7.3 不同任务 benchmark 需要各自的轨迹/上下文配置

- 用户需求原文:**不同的任务 benchmark,它们的难度、任务轨迹类型都不一样,
  所以希望能够通过配置文件的形式,调整这些跟轨迹、上下文长度等相关的配置。**
- 即:轨迹与上下文预算相关的配置项(轮数上限、单轮 token 上限、
  上下文/响应长度预算等)应支持 **per-benchmark 配置**,而不是全局单值。
- 当前状态(客观事实):`episode:` 是全局单段(`common/config.py:508` 的
  SECTIONS 列表),`alfworld:` 等环境段只含数据路径、超时、并发,不含任何
  轨迹或长度预算。

### 7.4 长轨迹必须保留训练梯度

- 用户要求:**agentic 任务确实会出现长轨迹,长的正常轨迹也应该有训练梯度,
  不能随意置为 0。**
- 涉及现有实现:`research_rl.research_max_sample_tokens` 超阈 ⇒
  `response_mask` 置全 0(`train/selfres.py:197`)。

### 7.5 上下文上限与轨迹有效长度必须是两个独立的量(2026-08-08)

用户需求原文与意图:

- **模型的总上下文,应当由一个参数配置来控制**,它是**一般不会达到的上限**
  (用户举例:比如 128K)。
- **推理生成以及训练时,单条轨迹的最大有效 tokens,应当由一个派生值
  `max_turn × per-turn max_tokens` 来控制。**
- 即:**上限(保险丝)与实际控制器是两个不同的量**,前者配置、后者派生;
  不应像现在这样由同一个量兼任两职。
- 用户对现状的推测:**「开发分支没这么做的原因可能是因为 verl 或者相关框架
  有一些约束」**。
- 用户要求:**请 CD-010 理解、分析、思考,设计恰当、干净的改进完善方案。**

本节适用于 researcher 与任务执行 agent 两侧;§7.3 的 per-benchmark 配置能力
要求同样适用于本节引入的参数。

### 7.6 任务执行 agent 也需要独立的 session_token_budget(2026-08-08)

用户需求原文:**「把 researcher 的 session_token_budget 设为 48K,
任务执行的 session_token_budget 设为 16K」**。

即:§7.5 给 researcher 落地的「控制器 / 保险丝解耦」,**任务执行 agent 侧需要
同样的独立控制器**,且两者取值互不相同、各自可配。

CD-006 侧现状核对(客观事实,不含方案):

- `EpisodeCfg` 现有键只有 `max_env_steps` / `max_assistant_turns` /
  `max_tokens_per_turn` / `protocol`,**无流预算键**;
- executor 的流预算取自 `train.response_length`
  (`train/agent_loop.py:325` → `episode_response_budget=self.rollout_config.response_length`);
- 同一个 `train.response_length` 同时承担 **verl padding 维度**
  (`scripts/launch_train.py:299` → `data.max_response_length`)与
  **executor 流预算**(`launch_train.py:356` →
  `actor_rollout_ref.rollout.response_length`)两职 —— 这两个是不同的 hydra 键,
  但 launcher 目前绑在同一个配置值上;
- `common/config.py` 的 §7.5 校验 `session_token_budget > t.response_length`
  也以该单一值为准。
- 因此在当前实现下,`train.response_length` 必须 ≥ researcher 的
  `session_token_budget`(padding 只补不截),无法同时等于 executor 想要的更小值
  —— 用户的「researcher 48K / executor 16K」在现有键位上不可表达。

**CD-006 按 CD-010「四类量」模型的实测复核(2026-08-08,c2273a8 之后)**

CD-010 给出的目标模型:① 保险丝 `rollout_max_model_len` / ② 有效生成预算
(派生 = turns × per-turn) / ③ 流预算 = verl padding 维度(executor:
`train.response_length`;researcher: `research.session_token_budget`)/
④ 单次调用上限。**③ 这一层在当前代码里尚未分离**,实测:

配置 `train.response_length: 16384` + `research.session_token_budget: 49152`
(即用户裁定的 executor 16K / researcher 48K),`load_config` 直接 FAIL:

```
research.session_token_budget(49152) 超过 train.response_length(16384)
——research 行 stream 进该 padding 维度且 verl 只补不截
```

同一次加载的 info 行确认 executor 侧语义是对的:
`executor 流预算 response_length=16384,轮×单轮乘积=40960(总预算 response_length 兜底)`。

事实定位:`scripts/launch_train.py:299` 的 `data.max_response_length`
(verl padding 维度)与 `:356` 的 `actor_rollout_ref.rollout.response_length`
(executor 流预算)是**两个不同的 hydra 键**,但当前都取自同一个
`train.response_length`;`common/config.py` 的 §7.5 校验也以该单值为准。
因此 `train.response_length` 一身两职,无法既等于 executor 想要的 16384、
又容下 researcher 的 49152。

请 CD-010 理解、分析、思考,设计恰当、干净的改进完善方案(§7.3 的
per-benchmark 配置能力要求同样适用于本节引入的参数)。

## 附:需求适用对象的现状映射(客观事实,供 CD-010 定位工作范围)

以下为 2026-08-08 时点 `src/trace_opd/envs/` 下的实际目录,仅作为把上述需求
落到具体对象的事实依据,不构成任何建议:

| 目录 | 是否在选定名单内 | 对应本文档条款 |
|---|---|---|
| `alfworld` | 是 | §3.1 保留 + 改 Function Call 协议 |
| `math`(TIR-Math) | 是(用户注明"当前代码中已经引入") | §3.1 |
| `spreadsheetbench` | 是 | §3.1 |
| `appworld` | **否** | §3.2 直接完全删除 |
| `_template` | —(环境模板,非 benchmark) | — |

选定名单中尚未在 `src/trace_opd/envs/` 下出现的:
**ScienceWorld、WebShop、SearchQA、SWE-bench** —— 适用 §4 的逐个引入流程。


## 附二:§6 相关的实测背景(客观事实,非建议)

以下为 2026-08-08 在 162 上做 researcher 探针时的实测数据,仅作为 §6 需求提出时
所依据的事实,不构成任何技术建议:

**text 协议下的工具调用被拒率**(ALFWorld,32 任务分层抽样,seed=42):

| | Qwen3.5-9B | Qwen3.5-4B |
|---|---|---|
| assistant 消息 | 416 | 312 |
| `TOOL ARGUMENTS ERROR` | 40 | 141 |
| 错误率 | 9.6% | 45.2% |
| 其中属 JSON 转义/截断 | 97.5% | 98.6% |

**同一模型换 Function Call 协议后**(服务端 `sglang --tool-call-parser qwen3_coder`):

| | text | Function Call |
|---|---|---|
| `TOOL ARGUMENTS ERROR` | 45.2%(4B) | **0** |
| assistant 带 tool_calls | 客户端正则提取 | 19/19 服务端解析 |
| doc 词数(该会话) | 530 | 1044 |

**现有代码中与 §6.2 直接相关的校验**(`src/trace_opd/common/config.py:619`):

- `research_rl 下 research.tool_protocol 必须为 text`
- `research.backend 必须为 external`

探针为绕开上述校验,另建了配置 `162-q35-researcher-probe-fncall.yaml`
(关闭 `research_rl`),该配置仅用于本次探针,不作为正式配置。

探针数据与脚本位置:`/data4/ws/PGS/researcher_probe_q35/`(162)。

## 附三:§7.5 相关的现状与实测(客观事实,非建议)

以下为 CD-006 2026-08-08 在 162 上核对代码与实测探针所得,仅作为 §7.5 提出时
的事实依据,不构成任何方案建议。

**1. 现状:两职由同一个量兼任**

```python
# src/trace_opd/train/agent_loop.py:467   researcher 侧
max_total_tokens = max(4096, self.cfg.train.rollout_max_model_len - 2048)
# src/trace_opd/train/backends.py:196     每轮的实际生成上限
budget = self._max_total_tokens - len(self.prompt_ids) - len(self._stream)
max_tokens = min(self._max_tokens_per_turn, budget)
```

`ResearchCfg` 无对应配置键;`max_turns × researcher_max_tokens` 目前只是
两个独立上限,系统内不存在「生成预算」这一累计量。

对照 executor 侧(`core/episode.py:178`):`budget = response_budget - len(stream) - 64`,
其中 `response_budget = train.response_length` 是**可配的**,且口径不含 prompt。
两侧口径不一致(researcher 扣 prompt,executor 不扣)。

**2. 现状:`stream` 同时装生成与工具返回**

`backends.py:165-166` 把工具返回并入 `stream`(mask=0)。因此
`max_total_tokens` 约束的是 `prompt + 生成 + 工具返回`,而
`max_turn × per-turn` 只覆盖其中的「生成」项。

**3. 实测(4B researcher 探针,ALFWorld 32 任务 ×2 会话 = 64 条,FC 协议,
   探针配置 `researcher_max_tokens: 8192 / max_turns: 30 / n_exp: 16`)**

| 项 | 中位 | p90 | max |
|---|---|---|---|
| 生成 token | 10034 | 26574 | 46624 |
| 工具返回 token | 8889 | 19904 | 29989 |
| prompt token | 1426 | 1535 | 1598 |
| 会话总 token | 24672 | 42122 | 58828 |
| 会话轮数 | 9 | 17 | 20 |

各类 assistant 轮的生成 token:`run_experiment` n=494 中位 680/p90 2324;
`finalize` n=70 中位 1976/p90 3063;`(无工具调用)` n=54 中位 2056/**p90 8192
(打满单轮上限)**。全部 626 轮中 ≥4096 的仅 35 条(5.6%)。

**4. 一处与张量维度相关的事实**

verl 的 `_pad_token_ids` 走 `tokenizer.pad(padding="max_length")`,**只补不截**
(CD-006 实测:输入 63488 / max_length 40960 → 输出仍 63488)。research 行在
`train/selfres.py:197` 只做超长零梯度降级(mask 置 0),不截断 stream;
executor 行则由 `core/episode.py:285` 的 `stream[:response_budget]` 自然不溢出。
因此 `data.max_response_length` 需覆盖 `生成 + 工具返回`,且它是**固定 padding
维度**(每行都补到该长度),直接决定 CPU 内存占用。

**5. 162 的资源事实**

总内存 503 GB(当前可用 414 GB)。1.7B v4 冒烟实测
`cpu_memory_used_gb = 177.2`,当时 `response_length = 38912`、384 行
(16 任务 × 24 行)。
