# 正式实验阶段:FC 协议统一 + benchmark 扩展 —— 实施计划(CD-010)

**依据**:用户需求文档 `docs/collab/cd010-cd006/requirements-formal-experiment-benchmarks.md`
(2026-08-08,含 §6 训练期转换)。本文档 = 该需求的实施分解与已确认裁定的留痕。

## 0. 用户四裁定(2026-08-08,AskUserQuestion 当面确认)

1. **§6.2 训练期转换 = SSB 模式**:wire format 始终是 chat template 原生 FC 形态
   (分布内);「转换」指训练通道(token-in-token-out,拿不到服务端 parser)由我们
   在文本/token 层自实现解析与 tool 响应渲染,生成的原始 token 原样进训练流。
   训练/推理分布严格一致。不采用「训练期换 text wire 格式」(分布不一致)。
2. **ALFWorld FC 工具形态 = 单工具 `execute(command)`**:命令语法/环境说明精心写进
   工具描述与系统提示;动作空间不变,GT walkthrough 与 researcher 资产零改动,
   与 text 协议历史读数可比性最好。
3. **新环境引入顺序 = ScienceWorld → WebShop → SearchQA → SWE-bench**;
   每个走一轮完整的「调研 → 设计 → 与用户协商(§4 七项)→ 确认 → 实现 → 探针」。
4. **在跑 9B 主线(text 协议)继续跑完**;协议改造在代码层并行,不动在跑 run;
   下一个正式 run 起换 FC 协议(熵门/θ_D 等标定届时在新协议语境重做)。

## 1. 现状差距(2026-08-08 盘点)

- **SSB 与 Math 已是 FC 目标形态**:`adapter.tools()`(OpenAI schema)经 chat
  template 渲染进 prompt + `protocol.parse_tool_calls` 文本层解析 +
  `_tool_resp` 按模板 tool-role 形态包裹。这就是 §6.2 的既有解法原型。
- **P0 风险 · 模板家族断点**:现有 FC 件是 Qwen3 hermes 形态
  (`<tool_call>{json}</tool_call>`);CD-006 实测 Qwen3.5 chat template 为
  `qwen3_coder` XML 形态(`<function=名><parameter=键>`,与 sglang 0.5.10 服务端
  parser 逐字一致;sglang 无 `qwen3_xml`)。SSB/Math 在 Qwen3.5 上从未验证,
  hermes 正则大概率解析不到 XML 输出 → acted=False 循环。与 EOS 事故同类病
  (协议件隐式绑定模型家族资产),同类修法(家族化 + 单一真源挂 template_delta
  + preflight 门)。
- ALFWorld 是 text 协议(`<reasoning>/<action>`),存量改造主体。
- researcher 工具链是 text 协议;CD-006 实测 4B 工具调用错误率 45.2%(9B 9.6%),
  97%+ 是 finalize 整篇文档进 JSON 的转义失败;换 qwen3_coder 后归零。
  `common/config.py:619` 两条校验(research_rl→text、backend→external)需重写。
- AppWorld 在名单外 → 完全删除(§3.2)。

## 2. 阶段计划

**M0 · FC 基座家族化**(先行)
1. A800 实测 Qwen3.5-9B/4B 与 Qwen3-4B 模板三件:tools 渲染形态、assistant
   tool_calls 渲染形态(=模型学到的输出形态)、tool role 消息渲染形态;
2. 读 sglang 0.5.10 `qwen3_coder` parser 源码,摘取解析语义作参照;
3. FC 三件家族化:解析器(hermes | qwen3_coder)按模板家族分派,单一真源挂
   template_delta;`_tool_resp` 包裹随家族;
4. 等价性对拍:同一批真实模型输出,文本层解析结果 == sglang 服务端 parser;
5. preflight 加 FC 渲染对拍门(新模型接入自动查);
6. SSB/Math 在 Qwen3.5 上复验探针(GPU 探针等卡窗口,tokenizer 层先行)。

**M1 · 存量环境整改**
ALFWorld FC 化(单 execute,描述精修)+ SSB/Math 描述精修 + AppWorld 全删。
验收:ALFWorld text vs FC zero-shot A/B(text 基线 = 9B raw 30.71%/26.87%)。

**M2 · researcher FC 化 + §6 落地**
external 后端走 OpenAI API + 服务端 parser;research_rl / v4 自体研究员走 SSB
模式;config 校验重写;双通道解析等价性测试。

**M3–M6 · 新环境逐个引入**(ScienceWorld → WebShop → SearchQA → SWE-bench)
每个产出协商文档(§4 七项:训练数据/测试数据/协议与系统提示/评估方案/GT 反馈
信息/任务特定信息/调研发现),用户确认后实现。SearchQA 的指代(2017 数据集 vs
Search-R1 风格检索 QA)在其调研轮首先与用户对齐。SWE-bench 以执行沙箱方案为
协商核心(容器成本画像见 `docs/qwen35-baseline-survey.md` 第三部分)。

## 3. 与既有工作的衔接

- 9B v33 主臂(A800, text 协议)照常;bare-first 生效读数与熵四分位重标仍等
  CD-006 回流,正式 9B 配置(text 语境)按原挂账出——它服务于当前 run 的延续,
  与 FC 切换后的新正式 run 是两代配置。
- 协议均在环境层/researcher 层,PGS 机制六件套(倾斜/门控/预算/SPS/W_beh)
  不感知协议——需求 §2「机制保持通用泛化」的边界即此。

---

# 进展台账(2026-08-08 收工快照)

## 已完成(M0–M3 + 两条穿插线,全部入库)

| 项 | 状态 | commit 链 | 要点 |
|---|---|---|---|
| **M0** FC 基座家族化 | ✅ | `3be537f`+`2602b57` | fc_dialect 行为探测(qwen35_xml/hermes);解析器与 sglang qwen3_coder golden 对拍 26 例;流等价探针(真机 3 模型双 PASS);preflight 三重门;SERVER_TOOL_CALL_PARSER 绑定(sglang 0.5.8/0.5.10 无 qwen3_xml,那是 vLLM 的) |
| **M1** 存量环境 | ✅ | `57d9955` | ALFWorld 单工具 execute FC(`alfworld.protocol` 开关,fncall 默认,进指纹条件追加);AppWorld 全删;SSB/Math 核过 |
| **M2** researcher FC 化 | ✅ | `5866adc` | TextProtocol → 客户端解析 FC(schema 交模板渲染,§6.2 SSB 模式);EngineResearcherChat 首调带 tools;端点带 parser 的 fail-loud;收编 CD-006 tool_schemas_for |
| **审阅修复**(xhigh 16 findings) | ✅ 14修/1留/1转CD-006 | `7da4bdb`+`779d2e1`+`3acf8e8` | 两个在跑 run P0:steered 退役 step-240 必崩已修(**须在 run 到 240 前同步**)、narration 通道补 stop(定性已更正为防御性——实测 204 条无越界);续写重放归一化、consumed 语义对齐、fingerprint 哨兵×2、preflight 扩面等 |
| **§7 预算系列** | ✅ | `9c9568a`+`4baa455`+`c2273a8` | 7.2 dead config 修(第三参强制);7.4 零梯度阈联动会话墙;7.5 保险丝/控制器解耦(session_token_budget/max_prompt_tokens/narration 两键,**必配**;允许 budget<轮×单轮;校验链抓出旧墙 38912>维度 32768 的存量矛盾) |
| **M3** ScienceWorld | ✅ | `76b1275`+`f35a501`+`cbbd0de` | 五裁定:easy/全局步预算(RLVMR 30)/终局二元(won=completed∧score≥100)/ETO 口径(24 任务 1483/194/211,indices vendored)/不注入动作表;JVM worker 冷启(实测 1.6s);真引擎本地 e2e 过;related_work/benchmarks/ 资产约定建立(五库五文 commit 钉) |

**测试**:442 passed / 3 skipped(起点 395)。

## 挂账(勿丢)

1. **FC 真实生成端到端探针**(卡窗口):ALFWorld FC zero-shot vs text 基线
   30.71%/26.87%、SSB/Math 短探针、v4 researcher smoke——changelog 已请求
   CD-006,或窗口出现自跑。
2. **CD-006 待办**(changelog 各条已明示):三个 yaml 的 `protocol: text`
   pin + §7.5 必配键提交(代补在工作树);step-240 前同步 `7da4bdb`;
   SciWorld 部署(worker venv + A800 conda openjdk)与 zero-shot 自建基线;
   §7.5 迁移(v4 语境 budget 定值,38912 与 response_length 的矛盾处置)。
3. 旧挂账:9B 熵标定回流 → text 语境正式配置(服务当前在跑 run 一代)。

## M4–M6 规划(逐个走 §4 流程:调研→协商文档→用户确认→实施)

**M4 · WebShop**(下一个)
- 资产:`related_work/benchmarks/webshop/` — 官方 princeton-nlp/WebShop、
  ETO 的 webshop 支持(repo 已在库)、Harness-R1(若开源;其口径 train
  10,480 / val 100 / test 500,9B baseline 31.2%)。
- 预判抉择(读码后定):①FC 工具形态:双工具 search(query)/click(target)
  vs 单 execute——WebShop 动作天然二分,双工具更贴 FC 语义(与"单工具"
  裁定的 ALFWorld/SciWorld 先例不同,需用户裁);②reward:官方 0-1 属性
  部分分 vs 二元(succ=reward==1,Harness-R1 口径;SciWorld 二元先例);
  ③数据口径:Harness-R1 vs ETO/AgentGym 划分;④observation:text 简化
  模式 vs html。
- 工程面:环境是**服务形态**(Flask + 检索引擎 + 商品库数 GB)——部署账
  (单服务多 episode 并发 vs per-worker)是与 ALFWorld/SciWorld 子进程
  模式的最大差异,协商文档重点。

**M5 · SearchQA**
- **第一步与用户对齐指代**:2017 Jeopardy SearchQA 数据集 vs Search-R1
  风格检索 agent(倾向后者,agentic 语境)。
- 若 Search-R1 风格:资产 PeterGriffinJin/Search-R1 + 2026 后继;抉择:
  检索栈(本地 wiki 索引 BM25/E5 vs 联网;索引内存账)、数据(NQ/HotpotQA
  等混合)、工具(search(query)+finish(answer) 双工具 FC)、评估(EM 二元
  reward 先例自然)、GT(gold answer;gold 文档可作特权信息——与机制的
  特权通道天然契合,协商点)。

**M6 · SWE-bench**(最重,放最后)
- 核心协商点 = **执行沙箱**:162/A800 无 root,docker 大概率不可用——
  候选:rootless podman / 预构建 conda 环境复原(SWE-smith 风格)/
  Dockerless 验证器路线(arXiv 2606.28436)。先查服务器容器可用性再定。
- 数据:评测 SWE-bench Verified(500,9B baseline ≈59.6%);训练候选
  SWE-gym / SWE-smith / R2E;范围裁剪(子集)待协商。
- 协议:多工具 FC(bash / 文件编辑 / submit,mini-swe-agent 形态)——
  工具面远大于既有环境,协议层压力测试。
- 资源账:每 rollout 容器内跑测试,墙钟/磁盘(镜像库百 GB 级)/并发形态
  (128 并发大概率不可行)是协商文档的硬核部分。
