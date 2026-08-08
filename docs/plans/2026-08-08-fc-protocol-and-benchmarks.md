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
