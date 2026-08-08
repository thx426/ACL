# FC 协议统一 · 实验侧运行指导(CD-010 → CD-006,2026-08-08)

**变更本体**:全 agentic 面统一 OpenAI 格式 Function Call(用户需求
`docs/collab/cd010-cd006/requirements-formal-experiment-benchmarks.md` §2/§3/§6;
四项用户裁定见 `docs/plans/2026-08-08-fc-protocol-and-benchmarks.md` §0)。
commit 链:`3be537f`(M0 家族化)→ `2602b57`(M0+ 渲染方向)→ `57d9955`
(M1 ALFWorld FC + AppWorld 删除)→ `5866adc`(M2 researcher)→ `7ada82d`
(模板注释)。本地 433 passed / 3 skipped;机制细节见 changelog 三条
GO-FC-* 条目,本文只写**你们要做的事**。

## 0. 结论先行(三件必做 + 一件勿做)

1. **凡 text 语境的旧配置,同步新代码后必须显式加 `alfworld: {protocol: text}`**
   ——不加则默认 fncall,objective_fingerprint 变化,resume 会被守卫拦下
   (防呆设计,不是故障)。涉及:`a800-q35-9b-smoke.yaml`、
   `a800-pgs-v33-4braw-asym.yaml`、162 的 v33/v4 配置等一切在跑/待续 run。
2. **external researcher(35B)服务起服必须带 `--tool-call-parser qwen3_coder`**
   (Qwen3.5 上唯一合法值,`protocol.SERVER_TOOL_CALL_PARSER` 绑定);
   **sftgen 外部采样端点必须不带 parser**(带了 extract 会 fail-loud——
   调用文本被服务端剥走,SFT 原始形态即丢失)。
3. **卡窗口出现时跑 §3 的三项验收探针**(或告知窗口由 CD-010 自跑)。
4. **勿在 Qwen3.5 上用旧代码(<3be537f)跑 SSB/Math**:旧 hermes 解析
   与 Qwen3.5 的 XML 输出不同族,executor 会全程 acted=False。反向兼容
   成立:新代码在 Qwen3(4B-2507/1.7B)上自动判 hermes,行为不变。

## 1. 同步与工作区状态

- 同步到 `7ada82d`(或更新)。你们未提交的 `research/prompts.py`、
  `research/session.py`、`tests/test_tool_schema_k_follows_config.py`
  已由 M2 **原样收编入库**(changelog 有声明)——同步后这三个文件应显示
  clean,无需再提交。`launch_train.py`(`++` 修复)、两个 a800 yaml、
  `test_162_v33_config.py`、162 yaml 仍归你们提交。
- AppWorld 已全删(envs/appworld、AppworldCfg、测试、两份实验配置)。
  凡配置含 `appworld:` 段会因 unknown section 报错——删掉该段即可。

## 2. 配置与部署面变化

| 项 | 变化 | 动作 |
|---|---|---|
| `alfworld.protocol` | 新键,默认 **fncall**;text 进不了新默认 | 旧 run 配置显式 `protocol: text`;新 FC run 不用写 |
| objective_fingerprint | fncall 时追加 `alfworld_protocol=fncall`;text 保历史指纹逐字节不变 | resume 被拦即检查此项 |
| `research.tool_protocol` | 键与取值语义不变(`research_rl→text`、`external→fncall`);text 的**实现**已升级为客户端解析 FC | 无配置动作 |
| preflight model 步 | 新打印 `fc_dialect=...; fc_stream=...`;FC 环境三门(dialect/roundtrip/流等价)不过即 FAIL | 首跑看它绿 |
| 起服参数 | 见 §0.2 | 起服脚本核对 |

预期 preflight 读数:Qwen3.5-9B/4B → `fc_dialect=qwen35_xml`、
`fc_stream={'tool_turn': 'PASS', 'full_stream': 'PASS'}`;Qwen3-4B-2507 →
`hermes` 双 PASS;Qwen3-1.7B(text 线)不触发 FC 门,若手动跑会报
`SKIPPED:think-strip`(该家族模板性质,非故障)。

## 3. 验收探针(等卡窗口;tokenizer 层双向等价已闭环,这是最后一环)

1. **ALFWorld FC zero-shot(9B raw)**:val 274 条 greedy,对照 text 基线
   seen **30.71%** / unseen **26.87%**(±5pp 误差棒,val greedy 非确定)。
   看三个数:①`invalid_format`/acted=False 率(FC 下应 ≈0,偏高=模型不按
   模板出调用,回报 CD-010);②won 不应显著低于 text 基线(协议切换
   ≠能力回退的判据);③turns 分布定性同 text。
2. **SSB / Math 短探针**(各 ~16 任务,9B):tool-call 解析成功率
   (acted=True 占比)与轨迹能走完;这两个环境在 Qwen3.5 上**从未跑过**,
   是家族断点修复的直接受益者。
3. **v4 researcher smoke**:1.7B(hermes 语境)跑一次确认升级零回归;
   Qwen3.5 语境的 v4(4B/9B)首跑时重点看 finalize 成功率——旧 text
   形态 4B 上 45.2% 工具调用失败,新形态预期归零(162 fncall 探针
   已证方向)。

## 4. 观察口径变化(dump/指标)

- **thought 记账**:FC 下 `thought` = 调用块**之前**的文本(与服务端
  parser 的 normal_text 语义一致);text 协议不变。**thought 逐字重复率
  指标在 FC 语境依然有效,但与 text 历史读数不可直接跨协议比较**。
- `invalid_format` / `DEGENERATE:format` 判定链(acted=False×3)不变;
  轮末哨兵三计数(`turns_stop_*`)不变。
- ALFWorld 观察在 FC 下经 `<tool_response>` 包裹后入流(与推理引擎对
  role=tool 的渲染逐 token 一致,流探针守护)——grep 轨迹文本时注意
  该包裹层。
- 首个 FC 正式 run 前,熵门/θ_D 等标定须在 FC 语境重做(任务表面变了,
  text 语境标定值不迁移)。

## 5. 红线(复述,防回归)

- golden fixture(`tests/fixtures/fc_qwen35_xml_golden.json`)是真
  detector 产物,**勿手改**;升级 sglang / 换 parser 名 / 换 vLLM 后端
  前先通知 CD-010 重新对拍。
- 训练流只用模型原始生成 token,任何场景不重渲染 assistant 轮。
- 在跑的 text run(9B formal / v33 各臂)按用户裁定跑完,不切协议。
