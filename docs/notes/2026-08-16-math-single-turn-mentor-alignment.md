# Math 环境对齐 MENTOR 单轮 setting(2026-08-16,CD-011)

**用户裁定**:把我们的数学任务环境 setting 与格式协议对齐到 MENTOR 的单轮
setting。三项当面裁定:① reward **完整对齐 9:1**;② 数据集**保留** DAPO-17k +
六源 valid(不换);③ 基座 **Qwen2.5-7B-Instruct**。

**对齐对象**:MENTOR *"Selective Expert Guidance for Effective and Diverse
Exploration in Reinforcement Learning of LLMs"*,arXiv 2510.04140,ICLR 2026;
代码 `github.com/Jiangzs1028/MENTOR` commit `b1cc4c6`(本地
`trace_opd-pgs/related_work/code/MENTOR`,逐行核实)。姊妹篇:
`2026-08-11-mentor-dosage-comparison.md`(剂量维度;本篇是环境与协议维度)。

---

## 1. MENTOR 的数学"环境"是什么(取证结论)

**没有环境层**。跑在 EasyR1(verl 轻量 fork,`requirements.txt` 锁
`vllm==0.9.2`)上,是标准单轮 RLHF 数据流:一个 prompt 进、一段 completion
出、规则判分。无 AgentLoop、无 observation 回注、无 turn 边界、无工具调用。
全部机制复杂度在采样端(教师混合),环境侧刻意压到最简 —— 只有三个部件:

| 部件 | MENTOR 的实现 | 出处 |
|---|---|---|
| 数据 | train MATH L3-5 去重 **8888 条**(论文写 8,889)/ val MATH-500 | `data/train/math3-5_dedup.json`、`data/test/test.jsonl` |
| prompt 渲染 | system 硬编码 + `apply_chat_template` | `verl/utils/dataset.py:174-176, 263` |
| reward | `0.9·accuracy + 0.1·format` | `examples/reward_function/math.py` |

**格式协议的位置是个坑**:`train.sh` 传的 `format_prompt=no.jinja` 内容是
`{{ content | trim }}`(什么都不加),看起来像"无格式约束";真正的格式指令
**硬编码在 `dataset.py:_build_messages()` 的 system message 里**。佐证:
`config.yaml` 默认指向的 `examples/format_prompt/math.jinja` 在仓库里根本
不存在 —— 上游 EasyR1 原本把指令放 jinja、`_build_messages` 只 return user
一条,MENTOR 把它挪进了 system 并留了个空模板顶位。

数据集自带的 `solution`(人写完整解答)**训练里完全没用到**(`answer_key:
answer` 只取最终答案)⇒ 他们的特权信息**只来自教师模型的能力**,不来自
参考解答。

---

## 2. 我们的实现:第三档协议 `math.protocol = single`

不改现有两档(`fncall` / `text` 多轮 TIR),**新增并列的第三档**。三档共享
判分器、数据、researcher/steering 全链路,只在任务表面上分叉。

### 2.1 逐项对齐表

| 维度 | MENTOR | 我们的 single 档 | 落点 |
|---|---|---|---|
| 轮次 | 单轮 completion | 单轮,`react` 恒 `done=True` | `adapter._react_single` |
| caps | — | `max_env_steps=1, max_assistant_turns=1`(配置层钉死,fail loud) | `config._validate` |
| 工具/沙箱 | 无 | `tools()→None`,不解析代码块 | `adapter.tools` |
| system prompt | R1 风格 4 句 | **逐字移植**(脚本对拍验证) | `SYSTEM_PROMPT_SINGLE` |
| 输出格式 | `<think></think>` + `\boxed{}` | 同 | — |
| format 判定 | `re.fullmatch(r"<think>.*</think>.*\\boxed\{.*\}.*", DOTALL)` + 空白预处理 `re.sub(r"\s*(<\|>\|/)\s*", r"\1", …)` | **逐字移植** | `mentor_format_ok` |
| reward | `0.9·acc + 0.1·fmt` | `(1-w)·acc + w·fmt`,`format_weight=0.1` | `MathCfg.format_weight` |
| 单轮 token | 8192 | 8192(实测生效,见 §4) | `episode.max_tokens_per_turn` |
| 训练温度 | 1.0 | 1.0 | 沿用 TIR 档 |
| batch × n / lr | 128 × 8 / 1e-6 | 同 | 沿用 TIR 档 |
| clip | low .2 / high .28 | 沿用我方既有档 | — |
| KL / 熵正则 | 全关 | 全关(Dr.GRPO 底座 IF-04) | — |

### 2.2 刻意**不**对齐的四处(经用户裁定,记账)

1. **基座** Qwen2.5-7B-**Instruct** vs 他们的 **Base**。他们用 base 是为展示
   "从 base 起 RL";我们的 researcher/doc/narration 链路都在 Instruct 语境
   验证过,换 base 是未验证语境。
2. **数据** DAPO-17k + 六源 valid(1304)vs MATH L3-5 8888 + MATH-500。
   保留既有数据 ⇒ **与 TIR 多轮臂同源,跨协议对照是单变量的**。
3. **判分** math-verify 子进程(我们既有的、防过 sympy 满核自旋那套)vs
   mathruler。★ **论文 Appx C 写的是 "Math-Verify",与其代码不符** ——
   代码里是 `from mathruler.grader import ...`,是两套等价性判定。
4. **评测** avg@3 @ temp 0.6(用户长期裁定,全任务通用)vs greedy@1 @ temp 0。

另有两处我方**更严**,非对齐缺口:
- 答案抽取用花括号平衡扫描(`extract_boxed`),MENTOR 是贪婪正则;未闭合的
  `\boxed{` 我们按无终答处理,不冒判。
- `won` 仍是 `score >= 1.0`(即格式与答案都对),但它已退化为观察指标 ——
  reward 走连续通道,见下。

---

## 3. reward 通道:为什么必须加开关

我们的训练 reward 一直是 `reward_score = 1.0 if traj.won else 0.0`
(`agent_loop.py`),而 `won = score >= 1.0`(`episode.py:260`)。把 MENTOR 的
连续 score 直接塞进来,**"答对但格式错"(0.9)会掉成 `won=False` → reward 0**,
比 MENTOR 更严 —— format 分不但没生效,还把正确答案打成完全失败。

故引入 `train.reward_from_score`(默认 `False` = 历史行为):

```python
reward_score=(float(traj.score) if self.cfg.train.reward_from_score
              else (1.0 if traj.won else 0.0))
```

三道配套护栏:
1. **配置校验**:`format_weight > 0` 且 `reward_from_score=False` ⇒ 装配期
   `ValueError`(静默降级的后果太隐蔽)。
2. **指纹**:`reward_from_score` 与 `format_weight` 都改变"优化什么" ⇒ 进
   `objective_fingerprint`,**条件追加**(False / 非 single 时不改 material)
   ⇒ 全部既有 run 的指纹逐字节不变,误用新代码续跑旧 run 会被 resume 守卫拦下。
3. **观察面**:`score` 进 `extra_fields` / `reward_extra_info` / dump /
   `run_eval` 的 `mean_score`。★ 加进 `reward_extra_info` 时**必须在
   executor 行与 research 行两侧对称补齐** —— 该字典是 per-key append,
   缺 key 会产生锯齿字典,`won[keep]` 布尔索引长度不匹配直接 IndexError
   崩整步(2026-08-07 162 v4 冒烟实测过)。

**score 四值可无损反推分项**:`{0, w, 1-w, 1} ↔ (acc, fmt)` 一一对应。
w=0.1 时:`0.0=(错,错)`、`0.1=(错,对)`、`0.9=(对,错)`、`1.0=(对,对)`。
⇒ dump 只带 score 即可事后完全还原,无需新增任务耦合字段。

**一个值得知道的等价性**:GRPO 的组内标准化对仿射变换不变 ⇒ **格式全对时
`0.9a+0.1` 与 `a` 给出完全相同的 advantage**。format 分只在格式失败的样本
上产生实质差异。

---

## 4. 验证记录(数字可溯)

### 4.1 单元测试:562 passed, 6 skipped(全量),math 18/18

新增 8 个 single 档用例(`tests/test_math_env.py`),含:
- `mentor_format_ok` 语义锁(fullmatch 起头约束、空白预处理容错、缺标签四种)
- score 四值映射 + 单轮不变量(`done/acted/consumed_env_step` 恒真,`feedback==""`)
- `format_weight=0` 退化为二值
- 未闭合 `\boxed{` ⇒ 无终答(不冒判)
- 任务表面:scaffold 逐字对拍 MENTOR、`tools()` 为 None、brief 不提工具、gt 不泄漏
- 端到端 caps 1/1 恰好一轮、非截断
- 两道配置 fail-loud、三档指纹两两不同

**顺带修了一个既存失败**:`tests/test_protocol.py` 仍断言
`configure_episode_caps(30,30,…)` 应抛错(注释 "turns must exceed steps"),
而 2026-08-13 用户裁定已把 `turns > steps` 放开为 `>=`,实现早已放行 ——
测试没跟上裁定。single 档(1/1)是 `turns == steps` 的**首个真实使用者**。

### 4.2 prompt 渲染验证(A800 CPU,零 GPU;脚本自包含不碰部署树)

```
[0] system prompt 与 MENTOR 逐字一致: True
[2] token 账:  system 60 | 题面 104 | 渲染总计 177
               + 500 词 doc 后 539   (max_prompt_length=4096,余量 3557)
[3] 渲染文本含工具块: False        反证(传 tools 时): True
[5] 生成起点: '…<|im_end|>\n<|im_start|>assistant\n'
```

配置加载时打印的预算账印证单轮上限生效:
`轮×单轮乘积=8192(先到);padding 维度=train.response_length=8704`。

**踩到的坑**:本版 transformers 的 `apply_chat_template(tokenize=True)` 返回
`Encoding` 对象而非 `list[int]`,`len()` 得到的是字段数 2(第一次跑报出
"渲染总计 2" 的假读数)。生产路径本就是 `tokenize=False` 取文本再 `encode`
(`template_delta.render_ids`,与 MENTOR `dataset.py:264-265` 同款),探针脚本
已改为同一路径。

### 4.3 ★ 未决风险:`<think>` 的 zero-shot 服从性

```
'<think>'      -> 3 tok  ['<th', 'ink', '>']
'</think>'     -> 3 tok  ['</', 'think', '>']
'\boxed{'      -> 3 tok  ['\\', 'boxed', '{']
--- 对照 ---
'<reasoning>'  -> 4 tok  ['<', 'reason', 'ing', '>']
```

`<think>` 在 Qwen2.5-Instruct 词表里**不是原子 token**,碎片结构与 ALFWorld
那套 `<reasoning>`/`<action>` 同类 —— 而后者的 zero-shot 服从率实测 **0%**
(274/274 全灭 DEGENERATE、零梯度),靠 system 内单条 one-shot 救回 99.9%。

MENTOR 的 system prompt 是 **zero-shot(无示例)**,他们能成立的三个条件:
① `<think>` 是 R1 蒸馏生态的高频模式,预训练语料里常见(与生造的
`<reasoning>` 先验不同);② format 分占 10% 直接给梯度;③ 教师
OpenR1-Qwen-7B 本身就产 `<think>`,混合采样等于在示范格式。

**我们缺第 ③ 条**(单轮档目前无教师混合),第 ① 条强度未知。

⇒ **拿到卡后的第一件事**:跑格式服从率探针(几十题 × 单轮生成,量
`mentor_format_ok` 通过率)。若显著偏低,是否加 one-shot **须经用户裁定**
—— 那会成为与 MENTOR 的第 5 处差异,且属于"影响效果的改动"。
**不擅自加**。

---

## 5. 改动清单

| 文件 | 改动 |
|---|---|
| `common/config.py` | `MathCfg.protocol` 增 `single` + 新增 `format_weight`;`TrainCfg.reward_from_score`;两道 single 档 fail-loud 校验;指纹条件追加两项 |
| `envs/math/adapter.py` | `SYSTEM_PROMPT_SINGLE` / `mentor_format_ok` / `single_think` / `_react_single`;`executor_scaffold`/`tools`/`task_brief`/`format_reminder`/`replay_action` 增 single 分支 |
| `train/agent_loop.py` | reward 通道开关;`score` 进 extra_fields / reward_extra_info(**两侧对称**)/ dump |
| `evalm/run_eval.py` | 汇总增 `mean_score` |
| `envs/math/build_dataset.py` | `--protocol` 参数(parquet prompt 字段按档选 scaffold) |
| `configs/classic/math_single.yaml` | 探针档(新) |
| `configs/classic/math_single_7b_{pgs,grpo}.yaml` | 训练双臂(新,程序化派生) |
| `tests/test_math_env.py` | +8 用例;fixture 补 `TRACE_35B_KEY` |
| `tests/test_protocol.py` | 修陈旧断言(`turns == steps` 合法) |

**配置的可复现真源**:`scratchpad/derive_math_single.py` —— 从
`math_7b_pgs.yaml` 逐条精确替换(10 条,**锚点不唯一即中止**),再从 pgs 档
派生 grpo 档(7 条)。机制段(steering/research/narration)与 lr/batch/n/temp
**逐键未动** ⇒ 与 TIR 多轮档的差异恰好等于协议维度本身。

**数据无需重建**:运行时只消费 parquet 的 `extra_info`(`run_eval.py:48`),
`prompt` 字段不进实际渲染(executor 的 system prompt 一律经
`adapter.executor_scaffold()` 现取)。`--protocol` 只是让事后读 parquet 的人
不被旧 scaffold 误导。

---

## 6. 未完成 / 阻塞

- **无 GPU 可用**:A800 8 卡跑 ALFWorld 双臂(用户裁定长期运行)、162 的 4 卡
  跑 35B researcher 服务(为该双臂供 doc)。真实 rollout 探针必须等卡。
- **服务器代码未同步**:bench 线部署在 `TRACE-OPD-TERM`,而该目录正跑着
  ALFWorld PGS 臂。按纪律不动在跑实验的部署树;发车前同步,且须先做 sha
  对拍(防覆盖服务器上的独有改动)。
- 格式服从率探针(见 §4.3)——发车前的第一道门。
