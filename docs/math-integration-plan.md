# Math 任务环境引入方案(DAPO-Math-17k × 6 评测集)

状态:调研完成、方案待用户裁定(2026-08-07)。
契约骨架:通用性契约(设计文档 §9)+ `envs/_template/` + SSB 先例(FC 协议
大量同构复用)。

## 0. 调研结论(全部实查,可溯)

**训练集** `BytedTsinghua-SIA/DAPO-Math-17k`(verl 格式,DAPO 论文配套):
- HF 实际行数 **1,791,700**——是每题重复展开版(动态采样 epoch 展开),
  **必须按题去重 → ~17k 唯一题**(去重键用题面文本 hash,extra_info.index
  不可信);
- prompt 固定三段模板:指令头("...The last line of your response should
  be of the form Answer: $Answer...")+ 题面 + 尾部提醒——**剥离锚点明确**
  (不匹配即 fail loud,防数据版本漂移);
- `reward_model.ground_truth` 为纯字符串(整数为主,DAPO 特意选整数答案
  题);`ability` 恒 "MATH",**无类别字段**(task_type 无自然来源)。

**测试集六源**(schema 各异,逐源适配;总计 ~1305 题):

| benchmark | HF id | 题/答案字段 | 规模 | 答案形态 |
| --- | --- | --- | --- | --- |
| AIME24 | math-ai/aime24 | problem / solution(`\boxed{204}` 需剥) | 30 | 整数 |
| AIME25 | math-ai/aime25 | (同 aime24 系,转换时验证) | 30 | 整数 |
| AIME26 | MathArena/aime_2026 | problem / answer(int;split=train) | 30 | 整数 |
| AMC23 | math-ai/amc23 | question / answer | 40 | 整数 |
| MATH-500 | HuggingFaceH4/MATH-500 | problem / answer(latex,如 `\left(3,\frac{\pi}{2}\right)`)| 500 | latex 表达式 |
| Olympiad | math-ai/olympiadbench | question / final_answer(**列表**) | ~675 | 混合(Text-only 子集) |

**验证器**:HF `math-verify`(pypi,parse+verify;latex/表达式/字符串等价,
RLVR 社区标准)——鉴于 MATH-500/Olympiad 的表达式答案,它是**必需件**;
DAPO 风格字符串精确匹配作 fallback。纯 CPU、轻量。

**旧项目资产**:两代旧项目均无 math 环境("OPD 常用领域"指文献惯例);
形态参考 = TIR 文献(ToRA/Qwen-Math-TIR:推理+代码执行交替)+ DAPO
(verl 系,单轮;我们改多轮工具化)。

## 1. 交互协议(用户初步想法的确认与补全)

**用户想法确认:python 工具做计算——够,但需补一个载答案的终局工具**:

- `run_python(code)`:沙箱执行(数值/sympy 符号/枚举验证/暴力搜索——
  覆盖 TIR 全部工具需求);SSB 的 subprocess 执行件同构复用(净化 env、
  timeout、截断、并发门);
- `submit(answer)`:终局,**答案走参数而非文本行**——零抽取歧义(不再
  依赖 "Answer:" 行解析),score 在 submit 时由 verifier 判;feedback 恒
  "Submitted."(零评分信息,信息状态红线同 SSB)。

形态:原生 FC(OpenAI 格式 `tools()` + 模板渲染 + 文本层解析),与 SSB
完全同构。**长推理写在 tool call 前的文本部分**(FC 形态天然支持);不需
计算的题可首轮直接 submit。无状态环境 ⇒ EnvHandle 极薄(临时 cwd 沙箱,
close 清理);重放 = 重跑工具调用(确定性 ✓ v4 续写零障碍)。

## 2. 评估与信息状态

- score 二元 {0,1}(math-verify 等价判定;Olympiad 多元素答案:逐元素
  verify 全对才 1,转换时统计多答案题占比);
- **ground_truth 随 extra_info 携带**(评分必需;载体与 SSB 的 golden
  文件不同但纪律同):adapter 保证零进提示词/反馈/researcher——
  `gt_text=None`(答案即测试答案本体),submit 反馈零信息;
- **oracle 探测风险**(数学特有的显形):researcher 可用实验 won 反馈
  验证候选答案再写进 doc——与 SSB 单 case 死记同框:训练题 doc 死记
  不污染 val(val 全新题),记录为已知特性、报告口径分层,不加机制。

## 3. 机制交互点核对

| 交互点 | Math 落点 | 机制改动 |
| --- | --- | --- |
| 装配 | registry 一行 + `MathCfg`(exec_concurrency/python_timeout_s≈30/obs_max_chars;**无 data_root**——任务全自足于 extra_info) | 零 |
| 工具协议 | tools() OpenAI 格式 + 模板渲染 + parse_tool_calls | 零(SSB 先例) |
| score | 二元;won=score>=1.0 | 零 |
| v4 三段/前缀续写 | 协议无关;重放平凡(无状态) | 零 |
| researcher 双模式 | enabled 开关(自体/外部 35B) | 零 |
| SFT 分层 | task_type:训练集常量 "math"(退化均匀抽样,协议允许);val 按 benchmark 名 | 零 |
| 数据 | 转换脚本(去重/剥模板/六源适配);**无文件资产、无服务器依赖**——本地即可构建 | 新脚本 |
| 依赖 | `math-verify` 进训练环境(env 入口纪律,CD-006 清单) | 新依赖 |

## 4. 工程清单

1. `envs/math/`:adapter.py(react=FC 分派;submit 走 verifier)、
   verifier.py(math-verify 封装 + 精确匹配 fallback + 多答案)、
   build_dataset.py(DAPO 去重剥模板 → train parquet;六源 → valid
   parquet,`data_source=math/<bench>` 分桶曲线)、README;
2. python 执行共享件:SSB 的 `_run_python` 提取为 `envs/pytool.py`
   (subprocess+净化+门),SSB/math 共用;
3. registry + MathCfg + 双 yaml(selfres/extres;episode caps 与 token
   预算见决策点);
4. 测试:verifier 行为快照(整数/latex/列表/错误形态)、契约面、episode
   端到端(真 subprocess:计算题经 run_python→submit 胜局;免工具题首轮
   submit)、续写端到端、信息状态红线(ground_truth 不出现在任何模型可见
   文本)。

## 5. 决策点(待用户裁定)

**a. 训练集规模**:17k 唯一题全量(推荐;RL 常态,~1000 step/epoch@bs16)
vs 固定 seed 抽子集(如 2k,epoch 语义与 ALFWorld/SSB 相近)。

**b. 训练期 val 口径**(表格口径 avg@32(AIME/AMC)/avg@8(M500/Oly)
是终局口径,训练 val 不可承受):
- 方案甲(推荐):valid parquet 六源全量 1305 题 greedy@1,每源独立
  data_source 曲线(AIME 30 题曲线噪声大,主看 MATH-500/Olympiad 大集;
  1305 rollout/val 点可承受);终局评测增强 run_eval 支持 `--n/--temperature`
  跑 avg@K 重口径(工程项,与文献可比);
- 方案乙:训练 val 只挂 MATH-500+Olympiad(稳),AIME/AMC 仅终局。

**c. 预算**:max_env_steps(工具调用数)默认 8?(TIR 常态 1-4 次+submit;
可配);**max_tokens_per_turn 需上调**(数学长推理轮,建议 8192,vs
ALFWorld/SSB 2048);response_length 32768 沿用。

**d. researcher 预算**:n_exp=16/k=4 沿用?(数学 episode 比 ALFWorld 短,
成本更低,沿用无压力)。
