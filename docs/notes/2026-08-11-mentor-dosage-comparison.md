# MENTOR 干预剂量取证与本 run 对照(2026-08-11,CD-010)

**目的**:回答"MENTOR 实验时分布干预有多大"并对照 `ssb_4b_selfres_engine` 的失控
剂量,为 PGS 修复方案提供文献锚点。姊妹篇:`docs/diagnosis/2026-08-11-ssb-engine-run-degradation.md`。

**来源**:论文 `related_work/papers/2510.04140_MENTOR_ICLR2026.pdf`(页码下注);
代码 `related_work/code/MENTOR`(commit b1cc4c6;在 trace_opd-pgs worktree,
engine worktree 仅有 INDEX)。

---

## 1. MENTOR 的实际干预剂量(五个维度,全部实数)

**混合数学**(Eq.8, p5;rejection_sampler.py:193-199):
`π_mix = (1−w_t)·π_θ + w_t·π*`,`w_t = min(1, H_t/γ_p)`——概率域线性软插值,
权重由学生 token 熵除以阈值决定。

| 维度 | MENTOR 实际值 | 出处 |
|---|---|---|
| **位置/强度** | γ_p = batch token 熵 **P95**(p=0.95)+ 代码地板 `max(q80_var, 1.5)`;⇒ 满权重(w=1)位 ≈**5%**,其余按 H/γ_p 稀释——低熵(自信)token w→0,教师无权干预 | Appx C p18;vllm_rollout_spmd.py:109 |
| **冷启动** | **首步 γ_p=999 ⇒ 第一步教师实质禁用**;之后每步用上一步 batch 熵分位直接 set(滞后一步,无 EMA)——分位数自标定,无收敛时延 | Appx C p18(论文明写 "As a special case");dp_actor.py:373 |
| **群体** | n_mixedpolicy=4 / n=8 = 50% 行混合(on 组 4 行保持纯净基线) | train.sh;Appx C |
| **时间** | α:1→0 cosine **120 步**;α=0 后 mixed rollout 停产(教师=有限期脚手架) | Appx C;ray_trainer.py:633 |
| **advantage** | mixed 行 `α·[R−mean(G_on)]₊/R_range`(Eq.13);**无 IS 校正**(π_old 全 batch 统一重算,ray_trainer.py:672);expert_cap 默认 99 不激活 | p6;core_algos.py:171 |

**执行**:draft=学生 π(K=4 spec tokens),verify=π_mix,标准接受/残差重采
——落地 token 分布=π_mix,但机制天然让大多数 token 是学生自己的提议。

**教师性质(与我们的本质差异)**:OpenR1-Qwen-7B——**能力型教师**(公开 R1
蒸馏推理模型,同 7B 规模,**无 per-task 特权信息**);任务=单轮 MATH(难度 3-5,
8889 题),reward=**二值**(Math-Verify 对/错)+格式分(9:1)。

## 2. 他们文里就有的"剂量-反应曲线"(对我们最有用的三块)

1. **α ablation(Table 4, p20)**:GRPO 76.8/14.2(MATH/AIME24);
   **α 固定 1.0:78.2/13.9——AIME24 低于 GRPO**;α 固定 0.5:80.4/16.1;
   α decay 1→0:81.4/18.3。论文结论原话:*"a lower fixed weight (α=0.5) yields
   stronger performance than an overly high weight (α=1)…preserving autonomy is
   necessary for genuine understanding rather than rote imitation."*
   ⇒ 在能力型教师+P95 稀疏门+二值 reward 的**最温和语境**下,α=1.0 常驻已翻负。
2. **guidance 位置 ablation(Appx D, Table 2, p19-20)**:每 token 20% 均匀随机
   路由教师几乎无效(77.6 vs 76.8);top-20% 高 PPL 硬路由**偶尔有害**(AIME24
   13.3<14.2);熵门 81.4。词云(Fig.5):熵门注入的是逻辑连接词(推理分叉),
   其余策略注入的 token "remain close to the base model"。⇒ **位置选择 > 干预总量**。
3. **失败模式的文献预演(§4.2 p7 + Fig.3 p9)**:LUFFY(全轨迹模仿)
   *"overfit superficial expert patterns and fall into suboptimal strategies"*、
   过度采纳 okay/wait 等表面 token;QuestA hint 过量对弱模型 **-1.6**。
   ——正是本 run val 崩塌(无差别秒交=表面模式过拟合)的同构病理。

**理论护栏的前提(Appx A.3, p16-17)——核实后修正(2026-08-11)**:[·]₊ 的
"自动过滤误导引导"证明依赖**二值 reward**。初判曾疑 SSB soft 分弱化该护栏,
**代码核实推翻**:训练 reward 自 Initial commit 起就是二值
(`agent_loop.py` `reward_score = 1.0 if traj.won else 0.0`;soft score 仅进
dump/审计,README 双轨裁定"soft→score,hard→won")。护栏在场且工作正常——
**但它只过滤失败的引导,挡不住成功特权轨迹的行为克隆**:本 run 的正推行
全部是 won 行(won 口径 ≈546 行,与诊断文档 soft 口径 524 同量级),
全在护栏之内。毒性核心因此不在 reward 结构,在**教师优势来源 × 剂量**。

## 3. 本 run 与 MENTOR 的剂量逐维对照

| 维度 | MENTOR | ssb_4b_selfres_engine 实测 | 倍数/性质 |
|---|---|---|---|
| 满强度位占比 | ≈5%(P95 分位) | 目标 0.05,实际 open 率 0.25–0.37 | **3–7×** |
| 冷启动 | 首步禁教师(γ_p=999) | θ=1.28 占位直接 0.33 门率 | **方向相反** |
| 阈值更新 | 每步分位直标(无时延) | EMA β=0.1,24 步未收敛 | 慢一个量级 |
| 干预 token/轨迹 | 轨迹~1751 tok、强干预仅熵尾 | 800–3600 tok/轨迹 | 量级差 |
| α(前 20 步) | 同款 cosine(120 步) | ≈0.91–1.0(123 步) | 同 |
| reward 过滤 | 二值:全对才正推 | 同为二值(won;soft 分仅审计) | 同,护栏在场 |
| IS 校正 | 无(靠稀疏门+[·]₊+α 兜) | 无(W off) | 同,但前提失守 |
| 教师优势来源 | **能力差**(可蒸馏) | **信息差**(per-task 答案 doc,不可蒸馏) | 本质差异 |
| 任务结构 | 单轮 math(无行为捷径) | 20-turn agentic(存在"秒交"捷径) | 捷径面差异 |

## 4. 对修复方案的落点(供裁定)

> 落点经用户裁定(2026-08-11)通用泛化后实施,定稿见
> `docs/plans/2026-08-11-dosage-control-generalized.md`;下表为取证时的
> 原始候选与最终处置。

1. **θ 标定换范式(第一优先,已实施)**:弃"率控 EMA 慢追",改分位直标 +
   首步零干预 warmup;绝对 floor(MENTOR 的 1.5 是 math 语境值)不照搬——
   通用化为 **ratchet**(θ 不低于首次标定值,自标定无量纲);rate_bounds
   告警接到熵门(此前只接 adaptive 硬门)。一次修掉占位起点、EMA 时延、
   防线缺口三个病,且全部无 benchmark 特定常数。
2. ~~soft reward 的过滤修复(won 门控)~~ **取消**:核实训练 reward 本就
   二值(见 §2 修正),[·]₊ 护栏在场;won 门控数学上恒等于现状,无需实施。
   该护栏挡不住的"成功特权轨迹克隆"由剂量控制(1)承担第一道防线;
   结构性方案(干预位/自由位分治 advantage)留作后续裁定项。
3. **α 消融臂(进验证方案)**:α0=0.5 对照(MENTOR Table 4 差值 2.2pp 的
   方向证据);配置层,无代码改动。
4. **W_beh 暂不默认开**:MENTOR 证明剂量正确时无 IS 也可工作;先修剂量,
   W 留作消融,避免多变量归因不清。
5. **监控固化(已实施)**:`scripts/rollout_health_check.py`——turns 分布/
   秒交率/mid-EOS/臂对照,吃统一 dump schema,benchmark 无关;本次
   "表面模式过拟合"的最早指纹是行为分布,不是 won。
6. **边界诚实**:MENTOR 两个维度仍比我们容易(能力型教师/无捷径单轮);
   其数字标定方向,不保证照抄即安全;各 benchmark 的安全剂量由分位自标定
   自动适配,但 target_rate=0.05 本身的合理性仍待跨 benchmark 验证。
