# v4 自体研究员生成段吞吐诊断(CD-010,2026-08-08)

**数据源**:162 `/data4/ws/PGS/runs/pgs_s4_smoke5` step_00001(384 行齐:16 任务
× [8 bare + 8 research + 8 steered],1.7B;取证脚本 `/tmp/v4_{timeline2,rows}.py`,
时序由 dump `ts`/`wall_s` 与 role 字段重构)。formal run(`pgs_s4_selfres_1p7b_162`)
step1 中断仅余 bare 行,不可用。

## 0. 结论先行

v4 的**调度骨架没有可挤的水分**——"依赖用 shared future,其余全并发"的红线
已兑现:bare 立即起跑(prefix 等待实测 ≈0)、研究会话与 bare 并行、k/续写全
gather、续写同 request_id 共享 teacher 前缀 KV。生成段墙钟(smoke5 实测
**63 min**)由两个**机制内生**因素主导,全部优化空间都在等待语义与会话预算
形态上,**没有零效果面的落码项**;三个候选全部需要用户裁定(§4)。结构性
下界:research 行是训练行,整步生成段 ≥ 最慢 research 行(必须跑完进 GRPO)。

## 1. Smoke5 实测时序(t0=11:13:44)

| 段 | 读数 |
|---|---|
| bare 段 | 128 条,起步即满载(首分钟并发 128),任务内 bare_last 中位 157s |
| research 段 | 每任务 8 行;行完成时刻同任务首尾差极大(首行最快 +0s,末行 +1387~+2200s) |
| argmax(=末行登记) | 中位 **3096s(51.6 min)**,最慢 3264s |
| steered 段 | 起跑时刻==argmax(逐字相等,全量阻塞实锤);每任务 2-6 min;总并发峰仅 36 |
| 生成段合计 | 11:13:44→12:17:21 ≈ **63 min**;11:50 后引擎只剩涓流 steered 拖 25 min |

配置(smoke5):research_n=8、n_exp=16、k_default=4、quantiles=5、
continuation_samples=4(每有 doc 行 20 条续写)、max_turns=30、
session_timeout_s=3600、max_total_tokens=38912(ctx 墙)。

## 2. 最慢行画像(argmax 的决定者)

16 个任务的最慢 research 行:**err=14/16、no_finalize=15/16**,response token
37k–39k(全部顶到 38912 的 ctx 墙),实验消耗 e=44–56(把 16×4 预算几乎烧满),
续写 c=0。即:**拖住 argmax 的行几乎全是"预算烧满→撞墙→没写出 doc"的
失败行,对 argmax 零贡献**(nf 行不 eligible),却让每任务的 steered 平均
多等 P50=459s / P90=818s / max=845s。对照:成功行(c=20)常在 5–7 turns、
数百秒内 finalize。

死法机理(代码坐实):token 墙是硬的——`backends.py:194-196` 流长逼近
max_total 即返回空 content 的 `finish_reason="length"` 轮;而 wrapup 软着陆
只有 turn 维度(`session.py:125`,max_turns−1−buffer)。**token 维度没有
wrapup**:上下文涨得快的会话在 ~15 turn 就撞墙,之后只能空转到轮数耗尽,
既慢又零产出。

## 3. 已核查、确认无空间/关闭的项

- prefix_future 等待:实测 ≈0(bare_last 中位 157s ≪ 会话完成时刻),关闭;
- 会话逐 turn 串行/每消息一个 tool call:协议设计,不动;
- 实验 k、续写 specs×samples:已 gather 并发;续写 make_generate 恒定
  request_id ⇒ teacher 前缀 KV 同 replica 复用,已优化;
- nf/degraded 行跳过续写:已是快速路径;
- 11:21–11:50 的 bare+steered 双零窗:引擎在跑 128 条 research 行的
  会话+实验,非空转;
- verl 全批 barrier / GRPO 组间流水:架构级,收益被下述裁定项覆盖,不做;
- v3 的 bare-first(b475747)对 v4 无涉(v4 bare 本就立即起跑)。

## 4. 三个裁定项(按建议优先级)

> **状态(2026-08-08 用户裁定)**:调度并发已吃满,V1/V2/V5 **暂不推进,
> 整体挂账**——不是否决;若后续 v4 正式跑的墙钟成为瓶颈,从 V2 起重启裁定。

**V2|token 维度 wrapup(压最慢行 + 救赎 nf 行;机制正面,首推)**
wrapup 判据加"token 用量 ≥ wrapup_token_frac(建议 0.75)×max_total"(与
turn 判据取或,注入同一 WRAPUP_REMINDER;0.75×38912≈29k 时提醒,余 ~9k
足够写 3000 词 doc)。效果面:改变研究会话预算语义——但对现状撞墙行是
纯改善(现在 0 产出)。预期:nf 率 15/16 显著下降、最慢行左移。
实现:EngineResearcherChat 暴露用量,session 判据一行,配置一键。

**V1|deadline-argmax(steered 藏进慢行影子)**
argmax_future 在 max(全部登记, 首行登记+Δ) resolve;deadline 后未登记的行
不参与 doc 选择(行本身照常完成进训练)。效果面:偶尔错过慢而好的 doc——
**本 smoke 数据上 15/16 任务最慢行是 nf,doc 选择结果逐字不变**,风险经验
上限 1/16。收益:steered 段与慢行重叠,生成段 ≈ 最慢 research 行本身。

**V5|session_timeout_s 3600→~2700(尾部保险)**
实测最慢行 2200s;3600 意味着一个真卡死会话拖整步至 1 小时。降到
~1.25×实测 max。超时语义已有(snapshot 降级),纯参数。

三项合并的外推(smoke5 口径,须重冒烟验证):生成段 63 → **~35 min**
(最慢正常行 21–31 min + steered)。剩余墙钟由研究预算(n_exp=16/k=4/
8 行)×模型解码速度决定——**这是 v4 与 v3.3(13 min/step@9B)差距的主体,
属预算裁定而非调度浪费**;9B 上未来跑 v4 时此表为预算定档的直接输入。

## 5. 与 v3.3 的结构对照(供预算裁定参考)

| | v3.3(9B/A800 实测) | v4(1.7B/162 smoke5 实测) |
|---|---|---|
| researcher | 35B 外部,每任务 1 会话(n_exp=8,k=2) | policy 自体,每任务 **8 行**会话(n_exp=16,k=4) |
| 生成段 | 13.2 min | 63 min |
| 主导项 | research 前置串行(已修 bare-first) | argmax 等失败行 + 失败行撞墙无软着陆 |
| 研究 episodes/任务 | ~16 | ~150–250(+每行 20 续写) |
