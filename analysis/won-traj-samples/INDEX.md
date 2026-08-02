# 代表性轨迹样本 — trace_opd_4b_rkl_seqmean_rl(won 分层复盘,2026-08-02)

来源:162 `/data3/wushang/workspace/TRACE-OPD/trace_opd/runs/trace_opd_4b_rkl_seqmean_rl/rollouts/`;
选样口径:各阶段按 thought 总字符取中位(median)或最大(extreme)。raw/ 下为
原始 dump json,每条另渲染为同名 .md(thought 全文、action 原文、observation 截断 500 字符)。

| 样本 | step | 结局 | tok | turns | env_steps | thought 字符 | 字符/turn | 看点 |
|---|---|---|---|---|---|---|---|---|
| [early_won](early_won.md) | 6 | WON | 1571 | 15 | 15 | 3865 | 258 | 早期健康形态(step 6):短、行动导向,thought 简洁定位式 |
| [phase_won](phase_won.md) | 9 | WON | 3188 | 20 | 20 | 9990 | 500 | 相变点(step 9):膨胀启动,thought 开始加长 |
| [mid_won](mid_won.md) | 13 | WON | 2174 | 12 | 12 | 7168 | 597 | 中期(step 13):thought/turn 已 ~2x 早期 |
| [late_won_median](late_won_median.md) | 20 | WON | 3750 | 5 | 5 | 13961 | 2792 | 崩塌窗中位(step 20):普通的成功轨迹也已叙述化 |
| [late_won_extreme](late_won_extreme.md) | 20 | WON | 12154 | 13 | 13 | 44420 | 3417 | 崩塌窗极端(step 20):汇报引用的 laptop 轨迹——无依据断言/自我说服全样 |
| [val16_won_extreme](val16_won_extreme.md) | 16v | WON | 20631 | 20 | 16 | 73840 | 3692 | val greedy 极端(step 16 val):贪心解码下的成功轨迹叙述爆炸 |
| [late_lost_extreme](late_lost_extreme.md) | 20 | LOST | 32704 | 14 | 13 | 119167 | 8512 | 失败对照(step 20 lost):撞满上下文的形态,与 won 极端条对照 |

阅读建议:early_won → late_won_extreme 对读同类任务(都是 pick_and_place)的
thought 形态变化;late_won_extreme 的中段 turn 是「无依据断言 + 循环自我说服」
的典型段;val16_won_extreme 看 greedy 下的极端形态;late_lost_extreme 看撞满
上下文时 thought 的终态。