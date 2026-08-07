# v4 self-researcher 冒烟诊断脚本(162 / Qwen3-1.7B)

只读 `runs/<run>/rollouts/step_XXXXX/*.json`,不触碰任何运行中状态。

| 脚本 | 回答什么 |
|---|---|
| `v4_smoke_accept.py <rollout_dir>` | 七项验收:24 行布局 / 分组隔离 / research reward / `won_bare_rate` 未被稀释 / schema / policy 版本证据 / 健康度。退出码 0 = 全 PASS |
| `v4_research_diag.py <rollout_dir>` | research 行的因果链:`valid_continuations` 双峰、`overlong×no_finalize×error` 交叉表、`r_raw>0` 与 `=0` 的分组对比、抬阈救回量 |
| `v4_ctx_wall.py <rollout_dir> [ctx]` | 会话是不是撞在 context 墙上:按 `respLen` 分档看 finalize 成功率,看有没有断崖与堆积 |

## 两个必须记住的口径坑

1. **字段带 `research_` 前缀且在 `research` 子字典内** —— `research.research_r_raw`
   而不是 `research.r_raw`。路径写错会得到"全 None",看起来像功能没实现。
2. **别按文件名排序取前 N 条**。`session_id` 与任务/时序相关,前 N 条系统性偏乐观:
   2026-08-07 我用前 32 条得出 `r_raw` 非零 66%/中位 0.350,全量 128 条复算是
   28%/0.000 —— 差到足以推翻结论。要抽样就 `random.sample`,否则全量。

## 2026-08-07 的主要发现

- `research_max_sample_tokens=24576` 把 **82%** 的 research 行 mask 成零梯度;
  但未超限的行读数很好(r_raw 非零 87%、中位 0.450)⇒ 砍掉的是信号不是噪声。
- 真正的物理根因是 **ctx 上限 40960**(Qwen3-1.7B 的 `max_position_embeddings`):
  `respLen` 落在 (36864, 40960] 的 78 条(61%)finalize 成功率 **0%**,
  而 ≤32768 的 40 条是 **85–93%**。会话是被截断的,不是不会做研究。

详见 `docs/collab/cd010-cd006/experiment-feedback.md` 2026-08-07 的相关条目。
