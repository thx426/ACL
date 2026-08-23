#!/usr/bin/env python3
"""从 alfworld_3b_pgs.yaml 派生纯 GRPO baseline。

规则照搬既有的 alfworld_7b_pgs → _grpo(逐条比对过):
  换 run/experiment 名、卡组注释、adv_estimator、agent_task_timeout_s;
  steering 全段塌成 enabled:false + doc_source:static;
  research.enabled:false,并删掉 narration/roles 两段(无外部角色 ⇒ 不必配)。
train 段其余键(batch/n/lr/caps/model/数据/eval/存盘)**逐键不动** ⇒ 与 PGS 臂
的差异恰好等于机制本身。锚点不唯一即中止。
"""
import pathlib, re, sys

CFG = pathlib.Path("/Users/shangwu/workspace/vibe_coding_projects/work/"
                   "TRACE-OPD/trace_opd-bench/configs/classic")
src = (CFG / "alfworld_3b_pgs.yaml").read_text(encoding="utf-8")

SUBS = [
    ("  name: classic_alf3b_pgs", "  name: classic_alf3b_grpo", "run 名"),
    ("  experiment_name: classic_alf3b_pgs",
     "  experiment_name: classic_alf3b_grpo", "experiment 名"),
    ("  n_gpus: 4                  # CVD 4-7(SEED 对照臂占 0-3)",
     "  n_gpus: 4                  # CVD 0-3(PGS 机制臂占 4-7)", "卡组注释"),
    ("  adv_estimator: pgs_asym_grpo",
     "  adv_estimator: grpo        # 基线臂:纯 GRPO 底座", "advantage 口径"),
    ("  agent_task_timeout_s: 5400",
     "  agent_task_timeout_s: 3600 # 基线无 research 段拖长", "任务超时"),
]


def apply(text, subs):
    for old, new, why in subs:
        n = text.count(old)
        if n != 1:
            sys.exit(f"FATAL: 锚点命中 {n} 次(需 1)——{why}\n  {old[:80]!r}")
        text = text.replace(old, new)
        print(f"  ✓ {why}")
    return text


out = apply(src, SUBS)

# steering 段:整段替换为关闭态(保留段本身,避免 schema 缺段)
i = out.index("steering:")
j = out.index("research:", i)
seg = out[i:j]
assert "gate_mode: entropy" in seg, "steering 段定位失败"
out = out[:i] + (
    "# ---- 基线臂:无 steering(纯 GRPO 底座)----\n"
    "steering:\n"
    "  enabled: false\n"
    "  doc_source: static         # enabled=false 下不消费;置 static 免 researcher 校验\n"
    "  doc_table: \"\"\n\n"
) + out[j:]
print("  ✓ steering 段 → 关闭态")

# research 段起至文件尾:替换为关闭态(删 narration/roles —— 无外部角色)
k = out.index("research:")
tail = out[k:]
assert "backend: external" in tail, "research 段定位失败"
out = out[:k] + (
    "# ---- 基线臂:无 researcher / 无 narration / 无外部角色 ----\n"
    "research:\n"
    "  enabled: false\n"
)
print("  ✓ research/narration/roles → 关闭态")

hdr_old = out[:out.index("run:")]
out = ("# ALFWorld × Qwen2.5-3B — **纯 GRPO 基线臂**(2026-08-16 用户指令)\n"
       "# 由 alfworld_3b_pgs.yaml 程序化派生(scripts/derive_alf3b_grpo.py):\n"
       "# train 段的 batch/n/lr/caps/model/数据/eval/存盘**逐键不动**,差异恰好\n"
       "# 等于机制本身 —— adv_estimator、steering、researcher/doc 三项。\n"
       "# 对照对象:classic_alf3b_pgs(CVD 4-7,同时在跑)。\n"
       "# 取代原 SEED loss 对照臂(classic_alf3b_seed,2026-08-16 19:30 停于 step 43,\n"
       "#   val@0/20/40 数据保留:seen 3.57→26.43→30.24%)。\n"
       "#\n" + "\n".join(
           ln for ln in hdr_old.splitlines()
           if ln.startswith("#") and "PGS" not in ln and "机制臂" not in ln
       ).strip() + "\n\n")[:0] + (
    "# ALFWorld × Qwen2.5-3B — **纯 GRPO 基线臂**(2026-08-16 用户指令)\n"
    "#\n"
    "# 由 alfworld_3b_pgs.yaml 程序化派生(scripts/derive_alf3b_grpo.py,锚点唯一即中止):\n"
    "# train 段的 batch/n/lr/caps/model/数据/eval/存盘**逐键不动**,与 PGS 臂的差异\n"
    "# 恰好等于机制本身 —— adv_estimator / steering / researcher-doc 三项。\n"
    "#\n"
    "# 对照对象:classic_alf3b_pgs(CVD 4-7,同时在跑)。\n"
    "# 取代原 SEED loss 对照臂(classic_alf3b_seed,2026-08-16 19:30 停于 step 43;\n"
    "# 其 val@0/20/40 = seen 3.57→26.43→30.24% 保留可引用)。\n"
    "#\n") + out[out.index("run:"):]

dst = CFG / "alfworld_3b_grpo.yaml"
dst.write_text(out, encoding="utf-8")
print(f"\n写出 {dst}")
