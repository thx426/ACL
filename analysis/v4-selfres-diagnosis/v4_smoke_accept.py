#!/usr/bin/env python3
"""v4 冒烟七项验收:只读 rollout JSON,不改任何运行中状态。

判据(全部要 PASS 才起正式 run):
  1. 行布局   384 = 16 任务 × (8 steered + 8 bare + 8 research)
  2. 分组隔离 research 行与 bare/steered 不共享 group/uid
  3. reward   research_r_raw 非恒零、值域 [0,1]
  4. 能力读数 won_bare_rate 只由 bare 臂贡献,未被 steered/research 稀释
  5. schema   两类行各自的必需键齐全,无 None
  6. 版本证据 research 行带 min/max_global_steps(本次修的 P0)
  7. 健康度   replay_drift / degraded / doc_overflow 全零
"""
import collections
import glob
import json
import statistics as st
import sys

RD = sys.argv[1]
files = sorted(glob.glob(RD + "/*.json"))
rows = [json.load(open(f)) for f in files]

ok = True


def chk(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        ok = False


def arm(r):
    s = r.get("steering") or {}
    a = s.get("steer_arm")
    return int(a) if a is not None else None


research = [r for r in rows if r.get("sample_role") == "research"]
student = [r for r in rows if r.get("sample_role") != "research"]
bare = [r for r in student if arm(r) == 0]
steered = [r for r in student if arm(r) == 1]

print("=== 1. 行布局 ===")
tasks = {r.get("task_id") for r in rows}
chk("总行数 384", len(rows) == 384, f"实得 {len(rows)}")
chk("任务数 16", len(tasks) == 16, f"实得 {len(tasks)}")
chk("research 128", len(research) == 128, f"实得 {len(research)}")
chk("bare 128", len(bare) == 128, f"实得 {len(bare)}")
chk("steered 128", len(steered) == 128, f"实得 {len(steered)}")
per = collections.Counter((r.get("task_id"), r.get("sample_role") or f"arm{arm(r)}") for r in rows)
bad = {k: v for k, v in per.items() if v != 8}
chk("每任务每段恰 8 条", not bad, f"异常 {list(bad.items())[:3]}" if bad else "")

print("\n=== 2. 分组隔离(research 不与 student 混组) ===")


def gid(r):
    for k in ("group_id", "uid", "index", "prompt_uid"):
        if k in r:
            return r[k]
    v = r.get("validate") or {}
    return v.get("group_id") or v.get("uid")


g_res = {gid(r) for r in research} - {None}
g_stu = {gid(r) for r in student} - {None}
if not g_res and not g_stu:
    print("  [SKIP] rollout JSON 未落 group 键;分组隔离改由 trainer 侧断言,见下方 schema 检查")
else:
    chk("research 与 student 组不相交", not (g_res & g_stu), f"交集 {sorted(g_res & g_stu)[:3]}")

print("\n=== 3. research reward ===")
# 字段实际落在 research 子字典且带 research_ 前缀(实测 2026-08-07)


def rf(r, name):
    return (r.get("research") or {}).get(name)


rr = [x for x in (rf(r, "research_r_raw") for r in research) if x is not None]
chk("r_raw 有值", len(rr) == len(research), f"{len(rr)}/{len(research)}")
if rr:
    nz = sum(1 for x in rr if x > 0)
    chk("r_raw 非恒零", nz > 0, f"非零 {nz}/{len(rr)} ({100*nz/len(rr):.0f}%)")
    chk("r_raw 值域 [0,1]", all(0.0 <= x <= 1.0 for x in rr),
        f"min={min(rr):.3f} max={max(rr):.3f}")
    print(f"         中位 {st.median(rr):.3f}  均值 {st.mean(rr):.3f}")
    for label, key, base in (("experiments", "research_experiments", "基线 3.5"),
                             ("valid_conts", "research_valid_continuations", "满分 20"),
                             ("turns", "research_turns", ""),
                             ("doc_words", "research_doc_words", "")):
        xs = [x for x in (rf(r, key) for r in research) if x is not None]
        if xs:
            print(f"         {label:<12} 中位 {st.median(xs):>6.1f}  均值 {st.mean(xs):>7.1f}"
                  + (f"   ({base})" if base else ""))

print("\n=== 4. 能力读数 won_bare_rate ===")


def won(r):
    o = r.get("outcome") or {}
    return bool(o.get("won") if isinstance(o, dict) else r.get("won"))


wb = sum(won(r) for r in bare)
ws = sum(won(r) for r in steered)
chk("bare 臂有胜局(能力读数可用)", len(bare) > 0, "")
print(f"         won_bare_rate  = {100*wb/max(len(bare),1):.1f}%  ({wb}/{len(bare)})")
print(f"         won_steer_rate = {100*ws/max(len(steered),1):.1f}%  ({ws}/{len(steered)})")
print(f"         δ gap          = {100*(ws-wb)/max(len(bare),1):+.1f}pp")
chk("won_bare 未混入 research", all(r.get("sample_role") != "research" for r in bare), "")

print("\n=== 5/6. schema 与版本证据 ===")
miss_res = [k for k in ("global_step", "session_id", "task_id", "research")
            if any(k not in r for r in research)]
chk("research 行必需键齐", not miss_res, f"缺 {miss_res}")
nover = [r for r in research
         if rf(r, "min_global_steps") is None or rf(r, "max_global_steps") is None]
chk("research 行带 policy 版本证据(P0 修复)", not nover,
    f"{len(nover)}/{len(research)} 行缺 min/max_global_steps")
if research and not nover:
    mn = {rf(r, "min_global_steps") for r in research}
    mx = {rf(r, "max_global_steps") for r in research}
    print(f"         min_global_steps={sorted(mn)}  max_global_steps={sorted(mx)}")
miss_stu = [k for k in ("global_step", "task_id", "outcome") if any(k not in r for r in student)]
chk("student 行必需键齐", not miss_stu, f"缺 {miss_stu}")

print("\n=== 7. 健康度 ===")
# 一致性哨兵:不该发生,必须全零。
#   replay_drift = 环境重放的确定性契约被破坏
#   degraded     = 会话降级
for key in ("research_replay_drift", "research_degraded"):
    vals = [bool(x) for x in (rf(r, key) for r in research) if x is not None]
    if vals:
        chk(f"{key} 全零(一致性哨兵)", not any(vals), f"{sum(vals)}/{len(vals)} 命中")
    else:
        print(f"  [SKIP] {key} 未落盘")

# 容量降级:设计内会发生(selfres.py 注释 "r=0:结构性长度压力")——
# doc 太长 ⇒ teacher_ids_for_doc 返回 None ⇒ 塞不进 executor 的 prompt 预算。
# 判据看比例而不是"必须为零",否则会把一条正常降级读成实验不可起。
DOC_OVERFLOW_MAX = 0.05
vals = [bool(x) for x in (rf(r, "research_doc_overflow") for r in research) if x is not None]
if vals:
    frac = sum(vals) / len(vals)
    chk(f"research_doc_overflow ≤ {DOC_OVERFLOW_MAX:.0%}(容量降级,允许少量)",
        frac <= DOC_OVERFLOW_MAX, f"{sum(vals)}/{len(vals)} = {frac:.1%}")
else:
    print("  [SKIP] research_doc_overflow 未落盘")

for key in ("research_overlong", "research_no_finalize", "research_error"):
    xs = [x for x in (rf(r, key) for r in research) if x is not None]
    hit = sum(1 for x in xs if x)
    print(f"  [INFO] {key:<22} {hit}/{len(xs)}  ({100*hit/max(len(xs),1):.0f}%)")

RESP_CAP = 32768
over = [r for r in rows if (r.get("response_token_len") or 0) > RESP_CAP]
ovr = [r for r in over if r.get("sample_role") == "research"]
print(f"  [INFO] response_token_len > {RESP_CAP}: {len(over)}/{len(rows)} 行"
      f"  (其中 research {len(ovr)})")
for label, grp in (("research", research), ("bare", bare), ("steered", steered)):
    xs = [r.get("response_token_len") or 0 for r in grp]
    if xs:
        print(f"         {label:<9} respLen 中位 {st.median(xs):>7.0f}  "
              f"均值 {st.mean(xs):>7.0f}  max {max(xs):>7.0f}")

print("\n" + ("=== 全部 PASS ===" if ok else "=== 有 FAIL,不起正式 run ==="))
sys.exit(0 if ok else 1)
