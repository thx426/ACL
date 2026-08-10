#!/usr/bin/env python3
"""9B/4B 作 researcher 的会话质量评估(可对比多个目录)。

字段来自 sftgen_researcher.py sample 的落盘 schema:
  doc / episodes_consumed / error / experiments_used / family /
  finalized / messages / meta / no_finalize / seed / session_idx /
  task_id / turns_used

用法: python3 quality.py <sessions_dir> [<sessions_dir2> ...]
"""
import glob, json, os, re, sys, statistics as S
from collections import Counter, defaultdict

# 文档里"有实质内容"的信号:可执行命令、位置断言、失效模式、概率性表述
CMD = re.compile(r"`(go to|take|move|use|open|close|clean|heat|cool|examine|look|inventory)\b[^`]*`", re.I)
PROB = re.compile(r"\b(most (frequently|reliable|common)|rarely|occasionally|usually|typically|always|never)\b", re.I)
FAIL = re.compile(r"\b(failure|fail|error|mistake|pitfall|do not|don't|avoid|caution|warning)\b", re.I)
SECT = re.compile(r"^#{1,3}\s+\S", re.M)


def load(d):
    rows = []
    for fp in sorted(glob.glob(os.path.join(d, "*", "s*.json"))):
        try:
            rows.append(json.load(open(fp)))
        except Exception:
            pass
    return rows


def report(name, rows):
    print("=" * 78)
    print(f"【{name}】  会话 {len(rows)}")
    if not rows:
        print("  (无数据)"); return

    fin = [bool(r.get("finalized")) for r in rows]
    err = [r for r in rows if r.get("error")]
    print(f"\n── 完成度 ──")
    print(f"  finalize 率      {sum(fin)/len(fin)*100:5.1f}%  ({sum(fin)}/{len(fin)})")
    print(f"  no_finalize      {sum(1 for r in rows if r.get('no_finalize'))}/{len(rows)}")
    print(f"  有 error 的会话  {len(err)}/{len(rows)}")
    if err:
        for k, v in Counter(str(r.get("error"))[:60] for r in err).most_common(3):
            print(f"      {v}× {k}")

    print(f"\n── 预算使用 ──")
    for key, label in (("experiments_used", "实验次数"), ("episodes_consumed", "episode 数"),
                       ("turns_used", "会话轮数")):
        v = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        if v:
            v = sorted(v)
            print(f"  {label:<12} 中位 {S.median(v):>5.1f}  min {v[0]}  max {v[-1]}")

    docs = [(r.get("doc") or "") for r in rows]
    nonempty = [d for d in docs if d.strip()]
    print(f"\n── 文档 ──")
    print(f"  非空文档        {len(nonempty)}/{len(docs)}")
    if nonempty:
        w = sorted(len(d.split()) for d in nonempty)
        print(f"  词数            中位 {S.median(w):>5.0f}  min {w[0]}  max {w[-1]}")
        print(f"  含可执行命令    {sum(1 for d in nonempty if CMD.search(d))}/{len(nonempty)}"
              f"   (命令数中位 {S.median([len(CMD.findall(d)) for d in nonempty]):.0f})")
        print(f"  含概率性表述    {sum(1 for d in nonempty if PROB.search(d))}/{len(nonempty)}"
              f"   ← 说明是从多次实验的分布里总结,而非单次照抄")
        print(f"  含失效模式/告诫  {sum(1 for d in nonempty if FAIL.search(d))}/{len(nonempty)}")
        print(f"  有 markdown 分节 {sum(1 for d in nonempty if len(SECT.findall(d)) >= 2)}/{len(nonempty)}")
        uniq = len(set(nonempty))
        print(f"  去重后           {uniq} 份不同 (共 {len(nonempty)};同任务多会话应不同)")

    print(f"\n── 按 family ──")
    fam = defaultdict(list)
    for r in rows:
        fam[r.get("family") or "?"].append(r)
    print(f"  {'family':<32} {'n':>3} {'finalize':>9} {'doc词中位':>10}")
    for k in sorted(fam):
        v = fam[k]
        f = sum(1 for x in v if x.get("finalized")) / len(v) * 100
        ws = [len((x.get("doc") or "").split()) for x in v if (x.get("doc") or "").strip()]
        print(f"  {k:<32} {len(v):>3} {f:>8.0f}% {S.median(ws) if ws else 0:>10.0f}")

    print(f"\n── 同任务多会话的一致性(方差信号) ──")
    bytask = defaultdict(list)
    for r in rows:
        bytask[r.get("task_id")].append(r)
    multi = {k: v for k, v in bytask.items() if len(v) > 1}
    if multi:
        difffin = sum(1 for v in multi.values()
                      if len({bool(x.get("finalized")) for x in v}) > 1)
        print(f"  有多会话的任务 {len(multi)};其中 finalize 结果不一致的 {difffin}")
        wspread = []
        for v in multi.values():
            ws = [len((x.get("doc") or "").split()) for x in v if (x.get("doc") or "").strip()]
            if len(ws) > 1: wspread.append(max(ws) - min(ws))
        if wspread:
            print(f"  同任务文档词数极差 中位 {S.median(wspread):.0f}")
    else:
        print("  (每任务仅一个会话)")


for d in sys.argv[1:]:
    report(os.path.basename(d.rstrip("/")), load(d))
