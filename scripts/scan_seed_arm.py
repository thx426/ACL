#!/usr/bin/env python3
"""停机前盘点:严格区分 SEED 臂(要停)与 PGS 臂(绝不能碰)。
判据用 /proc 直读,双路交叉:experiment_name 与 CUDA_VISIBLE_DEVICES。
只报告,不杀。"""
import os

SEED, PGS = "classic_alf3b_seed", "classic_alf3b_pgs"


def read(pid, what):
    try:
        with open(f"/proc/{pid}/{what}", "rb") as f:
            raw = f.read()
        sep = b" " if what == "cmdline" else b"\n"
        return raw.replace(b"\0", sep).decode("utf8", "replace")
    except Exception:
        return ""


me = {os.getpid(), os.getppid()}
rows = {"seed": [], "pgs": [], "cvd0123_other": [], "ambiguous": []}
for p in os.listdir("/proc"):
    if not p.isdigit():
        continue
    pid = int(p)
    if pid in me:
        continue
    c = read(pid, "cmdline")
    if not c or "scan_seed_arm" in c:
        continue
    e = read(pid, "environ")
    cvd = ""
    for ln in e.split("\n"):
        if ln.startswith("CUDA_VISIBLE_DEVICES="):
            cvd = ln.split("=", 1)[1]
            break
    by_seed = SEED in c
    by_pgs = PGS in c
    rss = 0
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmRSS:"):
                rss = int(ln.split()[1]) // 1024
                break
    except Exception:
        pass
    item = (pid, cvd or "-", rss, c[:110])
    if by_seed and by_pgs:
        rows["ambiguous"].append(item)
    elif by_seed:
        rows["seed"].append(item)
    elif by_pgs:
        rows["pgs"].append(item)
    elif cvd in ("0,1,2,3",) and ("python" in c or "ray" in c.lower()):
        rows["cvd0123_other"].append(item)

for k, title in (("seed", "SEED 臂(要停)"), ("pgs", "PGS 臂(★ 绝不能碰)"),
                 ("cvd0123_other", "CVD=0,1,2,3 但名字不带 experiment_name(ray/引擎层)"),
                 ("ambiguous", "★ 两个名字都命中(异常,须人工看)")):
    print(f"\n=== {title}: {len(rows[k])} 个 ===")
    for pid, cvd, rss, c in sorted(rows[k])[:14]:
        print(f"  pid={pid:<8} CVD={cvd:<10} RSS={rss:>6}M  {c}")
    if len(rows[k]) > 14:
        print(f"  … 另有 {len(rows[k]) - 14} 个")
