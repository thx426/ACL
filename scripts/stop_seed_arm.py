#!/usr/bin/env python3
"""只停 SEED 臂(CVD=0,1,2,3)。三重保护,绝不碰 PGS 臂与显微镜。

杀:  CVD == "0,1,2,3"  或  cmdline 含 classic_alf3b_seed
排除:cmdline 含 classic_alf3b_pgs / pgs_microscope / stop_seed_arm
     或 CVD 含 4,5,6,7
先 SIGTERM → 等 → SIGKILL → /proc 复核(记忆:ps|grep 会自匹配,只信 /proc)。
"""
import os, signal, sys, time

SEED, PGS = "classic_alf3b_seed", "classic_alf3b_pgs"
SELF_MARK = "stop_seed_arm"
me = {os.getpid(), os.getppid()}


def read(pid, what):
    try:
        raw = open(f"/proc/{pid}/{what}", "rb").read()
        return raw.replace(b"\0", b" " if what == "cmdline" else b"\n").decode("utf8", "replace")
    except Exception:
        return ""


def targets():
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        pid = int(p)
        if pid in me:
            continue
        c = read(pid, "cmdline")
        if not c or SELF_MARK in c:
            continue
        # --- 三重排除 ---
        if PGS in c or "pgs_microscope" in c:
            continue
        cvd = ""
        for ln in read(pid, "environ").split("\n"):
            if ln.startswith("CUDA_VISIBLE_DEVICES="):
                cvd = ln.split("=", 1)[1]
                break
        if cvd and any(g in cvd.split(",") for g in ("4", "5", "6", "7")):
            continue
        # --- 命中判据 ---
        if SEED in c or cvd == "0,1,2,3":
            out.append(pid)
    return out


t0 = targets()
print(f"命中 {len(t0)} 个进程(已排除 PGS 臂与显微镜)")
if not t0:
    sys.exit(0)
for pid in t0:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
print("SIGTERM 已发,等 20s 让 ray 自清 …")
time.sleep(20)
t1 = targets()
print(f"仍存活 {len(t1)} 个 → SIGKILL")
for pid in t1:
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
time.sleep(8)
t2 = targets()
print(f"第二轮后仍存活 {len(t2)} 个" + (f": {t2[:10]}" if t2 else " ✓ 清空"))
# 复核 PGS 臂仍在
alive_pgs = sum(1 for p in os.listdir("/proc") if p.isdigit() and PGS in read(int(p), "cmdline"))
print(f"★ PGS 臂进程仍在: {alive_pgs} 个(应 >0)")
