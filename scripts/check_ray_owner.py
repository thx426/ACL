#!/usr/bin/env python3
"""分清 CVD=0,1,2,3 的进程归属:SEED 臂(15:06 发车)还是旧残留。
判据:进程启动时间(/proc/<pid>/stat 的 starttime)+ ray tmpdir + 祖先链。"""
import os, time

HZ = os.sysconf("SC_CLK_TCK")
with open("/proc/uptime") as f:
    uptime = float(f.read().split()[0])
boot = time.time() - uptime


def info(pid):
    try:
        st = open(f"/proc/{pid}/stat").read()
        rp = st.rindex(")")
        fields = st[rp + 2:].split()
        ppid = int(fields[1])
        starttime = int(fields[19]) / HZ
        return ppid, boot + starttime
    except Exception:
        return None, None


def read(pid, what):
    try:
        raw = open(f"/proc/{pid}/{what}", "rb").read()
        sep = b" " if what == "cmdline" else b"\n"
        return raw.replace(b"\0", sep).decode("utf8", "replace")
    except Exception:
        return ""


def env_of(pid, key):
    for ln in read(pid, "environ").split("\n"):
        if ln.startswith(key + "="):
            return ln.split("=", 1)[1]
    return ""


groups = {}
for p in os.listdir("/proc"):
    if not p.isdigit():
        continue
    pid = int(p)
    c = read(pid, "cmdline")
    if not c or "check_ray_owner" in c:
        continue
    cvd = env_of(pid, "CUDA_VISIBLE_DEVICES")
    if cvd != "0,1,2,3":
        continue
    ppid, start = info(pid)
    if start is None:
        continue
    tmp = env_of(pid, "RAY_TMPDIR") or env_of(pid, "TMPDIR")
    key = (time.strftime("%m-%d %H:%M", time.localtime(start))[:11], tmp or "-")
    groups.setdefault(key, []).append((pid, c[:60]))

print("CVD=0,1,2,3 的进程,按【启动时刻(分钟) × RAY_TMPDIR】分组:")
print("  (SEED 臂发车 = 08-16 15:06)\n")
for (t, tmp), items in sorted(groups.items()):
    print(f"  启动 {t}  RAY_TMPDIR={tmp:<26} {len(items):>3} 个   例: {items[0][1]}")
