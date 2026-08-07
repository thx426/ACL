#!/usr/bin/env python3
"""从 verl 的 step 指标行提取趋势表(日志已去 Ray 前缀写到 /tmp/s3_steps.txt)。"""
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/s3_steps.txt"

rows = []
for line in open(PATH):
    d = {}
    for kv in line.strip().split(" - "):
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        v = v.replace("np.float64(", "").replace(")", "")
        try:
            d[k] = float(re.sub(r"[^0-9eE.+-]", "", v))
        except ValueError:
            pass
    if "step" in d:
        rows.append(d)

KEYS = [
    ("won", "rollout_extra/won/mean"),
    ("熵", "actor/entropy"),
    ("grad", "actor/grad_norm"),
    ("env步", "rollout_extra/env_steps/mean"),
    ("invalid", "rollout_extra/invalid_format/mean"),
    ("trunc", "rollout_extra/truncated/mean"),
    ("respLen", "response_length/mean"),
    ("IS_off1", "actor/rollout_is_weight_off1_frac"),
    ("D均", "rollout_extra/steer_d_mean/mean"),
    ("拒绝", "rollout_extra/steer_reject_count/mean"),
]

hdr = "step " + " ".join(f"{n:>8}" for n, _ in KEYS)
print(hdr)
for r in rows:
    s = int(r["step"])
    if s < 20 and s % 10:
        continue
    if 20 <= s < 40 and s % 5:
        continue
    cells = " ".join(f"{r.get(k, float('nan')):>8.3f}" for _, k in KEYS)
    print(f"{s:>4} {cells}")
