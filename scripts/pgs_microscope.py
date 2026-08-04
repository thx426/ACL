#!/usr/bin/env python3
"""PGS 显微镜 v2 —— 商业级逐 token 可视化(静态 SPA,file:// 直开,任意轨迹可点)。

    python scripts/pgs_microscope.py --run-dir <run> --tokenizer <hf-path>
           [--steps 1,2] [--last N] [--out-dir <run>/microscope]

产出:
  microscope/index.html        应用外壳(内联全部 CSS/JS,零外部依赖)
  microscope/data/run.js       run 级聚合(step 序列 + 机制曲线)
  microscope/data/<step>.js    每 step 的全量逐 token 数据(128 条轨迹全部可看)

与 pgs_viz.py 的关系:数据源相同(rollouts dump);本工具是交互升级版
(用户 2026-08-04 要求商业化精细度)。渲染端零 tokenizer 依赖——文本在
导出期预解码;普通连续段整段 decode(无损),特殊位置单 token decode
(英文语料下的可接受近似,与 pgs_viz 同口径)。
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

APP = Path(__file__).with_name("pgs_microscope_app.html")


def load_tok(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


class Dec:
    """单 token 解码缓存(vocab 重复率高,缓存后近似零成本)。"""

    def __init__(self, tok):
        self.tok = tok
        self.c: dict[int, str] = {}

    def one(self, tid: int) -> str:
        if tid not in self.c:
            self.c[tid] = self.tok.decode([tid])
        return self.c[tid]

    def seq(self, ids: list[int]) -> str:
        return self.tok.decode(ids) if ids else ""


def r(x, nd=4):
    return round(float(x), nd)


def load_docs(path: str | None) -> dict[str, str]:
    """doc_table jsonl(task_id→特权文档);缺文件返回空表(base 臂无文档)。"""
    docs: dict[str, str] = {}
    if path and Path(path).is_file():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                j = json.loads(line)
                docs[j["task_id"]] = j.get("doc", "")
    return docs


def _goal_of(obs: str) -> str:
    for ln in (obs or "").splitlines():
        if "your task is to" in ln.lower():
            return ln.strip()
    return ""


def build_traj(d: dict, dec: Dec, docs: dict[str, str] | None = None) -> dict:
    ids = d["response_token_ids"]
    s = d.get("steering") or {}
    theta = s.get("steer_theta_d", 0.0)

    recs = []
    pos2rec: dict[int, int] = {}
    n_rec = len(s.get("steer_positions", []))
    for i in range(n_rec):
        p = s["steer_positions"][i]
        dr, fi = s["steer_draft_token"][i], s["steer_final_token"][i]
        rec = dict(
            p=p,
            d=r(s["steer_d"][i]),
            eta=r(s["steer_eta"][i], 3),
            delta=r(s["steer_delta"][i], 3),
            lp=r(s["steer_log_pi"][i], 3),
            lb=r(s["steer_log_b"][i], 3),
            prop=r(s["steer_propensity"][i], 3),
            rej=int(s["steer_rejected"][i]),
            probe=int(s["steer_probe"][i]),
            cap=int(s["steer_cap_active"][i]),
            open=int(s["steer_base_open"][i]),
            after=r(s["steer_budget_after"][i], 2),
            draft=dec.one(dr) if dr >= 0 else "",
            final=dec.one(fi) if fi >= 0 else "",
            pi=[[dec.one(t), r(l, 2)] for t, l in s["steer_pi_top16"][i]],
            pt=[[dec.one(t), r(l, 2)] for t, l in s["steer_pt_top16"][i]],
        )
        recs.append(rec)
        pos2rec[p] = i

    cands = [[int(p), r(v)] for p, v in zip(s.get("steer_cand_positions", []), s.get("steer_d_all", []))]
    pos2cand = {p: i for i, (p, _) in enumerate(cands)}

    # 预算耗尽点(近似):最后一条记录后若余额不足一次整额扣费(delta≈2ηc)
    exhausted = -1
    if recs:
        last = recs[-1]
        full = max((rc["delta"] for rc in recs), default=0) or 2.5
        if last["after"] < full:
            exhausted = last["p"] + 1

    spans = d["turn_token_spans"]
    turns = []
    prev = 0
    for ti, (a, b) in enumerate(spans):
        obs = dec.seq(ids[prev:a]) if a > prev else ""
        segs: list[list] = []
        buf: list[int] = []

        def flush():
            if buf:
                segs.append([dec.seq(buf), 0, -1])
                buf.clear()

        i = a
        while i < b:
            if i in pos2rec:
                flush()
                rc = recs[pos2rec[i]]
                cls = 3 if rc["rej"] else (2 if rc["open"] and rc["eta"] > 0 else 4)
                segs.append([dec.one(ids[i]), cls, pos2rec[i]])
            elif i in pos2cand:
                flush()
                segs.append([dec.one(ids[i]), 1, pos2cand[i]])
            else:
                buf.append(ids[i])
            i += 1
        flush()
        n_iv = sum(1 for rc in recs if a <= rc["p"] < b)
        turns.append(dict(i=ti, a=a, b=b, iv=n_iv, obs=obs, segs=segs))
        prev = b
    tail = dec.seq(ids[prev:]) if prev < len(ids) else ""

    task = d.get("task_id", "?")
    traj_meta = d.get("trajectory") or {}
    iobs = traj_meta.get("initial_obs", "") or ""
    return dict(
        task=task,
        short=task.split("/")[-2] if "/" in task else task,
        won=bool(d.get("won")),
        outcome=d.get("outcome", ""),
        split=d.get("split", ""),
        goal=_goal_of(iobs),
        iobs=iobs,
        ghash=traj_meta.get("guidance_hash", ""),
        doc=(docs or {}).get(task, ""),
        len=d.get("response_token_len", len(ids)),
        env=d.get("env_steps", 0),
        theta=r(theta),
        exhausted=exhausted,
        m=dict(
            cand=s.get("steer_n_candidates", 0),
            open=s.get("steer_n_base_open", 0),
            tilt=s.get("steer_count", 0),
            rej=s.get("steer_reject_count", 0),
            spent=r(s.get("steer_budget_spent", 0), 2),
            bmax=20.0,
            bypass=s.get("steer_n_bypassed", 0),
            miss=s.get("steer_n_pt_window_miss", 0),
            probe=s.get("steer_n_probe_flipped", 0),
        ),
        turns=turns,
        tail=tail,
        recs=recs,
        cands=cands,
    )


def step_files(run: Path, steps_arg: str | None, last: int | None) -> list[Path]:
    all_steps = sorted(p for p in (run / "rollouts").iterdir() if p.name.startswith("step_") and not p.name.endswith("_val"))
    if steps_arg:
        want = {int(x) for x in steps_arg.split(",")}
        all_steps = [p for p in all_steps if int(p.name.split("_")[1]) in want]
    if last:
        all_steps = all_steps[-last:]
    return all_steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--doc-table", default=None, help="task_id→特权文档 jsonl(s1 臂)")
    a = ap.parse_args()

    run = Path(a.run_dir)
    out = Path(a.out_dir) if a.out_dir else run / "microscope"
    (out / "data").mkdir(parents=True, exist_ok=True)
    dec = Dec(load_tok(a.tokenizer))
    docs = load_docs(a.doc_table)

    series = []
    for sd in step_files(run, a.steps, a.last):
        files = sorted(glob.glob(str(sd / "*.json")))
        trajs = []
        agg = dict(won=0, cand=0, open=0, tilt=0.0, rej=0.0, spent=0.0, theta=0.0, bypass=0.0)
        for f in files:
            d = json.load(open(f))
            t = build_traj(d, dec, docs)
            trajs.append(t)
            agg["won"] += t["won"]
            agg["cand"] += t["m"]["cand"]
            agg["open"] += t["m"]["open"]
            agg["tilt"] += t["m"]["tilt"]
            agg["rej"] += t["m"]["rej"]
            agg["spent"] += t["m"]["spent"]
            agg["theta"] += t["theta"]
            agg["bypass"] += t["m"]["bypass"]
        n = max(len(trajs), 1)
        sid = int(sd.name.split("_")[1])
        series.append(
            dict(
                id=sid,
                dir=sd.name,
                n=len(trajs),
                won=r(agg["won"] / n, 4),
                gate=r(agg["open"] / max(agg["cand"], 1), 4),
                tilt=r(agg["tilt"] / n, 2),
                rej=r(agg["rej"] / n, 2),
                rejrate=r(agg["rej"] / max(agg["open"], 1), 4),
                spent=r(agg["spent"] / n, 2),
                theta=r(agg["theta"] / n, 4),
                bypass=r(agg["bypass"] / n, 1),
            )
        )
        payload = json.dumps({"trajs": trajs}, ensure_ascii=False, separators=(",", ":"))
        (out / "data" / f"{sd.name}.js").write_text(
            f'window.PGS_STEP_DATA=window.PGS_STEP_DATA||{{}};window.PGS_STEP_DATA["{sd.name}"]={payload};',
            encoding="utf-8",
        )
        print(f"[microscope] {sd.name}: {len(trajs)} trajs -> data/{sd.name}.js")

    runjs = json.dumps(dict(name=run.name, steps=series), ensure_ascii=False, separators=(",", ":"))
    (out / "data" / "run.js").write_text(f"window.PGS_RUN={runjs};", encoding="utf-8")
    shutil.copyfile(APP, out / "index.html")
    print(f"[microscope] app: {out / 'index.html'}  (steps={len(series)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
