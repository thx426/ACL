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
import math
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

APP = Path(__file__).with_name("pgs_microscope_app.html")
DEFAULT_BMAX = 20.0
MAX_DOC_BYTES = 2 * 1024 * 1024
MAX_META_BYTES = 1024 * 1024
_AUDIT_FIELDS = (
    "global_step",
    "policy_version",
    "experiments_used",
    "episodes_consumed",
    "turns_used",
    "finalized",
    "no_finalize",
    "doc_words",
    "error",
    "session_error",
    "gt",
    "gt_injected",
)
_STEERING_LINE = re.compile(r"^(?P<indent>[ \t]*)steering\s*:\s*(?:#.*)?$")
_TRAIN_LINE = re.compile(r"^(?P<indent>[ \t]*)train\s*:\s*(?:#.*)?$")
# v4 自体研究员:λ_r 等参数在 research_rl 段,researcher 会话上限在 research 段。
_RESRL_LINE = re.compile(r"^(?P<indent>[ \t]*)research_rl\s*:\s*(?:#.*)?$")
_RES_LINE = re.compile(r"^(?P<indent>[ \t]*)research\s*:\s*(?:#.*)?$")
_BMAX_LINE = re.compile(
    r"^(?P<indent>[ \t]*)b_max\s*:\s*(?P<value>[^#\s]+)\s*(?:#.*)?$"
)
_GATE_MODE_LINE = re.compile(
    r"^(?P<indent>[ \t]*)gate_mode\s*:\s*(?P<value>[A-Za-z_]+)\s*(?:#.*)?$"
)
_TRAIN_BATCH_LINE = re.compile(
    r"^(?P<indent>[ \t]*)train_batch_size\s*:\s*(?P<value>[^#\s]+)\s*(?:#.*)?$"
)
_ROLLOUT_N_LINE = re.compile(
    r"^(?P<indent>[ \t]*)rollout_n\s*:\s*(?P<value>[^#\s]+)\s*(?:#.*)?$"
)
_TOP_LEVEL_LINE = re.compile(r"^[^ \t#][^:]*\s*:")


def _kv_line(key: str, *, word: bool = False) -> re.Pattern:
    """段内 `key: value` 行的通用正则(v3.3 新键统一走这里,勿再手写常量)。"""
    val = r"[A-Za-z_]+" if word else r"[^#\s]+"
    return re.compile(rf"^(?P<indent>[ \t]*){key}\s*:\s*(?P<value>{val})\s*(?:#.*)?$")


#: v3.3 M6 fork-token 双带(MENTOR Fig.3 度量的移植;occurrence = 该词出现
#: 于轨迹 thought 的轨迹占比)。初始词表为通用英文推理分叉词;数据驱动重标
#: 用 --fork-lexicon 传 json {"redundancy": [...], "essence": [...]} 覆盖。
FORK_LEXICON = {
    "redundancy": ("wait", "however", "but", "alternatively", "actually",
                   "let me re", "hmm"),
    "essence": ("check", "verify", "confirm", "double-check"),
}
_FORK_PATTERNS: dict[str, re.Pattern] = {}


def set_fork_lexicon(lex: dict) -> None:
    """安装词表并预编译**词边界**匹配(审阅 L27:裸子串在 agentic 语境
    有系统性假阳——'wait' 命中任务字面义等待、'check'⊂'checkout')。
    多词短语按字面嵌入,首尾仍受 \b 约束。"""
    FORK_LEXICON.update({k: tuple(str(w).lower() for w in v) for k, v in lex.items()})
    for band, words in FORK_LEXICON.items():
        _FORK_PATTERNS[band] = re.compile(
            r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        )


set_fork_lexicon({})


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
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    j = json.loads(line)
                    docs[j["task_id"]] = j.get("doc", "")
    return docs


def _parse_yaml_float(raw: str) -> float | None:
    """YAML 的 `.inf` 是合法无穷(v3.2 maxopen 的 b_max),但 Python float() 不认
    这个写法 —— 直接 float(".inf") 抛 ValueError,会被静默吞成“配置缺失”并回退
    到 20,于是无上限预算被显示成“7263/20 接近上限”。"""
    s = raw.strip()
    if s.lstrip("+").lower() in (".inf", "inf"):
        return math.inf
    if s.lstrip("-").lower() in (".inf", "inf"):
        return -math.inf
    try:
        return float(s)
    except ValueError:
        return None


def load_run_config(run: Path) -> dict:
    """读取显微镜所需的少量实收配置，避免给实时服务增加 YAML 依赖。"""
    path = run / "effective_config.yaml"
    out = {
        "bmax": DEFAULT_BMAX,
        "bmaxUnbounded": False,
        # v3.2:always_on 档下开门率恒 1、θ_D 无语义。前端据此切换判读口径,
        # 否则 100% 开门会被按自适应档的 5-25% 绿带标成"严重超标"。
        "gateMode": "",
        "bmaxSource": "fallback",
        "bmaxWarning": "",
        "trainBatchSize": 0,
        "rolloutN": 0,
        "expectedTrainTrajs": 0,
        # v3.3:α 调度(重算曲线用)、熵门参考线、响应预算(lenmax 余量红条)
        "advMode": "",
        "steeredAdvWeight": 1.0,
        "steeredAdvDecay": 0,
        "entropyFloor": 0.0,
        "entropyThetaInit": 0.0,
        "responseLength": 0,
        # v4 researcher 环(缺省 0 = 该 run 无 research 行,前端据此隐藏面板)
        "researchRlWeight": 0.0,
        "researchN": 0,
        "contSamples": 0,
        "nExp": 0,
        "researchMaxTurns": 0,
    }
    if not path.is_file():
        out["bmaxWarning"] = "effective_config.yaml 不存在，预算上限回退为 20"
        return out

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        out["bmaxWarning"] = f"effective_config.yaml 读取失败: {type(exc).__name__}"
        return out

    sections: dict[str, tuple[int, dict[str, re.Pattern]]] = {
        "train": (-1, {"trainBatchSize": _TRAIN_BATCH_LINE, "rolloutN": _ROLLOUT_N_LINE,
                       "responseLength": _kv_line("response_length")}),
        "steering": (-1, {"bmax": _BMAX_LINE, "gateMode": _GATE_MODE_LINE,
                          "advMode": _kv_line("adv_mode", word=True),
                          "steeredAdvWeight": _kv_line("steered_adv_weight"),
                          "steeredAdvDecay": _kv_line("steered_adv_decay_steps"),
                          "entropyFloor": _kv_line("entropy_floor"),
                          "entropyThetaInit": _kv_line("entropy_theta_init")}),
        "research_rl": (-1, {"researchRlWeight": _kv_line("research_rl_weight"),
                             "researchN": _kv_line("research_n"),
                             "contSamples": _kv_line("continuation_samples")}),
        "research": (-1, {"nExp": _kv_line("n_exp"),
                          "researchMaxTurns": _kv_line("max_turns")}),
    }
    active = ""
    for line in lines:
        top = None
        for name, regex in (("train", _TRAIN_LINE), ("steering", _STEERING_LINE),
                            ("research_rl", _RESRL_LINE), ("research", _RES_LINE)):
            match = regex.match(line)
            if match:
                top = name
                sections[name] = (len(match.group("indent").expandtabs(8)), sections[name][1])
                break
        if top:
            active = top
            continue
        if not active:
            continue
        section_indent, patterns = sections[active]
        if _TOP_LEVEL_LINE.match(line) and len(line) - len(line.lstrip()) <= section_indent:
            active = ""
            continue
        for key, regex in patterns.items():
            match = regex.match(line)
            if not match:
                continue
            indent = len(match.group("indent").expandtabs(8))
            if indent <= section_indent:
                continue
            if key in ("gateMode", "advMode"):
                out[key] = match.group("value").strip()
                continue
            value = _parse_yaml_float(match.group("value"))
            if value is None:
                continue
            if key in ("researchN", "contSamples", "nExp", "researchMaxTurns"):
                if value.is_integer() and value >= 0:
                    out[key] = int(value)
                continue
            if key == "researchRlWeight":
                if math.isfinite(value) and value >= 0:
                    out[key] = value
                continue
            if key in ("responseLength", "steeredAdvDecay"):
                if value.is_integer() and value >= 0:
                    out[key] = int(value)
                continue
            if key in ("steeredAdvWeight", "entropyFloor", "entropyThetaInit"):
                if math.isfinite(value) and value >= 0:
                    out[key] = value
                continue
            if key == "bmax":
                if value == math.inf:
                    # v3.2 maxopen 档 b_max: .inf。JSON 无法表示 Infinity
                    # (json.dumps 会吐出非法的 `Infinity`,浏览器 JSON.parse 直接抛),
                    # 故用 null + 显式标志传达“无上限”,前端据此显示 ∞ 且不报预算告警。
                    out.update(
                        bmax=None,
                        bmaxUnbounded=True,
                        bmaxSource="effective_config.yaml:steering.b_max",
                    )
                elif math.isfinite(value) and value > 0:
                    out.update(bmax=value, bmaxSource="effective_config.yaml:steering.b_max")
            elif key in ("trainBatchSize", "rolloutN") and value.is_integer() and value > 0:
                out[key] = int(value)
    if out["trainBatchSize"] and out["rolloutN"]:
        out["expectedTrainTrajs"] = out["trainBatchSize"] * out["rolloutN"]
    if out["bmaxSource"] == "fallback":
        out["bmaxWarning"] = "steering.b_max 缺失或非法，预算上限回退为 20"
    return out


def _empty_privilege(state: str, *, source: str = "none", error: str = "") -> dict:
    return {
        "state": state,
        "source": source,
        "doc": "",
        "archive": "",
        "meta": {},
        "error": error,
        "usedByTeacher": False,
    }


def _safe_archive_dir(run: Path, archive: str) -> tuple[Path | None, str]:
    """把 rollout 的 archive 引用限制在当前 run/research；拒绝路径和 symlink 逃逸。"""
    if not isinstance(archive, str) or not archive.strip():
        return None, "research_archive 为空"
    if "\\" in archive:
        return None, "research_archive 含反斜杠"

    rel = PurePosixPath(archive)
    if rel.is_absolute() or not rel.parts or rel.parts[0] != "research":
        return None, "research_archive 必须是 research/ 下的相对路径"
    if any(part in ("", ".", "..") for part in rel.parts):
        return None, "research_archive 含不安全路径段"

    try:
        run_root = run.resolve(strict=False)
        research_link = run / "research"
        research_root = research_link.resolve(strict=False)
        research_root.relative_to(run_root)
        candidate = run.joinpath(*rel.parts).resolve(strict=False)
        candidate.relative_to(research_root)
    except (OSError, RuntimeError, ValueError):
        return None, "research_archive 或 research 根解析后逃出当前 run"
    return candidate, ""


def _safe_archive_member(run: Path, archive_dir: Path, name: str) -> tuple[Path | None, str]:
    try:
        research_root = (run / "research").resolve(strict=False)
        member = (archive_dir / name).resolve(strict=False)
        member.relative_to(research_root)
        if member.parent != archive_dir:
            return None, f"{name} symlink 指向当前 archive 外部"
    except (OSError, RuntimeError, ValueError):
        return None, f"{name} symlink 逃出当前 run/research"
    return member, ""


def _audit_meta(meta: object) -> dict:
    if not isinstance(meta, dict):
        raise ValueError("meta.json 顶层必须是对象")
    return {key: meta.get(key) for key in _AUDIT_FIELDS if key in meta}


def resolve_privilege(d: dict, run: Path | None, docs: dict[str, str] | None = None) -> dict:
    """解析单条轨迹实际使用的教师特权上下文，不把“无静态表”误判为 base。"""
    if d.get("validate"):
        return _empty_privilege("validation_bare")

    # 分臂采样(机制 v3.1):steer_arm=0 的副本是纯 π 的 bare 臂。它与同组
    # steered 副本共享同一条 per-task research 记录(archive 引用逐字相同),
    # 但教师从未介入这条轨迹(steer_count=0、guidance_chars=0)。只看 research
    # 字段会把两臂都标成“有特权文档”—— 正是分臂要修的读数污染在可视化层重演。
    # 文档仍加载出来供对照(“同组 steered 看到了什么、这条 bare 没看到”),
    # 但 state/usedByTeacher 必须如实反映本副本未被干预。
    arm = (d.get("steering") or {}).get("steer_arm")
    arm = int(arm) if arm is not None else None

    research = d.get("research") or {}
    if not isinstance(research, dict):
        return _empty_privilege("archive_error", source="dynamic_researcher", error="research 字段不是对象")

    archive = research.get("research_archive") or ""
    has_research_signal = bool(research) or bool(archive)
    if has_research_signal:
        if not archive:
            reason = "researcher 降级：未产生可用 archive"
            if research.get("research_error"):
                reason = "researcher 报错且未产生可用 archive"
            return _empty_privilege("researcher_degraded", source="dynamic_researcher", error=reason)
        if run is None:
            return _empty_privilege("archive_error", source="dynamic_researcher", error="显微镜未配置 run 目录")

        archive_dir, error = _safe_archive_dir(run, archive)
        if archive_dir is None:
            out = _empty_privilege("archive_error", source="dynamic_researcher", error=error)
            out["archive"] = archive
            return out

        out = _empty_privilege("dynamic_researcher", source="dynamic_researcher")
        out["archive"] = archive
        if not archive_dir.is_dir():
            out["state"] = "archive_missing"
            out["error"] = "archive 目录不存在"
            return out
        doc_path, doc_path_error = _safe_archive_member(run, archive_dir, "doc.md")
        meta_path, meta_path_error = _safe_archive_member(run, archive_dir, "meta.json")
        errors: list[str] = [item for item in (doc_path_error, meta_path_error) if item]
        try:
            if doc_path is None:
                pass
            elif not doc_path.is_file():
                errors.append("doc.md 不存在")
            elif doc_path.stat().st_size > MAX_DOC_BYTES:
                errors.append(f"doc.md 超过 {MAX_DOC_BYTES} bytes 安全上限")
            else:
                out["doc"] = doc_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"doc.md 读取失败: {type(exc).__name__}: {exc}")

        try:
            if meta_path is None:
                pass
            elif not meta_path.is_file():
                errors.append("meta.json 不存在")
            elif meta_path.stat().st_size > MAX_META_BYTES:
                errors.append(f"meta.json 超过 {MAX_META_BYTES} bytes 安全上限")
            else:
                out["meta"] = _audit_meta(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"meta.json 读取失败: {type(exc).__name__}: {exc}")

        meta_error = out["meta"].get("error") or out["meta"].get("session_error")
        rollout_degraded = bool(
            research.get("research_error") or research.get("research_doc_missing")
        )
        producer_degraded = bool(rollout_degraded or meta_error)
        if producer_degraded:
            out["state"] = "researcher_degraded"
        elif not out["doc"]:
            out["state"] = "archive_missing"
        if errors:
            out["state"] = "archive_missing" if all("不存在" in item for item in errors) else "archive_error"
            out["error"] = "; ".join(errors)
        elif meta_error:
            out["error"] = str(meta_error)
        out["usedByTeacher"] = bool(out["doc"]) and not producer_degraded
        out["arm"] = arm
        if arm == 0:
            # bare 臂:producer 侧的降级/archive 错误对本副本无影响(它没用文档),
            # 但必须明确它不是 steered ——否则整个 run 的一半会被读成有教师。
            out["state"] = "arm_bare"
            out["usedByTeacher"] = False
        return out

    task = d.get("task_id", "?")
    if docs:
        if task in docs and docs[task]:
            return {
                "state": "static_gt",
                "source": "static_gt",
                "doc": docs[task],
                "archive": "",
                "meta": {},
                "error": "",
                "usedByTeacher": True,
            }
        return _empty_privilege("static_missing", source="static_gt", error="该任务在静态 doc 表中无文档")
    return _empty_privilege("bare")


def _goal_of(obs: str) -> str:
    for ln in (obs or "").splitlines():
        if "your task is to" in ln.lower():
            return ln.strip()
    return ""


def build_rec_dists(d: dict, dec: Dec, ref: int) -> dict:
    """单条倾斜记录的 π/教师 top16 分布(轨迹页抽屉按需取)。

    ref = 记录在 steer_positions 中的下标,与 build_traj 的 recs 序一致。
    """
    s = d.get("steering") or {}
    n = len(s.get("steer_positions", []))
    if not (0 <= ref < n):
        return {"error": "rec ref out of range"}
    return {
        "ref": ref,
        "pi": [[dec.one(t), r(l, 2)] for t, l in s["steer_pi_top16"][ref]],
        "pt": [[dec.one(t), r(l, 2)] for t, l in s["steer_pt_top16"][ref]],
    }


def build_traj(
    d: dict,
    dec: Dec,
    docs: dict[str, str] | None = None,
    *,
    run: Path | None = None,
    bmax: float = DEFAULT_BMAX,
    with_dists: bool = True,
) -> dict:
    """with_dists=False 省略每条记录的 π/教师 top16 分布。

    v3.2 起 top16 对所有倾斜位全量携带,maxopen 下单轨迹 ~4000 条记录,
    两个分布数组占单条记录 582/620 B —— 整个 payload 的 95%。轨迹页首屏
    只需要 token 序列与干预标记,分布明细仅在点开某个位置的抽屉时才看,
    故实时后端默认走精简档,分布由 /api/rec 按需取(静态导出仍带全量)。
    """
    ids = d["response_token_ids"]
    s = d.get("steering") or {}
    theta = s.get("steer_theta_d", 0.0)

    recs = []
    pos2rec: dict[int, int] = {}
    n_rec = len(s.get("steer_positions", []))
    intensities = s.get("steer_intensity") or []
    entropies = s.get("steer_entropy") or []
    # 吸收进度中间量(v3.3 M6:拒绝位上 π 给 draft/final 的 logp 差 =
    # 「最后一公里」读数;final 是 π-top1(rank 0-based == 0)= 教师只在降噪)
    ab_dlogp: list[float] = []
    ab_top1 = 0
    ab_miss16 = 0
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
        )
        if i < len(intensities):
            rec["w"] = r(intensities[i], 3)
        if i < len(entropies):
            rec["h"] = r(entropies[i], 3)
        top = s["steer_pi_top16"][i] if i < len(s.get("steer_pi_top16", [])) else []
        if rec["rej"] and top:
            lut = {int(t): float(l) for t, l in top}
            if dr in lut and fi in lut:
                ab_dlogp.append(lut[dr] - lut[fi])
            else:
                # 删失计数(审阅 L26):远 rank(>top16)拒绝位算不出 Δlogπ,
                # 静默剔除会让曲线在"硬知识增多"时反向移动——份额必须可见。
                ab_miss16 += 1
            if int(top[0][0]) == fi:
                ab_top1 += 1
        if with_dists:
            rec["pi"] = [[dec.one(t), r(l, 2)] for t, l in top]
            rec["pt"] = [[dec.one(t), r(l, 2)] for t, l in s["steer_pt_top16"][i]]
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
    privilege = resolve_privilege(d, run, docs)
    return dict(
        task=task,
        short=task.split("/")[-2] if "/" in task else task,
        won=bool(d.get("won")),
        outcome=d.get("outcome", ""),
        split=d.get("split", ""),
        goal=_goal_of(iobs),
        iobs=iobs,
        ghash=traj_meta.get("guidance_hash", ""),
        doc=privilege["doc"],
        privilege=privilege,
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
            bmax=None if bmax is None or bmax == math.inf else r(bmax, 4),
            bypass=s.get("steer_n_bypassed", 0),
            miss=s.get("steer_n_pt_window_miss", 0),
            probe=s.get("steer_n_probe_flipped", 0),
            gbypass=s.get("steer_n_gate_bypassed", 0),
        ),
        arm=s.get("steer_arm"),
        # v4 行类权威判据(见 _meta_of 同注);research 行的观察量在此随行。
        role=d.get("sample_role"),
        research=d.get("research") or {},
        conts=d.get("continuations") or [],
        enttheta=r(s.get("steer_entropy_theta", 0.0), 3),
        entmean=r(s.get("steer_entropy_mean", 0.0), 3),
        wmean=r(s.get("steer_intensity_mean", 0.0), 3),
        absorb=dict(
            dlogp=[r(x, 3) for x in ab_dlogp],
            top1=ab_top1,
            miss16=ab_miss16,
            nrej=int(s.get("steer_reject_count", 0)),
        ),
        fork=fork_hits(traj_meta),
        turns=turns,
        tail=tail,
        recs=recs,
        cands=cands,
    )


def fork_hits(traj_meta: dict) -> dict:
    """fork-token 双带命中(v3.3 M6;MENTOR Fig.3 occurrence 口径:该词是否
    出现在轨迹的 thought 文本里;词边界匹配防 agentic 语境假阳)。"""
    text = " ".join(
        (st.get("thought") or "") for st in traj_meta.get("steps") or []
    ).lower()
    return {
        band: int(bool(pat.search(text)))
        for band, pat in _FORK_PATTERNS.items()
    }


def pgs_alpha(step_id: int, config: dict) -> float | None:
    """α_t 重算(与 verl 侧 pgs_steered_alpha 同公式;dump 不带 α,由实收
    配置确定性重建)。非 asym 档返回 None(前端不画)。"""
    if config.get("advMode") != "asym":
        return None
    a0 = float(config.get("steeredAdvWeight", 1.0))
    decay = int(config.get("steeredAdvDecay", 0))
    if decay <= 0:
        return a0
    if step_id >= decay:
        return 0.0
    return a0 * 0.5 * (1.0 + math.cos(math.pi * step_id / decay))


def _med(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    m = len(ys) // 2
    return float(ys[m]) if len(ys) % 2 else (float(ys[m - 1]) + float(ys[m])) / 2.0


def build_research_entry(res_rows: list[dict], steered: list[dict],
                         config: dict) -> dict | None:
    """researcher 环的 step 级观察量(v4 自体研究员)。

    组织顺序即因果链 —— 上游不通,下游读数无意义,排障时从①往下看:

      ① 管道:研究行有没有真的产生梯度。overlong ⇒ mask 全 0(selfres.py
         §研究行),行还在、reward 还进组内 baseline,但对 policy 零贡献。
         `lambdaEff = λ_r × validGradRate` 才是 researcher 在 loss 里的
         **实际**权重 —— 名义 λ_r 与它可以差一个数量级(162 实测 0.018
         vs 名义 0.1),只看配置里的 λ_r 会严重高估这一环的话语权。
      ② 成绩:r_raw = 前缀续写胜率 = "这份文档能不能让人从教师前缀赢"。
         `rZeroRate` 是崩塌前兆判据(恒零 ⇒ 研究信号消失);`rOneRate`
         贴顶则是任务侧饱和,此时 r 已无分辨力,别再据此调 λ_r。
      ③ 过程:研究员实际做了多少实验/轮次。`expMed` 随训练上移 = 研究
         行为在变充分;塌回 1-2 则是会话在退化(与②的零率互证)。
      ④ 产物与闭环:每任务 8 条研究行只有 argmax 那条的 doc 会被 steered
         臂真正消费,故 `argmaxRMean` 才是"实际投产的文档质量",与全体
         均值的差 `selGain` = 择优带来的增益。它与 wonSteered 并列看,
         就是自进化闭环有没有闭上的直读证据。

    `steered` 仅用于闭环对照(不参与任何分母)。无 research 行时返回 None。
    """
    if not res_rows:
        return None
    n = len(res_rows)
    def f(row: dict, k: str, default=0):
        return (row.get("research") or {}).get(k, default)
    def rate(k: str) -> float:
        return r(sum(1 for x in res_rows if f(x, k)) / n, 4)

    rs = [float(f(x, "research_r_raw", 0.0)) for x in res_rows]
    overlong = [bool(f(x, "research_overlong")) for x in res_rows]
    eff_rows = [x for x, o in zip(res_rows, overlong) if not o]
    eff_rate = len(eff_rows) / n
    lam = float(config.get("researchRlWeight") or 0.0)

    # 每任务 argmax:实际被 steered 臂消费的那份文档(coord 择优)。
    # 增益必须在**任务内**比较后再平均 —— argmax 天然按任务加权,若拿它减
    # 全行均值 rMean(按行加权),各任务行数不等时(partial step 的常态)会算出
    # 负增益,而 max ≥ mean 决定了真实增益不可能为负。故基线取"每任务均值
    # 的均值",与 argmax 同权。rMean 仍按行算,那是成绩水平线、不是增益基线。
    by_task: dict[str, list[float]] = {}
    for x, rv in zip(res_rows, rs):
        by_task.setdefault(x.get("task", "?"), []).append(rv)
    argmax_rs = [max(v) for v in by_task.values() if v]
    taskmean_rs = [sum(v) / len(v) for v in by_task.values() if v]

    conts = [c for x in res_rows for c in (x.get("conts") or [])]
    cn = max(len(conts), 1)
    won_steered = (sum(t["won"] for t in steered) / len(steered)) if steered else None

    return dict(
        n=n, nTasks=len(by_task),
        # ① 管道
        validGradRate=r(eff_rate, 4),
        lambdaNominal=r(lam, 4),
        lambdaEff=r(lam * eff_rate, 5),
        overlong=rate("research_overlong"),
        noFinalize=rate("research_no_finalize"),
        docOverflow=rate("research_doc_overflow"),
        degraded=rate("research_degraded"),
        error=rate("research_error"),
        replayDrift=rate("research_replay_drift"),
        warmup=rate("research_warmup_active"),
        # ② 成绩
        rMean=r(sum(rs) / n, 4),
        rMed=r(_med(rs), 4),
        rZeroRate=r(sum(1 for v in rs if v == 0.0) / n, 4),
        rOneRate=r(sum(1 for v in rs if v >= 1.0) / n, 4),
        rMeanEff=(r(sum(float(f(x, "research_r_raw", 0.0)) for x in eff_rows)
                    / len(eff_rows), 4) if eff_rows else None),
        # ③ 过程
        expMed=_med([float(f(x, "research_experiments")) for x in res_rows]),
        expMean=r(sum(float(f(x, "research_experiments")) for x in res_rows) / n, 2),
        turnsMed=_med([float(f(x, "research_turns")) for x in res_rows]),
        episodesMed=_med([float(f(x, "research_episodes")) for x in res_rows]),
        # ④ 产物与闭环
        docWordsMed=_med([float(f(x, "research_doc_words")) for x in res_rows]),
        docWordsMax=max((int(f(x, "research_doc_words")) for x in res_rows), default=0),
        contWonRate=r(sum(1 for c in conts if c.get("won")) / cn, 4) if conts else None,
        contTruncRate=r(sum(1 for c in conts if c.get("truncated")) / cn, 4) if conts else None,
        contErrRate=r(sum(1 for c in conts if c.get("error")) / cn, 4) if conts else None,
        contGenTokMed=_med([float(c.get("gen_tokens", 0)) for c in conts]),
        argmaxRMean=r(sum(argmax_rs) / len(argmax_rs), 4) if argmax_rs else None,
        taskMeanR=r(sum(taskmean_rs) / len(taskmean_rs), 4) if taskmean_rs else None,
        selGain=(r(sum(argmax_rs) / len(argmax_rs)
                   - sum(taskmean_rs) / len(taskmean_rs), 4)
                 if argmax_rs else None),
        wonSteeredRef=r(won_steered, 4) if won_steered is not None else None,
    )


def build_step_entry(sid: int, dirname: str, trajs: list[dict], config: dict) -> dict:
    """step 级序列条目(导出器与实时服务共用——两处聚合口径由本函数唯一
    定义,防双头漂移)。"""
    # 每个量的分母 = 该量**有定义**的行,不是落盘行总数。
    # v4 起每步落 3 类行:bare / steered / research(研究员会话)。research 行
    # 既无 won(恒 False)也无软门量(字段整个缺失,.get(...,0.0) 会补 0 而分母
    # 照加),混进分母会按臂占比整体缩放读数 —— 2026-08-09 实测:won 被压成
    # 两臂均值的 2/3、entMean/entTheta/wMean 精确稀释 2 倍(step1 entTheta
    # 真值 1.28 显示成 0.64,恰是配置初值的一半,足以让熵门标定整体标错一倍)。
    #
    # 行类判据用落盘顶层的 `sample_role`(权威)。曾按 `steer_arm != 0` 间接
    # 推断,而 research 行落盘时 steering 是空 dict ⇒ 取出 None ⇒ 恰好落进
    # "非 bare",正是上述污染的根源;内存态注释称其为 0,与落盘态不符,不要
    # 再据此推断。无 role 字段的旧 run 才回退 arm 推断。
    if any(t.get("role") is not None for t in trajs):
        res_rows = [t for t in trajs if t.get("role") == "research"]
        exec_rows = [t for t in trajs if t.get("role") != "research"]
    else:
        _b = [t for t in trajs if t.get("arm") == 0]
        _s = [t for t in trajs if t.get("arm") == 1]
        if _b or _s:                      # 分臂 run:arm=None 的是 research 行
            res_rows = [t for t in trajs if t.get("arm") is None]
            exec_rows = _b + _s
        else:                             # 历史全臂 run:全部是执行行,曲线不变
            res_rows, exec_rows = [], trajs
    bare = [t for t in exec_rows if t.get("arm") == 0]
    steered = [t for t in exec_rows if t.get("arm") == 1]
    _split = bool(bare or steered)
    soft_rows = steered if _split else exec_rows        # 有软门/干预量定义
    n = max(len(exec_rows), 1)
    n_soft = max(len(soft_rows), 1)
    def tot(fn, rows=None):
        return sum(fn(t) for t in (exec_rows if rows is None else rows))
    cand = tot(lambda t: t["m"]["cand"], soft_rows)
    opened = tot(lambda t: t["m"]["open"], soft_rows)
    rej = tot(lambda t: t["m"]["rej"], soft_rows)
    dlogp = sorted(x for t in soft_rows for x in t.get("absorb", {}).get("dlogp", []))
    top1 = tot(lambda t: t.get("absorb", {}).get("top1", 0), soft_rows)
    miss16 = tot(lambda t: t.get("absorb", {}).get("miss16", 0), soft_rows)
    nrej_ab = tot(lambda t: t.get("absorb", {}).get("nrej", 0), soft_rows)
    toks = tot(lambda t: t["len"], soft_rows)      # 干预密度的分母(仅 steered)
    toks_exec = tot(lambda t: t["len"])            # 执行轨迹长度
    entry = dict(
        id=sid,
        dir=dirname,
        n=len(exec_rows),          # 执行轨迹数(与 expectedTrainTrajs 同口径)
        nAll=len(trajs),           # 落盘行总数(含 research 行)
        won=r(tot(lambda t: t["won"]) / n, 4),
        gate=r(opened / max(cand, 1), 4),
        tilt=r(tot(lambda t: t["m"]["tilt"], soft_rows) / n_soft, 2),
        rej=r(rej / n_soft, 2),
        rejrate=r(rej / max(opened, 1), 4),
        spent=r(tot(lambda t: t["m"]["spent"], soft_rows) / n_soft, 2),
        theta=r(tot(lambda t: t["theta"], soft_rows) / n_soft, 4),
        bypass=r(tot(lambda t: t["m"]["bypass"], soft_rows) / n_soft, 1),
        # ---- v3.3 观察面 ----
        wonBare=r(sum(t["won"] for t in bare) / len(bare), 4) if bare else None,
        wonSteered=(r(sum(t["won"] for t in steered) / len(steered), 4)
                    if steered else None),
        nBare=len(bare) or None,
        nSteered=len(steered) or None,
        armGap=(r(sum(t["won"] for t in steered) / len(steered)
                  - sum(t["won"] for t in bare) / len(bare), 4)
                if bare and steered else None),
        # 软门观察量只在 steered 行有定义(bare 行携 0.0 哨兵、research 行
        # 字段整个缺失 ⇒ .get(...,0.0) 会补 0 而分母照加,是 2 倍稀释的来源)。
        # soft_rows 上面已按 _split 取定:分臂 run 只取 arm==1。
        entTheta=(r(sum(t.get("enttheta", 0.0) for t in soft_rows)
                    / len(soft_rows), 3) if soft_rows else 0.0),
        entMean=(r(sum(t.get("entmean", 0.0) for t in soft_rows)
                   / len(soft_rows), 3) if soft_rows else 0.0),
        wMean=(r(sum(t.get("wmean", 0.0) for t in soft_rows)
                 / len(soft_rows), 3) if soft_rows else 0.0),
        gateBypassRate=r(
            sum(t["m"].get("gbypass", 0) for t in soft_rows)
            / max(sum(t["len"] for t in soft_rows), 1), 4),
        alpha=pgs_alpha(sid, config),
        lenMean=r(toks_exec / n, 0),
        lenMax=max((t["len"] for t in exec_rows), default=0),
        # research 行(研究员会话)长度单列 —— 与执行轨迹不同量级,混进
        # lenMean 会把"执行轨迹在变长/变短"的判读带偏。
        lenResearchMean=(r(sum(t["len"] for t in res_rows) / len(res_rows), 0)
                         if res_rows else None),
        # v4 researcher 环(见 build_research_entry 的因果链说明)
        res=build_research_entry(res_rows, steered, config),
        # 吸收进度三曲线(docs/pgs-v33-mixed-policy-design.md M6)
        absorbDlogpMed=(r(dlogp[len(dlogp) // 2], 3) if dlogp else None),
        absorbTop1Rate=(r(top1 / nrej_ab, 4) if nrej_ab else None),
        absorbMiss16Rate=(r(miss16 / nrej_ab, 4) if nrej_ab else None),
        rejDensity=r(rej / max(toks, 1), 5),
        # fork-token 双带 occurrence(轨迹占比)
        forkRed=r(tot(lambda t: t.get("fork", {}).get("redundancy", 0)) / n, 4),
        forkEss=r(tot(lambda t: t.get("fork", {}).get("essence", 0)) / n, 4),
    )
    return entry


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
    ap.add_argument("--fork-lexicon", default=None,
                    help='fork 词表 json {"redundancy": [...], "essence": [...]}'
                         "(数据驱动重标后覆盖内置词表)")
    a = ap.parse_args()
    if a.fork_lexicon:
        set_fork_lexicon(json.load(open(a.fork_lexicon)))

    run = Path(a.run_dir)
    out = Path(a.out_dir) if a.out_dir else run / "microscope"
    (out / "data").mkdir(parents=True, exist_ok=True)
    dec = Dec(load_tok(a.tokenizer))
    docs = load_docs(a.doc_table)
    config = load_run_config(run)

    series = []
    for sd in step_files(run, a.steps, a.last):
        files = sorted(glob.glob(str(sd / "*.json")))
        trajs = []
        for f in files:
            d = json.load(open(f))
            trajs.append(build_traj(d, dec, docs, run=run, bmax=config["bmax"]))
        sid = int(sd.name.split("_")[1])
        series.append(build_step_entry(sid, sd.name, trajs, config))
        payload = json.dumps({"trajs": trajs}, ensure_ascii=False, separators=(",", ":"))
        (out / "data" / f"{sd.name}.js").write_text(
            f'window.PGS_STEP_DATA=window.PGS_STEP_DATA||{{}};window.PGS_STEP_DATA["{sd.name}"]={payload};',
            encoding="utf-8",
        )
        print(f"[microscope] {sd.name}: {len(trajs)} trajs -> data/{sd.name}.js")

    runjs = json.dumps(
        dict(
            name=run.name,
            steps=series,
            config=config,
            docs={"staticCount": len(docs), "dynamicArchives": (run / "research").is_dir()},
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    (out / "data" / "run.js").write_text(f"window.PGS_RUN={runjs};", encoding="utf-8")
    shutil.copyfile(APP, out / "index.html")
    print(f"[microscope] app: {out / 'index.html'}  (steps={len(series)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
