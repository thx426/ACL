import importlib.util
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("pgs_microscope")
S = _load("pgs_microscope_server")


class FakeDec:
    def one(self, token_id):
        return f"<{token_id}>"

    def seq(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


def rollout(**overrides):
    row = {
        "task_id": "train/pick/task-1",
        "validate": False,
        "won": True,
        "outcome": "won",
        "split": "train",
        "response_token_ids": [1, 2],
        "response_token_len": 2,
        "turn_token_spans": [[0, 2]],
        "trajectory": {"initial_obs": "Your task is to test the microscope."},
        "steering": {},
        "research": {},
    }
    row.update(overrides)
    return row


def make_archive(run: Path, *, doc="dynamic guidance", meta=None):
    rel = Path("research/task--abc/20260805_152705_v0")
    archive = run / rel
    archive.mkdir(parents=True)
    if doc is not None:
        (archive / "doc.md").write_text(doc, encoding="utf-8")
    if meta is not None:
        (archive / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return rel.as_posix()


def test_dynamic_archive_and_budget_flow_into_trajectory(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "effective_config.yaml").write_text(
        "run:\n  name: demo\nsteering:\n  enabled: true\n  b_max: 400.0  # formal budget\ntrain:\n  b_max: 7\n",
        encoding="utf-8",
    )
    archive = make_archive(
        run,
        meta={
            "global_step": 1,
            "policy_version": 0,
            "experiments_used": 16,
            "episodes_consumed": 64,
            "turns_used": 20,
            "finalized": True,
            "doc_words": 2,
            "gt": "off",
            "gt_injected": False,
            "researcher_model": "must-not-leak-unlisted-fields",
        },
    )
    row = rollout(
        research={
            "research_archive": archive,
            "research_error": 0,
            "research_doc_missing": 0,
        }
    )

    config = M.load_run_config(run)
    traj = M.build_traj(row, FakeDec(), run=run, bmax=config["bmax"])

    # 子集断言:v3.3 起 config 还带 advMode/entropy* 等新键(取默认值),
    # 全等断言会随每次键面演进碎一次——只锁本用例关心的键。
    assert {
        "bmax": 400.0,
        "bmaxUnbounded": False,
        "gateMode": "",
        "bmaxSource": "effective_config.yaml:steering.b_max",
        "bmaxWarning": "",
        "trainBatchSize": 0,
        "rolloutN": 0,
        "expectedTrainTrajs": 0,
    }.items() <= config.items()
    assert traj["m"]["bmax"] == 400.0
    assert traj["doc"] == "dynamic guidance"
    assert traj["privilege"]["state"] == "dynamic_researcher"
    assert traj["privilege"]["usedByTeacher"] is True
    assert traj["privilege"]["archive"] == archive
    assert traj["privilege"]["meta"]["global_step"] == 1
    assert traj["privilege"]["meta"]["policy_version"] == 0
    assert traj["privilege"]["meta"]["experiments_used"] == 16
    assert "researcher_model" not in traj["privilege"]["meta"]


def test_arm_split_bare_replica_is_not_reported_as_steered(tmp_path):
    """分臂 run:bare 副本与同组 steered 共享 archive 记录,但教师未介入它。

    只看 research 字段会把整个 run 的一半误读成“有特权文档”,正是分臂
    要修的读数污染在可视化层重演。
    """
    run = tmp_path / "run"
    run.mkdir()
    archive = make_archive(run, meta={"global_step": 1, "policy_version": 0})
    research = {"research_archive": archive, "research_error": 0, "research_doc_missing": 0}

    bare = M.resolve_privilege(
        rollout(research=research, steering={"steer_arm": 0, "steer_count": 0}), run
    )
    assert bare["state"] == "arm_bare"
    assert bare["usedByTeacher"] is False
    assert bare["arm"] == 0
    # 文档仍载入供对照(“同组 steered 看到了什么”),但不得标为已进入教师
    assert bare["doc"] == "dynamic guidance"
    assert bare["archive"] == archive

    steered = M.resolve_privilege(
        rollout(research=research, steering={"steer_arm": 1, "steer_count": 3990}), run
    )
    assert steered["state"] == "dynamic_researcher"
    assert steered["usedByTeacher"] is True
    assert steered["arm"] == 1

    # 非分臂 run(历史全臂,无 steer_arm)行为逐字不变
    legacy = M.resolve_privilege(rollout(research=research), run)
    assert legacy["state"] == "dynamic_researcher"
    assert legacy["usedByTeacher"] is True
    assert legacy["arm"] is None


def test_api_run_reports_per_arm_win_rates(tmp_path):
    """混合 won 既不是能力也不是教师上限,聚合必须给出分臂条件率与 δ gap。"""
    run = tmp_path / "formal"
    step = run / "rollouts" / "step_00001"
    step.mkdir(parents=True)
    # 4 bare(1 胜) + 4 steered(3 胜) => bare 25%, steered 75%, gap +50%
    for i in range(4):
        (step / f"b{i}.json").write_text(
            json.dumps(rollout(won=(i == 0), steering={"steer_arm": 0}))
        )
    for i in range(4):
        (step / f"s{i}.json").write_text(
            json.dumps(rollout(won=(i < 3), steering={"steer_arm": 1}))
        )
    S.RUN = run
    S.DOCS = {}
    S.CONFIG = {"bmax": 0.0, "expectedTrainTrajs": 8}
    S._cache.clear()
    row = S.api_run()["steps"][0]
    assert row["nBare"] == 4 and row["nSteered"] == 4
    assert row["wonBare"] == 0.25
    assert row["wonSteered"] == 0.75
    assert row["armGap"] == 0.5
    assert row["won"] == 0.5          # 混合率仍保留,但不再是唯一读数


def test_api_run_omits_arm_fields_for_non_split_runs(tmp_path):
    run = tmp_path / "legacy"
    step = run / "rollouts" / "step_00001"
    step.mkdir(parents=True)
    for i in range(4):
        (step / f"{i}.json").write_text(json.dumps(rollout()))
    S.RUN = run
    S.DOCS = {}
    S.CONFIG = {"bmax": 0.0, "expectedTrainTrajs": 4}
    S._cache.clear()
    row = S.api_run()["steps"][0]
    # v3.3 契约演进:step 条目 schema 恒齐(build_step_entry 单一口径),
    # 非分臂 run 的臂字段以 None 表缺席(前端 `!=null` 判有臂)。
    assert row["wonBare"] is None and row["armGap"] is None


def test_static_gt_validation_and_true_bare_states(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    task = rollout()["task_id"]

    static = M.resolve_privilege(rollout(), run, {task: "GT guidance"})
    assert static["state"] == "static_gt"
    assert static["doc"] == "GT guidance"

    missing = M.resolve_privilege(rollout(), run, {"other": "GT"})
    assert missing["state"] == "static_missing"

    validation = M.resolve_privilege(rollout(validate=True), run, {})
    assert validation["state"] == "validation_bare"

    bare = M.resolve_privilege(rollout(), run, {})
    assert bare["state"] == "bare"


def test_research_degraded_and_archive_failures_are_visible(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    degraded = M.resolve_privilege(
        rollout(research={"research_error": 1, "research_doc_missing": 1}), run
    )
    assert degraded["state"] == "researcher_degraded"
    assert "报错" in degraded["error"]

    absent = M.resolve_privilege(
        rollout(research={"research_archive": "research/task/absent"}), run
    )
    assert absent["state"] == "archive_missing"
    assert absent["error"] == "archive 目录不存在"

    missing_rel = make_archive(run, doc=None, meta={"global_step": 1})
    missing = M.resolve_privilege(
        rollout(research={"research_archive": missing_rel}), run
    )
    assert missing["state"] == "archive_missing"
    assert "doc.md 不存在" in missing["error"]

    bad_rel = make_archive(run / "other", meta={"global_step": 1})
    bad_archive = run / "other" / bad_rel
    (bad_archive / "meta.json").write_text("{not-json", encoding="utf-8")
    bad = M.resolve_privilege(
        rollout(research={"research_archive": bad_rel}), run / "other"
    )
    assert bad["state"] == "archive_error"
    assert bad["doc"] == "dynamic guidance"
    assert bad["usedByTeacher"] is True
    assert "meta.json 读取失败" in bad["error"]


def test_dynamic_meta_error_marks_degraded_but_keeps_document(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    archive = make_archive(
        run,
        meta={"global_step": 3, "policy_version": 2, "error": "research timeout"},
    )
    result = M.resolve_privilege(
        rollout(research={"research_archive": archive, "research_error": 1}), run
    )
    assert result["state"] == "researcher_degraded"
    assert result["doc"] == "dynamic guidance"
    assert result["usedByTeacher"] is False
    assert result["error"] == "research timeout"


@pytest.mark.parametrize(
    "archive",
    [
        "/etc",
        "../outside",
        "research/../outside",
        "other/task/archive",
        r"research\\task\\archive",
    ],
)
def test_archive_path_rejects_unsafe_references(tmp_path, archive):
    run = tmp_path / "run"
    run.mkdir()
    result = M.resolve_privilege(
        rollout(research={"research_archive": archive}), run
    )
    assert result["state"] == "archive_error"
    assert not result["doc"]


def test_archive_path_rejects_symlink_escape(tmp_path):
    run = tmp_path / "run"
    outside = tmp_path / "outside"
    (run / "research").mkdir(parents=True)
    outside.mkdir()
    (outside / "doc.md").write_text("secret", encoding="utf-8")
    (outside / "meta.json").write_text("{}", encoding="utf-8")
    try:
        os.symlink(outside, run / "research" / "escape")
    except (OSError, NotImplementedError):
        pytest.skip("symlink is unavailable")

    result = M.resolve_privilege(
        rollout(research={"research_archive": "research/escape"}), run
    )
    assert result["state"] == "archive_error"
    assert result["doc"] == ""


def test_research_root_rejects_symlink_escape(tmp_path):
    run = tmp_path / "run"
    outside = tmp_path / "outside-research"
    run.mkdir()
    archive = outside / "task" / "archive"
    archive.mkdir(parents=True)
    (archive / "doc.md").write_text("secret root", encoding="utf-8")
    (archive / "meta.json").write_text("{}", encoding="utf-8")
    try:
        os.symlink(outside, run / "research")
    except (OSError, NotImplementedError):
        pytest.skip("symlink is unavailable")

    result = M.resolve_privilege(
        rollout(research={"research_archive": "research/task/archive"}), run
    )
    assert result["state"] == "archive_error"
    assert result["doc"] == ""
    assert "research 根" in result["error"]


def test_archive_member_rejects_symlink_escape(tmp_path):
    run = tmp_path / "run"
    outside = tmp_path / "outside.md"
    archive = run / "research" / "task" / "archive"
    archive.mkdir(parents=True)
    outside.write_text("secret member", encoding="utf-8")
    (archive / "meta.json").write_text("{}", encoding="utf-8")
    try:
        os.symlink(outside, archive / "doc.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlink is unavailable")

    result = M.resolve_privilege(
        rollout(research={"research_archive": "research/task/archive"}), run
    )
    assert result["state"] == "archive_error"
    assert result["doc"] == ""
    assert "symlink" in result["error"]


def test_unbounded_budget_is_not_mistaken_for_missing_config(tmp_path):
    """v3.2 maxopen 的 `b_max: .inf`:float(".inf") 抛 ValueError,旧逻辑会把它
    当成“配置缺失”回退到 20,于是无上限预算被显示成“7263/20 接近上限”。"""
    run = tmp_path / "run"
    run.mkdir()
    (run / "effective_config.yaml").write_text(
        "train:\n  train_batch_size: 16\n  rollout_n: 16\nsteering:\n  b_max: .inf\n",
        encoding="utf-8",
    )
    config = M.load_run_config(run)
    assert config["bmaxUnbounded"] is True
    assert config["bmax"] is None          # JSON 不能表示 Infinity
    assert config["bmaxSource"] == "effective_config.yaml:steering.b_max"
    assert config["bmaxWarning"] == ""     # 无上限不是告警
    assert config["expectedTrainTrajs"] == 256

    # 整条链路必须可 JSON 序列化(Infinity 会让浏览器 JSON.parse 直接抛)
    traj = M.build_traj(rollout(), FakeDec(), run=run, bmax=config["bmax"])
    assert traj["m"]["bmax"] is None
    assert "Infinity" not in json.dumps({"config": config, "traj": traj})


def test_always_on_gate_mode_switches_readout_semantics(tmp_path):
    """always_on 档开门率恒 1、θ_D 恒 0 是正确常态,不能按自适应档的绿带报警。"""
    run = tmp_path / "run"
    run.mkdir()
    (run / "effective_config.yaml").write_text(
        "steering:\n  b_max: .inf\n  gate_mode: always_on\n", encoding="utf-8"
    )
    assert M.load_run_config(run)["gateMode"] == "always_on"

    adaptive = tmp_path / "adaptive"
    adaptive.mkdir()
    (adaptive / "effective_config.yaml").write_text(
        "steering:\n  b_max: 400\n  gate_mode: adaptive\n", encoding="utf-8"
    )
    cfg = M.load_run_config(adaptive)
    assert cfg["gateMode"] == "adaptive" and cfg["bmax"] == 400.0

    html = (SCRIPTS / "pgs_microscope_app.html").read_text(encoding="utf-8")
    assert "const isAlwaysOn" in html
    assert "恒 100% 是正确常态" in html
    # 绿带判定必须先过 always_on 分支,否则 100% 恒标 warn
    assert 'const gateCls = isAlwaysOn() ? "good"' in html


def test_frontend_treats_null_bmax_as_unbounded_not_missing():
    html = (SCRIPTS / "pgs_microscope_app.html").read_text(encoding="utf-8")
    assert "const isUnbounded" in html
    assert "const fmtB" in html
    assert "const nearCap" in html
    # 旧的裸比较会在 bmax=null 时把 NaN 当成"接近上限"
    assert "t.m.spent>=BMAX-1" not in html
    assert "L.spent>=BMAX-1" not in html


@pytest.mark.parametrize(
    ("text", "warning"),
    [
        ("steering:\n  b_max: nope\n", "非法"),
        ("steering:\n  b_max: -1\n", "非法"),
        ("train:\n  b_max: 400\n", "缺失"),
    ],
)
def test_budget_config_has_explicit_fallback(tmp_path, text, warning):
    run = tmp_path / "run"
    run.mkdir()
    (run / "effective_config.yaml").write_text(text, encoding="utf-8")
    config = M.load_run_config(run)
    assert config["bmax"] == 20.0
    assert config["bmaxSource"] == "fallback"
    assert warning in config["bmaxWarning"]


def test_run_config_reads_expected_train_trajectory_count(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "effective_config.yaml").write_text(
        "train:\n  train_batch_size: 16\n  rollout_n: 8\n"
        "steering:\n  b_max: 81920.0\n",
        encoding="utf-8",
    )
    config = M.load_run_config(run)
    assert config["trainBatchSize"] == 16
    assert config["rolloutN"] == 8
    assert config["expectedTrainTrajs"] == 128
    assert config["bmax"] == 81920.0


def test_api_run_marks_first_incomplete_step_partial_from_config(tmp_path):
    run = tmp_path / "formal"
    step = run / "rollouts" / "step_00001"
    step.mkdir(parents=True)
    for index in range(64):
        (step / f"{index:03d}.json").write_text(json.dumps(rollout()))
    S.RUN = run
    S.DOCS = {}
    S.CONFIG = {
        "bmax": 81920.0,
        "trainBatchSize": 16,
        "rolloutN": 8,
        "expectedTrainTrajs": 128,
    }
    S._cache.clear()
    data = S.api_run()
    assert data["steps"][0]["n"] == 64
    assert data["steps"][0]["expected"] == 128
    assert data["steps"][0]["partial"] is True

    for index in range(64, 128):
        (step / f"{index:03d}.json").write_text(json.dumps(rollout()))
    data = S.api_run()
    assert data["steps"][0]["n"] == 128
    assert data["steps"][0]["partial"] is False


def steered_rollout(n_rec=3, **overrides):
    """带 steering 位置级记录的 rollout(maxopen 档:每个候选位置都有记录)。"""
    row = rollout(
        response_token_ids=list(range(n_rec)),
        response_token_len=n_rec,
        turn_token_spans=[[0, n_rec]],
        steering={
            "steer_arm": 1,
            "steer_theta_d": 0.0,
            "steer_count": n_rec,
            "steer_n_candidates": n_rec,
            "steer_positions": list(range(n_rec)),
            "steer_d": [0.1 * (i + 1) for i in range(n_rec)],
            "steer_eta": [0.5] * n_rec,
            "steer_delta": [2.5] * n_rec,
            "steer_log_pi": [-0.5] * n_rec,
            "steer_log_b": [-0.3] * n_rec,
            "steer_propensity": [1.0] * n_rec,
            "steer_rejected": [0] * n_rec,
            "steer_probe": [0] * n_rec,
            "steer_cap_active": [0] * n_rec,
            "steer_base_open": [1] * n_rec,
            "steer_budget_after": [100.0] * n_rec,
            "steer_draft_token": list(range(n_rec)),
            "steer_final_token": list(range(n_rec)),
            "steer_pi_top16": [[[t, -0.5 * t] for t in range(16)] for _ in range(n_rec)],
            "steer_pt_top16": [[[t, -0.4 * t] for t in range(16)] for _ in range(n_rec)],
            "steer_cand_positions": list(range(n_rec)),
            "steer_d_all": [0.1 * (i + 1) for i in range(n_rec)],
        },
    )
    row.update(overrides)
    return row


def test_live_traj_omits_top16_dists_and_rec_endpoint_serves_them(tmp_path):
    """v3.2 全量携带 top16 后,分布占单轨迹 payload 的 95%(实测 3.15/3.31 MB)。
    轨迹页首屏用不到它们,必须拆成按需接口,否则每开一条都要传几 MB。"""
    run = tmp_path / "run"
    step = run / "rollouts" / "step_00001"
    step.mkdir(parents=True)
    dump = steered_rollout(n_rec=4)
    (step / "0.json").write_text(json.dumps(dump))

    full = M.build_traj(dump, FakeDec(), run=run)
    lean = M.build_traj(dump, FakeDec(), run=run, with_dists=False)
    assert all("pi" in rc and "pt" in rc for rc in full["recs"])      # 静态导出保持全量
    assert all("pi" not in rc and "pt" not in rc for rc in lean["recs"])
    # 除分布外两档必须逐字节一致,精简不能悄悄丢别的字段
    assert [{k: v for k, v in rc.items() if k not in ("pi", "pt")} for rc in full["recs"]] == lean["recs"]
    assert len(json.dumps(lean)) < len(json.dumps(full)) / 2

    S.RUN = run
    S.DEC = FakeDec()
    S.DOCS = {}
    S.CONFIG = {"bmax": None, "bmaxUnbounded": True, "expectedTrainTrajs": 1}
    S._cache.clear()
    S._dump_cache.clear()

    served = S.api_traj("step_00001", 0)
    assert all("pi" not in rc for rc in served["recs"])               # 实时档默认精简

    dists = S.api_rec("step_00001", 0, 2)
    assert dists["ref"] == 2
    assert len(dists["pi"]) == 16 and len(dists["pt"]) == 16
    assert dists["pi"] == full["recs"][2]["pi"]                       # 与全量档同值
    assert "error" in S.api_rec("step_00001", 0, 99)                  # 越界 fail loud
    assert "error" in S.api_rec("step_00001", 5, 0)


def test_dump_cache_avoids_reparsing_between_traj_and_rec_calls(tmp_path):
    run = tmp_path / "run"
    step = run / "rollouts" / "step_00001"
    step.mkdir(parents=True)
    path = step / "0.json"
    path.write_text(json.dumps(steered_rollout(n_rec=3)))
    S.RUN = run
    S.DEC = FakeDec()
    S.DOCS = {}
    S.CONFIG = {"bmax": None, "expectedTrainTrajs": 1}
    S._dump_cache.clear()

    S.api_traj("step_00001", 0)
    S.api_rec("step_00001", 0, 0)
    S.api_rec("step_00001", 0, 1)
    assert len(S._dump_cache) == 1                                    # 三次调用只解析一次

    # 文件改写后必须失效,否则运行中的 step 会读到旧内容
    changed = steered_rollout(n_rec=3)
    changed["won"] = False
    path.write_text(json.dumps(changed))
    os.utime(path, (0, 0))
    assert S.api_traj("step_00001", 0)["won"] is False


def test_in_progress_step_only_parses_newly_arrived_files(tmp_path, monkeypatch):
    """盯在跑的 step 时,每来一条新轨迹整步指纹就变。旧实现把全部文件重解析
    (maxopen 下 256×2.9MB,实测 7.5s/次),而写完的 dump 不会再变。"""
    run = tmp_path / "run"
    step = run / "rollouts" / "step_00001"
    step.mkdir(parents=True)
    for i in range(8):
        (step / f"{i:03d}.json").write_text(json.dumps(steered_rollout(n_rec=2)))
    S.RUN = run
    S.CONFIG = {"bmax": None, "expectedTrainTrajs": 16}
    S._cache.clear()
    S._meta_cache.clear()

    parsed = []
    real_meta_of = S._meta_of
    monkeypatch.setattr(S, "_meta_of", lambda d: (parsed.append(1), real_meta_of(d))[1])

    S.api_run()
    assert len(parsed) == 8                      # 首次:全解析
    parsed.clear()

    S.api_run()
    assert len(parsed) == 0                      # 无变化:整步聚合命中
    S.api_step("step_00001")
    assert len(parsed) == 0                      # 列表复用同一份 memo

    # 新到 2 条 → 只解析这 2 条,而不是 10 条
    for i in range(8, 10):
        (step / f"{i:03d}.json").write_text(json.dumps(steered_rollout(n_rec=2)))
    S.api_run()
    assert len(parsed) == 2

    # 已有文件被改写必须失效(补写/重跑场景)
    parsed.clear()
    changed = steered_rollout(n_rec=2)
    changed["won"] = False
    (step / "000.json").write_text(json.dumps(changed))
    os.utime(step / "000.json", (0, 0))
    S.api_run()
    assert len(parsed) == 1


def test_frontend_caches_trajectories_and_fetches_dists_on_demand():
    html = (SCRIPTS / "pgs_microscope_app.html").read_text(encoding="utf-8")
    assert "const trajCache" in html
    assert "prefetchNeighbors" in html
    assert "getRecDists" in html
    assert "api/rec/" in html
    assert "needDists" in html
    # 进行中的 step 数据仍在变,缓存会喂旧读数
    assert "if(!isPartial(dir)) cachePut" in html


def test_cache_fingerprint_detects_rewrite_without_file_count_change(tmp_path):
    step = tmp_path / "step_00001"
    step.mkdir()
    dump = step / "one.json"
    dump.write_text('{"v": 1}', encoding="utf-8")
    calls = []

    def build(files):
        calls.append(1)
        return json.loads(Path(files[0]).read_text())

    S._cache.clear()
    assert S._cached("x", step, build) == {"v": 1}
    assert S._cached("x", step, build) == {"v": 1}
    dump.write_text('{"v": 200}', encoding="utf-8")
    assert S._cached("x", step, build) == {"v": 200}
    assert len(calls) == 2


def test_cache_single_flight_builds_once_for_concurrent_requests(tmp_path):
    step = tmp_path / "step_00001"
    step.mkdir()
    (step / "one.json").write_text('{"v": 1}', encoding="utf-8")
    calls = []
    release = threading.Event()

    def build(files):
        calls.append(1)
        release.wait(timeout=2)
        return json.loads(Path(files[0]).read_text())

    S._cache.clear()
    S._inflight.clear()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(S._cached, "single", step, build) for _ in range(8)]
        for _ in range(100):
            if calls:
                break
            time.sleep(.005)
        release.set()
        results = [future.result(timeout=2) for future in futures]
    assert results == [{"v": 1}] * 8
    assert len(calls) == 1
    assert not S._inflight


def test_cache_single_flight_releases_waiters_after_error(tmp_path):
    step = tmp_path / "step_00001"
    step.mkdir()
    (step / "one.json").write_text('{"v": 1}', encoding="utf-8")
    S._cache.clear()
    S._inflight.clear()
    calls = []

    def build(files):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("first build failed")
        return {"v": 1}

    with pytest.raises(ValueError, match="first build failed"):
        S._cached("retry", step, build)
    assert S._cached("retry", step, build) == {"v": 1}
    assert len(calls) == 2
    assert not S._inflight


def test_server_rejects_step_path_traversal(tmp_path):
    run = tmp_path / "run"
    (run / "rollouts" / "step_00001").mkdir(parents=True)
    S.RUN = run
    assert S._step_dir("step_00001") == run / "rollouts" / "step_00001"
    assert S._step_dir("../research") is None
    assert S._step_dir("step_00001/../../research") is None


def test_live_http_api_returns_dynamic_doc_and_run_budget(tmp_path):
    run = tmp_path / "formal"
    step = run / "rollouts" / "step_00001"
    step.mkdir(parents=True)
    (run / "effective_config.yaml").write_text(
        "steering:\n  b_max: 400.0\n", encoding="utf-8"
    )
    archive = make_archive(
        run,
        meta={"global_step": 1, "policy_version": 0, "experiments_used": 16},
    )
    (step / "0.json").write_text(
        json.dumps(rollout(research={"research_archive": archive})), encoding="utf-8"
    )
    S.RUN = run
    S.DEC = FakeDec()
    S.DOCS = {}
    S.CONFIG = M.load_run_config(run)
    S._cache.clear()
    server = S.ThreadingHTTPServer(("127.0.0.1", 0), S.H)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/run") as response:
            api_run = json.load(response)
        with urllib.request.urlopen(base + "/api/traj/step_00001/0") as response:
            traj = json.load(response)
        with urllib.request.urlopen(base + "/data/run.js") as response:
            assert response.status == 200
            assert response.read() == b""
        with urllib.request.urlopen(base + "/favicon.ico") as response:
            assert response.status == 204
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert api_run["config"]["bmax"] == 400.0
    assert api_run["docs"]["dynamicArchives"] is True
    assert traj["m"]["bmax"] == 400.0
    assert traj["privilege"]["state"] == "dynamic_researcher"
    assert traj["privilege"]["doc"] == "dynamic guidance"
    assert traj["privilege"]["meta"]["policy_version"] == 0


def test_frontend_separates_arms_in_overview_and_list():
    """分臂 run 的能力读数必须与教师上限分开呈现,混合率不得占据能力位。"""
    html = (SCRIPTS / "pgs_microscope_app.html").read_text(encoding="utf-8")
    assert "const HAS_ARMS" in html
    assert "能力 (bare 臂)" in html          # KPI 位给 bare,不给混合
    assert "训练成功率:分臂" in html          # 三线图
    assert "δ gap = steered − bare" in html   # 教师增益独立曲线
    assert "arm_bare" in html                 # 轨迹页状态
    assert "未被干预" in html                 # bare 副本徽章
    assert "F.arm" in html                    # 列表臂筛选


def test_frontend_uses_structured_privilege_and_dynamic_budget():
    html = (SCRIPTS / "pgs_microscope_app.html").read_text(encoding="utf-8")
    assert "renderPrivilege(t)" in html
    assert "dynamic_researcher" in html
    assert "validation_bare" in html
    assert "R.hasDocs===false" not in html
    assert "对照臂(纯 GRPO)" not in html
    assert "const runBmax" in html
    assert "EXPECT_TILT" not in html
    assert "const BMAX=20" not in html


# ------------------------------------------------- v3.3 监控/可视化观察面

def _v33_steering(arm, *, rej=False):
    return {
        "steer_arm": arm, "steer_positions": [1], "steer_d": [0.5],
        "steer_eta": [0.25], "steer_delta": [0.6], "steer_log_pi": [-1.2],
        "steer_log_b": [-0.9], "steer_propensity": [1.0],
        "steer_rejected": [int(rej)], "steer_probe": [0],
        "steer_cap_active": [0], "steer_base_open": [1],
        "steer_budget_after": [5.0], "steer_draft_token": [7],
        "steer_final_token": [9 if rej else 7],
        "steer_pi_top16": [[[9, -0.3], [7, -1.2]]],
        "steer_pt_top16": [[[9, -0.1]]],
        "steer_intensity": [0.5], "steer_entropy": [1.1],
        "steer_n_candidates": 4, "steer_n_base_open": 2, "steer_count": 1,
        "steer_reject_count": int(rej), "steer_budget_spent": 0.6,
        "steer_n_bypassed": 0, "steer_n_pt_window_miss": 0,
        # bare 行的软门量是 0.0 哨兵(真实形态:bare 无 steering state)
        "steer_n_gate_bypassed": 3 if arm == 1 else 0,
        "steer_theta_d": 0.03,
        "steer_entropy_theta": 1.28 if arm == 1 else 0.0,
        "steer_entropy_mean": 0.4 if arm == 1 else 0.0,
        "steer_intensity_mean": 0.5 if arm == 1 else 0.0,
        "steer_cand_positions": [1], "steer_d_all": [0.5],
    }


def _v33_row(arm, won, *, rej=False):
    return rollout(
        won=won,
        outcome="won" if won else "lost",
        steering=_v33_steering(arm, rej=rej),
        trajectory={
            "initial_obs": "x",
            "steps": [{"thought": "Wait, let me verify the plan",
                       "action_raw": "go", "observation": "ok"}],
        },
    )


_V33_CFG = {"bmax": 20.0, "advMode": "asym", "steeredAdvWeight": 1.0,
            "steeredAdvDecay": 100, "responseLength": 32768,
            "entropyFloor": 0.6, "entropyThetaInit": 1.28}


def test_v33_step_entry_absorb_fork_alpha_and_soft_gate_fields():
    t_bare = M.build_traj(_v33_row(0, True), FakeDec(), bmax=20.0)
    t_st = M.build_traj(_v33_row(1, False, rej=True), FakeDec(), bmax=20.0)
    # 行级观察面:软强度 w/h 进 rec;吸收中间量按 0-based rank(top1=首位)
    assert t_st["recs"][0]["w"] == 0.5 and t_st["recs"][0]["h"] == 1.1
    assert t_st["absorb"] == {"dlogp": [-0.9], "top1": 1, "miss16": 0, "nrej": 1}
    assert t_bare["fork"] == {"redundancy": 1, "essence": 1}
    e = M.build_step_entry(50, "step_00050", [t_bare, t_st], _V33_CFG)
    assert e["alpha"] == 0.5                       # cosine(50/100)
    # 审阅 C3/C6:软门量只在 steered 行上聚合(bare 的 0.0 哨兵不进分母)
    assert e["entTheta"] == 1.28 and e["wMean"] == 0.5
    assert e["gateBypassRate"] == round(3 / 2, 4)  # steered 行 3 bypass / 2 tok
    assert e["absorbDlogpMed"] == -0.9 and e["absorbTop1Rate"] == 1.0
    assert e["forkRed"] == 1.0 and e["forkEss"] == 1.0
    assert e["armGap"] == -1.0 and e["lenMax"] == 2


def test_v33_exporter_and_server_step_aggregation_identical():
    """聚合口径单一定义红线:_meta_of 轻量路径与 build_traj 全量路径喂
    build_step_entry 必须产出逐字段相同的 step 条目。"""
    rows = [_v33_row(0, True), _v33_row(1, False, rej=True)]
    full = [M.build_traj(r, FakeDec(), bmax=20.0) for r in rows]
    lite = [S._meta_of(r) for r in rows]
    e1 = M.build_step_entry(7, "step_00007", full, _V33_CFG)
    e2 = M.build_step_entry(7, "step_00007", lite, _V33_CFG)
    assert e1 == e2


def test_v33_alpha_recompute_matches_schedule_edges():
    assert M.pgs_alpha(0, _V33_CFG) == 1.0
    assert M.pgs_alpha(100, _V33_CFG) == 0.0
    assert M.pgs_alpha(500, _V33_CFG) == 0.0
    assert M.pgs_alpha(10, {"advMode": "legacy_shared"}) is None
    assert M.pgs_alpha(10, {"advMode": "asym", "steeredAdvWeight": 0.7,
                            "steeredAdvDecay": 0}) == 0.7


def test_v33_run_config_reads_new_keys(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "effective_config.yaml").write_text(
        "train:\n  response_length: 32768\n"
        "steering:\n  b_max: .inf\n  gate_mode: entropy\n  adv_mode: asym\n"
        "  steered_adv_weight: 1.0\n  steered_adv_decay_steps: 200\n"
        "  entropy_floor: 0.6\n  entropy_theta_init: 1.28\n",
        encoding="utf-8",
    )
    c = M.load_run_config(run)
    assert c["advMode"] == "asym" and c["gateMode"] == "entropy"
    assert c["steeredAdvDecay"] == 200 and c["responseLength"] == 32768
    assert c["entropyFloor"] == 0.6 and c["entropyThetaInit"] == 1.28
