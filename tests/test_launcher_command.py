"""build_command — the yaml->verl translation must carry the mechanism
switches faithfully: rollout_n and the steering→max_logprobs linkage always,
and the ulysses_sp>1 three-way linkage (SP size + use_remove_padding=True +
flash_attention_2) versus the sdpa profile at sp=1. PGS: no distillation
overrides may appear anywhere in the command.
"""
import importlib.util
from pathlib import Path

from trace_opd.common.config import load_config

_LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "launch_train.py"
_spec = importlib.util.spec_from_file_location("launch_train", _LAUNCHER)
launch_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_train)

YAML = """
run: {name: t, runs_root: runs}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet]%s}
eval: {}
%s
"""


def _cmd(tmp_path, train_extra="", steering_extra=""):
    path = tmp_path / "c.yaml"
    path.write_text(YAML % (train_extra, steering_extra))
    cfg = load_config(path)
    return launch_train.build_command(cfg, tmp_path / "run")


def test_default_profile_and_rollout_n(tmp_path):
    cmd = _cmd(tmp_path)
    assert "actor_rollout_ref.rollout.n=8" in cmd
    assert "actor_rollout_ref.model.use_remove_padding=False" in cmd
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in cmd
    assert not any("ulysses_sequence_parallel_size" in c for c in cmd)
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in cmd
    # PGS 默认 grpo_loo(规范 §2.6 条件 iv);对照臂经 train.adv_estimator=grpo
    assert "algorithm.adv_estimator=grpo_loo" in cmd
    # Collapse-watch entropy curve (metric only, entropy_coeff stays 0).
    assert "actor_rollout_ref.actor.calculate_entropy=True" in cmd
    # Micro-batch packing budgets decoupled from the 34816 capacity declaration
    # (feedback P0: 34816 fed 21.2 GiB fp32 logits into both phases).
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192" in cmd
    assert not any(c.endswith("token_len_per_gpu=34816") for c in cmd)


def test_v4_research_rows_extend_rollout_n(tmp_path):
    # v4:rollout.n = rollout_n + research_n(三段布局);关断时保持原值。
    steer = (
        "steering: {enabled: true, doc_source: researcher, theta_d: 0.8}\n"
        "research: {enabled: true, backend: engine, tool_protocol: text, gt: \"off\"}\n"
        "narration: {backend: engine}\n"
        "research_rl: {enabled: true, research_n: 8}"
    )
    cmd = _cmd(tmp_path, ", rollout_n: 16, agent_task_timeout_s: 0", steer)
    assert "actor_rollout_ref.rollout.n=24" in cmd
    cmd0 = _cmd(tmp_path, ", rollout_n: 16")
    assert "actor_rollout_ref.rollout.n=16" in cmd0


def test_pgs_loss_surface_is_plain_grpo(tmp_path):
    # PGS: the loss is plain GRPO — a distillation KEY in the launch command
    # means dead config resurrected somewhere. (Match on the key side only:
    # tmp_path values may contain the word via the test name.)
    cmd = _cmd(tmp_path)
    assert not any("distillation" in c.split("=", 1)[0] for c in cmd)


def test_steering_lifts_vllm_max_logprobs(tmp_path):
    # SPS verify needs per-position top-K prompt_logprobs; vLLM's max_logprobs
    # is a LAUNCH-time parameter and must be >= steering.topk.
    cmd = _cmd(
        tmp_path,
        steering_extra=(
            "steering: {enabled: true, doc_table: /tmp/d.jsonl, theta_d: 0.8, topk: 64}"
        ),
    )
    # 联动源是 obs_topk(观测窗)而非 topk —— hook 的 draft logprobs=obs_topk
    # 与 verify prompt_logprobs=obs_topk 都要过 vLLM 校验(codex 审阅 P0)。
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=256" in cmd
    # off by default: no max_logprobs key at all
    cmd0 = _cmd(tmp_path)
    assert not any("max_logprobs" in c for c in cmd0)


def test_ulysses_sp_flips_padding_and_attention(tmp_path):
    cmd = _cmd(tmp_path, ", ulysses_sp: 2")
    assert "actor_rollout_ref.actor.ulysses_sequence_parallel_size=2" in cmd
    assert "actor_rollout_ref.model.use_remove_padding=True" in cmd
    assert (
        "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2"
        in cmd
    )
    assert not any(c.endswith("=sdpa") for c in cmd)


def test_rollout_n_override_flows_through(tmp_path):
    cmd = _cmd(tmp_path, ", rollout_n: 4")
    assert "actor_rollout_ref.rollout.n=4" in cmd


def test_batched_tokens_override_flows_through(tmp_path):
    cmd = _cmd(tmp_path, ", rollout_max_num_batched_tokens: 4096")
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=4096" in cmd


def test_steering_config_rejects_nonfinite_and_bad_structure(tmp_path):
    # 审阅 IF-13:NaN/inf/topk=0/坏 rate bounds 必须在启动时死,不能等到
    # rollout 中段。
    import pytest

    def _load(steering):
        path = tmp_path / "c.yaml"
        path.write_text(YAML % ("", steering))
        return load_config(path)

    base = 'steering: {enabled: true, doc_table: /tmp/d.jsonl, theta_d: 0.8, %s}'
    for bad in ("c: .inf", "eta_max: .nan", "topk: 0", "r_max: -1",
                "gate_rate_bounds: [0.5, 0.2]", "gate_rate_bounds: [0.0, 0.3, 1.0]"):
        with pytest.raises(ValueError):
            _load(base % bad)


def test_norm_adv_by_std_defaults_off_and_flows_through(tmp_path):
    # 审阅 IF-04(方案 A):主档不做组内 std 归一化(scale≡1,条件 iv 严格成立)
    cmd = _cmd(tmp_path)
    assert "algorithm.norm_adv_by_std_in_grpo=False" in cmd
    cmd2 = _cmd(tmp_path, ", norm_adv_by_std: true")
    assert "algorithm.norm_adv_by_std_in_grpo=True" in cmd2


def test_fingerprint_moves_with_objective_fields(tmp_path):
    # 审阅 IF-10:改 adv_estimator / norm_adv_by_std / steering 参数必须换指纹
    from trace_opd.common.config import objective_fingerprint

    def _fp(train_extra=""):
        path = tmp_path / f"fp{abs(hash(train_extra))}.yaml"
        path.write_text(YAML % (train_extra, ""))
        return objective_fingerprint(load_config(path))

    base = _fp()
    assert _fp(", adv_estimator: grpo") != base
    assert _fp(", norm_adv_by_std: true") != base
    assert _fp(", lr: 2.0e-6") != base



def _dynamic_cfg(tmp_path, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("TEST_PREFLIGHT_KEY", "test-key")
    else:
        import os
        os.environ.setdefault("TEST_PREFLIGHT_KEY", "test-key")
    text = YAML % (
        ", train_batch_size: 16, agent_num_workers: 16, agent_task_timeout_s: 5400",
        """
steering: {enabled: true, doc_source: researcher, doc_table: '', theta_d: 0.8}
research: {enabled: true, backend: external, tool_protocol: fncall, gt: 'off', session_timeout_s: 3600}
narration: {backend: engine}
roles:
  researcher:
    endpoint: 'http://one/v1,http://two/v1'
    model: qwen3.6-35b-a3b
    api_key_env: TEST_PREFLIGHT_KEY
    mode: external_pool
    sidecar_sleep: false
    dp_size: 0
    max_concurrency: 64
    max_tokens: 8192
    timeout_s: 600
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
""",
    )
    path = tmp_path / "dynamic.yaml"
    path.write_text(text)
    return load_config(path)


def test_preflight_all_endpoints_model_and_auth(tmp_path, monkeypatch):
    import io
    import json
    import pytest

    cfg = _dynamic_cfg(tmp_path, monkeypatch)
    cfg.roles["researcher"].api_key = "secret"
    seen = []

    class Resp:
        def __init__(self, model): self.model = model
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"data": [{"id": self.model}]}).encode()

    def good(req, timeout):
        seen.append((req.full_url, req.get_header("Authorization"), timeout))
        return Resp("qwen3.6-35b-a3b")

    async def protocol_ok(endpoint, role):
        seen.append((endpoint + "/chat/completions", "protocol", role.extra_body))

    monkeypatch.setattr(launch_train.urllib.request, "urlopen", good)
    monkeypatch.setattr(launch_train, "_preflight_researcher_protocol", protocol_ok)
    launch_train.preflight_researcher(cfg)
    assert seen == [
        ("http://one/v1/models", "Bearer secret", 15),
        (
            "http://one/v1/chat/completions",
            "protocol",
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        ("http://two/v1/models", "Bearer secret", 15),
        (
            "http://two/v1/chat/completions",
            "protocol",
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
    ]

    def half(req, timeout):
        return Resp("qwen3.6-35b-a3b" if "one" in req.full_url else "wrong")

    monkeypatch.setattr(launch_train.urllib.request, "urlopen", half)
    with pytest.raises(SystemExit, match="not fully healthy"):
        launch_train.preflight_researcher(cfg)


def test_protocol_preflight_rejects_thinking_bad_calls_and_bad_followup(monkeypatch):
    import asyncio
    import pytest
    from trace_opd.common.llm import ChatResult, ToolCall

    class Role:
        model = "qwen3.6-35b-a3b"
        api_key = "secret"
        timeout_s = 600
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def completion(*, content=None, calls=None, **message_extra):
        calls = calls or []
        raw_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments,
                },
            }
            for call in calls
        ]
        raw = {"role": "assistant", "content": content, **message_extra}
        if raw_calls:
            raw["tool_calls"] = raw_calls
        return ChatResult(content=content, tool_calls=calls, raw_message=raw)

    good_call = ToolCall(
        id="call_1",
        name="trace_preflight_echo",
        arguments={"nonce": launch_train._PREFLIGHT_NONCE},
        raw_arguments='{"nonce":"trace-researcher-fncall-v1"}',
    )

    class FakeClient:
        replies = []
        seen = []
        closed = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def chat(self, messages, **kwargs):
            type(self).seen.append((messages, kwargs, self.kwargs))
            return type(self).replies.pop(0)

        async def aclose(self):
            type(self).closed = True

    monkeypatch.setattr(launch_train, "OpenAICompatClient", FakeClient)

    async def run(first, second=None):
        FakeClient.replies = [first] + ([second] if second is not None else [])
        FakeClient.seen = []
        FakeClient.closed = False
        await launch_train._preflight_researcher_protocol("http://one/v1", Role())

    asyncio.run(run(completion(calls=[good_call]), completion(content="PREFLIGHT_OK")))
    assert FakeClient.closed
    assert len(FakeClient.seen) == 2
    second_messages = FakeClient.seen[1][0]
    assert second_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"nonce":"trace-researcher-fncall-v1"}',
    }
    assert FakeClient.seen[1][1]["tools"] == [launch_train._PREFLIGHT_TOOL]
    assert FakeClient.seen[0][2]["extra_body"] == Role.extra_body
    assert FakeClient.seen[0][2]["api_key"] == "secret"
    wire_data = [(messages, kwargs) for messages, kwargs, _ in FakeClient.seen]
    assert "secret" not in str(wire_data)

    with pytest.raises(RuntimeError, match="thinking mode"):
        asyncio.run(
            run(
                completion(calls=[good_call], reasoning_content="hidden thinking"),
            )
        )
    with pytest.raises(RuntimeError, match="malformed tool call"):
        asyncio.run(
            run(
                completion(
                    calls=[
                        ToolCall(
                            id="call_2",
                            name="wrong",
                            arguments={},
                            raw_arguments="{}",
                        )
                    ]
                )
            )
        )
    with pytest.raises(RuntimeError, match="not exact PREFLIGHT_OK"):
        asyncio.run(run(completion(calls=[good_call]), completion(content="done")))
    with pytest.raises(RuntimeError, match="not exact PREFLIGHT_OK"):
        asyncio.run(
            run(
                completion(calls=[good_call]),
                completion(content="analysis then PREFLIGHT_OK"),
            )
        )


def test_dynamic_dry_run_has_no_network_or_filesystem_side_effects(tmp_path, monkeypatch):
    _dynamic_cfg(tmp_path, monkeypatch)
    config_path = tmp_path / "dynamic.yaml"
    run_name = "dry-unique"
    run_dir = launch_train.ROOT / "runs" / run_name
    assert not run_dir.exists()
    called = []
    monkeypatch.setattr(launch_train, "preflight_researcher", lambda cfg: called.append(1))
    monkeypatch.setattr("sys.argv", ["launch_train.py", "--config", str(config_path), "--run-name", run_name, "--dry-run"])
    assert launch_train.main() == 0
    assert called == [] and not run_dir.exists()


def test_preflight_missing_auth_fails_before_network(tmp_path, monkeypatch):
    cfg = _dynamic_cfg(tmp_path, monkeypatch)
    cfg.roles["researcher"].api_key = ""
    called = []
    monkeypatch.setattr(launch_train.urllib.request, "urlopen", lambda *a, **k: called.append(1))
    import pytest
    with pytest.raises(SystemExit, match="unset or empty"):
        launch_train.preflight_researcher(cfg)
    assert called == []


def test_static_doc_preflight_is_lightweight_and_strict(tmp_path):
    import json
    import pytest
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps({'task_id': 't', 'doc': 'd'}) + '\n')
    launch_train.preflight_static_doc_table(str(good))
    for name, content in (('empty', ''), ('bad', 'not-json\n'), ('missing', json.dumps({'doc': 'd'}) + '\n')):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(content)
        with pytest.raises(SystemExit):
            launch_train.preflight_static_doc_table(str(path))


def test_dry_run_still_rejects_objective_passthrough(tmp_path, monkeypatch):
    _dynamic_cfg(tmp_path, monkeypatch)
    config_path = tmp_path / "dynamic.yaml"
    monkeypatch.setattr("sys.argv", ["launch_train.py", "--config", str(config_path), "--dry-run", "algorithm.adv_estimator=grpo"])
    import pytest
    with pytest.raises(SystemExit, match="bypasses the fingerprint"):
        launch_train.main()


def test_v33_alpha_schedule_params_always_forwarded(tmp_path):
    # v3.3 M2:α 三参恒透传(legacy 默认值时 trainer 分支不读,行为零漂移;
    # asym 档由同一通道生效——单一传参路径防"配置了但没送到")。
    cmd = _cmd(tmp_path)
    assert "+algorithm.steered_adv_weight=1.0" in cmd
    assert "+algorithm.steered_adv_decay_steps=0" in cmd
    assert "+algorithm.steered_adv_cap=0.0" in cmd


def test_fused_kernels_default_off_and_flows_through(tmp_path):
    # 默认关:命令里不得出现任何 fused 开关(既有 run 的命令面逐字节不变)。
    cmd = _cmd(tmp_path)
    assert not any("use_fused_kernels" in c for c in cmd)
    assert not any("fused_kernel_options" in c for c in cmd)
    # 开:两参成对出现,backend 走 fused_kernel_options.impl_backend(verl
    # transformer_impl.py:295-298 的实际消费路径)。
    cmd = _cmd(tmp_path, train_extra=", use_fused_kernels: true")
    assert "actor_rollout_ref.model.use_fused_kernels=True" in cmd
    assert "+actor_rollout_ref.model.fused_kernel_options.impl_backend=torch" in cmd
    # backend 枚举校验在加载期失败(不是起跑后)。
    import pytest
    with pytest.raises(ValueError, match="fused_kernels_backend"):
        _cmd(tmp_path, train_extra=", use_fused_kernels: true, fused_kernels_backend: cuda")
