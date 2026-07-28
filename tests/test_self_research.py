"""Self-research channel (docs/self-research-design.md): config contract,
launcher wiring, teacher admission, the token-mode researcher chat, and the
NUMERIC truth of the research PG term executed against the VENDORED verl tree
(stub-exec technique from test_kl_modes_numeric.py — the math under test is
the shipped math, not a transcription)."""
import ast
import asyncio
import sys
import types
from pathlib import Path

import pytest

from trace_opd.common.config import load_config, objective_fingerprint
from trace_opd.train.research_group import select_best_doc
from trace_opd.train.backends import EngineResearcherTokenChat, ResearchBudgetExceeded
from trace_opd.train.curve import build_curve_metrics

_ROOT = Path(__file__).resolve().parents[1]
_VERL = _ROOT / "verl" / "verl"


def _cfg(**research_rl):
    overrides = [f"research_rl.{k}={v}" for k, v in research_rl.items()]
    if research_rl.get("weight", 0):
        overrides += [
            "research.backend=engine",
            "research.tool_protocol=text",
            # the template pins the EXTERNAL researcher's sidecar; the engine
            # form has no external server to drive
            "roles.researcher.sidecar_sleep=false",
        ]
    return load_config(_ROOT / "configs" / "trace.yaml", overrides)


# ------------------------------------------------------------------ config

def test_channel_is_off_by_default_and_ships_nothing_into_the_fingerprint():
    cfg = _cfg()
    assert cfg.research_rl.weight == 0.0
    # The legacy material must stay byte-stable with the channel off: three
    # production arms have this fingerprint in their manifests.
    base = objective_fingerprint(cfg)
    cfg.research_rl.n_docs = 4      # dormant knobs must not touch the hash
    cfg.research_rl.eval_k = 2
    cfg.research_rl.update_mode = "separate"
    assert objective_fingerprint(cfg) == base


def test_enabling_the_channel_requires_the_engine_text_researcher():
    with pytest.raises(ValueError, match="on-policy"):
        load_config(_ROOT / "configs" / "trace.yaml", ["research_rl.weight=1.0"])


def test_group_needs_at_least_two_documents():
    with pytest.raises(ValueError, match="n_docs"):
        _cfg(weight=1.0, n_docs=1)


def test_update_mode_is_a_closed_enum():
    with pytest.raises(ValueError, match="update_mode"):
        _cfg(weight=1.0, update_mode="interleaved")


def test_fingerprint_separates_the_channel_and_its_shape():
    off = objective_fingerprint(_cfg())
    on = _cfg(weight=1.0)
    fp_on = objective_fingerprint(on)
    assert fp_on != off
    on.research_rl.update_mode = "separate"
    assert objective_fingerprint(on) != fp_on
    on.research_rl.update_mode = "joint"
    on.research_rl.n_docs = 4
    assert objective_fingerprint(on) != fp_on


# ----------------------------------------------------------------- launcher

def test_launcher_ships_nothing_with_the_channel_off(tmp_path):
    from scripts.launch_train import build_command

    cfg = _cfg()
    cfg.train.model_path = "m"
    cfg.train.train_file = "t"
    cmd = build_command(cfg, tmp_path)
    joined = " ".join(cmd)
    assert "research_loss_coef" not in joined
    n = next(a for a in cmd if a.startswith("actor_rollout_ref.rollout.n="))
    assert n.endswith(f"={cfg.train.rollout_n}")  # sft off, research off


def test_launcher_ships_the_channel_and_extends_rollout_n(tmp_path):
    from scripts.launch_train import build_command

    cfg = _cfg(weight=0.5, n_docs=8)
    cfg.train.model_path = "m"
    cfg.train.train_file = "t"
    cmd = build_command(cfg, tmp_path)
    joined = " ".join(cmd)
    assert "+distillation.distillation_loss.research_loss_coef=0.5" in joined
    assert "+distillation.distillation_loss.research_update_mode=joint" in joined
    assert "+distillation.distillation_loss.research_adv_clip=2.0" in joined
    assert "+distillation.distillation_loss.research_warmup_steps=30" in joined
    n = next(a for a in cmd if a.startswith("actor_rollout_ref.rollout.n="))
    assert n.endswith(f"={cfg.train.rollout_n + 8}")


def test_launcher_rejects_a_role_split_that_leaves_a_dp_rank_short(tmp_path):
    from scripts.launch_train import build_command

    cfg = _cfg(weight=0.5, n_docs=3)  # 16 tasks x 3 docs = 48; n_gpus=4 -> ok
    cfg.train.model_path = "m"
    cfg.train.train_file = "t"
    cfg.train.n_gpus = 5              # 48 % 5 != 0
    with pytest.raises(SystemExit, match="divisible"):
        build_command(cfg, tmp_path)


# -------------------------------------------------------- teacher admission

def test_teacher_admission_argmax_short_tie_and_zero_gate():
    a = {"doc": "long document here", "p": 0.5}
    b = {"doc": "short", "p": 0.5}
    c = {"doc": "best but rare", "p": 0.75}
    assert select_best_doc([a, b, c]) is c          # argmax p
    assert select_best_doc([a, b]) is b             # tie -> shorter doc
    assert select_best_doc([{"doc": "d", "p": 0.0}]) is None   # p=0: no teacher
    assert select_best_doc([{"doc": None, "p": 1.0}]) is None  # no doc: never
    assert select_best_doc([]) is None


# ------------------------------------------------------------------- curve

def test_curve_keys_are_homogeneous_across_roles_under_the_research_flag():
    research_row = {
        "env_steps": 0, "invalid_format": 0, "truncated": 0,
        "research_rl_role": 1, "research_rl_reward": 0.75,
    }
    bare_row = {"env_steps": 5, "invalid_format": 0, "truncated": 0}
    kw = dict(validate=False, research_enabled=True, research_rl=True)
    assert set(build_curve_metrics(fields=research_row, **kw)) == set(
        build_curve_metrics(fields=bare_row, **kw)
    )
    # and the flag off must not leak any research_rl_* key
    off = build_curve_metrics(fields=bare_row, validate=False, research_enabled=True)
    assert not [k for k in off if k.startswith("research_rl_")]


# ------------------------------------------------- token-mode researcher chat

class _FakeTok:
    """Deterministic toy tokenizer: one token per character codepoint."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class _FakeDelta:
    post_obs = [1, 2]

    def after_generation(self, gen_ids):
        return [3]


class _FakeOut:
    def __init__(self, ids):
        self.token_ids = ids
        self.stop_reason = None


class _FakeServer:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.prompts.append(list(prompt_ids))
        return _FakeOut(self.replies.pop(0))


def _mk_chat(server, budget=10_000):
    import trace_opd.train.backends as backends

    chat = EngineResearcherTokenChat(
        server, _FakeTok(), _FakeDelta(),
        temperature=0.7, max_tokens=64, max_sample_tokens=budget,
        request_id="t",
    )
    # render_ids is monkey-free: the fake tokenizer has no chat template, so
    # patch the module-level renderer to a flat deterministic encoding.
    backends.render_ids, chat._render = (
        lambda tok, messages, add_gen: [ord(c) for m in messages for c in m["content"]],
        backends.render_ids,
    )
    return chat, backends


def test_token_chat_streams_engine_ids_verbatim_and_masks_the_glue():
    server = _FakeServer([[65, 66], [67]])
    chat, backends = _mk_chat(server)
    try:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "k"}]
        r1 = asyncio.run(chat(messages, None))
        messages += [
            {"role": "assistant", "content": r1.text},
            {"role": "user", "content": "obs"},
            {"role": "user", "content": "wrap"},
        ]
        asyncio.run(chat(messages, None))
        prompt, resp, mask = chat.token_sample()
        assert prompt == [ord("s"), ord("k")]
        # gen1 | glue + merged("obs\n\nwrap") + post_obs | gen2
        merged = [ord(c) for c in "obs\n\nwrap"]
        assert resp == [65, 66] + [3] + merged + [1, 2] + [67]
        assert mask == [1, 1] + [0] * (1 + len(merged) + 2) + [1]
        # the engine saw exactly prompt + stream-so-far
        assert server.prompts[1] == prompt + [65, 66, 3] + merged + [1, 2]
    finally:
        backends.render_ids = chat._render


def test_token_chat_fails_loud_when_the_archive_forks_from_the_sample():
    server = _FakeServer([[65]])
    chat, backends = _mk_chat(server)
    try:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "k"}]
        asyncio.run(chat(messages, None))
        messages += [
            {"role": "assistant", "content": "REWRITTEN"},
            {"role": "user", "content": "obs"},
        ]
        with pytest.raises(RuntimeError, match="differs from the engine generation"):
            asyncio.run(chat(messages, None))
    finally:
        backends.render_ids = chat._render


def test_token_chat_enforces_the_sample_budget():
    server = _FakeServer([[65] * 60])
    chat, backends = _mk_chat(server, budget=70)
    try:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "k"}]
        asyncio.run(chat(messages, None))  # 2 prompt + 60 gen = 62; budget 70
        messages += [
            {"role": "assistant", "content": chat._last_gen_text},
            {"role": "user", "content": "obs"},
        ]
        with pytest.raises(ResearchBudgetExceeded):
            asyncio.run(chat(messages, None))  # 62 + glue(1)+3+post(2) = 68 -> <64 left
    finally:
        backends.render_ids = chat._render


# ------------------------------------- research PG term (vendored, numeric)

@pytest.fixture()
def research_pg():
    """ast-extract the shipped research_pg_loss with its light deps stubbed."""
    torch = pytest.importorskip("torch")
    src = (_VERL / "trainer" / "distillation" / "losses.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "research_pg_loss"
    )

    class _Metric:
        def __init__(self, value, aggregation=None):
            self.value = value

    ns = {
        "torch": torch,
        "Metric": _Metric,
        "AggregationType": types.SimpleNamespace(SUM="sum"),
        "no_padding_2_padding": lambda log_probs, data: log_probs,
        "TensorDict": dict,
        "Any": object,
        "tuple": tuple,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "vendored_research_pg", "exec"), ns)
    return ns["research_pg_loss"], torch


class _LossCfg:
    research_loss_coef = 1.0
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2


class _Data(dict):
    """dict with .get/.keys like the TensorDict surface the function touches."""


def _mk_data(torch, mask, adv, old_lp, t_global):
    return _Data(
        research_mask=mask, research_advantage=adv,
        old_log_probs=old_lp, research_num_tokens=t_global,
    )


def test_research_pg_matches_hand_computed_reinforce_at_ratio_one(research_pg):
    fn, torch = research_pg
    lp = torch.tensor([[-1.0, -2.0, -0.5], [-0.3, -0.7, -1.5]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    adv = torch.tensor([[0.5, 0.5, 0.5], [-0.25, -0.25, -0.25]])
    data = _mk_data(torch, mask, adv, lp.detach().clone(), t_global=4.0)
    loss, metrics = fn(_LossCfg(), {"log_probs": lp}, data)
    # ratio == 1 -> loss = -sum(A * mask)/T = -(0.5+0.5-0.25-0.25)/4
    assert torch.allclose(loss, torch.tensor(-(0.5 + 0.5 - 0.25 - 0.25) / 4.0))
    loss.backward()
    # d/dlp of -(A*ratio)/T at ratio=1: -A/T on masked tokens, 0 elsewhere
    want = -(adv * mask) / 4.0
    assert torch.allclose(lp.grad, want)
    assert metrics["research/tokens"].value.item() == 4


def test_research_pg_is_additive_across_micro_batches(research_pg):
    """The packing-invariance contract: numerator/T_global per micro-batch,
    summed over micro-batches, equals the single-batch value."""
    fn, torch = research_pg
    g = torch.Generator().manual_seed(0)
    lp = torch.randn(4, 5, generator=g)
    old = lp + 0.05 * torch.randn(4, 5, generator=g)
    mask = (torch.rand(4, 5, generator=g) > 0.4).int()
    adv = torch.randn(4, 1, generator=g).expand(4, 5).contiguous()
    t = float(mask.sum())
    whole, _ = fn(_LossCfg(), {"log_probs": lp}, _mk_data(torch, mask, adv, old, t))
    parts = []
    for rows in ([0, 1], [2, 3]):
        part, _ = fn(
            _LossCfg(), {"log_probs": lp[rows]},
            _mk_data(torch, mask[rows], adv[rows], old[rows], t),
        )
        parts.append(part)
    assert torch.allclose(whole, sum(parts), atol=1e-6)


def test_research_pg_clips_the_stale_ratio(research_pg):
    fn, torch = research_pg
    lp = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-2.0]])            # ratio = e^2 ~ 7.39 >> 1.2
    mask = torch.tensor([[1]])
    adv = torch.tensor([[1.0]])
    data = _mk_data(torch, mask, adv, old, 1.0)
    loss, metrics = fn(_LossCfg(), {"log_probs": lp}, data)
    # positive advantage + over-high ratio -> min picks the clipped branch
    assert torch.allclose(loss, torch.tensor(-1.2))
    loss.backward()
    assert torch.allclose(lp.grad, torch.tensor([[0.0]]))  # clipped: no gradient
    assert metrics["research/ratio_clipped_frac"] == pytest.approx(1.0)


def test_research_pg_fails_loud_without_its_fields(research_pg):
    fn, torch = research_pg
    lp = torch.zeros(1, 2)
    with pytest.raises(RuntimeError, match="research_mask"):
        fn(_LossCfg(), {"log_probs": lp}, _Data(old_log_probs=lp))
    with pytest.raises(RuntimeError, match="research_num_tokens"):
        fn(
            _LossCfg(), {"log_probs": lp},
            _Data(
                research_mask=torch.ones(1, 2, dtype=torch.int64),
                research_advantage=torch.zeros(1, 2),
                old_log_probs=lp,
            ),
        )
