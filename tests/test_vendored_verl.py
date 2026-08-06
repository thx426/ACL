"""Vendored-verl export integrity — every fact PGS relies on must hold in the
exported tree ON ITS ACTUAL EXECUTION PATH.

Lesson from the f8f26d5 incident (CD-006, 2026-07-24): a patch sat only in the
V0 trainer (ray_trainer.py) while training actually runs the V1 trainer
(TaskRunnerV1 -> v1/trainer_base.py), so it silently never executed. These
assertions grep the vendored files the launcher puts on PYTHONPATH, so a stale
export or a patch landing on a dead path fails loudly here.

PGS(docs/pgs-final-spec.md)对 verl 的依赖面分三组:
  1. 训练面:原生 ppo_loss + vanilla policy loss 的 rollout_is_weights 通道
     (W_beh 乘在 PPO clip 之外 —— NF02 分离式校正,verl 原生即正确形态)。
  2. 采样面:vLLM server 的 logprobs / prompt_logprobs 通道(SPS verify 用)。
  3. 根除面:旧蒸馏机制(teacher/distillation/mixed-objective)必须不存在
     ——半死不活的残留分支比干净的缺失更危险。
"""
from pathlib import Path

_VERL = Path(__file__).resolve().parents[1] / "verl" / "verl"


def _read(rel):
    return (_VERL / rel).read_text(encoding="utf-8")


# --------------------------------------- 通用基建守护(继承,均有事故背书)

def test_rollout_extra_curves_on_v1_trainer_path():
    # e8f697c: import + call site must both be in the V1 trainer, which is
    # what training executes (V0 keeps its copy for parity, not sufficient).
    src = _read("trainer/ppo/v1/trainer_base.py")
    assert src.count("compute_rollout_extra_metrics") >= 2
    assert "def compute_rollout_extra_metrics" in _read("trainer/ppo/metric_utils.py")  # f8f26d5


def test_micro_batch_budget_is_packing_target_not_assert():
    # 6d77dc7: the old hard assert would kill the whole run on one long-tail
    # trajectory; the export must carry the clamp+warning form.
    assert "below longest sequence" in _read("utils/seqlen_balancing.py")


def test_packing_budget_not_raised_to_longest_sequence():
    # c48568e (R1, amends 6d77dc7): raising the budget shrinks
    # num_micro_batches for the whole batch (~1.3x longest-sequence peak, the
    # step-5 OOM amplifier). The raise line must stay gone.
    src = _read("utils/seqlen_balancing.py")
    assert "max_token_len = int(max_seq_len)" not in src
    assert "packing budget stays" in src


def test_wake_up_hands_cache_back_before_resume_weights():
    # c48568e (R2): resume weights must be preceded by aggressive_empty_cache,
    # or vLLM's CuMemAllocator cannot get physical pages (cumem OOM, step-5).
    src = _read("workers/engine_workers.py")
    before, _, after = src.partition("# 1. resume rollout memory")
    assert after, "resume-weights anchor comment disappeared — re-check R2"
    assert "aggressive_empty_cache(force_sync=True)" in before.rsplit("def ", 1)[-1]


def test_agent_loop_hang_guard_on_tq_path():
    # 58cd07e (R3): per-prompt wall-clock timeout wraps the gather on the TQ
    # path (the executed one); a permanent vLLM hang must cost one prompt,
    # not deadlock the run.
    src = _read("trainer/ppo/v1/agent_loop_tq.py")
    assert "TRACE_AGENT_TASK_TIMEOUT_S" in src
    assert "asyncio.wait_for(asyncio.gather(*tasks)" in src


def test_identity_temperature_division_is_skipped():
    # f88db6c (617817c): logits / 1.0 allocated a second full-size fp32 tensor
    # (13.5GiB at 22k tokens) — the step-16 OOM trigger.
    src = _read("workers/engine/fsdp/transformer_impl.py")
    assert "logits = logits / temperature.clamp" not in src
    assert "logits = logits / _temp" in src


# --------------------------------------- PGS 训练面:W_beh 通道

def test_ppo_loss_carries_rollout_is_weights_channel():
    # PGS W_beh(π_old/b 行为校正)进训练的唯一通道:batch 带
    # rollout_is_weights 字段 ⇒ 原生 ppo_loss 选取并传给 policy_loss_fn。
    src = _read("workers/utils/losses.py")
    assert 'if "rollout_is_weights" in data:' in src
    assert 'fields.append("rollout_is_weights")' in src
    assert "rollout_is_weights=rollout_is_weights" in src


def test_ppo_loss_reports_wbeh_consumption():
    # 运行时对账通道:rollout 侧有 steer 的 step,训练端 off1_frac 必须 >0;
    # 该指标缺席 = W_beh 静默丢失类故障(IF-09)无法在线暴露。
    src = _read("workers/utils/losses.py")
    assert "actor/rollout_is_weight_mean" in src
    assert "actor/rollout_is_weight_off1_frac" in src


def test_vanilla_loss_multiplies_weights_outside_the_clip():
    # NF02 分离式校正的实现事实:W 乘在 dual-clip 之后的 pg_losses 上,
    # 绝不能折进 ratio 再被 clip(那会在最需要校正处零梯度)。
    src = _read("trainer/ppo/core_algos.py")
    i = src.index("def compute_policy_loss_vanilla")
    body = src[i:]
    body = body[: body.index("\ndef ") if "\ndef " in body else len(body)]
    assert "pg_losses = pg_losses * rollout_is_weights" in body
    # ratio 本身必须仍是 π_θ/π_old(log_prob - old_log_prob),与 W 无关
    assert "negative_approx_kl = log_prob - old_log_prob" in body


def test_grpo_loo_estimator_registered():
    # 规范 §2.6 条件 iv:baseline+scale LOO(scale 带全组 std floor,封
    # 二元 reward 孤例 std_loo=0 的爆炸 edge)。经 registry 走 else 分支。
    src = _read("trainer/ppo/core_algos.py")
    assert '@register_adv_est("grpo_loo")' in src
    assert "torch.maximum(std_loo, std_full)" in src


def test_wbeh_expansion_wired_on_the_tq_put_path():
    # PGS W_beh:agent_loop_tq 在 TQ put 前把稀疏 steering ledger 展开成
    # 稠密 rollout_is_weights(唯一写入点),并与原生 rollout_correction 互斥。
    tq = _read("trainer/ppo/v1/agent_loop_tq.py")
    assert "expand_behavior_weights" in tq
    assert 'field["rollout_is_weights"]' in tq
    # 互斥判定必须是「真开启」口径:hydra 默认把 rollout_correction 物化成
    # 全 null 子键的 mapping,`is not None` 会误伤一切默认配置(codex 审阅 P0)。
    assert 'corr.get("rollout_is") or corr.get("rollout_rs")' in tq
    # IF-09:判定口径=「协议在场」(steer_count 恒在),空账展开全 1 ——
    # 批内 schema 恒一致,混合批不丢权重字段。
    assert '"steer_count" in extra' in tq
    sw = _read("trainer/ppo/steering_weights.py")
    assert "def expand_behavior_weights" in sw
    assert "per_decision" in sw and "vtrace" in sw


def test_arm_split_session_identity_reaches_the_agent_loop():
    # 分臂采样(v3.1)依赖 verl TQ 层逐副本传 session_id,且基类 **kwargs
    # 透传进 agent_loop.run() —— 任一环断掉,fraction<1 的 run 会在首个
    # rollout fail loud(arms.is_steered_arm),这里把链锁在导出树上。
    tq = _read("trainer/ppo/v1/agent_loop_tq.py")
    assert "session_id=i" in tq
    base = (_VERL / "experimental" / "agent_loop" / "agent_loop.py").read_text(
        encoding="utf-8"
    )
    i = base.index("async def _run_agent_loop")
    body = base[i : i + 3000]
    assert "await agent_loop.run(" in body and "**kwargs," in body


def test_rollout_extra_metrics_emit_per_arm_rates():
    # 分臂曲线:聚合端必须产出 bare/steered 条件率与 gap(won/mean 已不可
    # 作为能力读数,这三条是去混杂后的替代读数)。
    src = _read("trainer/ppo/metric_utils.py")
    for key in ("won_bare_rate", "won_steered_rate", "steer_arm_gap"):
        assert key in src, key


def test_engine_workers_loss_is_plain_ppo():
    # 训练 loss 挂接点:无蒸馏分支,恒 ppo_loss。
    src = _read("workers/engine_workers.py")
    assert "self.loss_fn = partial(ppo_loss, config=actor_config)" in src
    assert "distillation_ppo_loss" not in src


# --------------------------------------- PGS 采样面:SPS 的引擎能力

def test_vllm_server_logprobs_passthrough_exists():
    # SPS draft 侧:sampled-token logprob 通道(现形态仅 0/None 两态 —— P2
    # 若需 top-K 采样分布需扩展此行,风险项记 docs/pgs-impl-design.md)。
    src = _read("workers/rollout/vllm_rollout/vllm_async_server.py")
    assert 'sampling_params["logprobs"] =' in src
    # verify 侧:对给定序列的 per-position top-K 打分(双前缀评分)。
    assert "extract_prompt_logprobs(" in src
    assert "num_prompt_logprobs=sampling_params.prompt_logprobs" in src


def test_token_output_carries_log_probs_field():
    src = _read("workers/rollout/replica.py")
    assert "log_probs: Optional[list[float]]" in src


# --------------------------------------- PGS 根除面:旧机制必须不存在

def test_distillation_machinery_is_fully_removed():
    # 目录级:旧蒸馏包与 teacher 服务不复存在。
    assert not (_VERL / "trainer" / "distillation").exists()
    assert not (_VERL / "experimental" / "teacher_loop").exists()
    assert not (_VERL / "workers" / "config" / "distillation.py").exists()


def test_no_live_references_to_removed_machinery():
    # 引用级:执行路径文件里不残留会炸的 import 或半死分支。
    for rel in [
        "workers/engine_workers.py",
        "workers/utils/padding.py",
        "experimental/agent_loop/agent_loop.py",
        "trainer/ppo/v1/trainer_base.py",
        "trainer/ppo/v1/agent_loop_tq.py",
        "trainer/ppo/ray_trainer.py",
        "trainer/ppo/utils.py",
        "trainer/ppo/padding_utils.py",
        "trainer/main_ppo.py",
        "trainer/main_ppo_v0.py",
        "experimental/fully_async_policy/fully_async_rollouter.py",
        "experimental/fully_async_policy/fully_async_trainer.py",
    ]:
        src = _read(rel)
        assert "trainer.distillation" not in src, rel
        assert "teacher_loop" not in src, rel
        assert "is_distillation_enabled" not in src, rel
        assert "DistillationConfig" not in src, rel
        assert "teacher_logprobs" not in src, rel


def test_distillation_config_surface_is_gone():
    # 审阅 IF-14:hydra 默认组合不得再指向已删除的 Distillation*Config;
    # 生成配置不得继续宣称一个无法实例化的功能。
    assert not (_VERL / "trainer" / "config" / "distillation").exists()
    assert "distillation@distillation" not in _read("trainer/config/ppo_trainer.yaml")
    for gen in (
        "trainer/config/_generated_ppo_trainer.yaml",
        "trainer/config/_generated_ppo_megatron_trainer.yaml",
        "trainer/config/_generated_ppo_veomni_trainer.yaml",
        "trainer/config/_generated_ppo_torchtitan_trainer.yaml",
    ):
        assert "\ndistillation:" not in _read(gen), gen


def test_engine_impls_have_no_distillation_branches():
    # 审阅 IF-14:engine 层半死分支(default=False 永不执行,但会诱导误接回
    # 教师 loss)必须出清。
    for rel in (
        "workers/engine/fsdp/transformer_impl.py",
        "workers/engine/megatron/transformer_impl.py",
        "workers/engine/veomni/transformer_impl.py",
    ):
        src = _read(rel)
        assert "distillation_use_topk" not in src, rel
        assert "distillation_only" not in src, rel
        assert "teacher_topk" not in src, rel
    consts = _read("trainer/constants_ppo.py")
    assert "TRACE_DISTILL" not in consts
    assert "TRACE_TEACHER_SCORING_BACKEND" not in consts
    assert "TRACE_OBS_DIR" not in consts


def test_trainer_base_has_no_sample_role_machinery():
    # 旧 mixed-objective 的 GRPO 分组特判必须整体消失:PGS 的每个任务组
    # 全部是 bare student rollout,标准 GRPO 分组即正确语义。
    src = _read("trainer/ppo/v1/trainer_base.py")
    assert "sample_role" not in src
    assert "_mixed_objective_enabled" not in src
    assert "research_advantage" not in src
    tq = _read("trainer/ppo/v1/agent_loop_tq.py")
    assert "sft_mask" not in tq and "tg_mask" not in tq
