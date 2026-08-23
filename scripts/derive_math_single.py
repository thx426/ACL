#!/usr/bin/env python3
"""从 math_7b_pgs.yaml(TIR 多轮档)派生 MENTOR 单轮档的训练配置。

纪律(沿用 derive_seed_3b.py):每条替换的锚点必须**恰好命中一次**,否则中止
——保住原文件每条超参的裁定注释,并让"派生自哪个档"可复现。

产出:
  configs/classic/math_single_7b_pgs.yaml   PGS 机制臂
  configs/classic/math_single_7b_grpo.yaml  GRPO 基线臂(再从 pgs 派生,
                                            差异仅 run/experiment 名、
                                            adv_estimator、steering.enabled、
                                            agent_task_timeout_s ——
                                            与既有 math_7b_grpo 同款做法)
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
CFG = pathlib.Path("/Users/shangwu/workspace/vibe_coding_projects/work/"
                   "TRACE-OPD/trace_opd-bench/configs/classic")

HEADER = """# Math × Qwen2.5-7B-Instruct — **MENTOR 单轮档** PGS 机制臂
# (2026-08-16 用户裁定;由 math_7b_pgs.yaml 程序化派生,除协议维度外逐键一致)
#
# 对齐 MENTOR(arXiv 2510.04140 / ICLR 2026,代码 commit b1cc4c6 逐行核实):
#   单轮 completion,无工具/无沙箱/无 observation(EasyR1 单轮 RLHF 数据流)
#   system prompt 逐字移植(其 verl/utils/dataset.py:174-176 硬编码,不在
#     jinja 里 —— train.sh 传的 no.jinja 是空模板;论文 Appx C p18 复述)
#   <think></think> + \\boxed{} 两标签;reward = 0.9·acc + 0.1·format
#   单轮 8192 token(其 train.sh data.max_response_length)
#   训练 temperature 1.0 / batch 128 × n8 / lr 1e-6(其 config.yaml + 论文)
#
# 刻意不对齐(用户裁定,记账):基座 Instruct(非 Base)/ 数据 DAPO-17k+六源
#   (非 MATH L3-5+MATH-500,与 TIR 多轮臂同源以便跨协议对照)/ 判分
#   math-verify 子进程(非 mathruler)/ 评测 avg@3 @0.6(非 greedy@1 @0)
#
# 与 math_7b_pgs.yaml(TIR 多轮档)的**全部**差异 = 协议维度本身:
#   protocol text→single, caps 6/8→1/1, 单轮 4096→8192,
#   response 16384→8704, 相应的 token/gpu 维度, + format_weight/reward_from_score
# 机制段(steering/research/narration)与 lr/batch/n/temp 逐键相同 ⇒
# 跨协议对照是单变量的。
"""

# (锚点, 替换, 理由)
SUBS = [
    # ---- 头部注释整体换掉(原注释讲的是 TIR 口径)----
    ("""# Math(DAPO-17k × 六竞赛源)× Qwen2.5-7B-Instruct — PGS 机制臂正式训练草案
# (2026-08-13 冒烟全绿后用户确认为**默认配置**;运行脚本=probe/launch_dual_math7b.sh)
#
# 口径对照(TIR 相关工作,一手核实):
#   ToRL(2503.23383):Qwen2.5-Math base、batch128 × n16、C=1(每响应 1 次
#     工具调用;C=2 +2% 但墙钟 +21%)、temp1、无 KL、reward ±1
#   SimpleTIR(2509.02479):Qwen2.5-7B base、batch512、max 5 交互轮、
#     响应 16384、void-turn 轨迹过滤
#   本 run:batch128 × n8(与 SearchQA/ALFWorld 自洽)、2 epochs
#     (17255/128 ≈ 135 步/epoch ⇒ ≈270 步)、val 全量 1304 六源 greedy@1
#     (CD-010 裁定沿用)、无 KL(Dr.GRPO 底座 IF-04)
""", HEADER, "头部注释:TIR 口径 → MENTOR 单轮口径"),

    ("  name: classic_math7b_pgs",
     "  name: classic_math_single_7b_pgs", "run 名"),

    # ---- episode:多轮 → 单轮 ----
    ("""episode:
  # 预算档(2026-08-13 用户确认):ToRL C=1-2 / SimpleTIR 5 轮 / 探针档 12 步16轮。
  # 训练草案取 6 步 8 轮:比 SimpleTIR 宽(给 RL 长出多轮验算行为的空间),
  # 比探针档省一半最坏长度。探针实测(strat60):轮数中位 1-2、码轮/ep ≤0.6,
  # 6 步在行为分布的 p99 之外。
  max_env_steps: 6
  max_assistant_turns: 8
  max_tokens_per_turn: 4096   # 数学长推理单轮(math.yaml 探针同款)""",
     """episode:
  # 单轮:MENTOR 的 setting 里一次 completion 即终局。config._validate 钉死
  # 1/1(adapter 恒 done=True,更大的 caps 不改变行为却会让配置读起来像多轮)。
  max_env_steps: 1
  max_assistant_turns: 1
  max_tokens_per_turn: 8192   # = MENTOR train.sh 的 data.max_response_length""",
     "episode caps → 单轮 1/1,单轮上限 8192"),

    # ---- math 段:协议 + 格式分 ----
    ("""math:
  protocol: text              # 2026-08-13 用户裁定:文本 TIR(ToRL/SimpleTIR 型)
  python_timeout_s: 30
  exec_concurrency: 32        # ⇒ 2/worker;子进程勿饿死引擎(CPU 纪律)""",
     """math:
  protocol: single            # 2026-08-16 用户裁定:MENTOR 单轮档
  format_weight: 0.1          # MENTOR outcome:format = 9:1(其 compute_score)
  # 单轮无工具 ⇒ run_python 不会被调用;两键留着只为 MathCfg 字段完整,
  # 且 math-verify 判分子进程仍复用 exec_concurrency 这道并发门。
  python_timeout_s: 30
  exec_concurrency: 32""",
     "protocol → single,加 format_weight"),

    # ---- eval:预算 ----
    ("""eval:
  # 16384(SimpleTIR 同款;2026-08-13 用户确认)。探针在 32768 下实测:strat60 截断仅
  # 1/60,轨迹长度分布远低于 16384(math500 全量的 token 分布见探针报告)。
  # 旧 math.yaml 注释"16384 撞墙"是 FC 档 12 步的行为,文本档不复现。
  response_budget: 16384""",
     """eval:
  # 单轮 ⇒ 总预算 = 单轮上限 8192 + slack(episode.py 的 budget 检查扣 64;
  # 512 余量保证首轮能满额生成 8192)。
  response_budget: 8704""",
     "response_budget 16384 → 8704"),

    ("  experiment_name: classic_math7b_pgs",
     "  experiment_name: classic_math_single_7b_pgs", "experiment 名"),

    # ---- train:长度维度 ----
    ("  response_length: 16384      # 与 eval.response_budget 同源(已确认)",
     """  response_length: 8704       # 与 eval.response_budget 同源
  # format_weight>0 的前置条件:二值通道会把「答对但格式错」(0.9)判成
  # reward 0,比 MENTOR 更严 ⇒ _validate 在 weight>0 而此键 false 时 fail loud。
  # 开启后 won(=score>=1.0)退化为观察指标,曲线看 rollout_extra/score。
  reward_from_score: true""",
     "response_length 16384 → 8704,并挂上 reward_from_score"),

    ("  max_token_len_per_gpu: 20480       # = prompt 4096 + response 16384",
     "  max_token_len_per_gpu: 12800       # = prompt 4096 + response 8704",
     "max_token_len_per_gpu 随总长缩短"),

    ("  rollout_max_model_len: 20480",
     "  rollout_max_model_len: 12800", "rollout_max_model_len 随总长缩短"),

    ("  rollout_max_num_seqs: 256   # 序列比 SearchQA 长(16k vs 8k),KV 预算减半持平",
     "  rollout_max_num_seqs: 512   # 单轮序列短一半(8.7k vs 16k),KV 预算回到 512",
     "rollout_max_num_seqs 随序列缩短回调"),
]

GRPO_SUBS = [
    ("# Math × Qwen2.5-7B-Instruct — **MENTOR 单轮档** PGS 机制臂",
     "# Math × Qwen2.5-7B-Instruct — **MENTOR 单轮档** GRPO 基线臂", "标题"),
    ("  name: classic_math_single_7b_pgs",
     "  name: classic_math_single_7b_grpo", "run 名"),
    ("  experiment_name: classic_math_single_7b_pgs",
     "  experiment_name: classic_math_single_7b_grpo", "experiment 名"),
    ("  n_gpus: 4                   # 机制臂 CVD 0-3",
     "  n_gpus: 4                   # 基线臂 CVD 4-7", "卡组注释"),
    ("  adv_estimator: pgs_asym_grpo", "  adv_estimator: grpo", "advantage 口径"),
    ("  agent_task_timeout_s: 5400  # > research.session_timeout_s(2400)",
     "  agent_task_timeout_s: 1800  # 无 researcher,纯 rollout", "任务超时"),
    ("  enabled: true\n  doc_source: researcher", "  enabled: false\n  doc_source: researcher",
     "steering 关闭"),
]


def apply(text: str, subs: list[tuple[str, str, str]], label: str) -> str:
    for old, new, why in subs:
        n = text.count(old)
        if n != 1:
            sys.exit(f"FATAL [{label}]: 锚点命中 {n} 次(需恰好 1)——{why}\n"
                     f"  锚点前 90 字: {old[:90]!r}")
        text = text.replace(old, new)
        print(f"  ✓ {why}")
    return text


def main() -> int:
    src = (CFG / "math_7b_pgs.yaml").read_text(encoding="utf-8")
    print("[1/2] math_7b_pgs.yaml -> math_single_7b_pgs.yaml")
    pgs = apply(src, SUBS, "pgs")
    (CFG / "math_single_7b_pgs.yaml").write_text(pgs, encoding="utf-8")

    print("[2/2] math_single_7b_pgs.yaml -> math_single_7b_grpo.yaml")
    grpo = apply(pgs, GRPO_SUBS, "grpo")
    # 基线臂不需要 research/narration 段,但留着是无害的(steering.enabled=false
    # 时 agent_loop 不会起 researcher);与既有 math_7b_grpo.yaml 的做法一致。
    (CFG / "math_single_7b_grpo.yaml").write_text(grpo, encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
