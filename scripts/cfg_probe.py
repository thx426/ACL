#!/usr/bin/env python3
"""配置探针 —— 供启动器门禁调用,一次一项,只往 stdout 打一行结果。

把门禁逻辑从 shell 的 `python -c "..."` 里抽出来:内联 python 经过
ssh/heredoc 多层引号后极易被静默改写(2026-08-08 实测:`load_config('$CFG')`
被引号嵌套吃成 `load_config()`,门禁因此假失败)。

用法: python cfg_probe.py <config.yaml> {fingerprint|perturn|budget}
  datafiles    断言配置实收的 model_path/train_file/val_files 真实存在
  mech         机制档位判词:GRPO | V33_SELFRES | OTHER:<明细>
  runname      打印 run.name(启动器据此定位 run 目录与日志,单一真源)
  fingerprint  打印 objective_fingerprint(加载失败则无输出且退出码非 0)
  perturn      P0 哨兵:episode.max_tokens_per_turn 是否真的注入 protocol
               —— 断言注入后的行为,不是断言配置读到了值(CD-006 2026-08-08:
               该键曾是 dead config,而当时的测试只断言 `cfg.x == v` 故未拦住)
  budget       预算链自洽性:轮预算 <= response_length,ctx 容得下,SP 容量够
"""
import contextlib
import os
import sys

from trace_opd.common import config as C
from trace_opd.core import protocol as P


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    path, what = sys.argv[1], sys.argv[2]
    # load_config 会往 stdout 打预算主导等 info(config.py:530 起)。探针的
    # stdout 是给调用方 shell 做 [ -n "$FP" ] 判定用的,必须只含结果一行,
    # 否则 info 被当成结果(2026-08-08 实测:门禁因此假失败)。info 转 stderr
    # 保留可见性,不吞掉。
    with contextlib.redirect_stdout(sys.stderr):
        cfg = C.load_config(path)

    if what == "datafiles":
        # 断言**配置解析后的实收路径**存在,而不是断言某个环境变量存在。
        # 2026-08-09 事故:配置写 ${SSB_TRAIN_FILE} 而 env 只导出 TRACE_TRAIN_FILE,
        # 展开成空串;启动器当时检的是后者,门全绿却在 launcher 里必填校验失败。
        # 变量名可以变、可以有别名,路径存不存在才是唯一可信的判据。
        t = cfg.train
        bad = []
        for label, path in [("model_path", t.model_path),
                            ("train_file", t.train_file)] + \
                           [(f"val_files[{i}]", v) for i, v in enumerate(t.val_files)]:
            if not path:
                bad.append(f"{label}=<空,变量未展开?>")
            elif not os.path.exists(path):
                bad.append(f"{label}不存在:{path}")
        print("OK" if not bad else "; ".join(bad))
        return 0

    if what == "mech":
        # 机制档位判词。启动器据此断言"臂身份"—— 对照实验的科学性完全建立在
        # 机制确实按预期开/关上,而这一判定不该用 shell 内联 python 表达
        # (heredoc 经多层引号极易被静默改写,2026-08-09 连踩三次)。
        st, t, rc, rrl = cfg.steering, cfg.train, cfg.research, cfg.research_rl
        if (not st.enabled and t.adv_estimator == "grpo_loo"
                and st.adv_mode in ("", "legacy_shared")
                and not rc.enabled and not rrl.enabled):
            print("GRPO")
        elif (st.enabled and t.adv_estimator == "pgs_asym_grpo"
                and st.adv_mode == "asym"            # M1
                and st.gate_mode == "entropy"        # M4 熵门(非 always_on)
                and st.w_beh_mode == "off"           # M3 新默认
                and st.steered_adv_decay_steps > 0   # M2 α 有限期
                and rc.enabled and rc.backend == "engine"
                and rrl.enabled):
            print("V33_SELFRES")
        else:
            print("OTHER:steering=%s est=%s mode=%s gate=%s wbeh=%s decay=%s "
                  "research=%s/%s rrl=%s" % (st.enabled, t.adv_estimator, st.adv_mode,
                  st.gate_mode, st.w_beh_mode, st.steered_adv_decay_steps,
                  rc.enabled, rc.backend, rrl.enabled))
        return 0

    if what == "runname":
        print(cfg.run.name)
        return 0

    if what == "runsroot":
        # run 目录必须由**配置**决定。启动脚本自行拼 runs_root 会让"formal
        # 拒绝覆盖已有 run"这道门检查一个永不存在的路径而恒真(2026-08-09 实测)。
        print(cfg.run.runs_root)
        return 0

    if what == "ssbdata":
        # SSB 专用:断言实收的 train/val 就是**当前清单**过滤出来的那份数据集。
        # 用户裁定 2026-08-09:SpreadsheetBench 只跑 372 个任务,verified-400 与
        # 中途那版 373 一并作废。
        #
        # 判据 = manifest 记录的 excluded task_ids **逐个等于** 当前
        # unreliable_tasks.json 的集合。不看目录名(改名就骗过去)、不看行数
        # (清单一变行数就变)、也不能只看"有没有 unreliable 标记"—— 作废的 373
        # 版同样带标记,只有比对集合本身才能把它区分出来。
        import json as _json
        import pathlib as _pl
        import sys as _sys
        _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "src"))
        from trace_opd.envs.spreadsheetbench.build_dataset import (  # noqa: PLC0415
            load_unreliable)
        want, _doc = load_unreliable()
        files = [cfg.train.train_file, *cfg.train.val_files]
        bad = []
        for f in files:
            mf = _pl.Path(f).parent / "split_manifest.json"
            if not mf.is_file():
                bad.append(f"{_pl.Path(f).parent.name}: 无 split_manifest.json")
                continue
            try:
                doc = _json.loads(mf.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                bad.append(f"{_pl.Path(f).parent.name}: manifest 读取失败 "
                           f"{type(exc).__name__}")
                continue
            mark = doc.get("unreliable_excluded") or doc.get("unreliable_list") or {}
            got = set(map(str, mark.get("task_ids") or []))
            if not got:
                bad.append(f"{_pl.Path(f).parent.name}: 未剔除 reward 恒零任务")
            elif got != want:
                bad.append(
                    f"{_pl.Path(f).parent.name}: 剔除集合与当前清单不符"
                    f"(数据集剔了 {len(got)} 个,清单要求 {len(want)} 个;"
                    f"少剔 {sorted(want - got)[:4]} 多剔 {sorted(got - want)[:4]})")
        print("OK" if not bad else "BAD " + " | ".join(bad))
        return 0

    if what == "fingerprint":
        print(C.objective_fingerprint(cfg))
        return 0

    if what == "perturn":
        e = cfg.episode
        P.configure_episode_caps(
            e.max_env_steps, e.max_assistant_turns, e.max_tokens_per_turn
        )
        print(int(P.GEN_MAX_TOKENS_PER_TURN == e.max_tokens_per_turn))
        return 0

    if what == "budget":
        # config.py 的 _validate 已 fail-closed 覆盖:各角色流预算 <= padding 维度、
        # 流预算+prompt <= 保险丝、research prompt <= prompt 维度。这里只补它不查的。
        t, e, rrl, rc = cfg.train, cfg.episode, cfg.research_rl, cfg.research
        bad = []

        # ① SP 容量必须容得下**实际最长序列**(prompt 维度 + response 维度),
        #    而不是容得下保险丝 —— 保险丝按定义就不该被触及,拿它做分母会
        #    逼出无谓的大 max_token_len_per_gpu(2026-08-09 口径修正)。
        longest = t.max_prompt_length + t.response_length
        if t.max_token_len_per_gpu * t.ulysses_sp < longest:
            bad.append(
                f"SP容量({t.max_token_len_per_gpu}x{t.ulysses_sp}="
                f"{t.max_token_len_per_gpu * t.ulysses_sp})<最长序列({longest})"
            )

        # ② Qwen3.5 级词表不开 fused CE 必爆显存
        if not t.use_fused_kernels:
            bad.append("use_fused_kernels=False(大词表 logits 显存刚需)")

        # ③ 保险丝要名副其实:不比"流预算+prompt"宽裕就退化成了控制器
        if rc.enabled and rc.session_token_budget:
            head = rc.session_token_budget + rc.max_prompt_tokens
            if t.rollout_max_model_len < head * 1.1:
                bad.append(
                    f"保险丝({t.rollout_max_model_len})相对 researcher 流预算+prompt"
                    f"({head})余量<10%,失去「一般不触发」语义"
                )

        # ④ §7.4:显式阈不得小于流预算,否则预算内的正常轨迹被零梯度
        if rrl.enabled and rrl.research_max_sample_tokens and rc.session_token_budget:
            if rrl.research_max_sample_tokens < rc.session_token_budget:
                bad.append(
                    f"research_max_sample_tokens({rrl.research_max_sample_tokens})"
                    f"<session_token_budget({rc.session_token_budget})"
                )

        # ⑤ 训评同构:评测只跑 executor,预算须与 executor 流预算一致
        exec_budget = e.session_token_budget or t.response_length
        if cfg.eval.response_budget != exec_budget:
            bad.append(
                f"eval.response_budget({cfg.eval.response_budget})!="
                f"executor 流预算({exec_budget}),训评不同构"
            )

        print("OK" if not bad else "; ".join(bad))
        return 0

    print(f"unknown probe {what!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
