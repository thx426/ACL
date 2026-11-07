# TRACE-OPD — 正式机制(CD-005)

Researcher-in-the-loop on-policy self-distillation:每个训练 step 的 batch 任务,
researcher agent 现场做真实实验产出该任务的经验文档 → 文档作特权信息构成
self-teacher → 对 student 裸 on-policy rollout 做标准 OPD 蒸馏内化。按 step
迭代,与 RL 训练循环同构。

- 设计真源:`docs/design.md`
- 配置真源(全部超参数,带注释):`configs/trace.yaml`
- 可选的错误定位—局部修订—独立验证：[Skill 演化说明](docs/skill-evolution.md)
- verl:仓库外 `verl/` 克隆的 **trace-opd 分支**(h3-pilot 5 patch + validate 传递 patch)

## 布局

    src/trace_opd/
      core/       契约 SSOT(protocol/types/template_delta)+ 统一 EpisodeRunner
      envs/alfworld/  TaskAdapter 实现(worker 子进程、GT 提取)
      research/   researcher 会话(session/tools/prompts/toolproto 双协议)
      narrate/    叙述管线(preprocess/factcard/assertions/narrator)
      docstore/   研究归档(只写不回读)
      evalm/      独立评测器(token-ids greedy,bare/+doc 臂)
      train/      verl 集成(TraceAgentLoop、backends、build_datasets)
    train/        agent_loop_config.yaml(verl 注册)
    scripts/      launch_train.py(yaml → verl 命令,单真源)
    configs/      trace.yaml
    tests/        单测(mock env/llm/tokenizer,本地可跑)

## 快速开始(服务器)

```bash
# 1) 数据(一次性;GT 预提取内含)
python -m trace_opd.train.build_datasets \
    --data-root $ALFWORLD_DATA --worker-python $ALFWORLD_WORKER_PYTHON \
    --out-dir trace_opd/data/out            # 冒烟加 --n-train 32

# 2) researcher 服务(任一机器,OpenAI-compat;例:27B TP2)
#    export TRACE_RESEARCHER_ENDPOINT=http://<host>:<port>/v1
#    export TRACE_RESEARCHER_MODEL=Qwen3.6-27B

# 3) 训练(所有参数出自 configs/trace.yaml)
python trace_opd/scripts/launch_train.py --config trace_opd/configs/trace.yaml [--dry-run]

# 4) 独立评测(base 复测 / 训后 checkpoint 正式数字)
python -m trace_opd.evalm.run_eval --config trace_opd/configs/trace.yaml \
    --model-path <hf_dir> --api-base http://127.0.0.1:8501/v1 --served-model <name> \
    --parquet .../valid_seen.parquet .../valid_unseen.parquet --arm bare --out-dir <out>
```
