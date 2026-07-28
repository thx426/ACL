# Researcher-SFT 离线采样 pipeline(sftgen)

> 2026-07-28,CD-005。用户需求裁定:冷启动数据**不取**训练过程的 research 归档,
> 用专门的离线 pipeline 采样——成分、协议、执行者版本全部可控;且结构必须
> benchmark 可扩展。实现:`src/trace_opd/sftgen/`,入口
> `scripts/sample_researcher_sft.py`,导出 `python -m trace_opd.sftgen.export`。

## 形态(一图)

```
train parquet ──按 seed 分层采 T 任务──►  每任务 M 个 researcher 会话
   (与训练同源)   (strata.py, per-benchmark)      │
                                                researcher = 35B-A3B(roles.researcher,
                                                  标准 OpenAI fncall 面 → 任意采样模型可换)
                                                executor episodes = 初始 policy
                                                  (Qwen3-4B-Instruct-2507 vLLM 服务,
                                                   evalm 的 token 级 GenerateFn,协议同构)
                                                narrator = 同一 policy 服务(关思考)
                                                  │
              doc utility 评测(eval_k 带doc + bare_k 裸,机制温度)──► gain 标注
                                                  │
              归档 tasks/<task>/s<i>/{session,doc,evals,meta,DONE}(断点续跑幂等)
                                                  │
              export.py:fncall→text 确定性转换 + 过滤 ──► verl multiturn-SFT parquet
                                                            (messages 列)
```

## 三条用户裁定与其实现

1. **采样模型通用性**:采样侧走标准 OpenAI function-calling(roles.researcher,
   任何兼容端点即插);policy 训练用的 text 协议由 `convert.py` 在**导出时**做
   确定性转换——round-trip 测试钉死(转换后的 assistant 轮经 TextProtocol.extract
   重新解析 == 原始 ToolCall;system prompt == TextProtocol.prepare 的运行时输出)。
   research.tool_protocol=text 时原生直采、导出恒等透传。
2. **gt 单一配置键**:`research.gt` 是 SFT 采样与 self-evolve 训练共读的同一个键
   (语义:只控制参考解是否进 researcher prompt;run_experiment 的成败反馈是环境
   信号,始终存在)。per-benchmark 的 setting 改这一处,两侧自动一致。
3. **模型定版**:初始 policy = Qwen3-4B-Instruct-2507(executor+narrator 共用一个
   vLLM 服务);采样模型 = 35B-A3B 复用现有 roles.researcher 端点。

## 关键设计点

- **分布同构三件套**:任务(同一 train parquet)/ 输入(同一 research 配置节:
  n_exp/max_turns/doc_max_words/gt)/ 反馈(executor=初始 policy ⇒ researcher 看到
  的实验反馈分布 = 训练启动时的真实分布)。SFT 数据是 executor-conditioned 的——
  换执行者尺寸(如 1.7B)理论上应各自采一批,后续按需加采。
- **评测内嵌只标不筛**:gain 标注留给 export 的 `--min-gain`;归档全保 ⇒ 换过滤
  条件 = 重导出,不重采。summary.json(finalize 率/增益分布)= **M0 探针的 35B 侧
  基线**,冷启动后 policy 用 evalm 同口径对标。
- **评测口径**:机制温度(research.episode_temperature=1.0)、K=8——直接对标训练时
  reward p_i,刻意不用 evalm 的 greedy 口径。
- **benchmark 扩展** = envs/<new>/ 实现 adapter + train parquet + strata.py 注册
  一个分层函数;pipeline 本体零改动。ALFWorld 分层键 = task_id 前缀(六类任务型)。
- **多 tool-call 轮**:转换保真渲染全部(不做行为编辑);`--drop-multi-call`
  作为导出过滤选项(runtime 只执行第一个 call,单 call 是更干净的教学行为)。

## 用法(服务器)

```bash
# 1) 起初始 policy 服务(127.0.0.1;executor+narrator 共用)
# 2) export 训练同款的 researcher 端点环境变量(35B)
PYTHONPATH=src python3 scripts/sample_researcher_sft.py \
    --config configs/trace.yaml --out-dir /abs/runs/sftgen_v1 \
    --executor-endpoint http://127.0.0.1:8311/v1 \
    --executor-model Qwen3-4B-Instruct-2507 \
    --tasks 32                        # 首跑 32 任务标定(成本/协议错误率/finalize)
# 全量:--tasks 按用户制定;断点续跑直接重跑同命令(DONE 幂等)
PYTHONPATH=src python3 -m trace_opd.sftgen.export \
    --archive-dir /abs/runs/sftgen_v1 --out /abs/runs/sftgen_v1/sft.parquet \
    --min-gain 0.0                    # 过滤条件后议;不过滤先看分布
# --dry-run 只打印分层任务清单(不需要任何服务)
```

## 挂账

- verl SFT 训练配置(fsdp_sft_trainer + multiturn messages_key)与训练脚本:M0
  下一环,pipeline 产物 schema 已对齐(messages 列)。
- 35B 在 fncall 下的会话质量参照 H1(27B 实测 finalize 100%);text 直采臂与
  gt=on 对照臂按需小批量跑。
