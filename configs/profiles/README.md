# 配置 profile:从默认模板派生出一次具体实验

`configs/trace.yaml` 是**机制的默认配置模板**——所有键、推荐默认值、以及每个值
背后的理由都在那里,并**随机制演进而更新**。它不描述任何一次具体实验。

一次具体实验 = 模板 + 若干 profile(+ 启动时的 `--set`)。profile **只写与模板的差异**:

```bash
python scripts/launch_train.py --config configs/trace.yaml \
    --profile configs/profiles/machine-162.yaml \
    --profile configs/profiles/arm-a3.yaml \
    --run-name trace_opd_1p7b_a3
```

优先级:模板 < profile(按给出顺序) < `--set`。

## 为什么是这个形状

**模板改进自动到达每个实验**:profile 不重复模板里的值,所以机制侧调整一个默认值
(比如把 `vllm_rpc_timeout_ms` 从 60s 提到 300s),所有 profile 立刻继承,不必逐个改。

**profile 短到能当实验说明读**:它列出的就是"这次实验与默认有什么不同"。

**没有静默覆盖**:profile 里出现模板未定义的键会**报错**——这样打字错误、以及机制
改名后残留的旧键,都会在启动时失败,而不是变成一个没人读的值。这条纪律是有来历的:
启动脚本曾用 `sed` 无条件改写 yaml,谁在配置文件里设了值都会被悄悄改回默认
(`MIX_RL`、`TOKLEN` 两次同型事故)。

**一次解析,处处一致**:launcher 把模板+profile+`--set` 解析成
`<run_dir>/effective_config.yaml`,**训练进程与所有 rollout worker 都读这一份**。
在此之前 worker 会重新读模板,看不到 `--set`——一个用 `--set` 开了 SFT 臂的 run,
trainer 期待专家样本而 worker 一条都不产。manifest 记录派生链与 objective fingerprint。

## 分类

- `machine-*.yaml`:机器差异(卡数、并行度、显存比例、researcher 端点、通信后端)。
  同一份模板要在多台机器上跑,这些**不是**实验设计参数。
- `arm-*.yaml`:消融臂(混合目标权重等)。它们定义 run 在优化什么目标,会进
  objective fingerprint;**每个臂必须用不同的 `--run-name`**,否则 resume 会拿另一个臂的
  checkpoint 继续跑(launcher 会拦住并报错)。
- `smoke-*.yaml`:冒烟/调试(小数据集、少步数、跳过 val)。
