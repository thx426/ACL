# 实验配置:每个实验一份完整副本

这里的每个 `.yaml` 都是**一次实验的完整定义**——由 `config_tool new` 从
`../trace.yaml` 模板逐字复制而来,之后独立演化。

```bash
python -m trace_opd.tools.config_tool new  162-4b-a3    # 复制模板,建新实验
python -m trace_opd.tools.config_tool diff 162-4b-a3    # 与模板比键集
python -m trace_opd.tools.config_tool list              # 所有实验的漂移与可加载性
```

## 为什么是完整副本而不是差异层

一次实验的配置应当**自足**:半年后单独打开就能读懂,能与任何另一个实验直接 diff,
而且**模板日后怎么改都不会追溯性地改变一个已经跑完的实验的含义**。
差异层(base + overlay)做不到最后这条——要理解一份 overlay,必须先还原它当时叠加的
模板版本,而那个版本可能已经被改过很多次了。

代价是漂移:模板新增一个键,已有副本不会自动获得。所以模板同时是**基准**——
`config_tool diff` 与 launcher 启动日志都会把差异说出来,缺一个键意味着这个 run
用的是代码里的默认值,不一定是模板推荐的值。补键是手工粘贴,不做自动改写
(模板的价值大半在注释,解析回写会把注释全丢掉)。

## 命名

`<机器>-<基座>-<臂>`,例如 `162-4b-a0` / `a800-4b-a0` / `162-1p7b-a3`。
**不同消融臂必须用不同的 `run.name`**:臂进 objective fingerprint,同名 resume
会被 launcher 拦下(否则就是拿一个臂的 checkpoint 继续跑另一个臂的目标)。

## 启动

```bash
# A800 脚本按名字找配置
CONFIG_NAME=a800-4b-a0 bash scripts/run_a800_4b.sh
# 或直接调 launcher
python scripts/launch_train.py --config configs/experiments/162-4b-a0.yaml --run-name trace_opd_4b_a0
```

launcher 把配置(加上 `--run-name`、`--set` 与 `${VAR}` 展开)解析**一次**,写成
`<run_dir>/effective_config.yaml`,**训练进程与每个 rollout worker 都读那一份**——
两个进程各自解析同一份配置文件却得到不同结果,是一类很难查的故障。
