# AppWorld 任务环境引入方案(自旧引入调整完善)

状态:定稿实施(2026-08-07)。第四个 benchmark;旧引入 = skills_evolve
`css/envs/appworld/`(2026-07-04/05 谈判定稿的设计,深度考古后继承+按最新
契约调整)。

## 0. 旧引入考古结论(继承什么)

- **worker 子进程架构(硬约束,原样继承)**:AppWorld unified mode 每 OS
  进程只能一个 world(freezegun 进程级 mock 时间,官方文档约束),每
  episode 一个 worker;appworld 需 **Python ≥3.11**,worker 解释器独立配置
  (`worker_python`,与 ALFWorld 同款模式);JSON lines 协议
  (execute/evaluate/close)移植。**RAM 账:每 world ~300-500MB**——并发
  上限是内存约束(engine_slots)。
- **划分(旧 2026-07-04 定,继承)**:train 90 / dev 57 / test_normal 168 /
  **test_challenge 417 永不触碰**。
- **prompt 设计(旧谈判定稿,继承其原则)**:OFFICIAL-ONBOARDING 风格——
  只教 REPL 协议、三个 api_docs 元 API、完成契约;API 文档由 agent 运行时
  自己探索(探索是被训练的行为);官方 worked example(Spotify 密码)内嵌;
  单代码块响应契约;GT 防火墙(难度/答案零泄漏)。
- 评估:worker evaluate → {success, report(per-requirement passes/
  failures)};旧口径 hard=success(TGC)。

## 1. 对最新契约的调整点(与前三环境的差异及理由)

1. **交互协议 = fenced Python code block(react 第三形态)**:AppWorld 的
   代码就是动作本身(每轮必然写代码),FC 的 JSON 包装对多行代码是纯转义
   噪声;官方/社区标准形态即裸代码块,4B 先验最强。react 解析
   ```` ```python ... ``` ````(最后一个块;块外文本=thought);
   `tools()=None`。**终局内生**:代码调 `apis.supervisor.complete_task()`
   → worker 返回 completed=True → done+评估,无需 submit 工具。
2. **REPL 有状态(与 SSB/math 的无状态 run_python 相反)**:变量跨轮保留
   (notebook 语义)——scaffold 明示;worker 进程即状态容器。
3. **score = requirement 通过比例**(completed 时 evaluate;won=success=
   TGC 二元):部分分抽象的第三个实例;未 completed(步数/token 预算尽)
   score=0(与官方 TGC 精神一致——必须声明完成)。
4. **预算适配(旧 50 步 → 默认 30 步)**:旧引入是 API 客户端形态无总
   token 预算;本机制 response_length=32768 是硬预算,50 步 × (生成+长
   观察) 必然溢出。默认 30 步 + obs 截断 3000 chars(旧裁定"不截断"在
   token 流形态下不可持续);截断率入监控,按数据回调。
5. **gt_text=None 起步**:solution code 是过程性参考(ALFWorld walkthrough
   同型),但现行全线 gt=off;要开时走 build 捕获再议。GT 防火墙纪律照旧。
6. **重放**:fresh worker + 逐 execute 重放(freezegun 固定时钟,官方可
   复现性保证)→ v4 前缀续写成立。

## 2. 数据与装配

- `build_dataset`:读 `$APPWORLD_ROOT/data/datasets/{train,dev,
  test_normal}.txt`,每任务开 worker 捕获 instruction/supervisor 元信息/
  difficulty → initial_obs(确定性锚点);train.parquet=90,valid.parquet=
  dev+test_normal(`data_source=appworld/{dev,test_normal}` 双桶曲线);
  task_type=difficulty 桶(旧实现同款,per-difficulty 曲线免费)。
- `AppworldCfg`:appworld_root(env `APPWORLD_ROOT`)/ worker_python(env
  `APPWORLD_WORKER_PYTHON`)/ engine_slots(默认 64,RAM-bound,per-worker
  折算)/ start_timeout_s / exec_timeout_s / obs_max_chars。
- 服务器前置(CD-006):py3.11 venv 装 appworld + `appworld install`
  数据;真引擎冒烟(本地无 appworld,测试用 stub worker 锁协议)。

## 3. 红线与测试

- 评估信息零反馈(evaluate 的 report/success 不进 executor 流;completed
  的反馈是 execute 输出本身);难度/GT 零进 executor(旧防火墙);
- stub-worker 测试:driver 协议(spawn/ready/execute/evaluate/close/超时)、
  react 代码块解析(最后块/无块 reminder/completed 语义)、score 口径、
  截断、契约面;真引擎端到端归服务器冒烟。
