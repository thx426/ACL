# GT-doc 教师消融实验(ALFWorld 专属验证;2026-08-03 用户裁定)

> **验证假设 H_doc:researcher 产出文档的内容/形态是 OPD 病理的作用变量,
> 而非 OPD 结构本身。** 方法:教师特权信息从"researcher 现产经验文档"换成
> "任务 ground-truth 参考解"(最正确、零叙述、零随机),其余一切——注入协议
> (GUIDANCE_HEADER)、教师打分、OPD、训练配置——逐字节不动。
> 这是**针对 ALFWorld 的特定验证试验,不是通用方案机制**;实现按最小旁路
> 原则隔离(见 §3),勿在此基础上扩展通用能力。

## 1. 判读逻辑(预注册)

与 B0(researcher-doc 同底座)构成 doc 消融对,2×2:

| | G0 健康 | G0 崩 |
|---|---|---|
| **B0 崩** | H_doc 成立:researcher doc 是病因 | 病在 OPD×冷启动结构,doc 内容无关 |
| **B0 不崩** | 病在 won 分割/tg(回 T1′/G1 线) | 反直觉强信号:完美信息更毒 ⇒ 信息不对称机理的硬证据 |

附带产出:**"完美特权信息下 OPD 的上限"锚点**(headroom 读数,此前没有);
以及 **GT 注入 K=8 成功率**(4B 拿着逐字可执行解照做的能力,独立读数,
离线 doc_lift 口径量)。

**宏变量绑定(诚实预注册)**:"doc 来源"一次切换同时改变四件事——内容正确
性(恒真 vs 可错)、形态(纯命令序列 vs 自然语言叙述)、长度(~100–200
token vs 数百)、动态性(静态 vs 逐步适配策略弱点)。四者绑定不可再分,这
正是"researcher 整个环节"作为一个消融单元;若 GT 臂胜出,细分(如 GT+叙述
包装中间臂)是后续消融。

## 2. 臂定义

- **G0** = 冷启动(`sft_4b_coldstart_v1/hf/step_6`)+ 纯 rkl 全轨迹 OPD +
  `research.static_doc_source: "gt"`。与 B0 唯一差异 = doc 来源。
- **G1(可选,等 G0×B0 出结果再定)** = GT 版 t1(+ `mix_tg_weight` +
  `opd_lost_only`),直接判 doc 对 T1 崩塌形态的贡献。
- **成本红利**:GT 臂无 researcher(t1_cs6 实测 gen 占整步 77%、其中
  research episodes 占解码量 68–76%)⇒ 单步 ~19 分 → 预计 5–7 分,30 步
  约 3–4 小时(vs B0 16–21h)——全桌最便宜的一跑,且不需要 35B 隧道。

## 3. 实现(最小旁路,默认关断;commit 见 git log)

- `research.static_doc_source: null | "gt"`(默认 null = 逐字节现状)。
  校验:与 `research.gt: "on"` 互斥(那是喂 researcher 的开关)、与
  research_rl 互斥(静态源下无 researcher 可训)、要求 research.enabled
  (doc 注入管线);非 null **条件式进 fingerprint**(教师信息源 =
  objective 语义,拒跨臂 resume)。
- `agent_loop._research` 入口短路(~8 行,延迟 import,不背 ALFWorld 依赖):
  `doc = render_gt_as_doc(adapter.gt_text())`,fields = `static_doc_fields`
  (键集与真实会话路径一致,曲线键批一致纪律;
  **`research_final_success` 在静态源下恒 0 无实测语义,判读忽略该曲线**)。
- `envs/alfworld/gt.py::render_gt_as_doc`(ALFWorld 专属模块):**只认引擎
  walkthrough 形态**(playable commands,实例编号与环境一致);high_pddl
  高层计划(`GotoLocation(shelf)`,无实例编号不可执行)返回 None 走既有
  nodoc 降级。文档措辞最小化、不加执行指令(保持与 researcher 文档同一
  guidance 注入姿态,不引入语气变量):

  ```
  A verified step-by-step reference solution for this task:
  1. go to shelf 2
  2. take alarmclock 1 from shelf 2
  ...
  ```

- 零 verl 改动;tg 兼容自动获得(tg 行 doc 前缀走同一管线,G1 零额外实现)。

## 4. 起跑前预检(地基,`scripts/gt_doc_precheck.py`)

1. **普查**(离线,只读 parquet):gt_text 覆盖率、形态构成(walkthrough /
   high_pddl / 其他)、渲染成功率、doc 长度分布。**walkthrough 渲染成功率
   <95% ⇒ 用 build_datasets 的 walkthrough 通道重建数据**,不许带高层计划
   形态起跑(nodoc 降级行会稀释消融)。
2. **重放验证**(`--replay N --config <yaml>`,需环境,训练机上跑):抽 N
   个任务真实环境逐条执行 walkthrough,成功率必须 ~100%——GT 不重放验证
   等于拿假 GT 做消融。

## 5. 观察面

全套既有产物照用(val 配对、won 分层、重复率/熵/grad_norm 坍缩三信号、
批 1 修复后的 obs 段分解);外加:GT 注入 K=8 成功率(离线 doc_lift 口径,
教师上限读数)。判停与自基线原则照 runbook v3。
