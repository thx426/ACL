# 论文主实验总控:经典三域 × 双模型 × PGS(单臂)

> 2026-08-23 用户三裁定:①主实验 = **Qwen2.5-3B/7B × ALFWorld/SearchQA/WebShop 共 6 格,只跑 PGS 机制臂**;
> GRPO 及其他 baseline **直接引用 SEED 论文报告数字**(arXiv 2607.14777,本项目立项即以其为对拍目标,
> setting 同源);②**维持现训练协议**(ALF 444 步 16×8 / WS 200 步 32×16 / SQA ~2650 步 128×8),
> 主表脚注与 SEED 的预算差异;③主表评测口径 = **我方 avg@3 @ temp0.6**,脚注 SEED 口径差异。
> 全 6 格按统一协议复跑(旧 run 全部退火 bug 版,降级为辅助证据)。

## 1. 六格矩阵与状态(2026-08-23 建;WS7B 2026-08-25 收官)

> **终值数字单一真源 = `docs/classic-main-results.md`**(结果台账,每格含口径/路径/SEED 对拍)。

| 格 | 配置(configs/classic/) | 指纹 | 协议 | 状态 |
|---|---|---|---|---|
| ALF3B | alfworld_3b_pgs.yaml | 88a3c7172848 | 16×8,444 步(2ep),decay470 | 待发 |
| ALF7B | alfworld_7b_pgs.yaml | e914221e53da | 同上 | 待发 |
| SQA3B | searchqa_3b_pgs.yaml | de57cfcca310 | 128×8,~2650 步(2ep),decay2800 | 待发(格新建 2026-08-23) |
| SQA7B | searchqa_7b_pgs.yaml | 47c6734f849b | 同上 | 待发 |
| WS3B | webshop_3b_pgs.yaml | 0a2c352725c3 | 32×16,200 步(1ep),decay210 | 待发 |
| WS7B | webshop_7b_pgs.yaml(162 变体 `_162`) | 463b32932ef9(162: f2bb626e6f2d,仅宿主路径异) | 同上 | ✅ **完成** 2026-08-25(200/200 步)**Succ 83.78 / Score 89.49**,超 SEED 本尊 Succ +5.7pp;数字与路径见 [结果台账](classic-main-results.md#ws7b--webshop--qwen25-7b-instruct--pgs-) |

统一机制参数(全 6 格同):`pgs_asym_grpo` + 熵门 v3.4 自标定(theta_init 0 / ratchet)+
steered_fraction 0.5 + decay=总步数×1.05 + **epsilon 0.0**(2026-08-17 裁定全线关探针;
旧 run 均 0.05,复跑口径变化点之一)+ w_beh off + lr 1e-6 constant + Dr.GRPO(norm_adv_by_std false,无 KL)。
运维统一:max_ckpt_to_keep=1(硬纪律)/ test_freq=save_freq=20 / avg@3@0.6 /
35B 端点一律 `127.0.0.1:8880-8882`(A800 本机直连;162 宿主经反向隧道同地址,yaml 零改)。

## 2. SEED 引用口径对拍(主表脚注清单)

SEED(2607.14777)Table 1 提供全部 6 格的 baseline:Vanilla/Skill-Prompt*/OPSD/GRPO/
Skill-GRPO(*)/GRPO+OPSD/Skill-SD/RLSD/SDAR/Seed。锚点(Avg):

| | ALF(seen macro) | SQA(7集 macro) | WS Score/Succ |
|---|---|---|---|
| 3B GRPO / Seed | 75.0 / 91.8 | 36.4 / 45.7 | 79.8-63.3 / 88.5-78.9 |
| 7B GRPO / Seed | 81.2 / 96.1 | 42.0 / 48.6 | 80.9-72.6 / 89.7-78.1 |
| 附录 Unseen 3B(GRPO/Seed) | 70.9 / 86.2 | — | — |

**已对齐**(立项时即对着 SEED 搭):交互步数 ALF30/WS15/SQA4、lr 1e-6、ALF batch16×8、
SQA batch128×8、SQA 数据源(Search-R1 processed)、WS AgentGym 栈、文本标签协议+one-shot。

**差异(全部进主表脚注)**:
1. 训练预算:SEED 全线 150 updates;我们 444/200/~2650(用户裁定维持)。
2. WS batch:我们 32×16(每步 512 eps)vs SEED 16×8(128 eps)。
3. 评测:我们 avg@3@temp0.6;SEED temp0.4 单次。
4. 测试集:ALF 我们 valid_seen+valid_unseen 双报 vs SEED eval 128;WS 我们
   test 500 goals vs SEED 标准 128 tasks;SQA 我们 val_shuf2048(七源混采固定)vs SEED
   test 全量按七集分项。→ 终评须补分项/子集读数(§4)。
5. SEED 有 SFT 冷启动(1440 条 GLM-5.2 标注 skill 轨迹);我们零 SFT 纯 RL(对我方有利,明写)。
6. KL:SEED 0.01,我们无 KL(Dr.GRPO 底座);SEED lr warmup 0.1,我们 constant。

## 3. 发车手册(宿主无关;2026-08-23 起统一)

**同一份配置在 A800 / 162 发车零改动**:yaml 内一切宿主路径为 `${TRACE_MODELS}/${TRACE_DATA}`
插值(config `_expand` 于 load 时展开);宿主字面量只存在于 `scripts/classic/host_{a800,162}.env`
(树/env 入口/venv python/模型与数据根/ALF 资产);发射器自动探测宿主并注入。
`_162` 变体 yaml 自此废弃(在跑 WS7B 的现场副本除外,服役至 run 完成)。

统一发射器(库内 `scripts/classic/`,两机部署于 `$TRACE_TREE/scripts/classic/`):
```
launch_classic.sh <run_name> <cfg_stem> <arm> <cvd> <sglport> <raytag>
# 例:  launch_classic.sh classic_alf3b_pgs alfworld_3b pgs 4,5,6,7 34000 alf3b
# 干跑: DRY_RUN=1 launch_classic.sh ...(不写盘不占卡;预检+指纹全链照走)
# 宿主: 自动探测;TRACE_HOST=a800|162 可显式覆盖
# 看护: RUN_PREFIX=classic_alf3b CFG_STEM=alfworld_3b TOTAL_STEPS=444 ARMS=pgs \
#        CVD_PGS=4,5,6,7 PORT_PGS=34000 setsid bash watchdog_single.sh &
#        (watchdog 重启统一走 launch_classic.sh,发车逻辑单一真源)
```
内置预检(fail-loud,dry 也全走):模型/数据路径(插值展开后存在性)→ 35B 三端点
**真实往返**(127.0.0.1:8880-8882,两宿主同地址:A800 直连/162 经隧道)→ 域预检
(alfworld*=资产存在性;webshop*=逐分片 /meta+option_fix+槽位;searchqa*=retriever
真实查询)→ GPU 清场门(>500MiB 拒发)。

双机验收记录(2026-08-23):A800 DRY alfworld_3b → 指纹 88a3c7172848 ✓;
162 DRY webshop_7b → 指纹 **f2bb626e6f2d(与在跑 run 逐位一致)**+ 32 分片/2048 槽位 ✓。

发车前人工项(发射器管不到的):
- [ ] `SWANLAB_API_KEY` export 真值(库内脱敏,缺失即拒发)
- [ ] 162 宿主:A800 侧 tunnel_guard 在位(35B 预检失败时先查隧道)
- [ ] 162 首发 ALF 格前:ALF 数据+worker venv 落地并填 host_162.env(现留空,硬门拒发)
- [ ] resume 用同 run_name(指纹守门 FATAL = 保护,绝不 --set 绕过)
- [ ] 跨宿主指纹不同(路径展开进指纹)= 预期;单 run 一生只在一台宿主 resume

### 3.1 SQA 发车专项(拓扑硬约束)

retriever(wiki-18 e5 faiss,GPU 服务)历史部署在 162 四卡;当前 162 四卡被 WS7B 训练占用,
35B 已迁 A800。**SQA 两格必须等 162 有卡跑 retriever**(或届时另行裁定 retriever 宿主)。
yaml 的 `retriever_url` 仍写 162:8000,发车前:
- [ ] retriever 服务真实往返(非 /health;批查询压测,防"200≠可用")
- [ ] 35B researcher+narrator 并发档核对(researcher 128 / narrator 512,SQA 负载 11×)
- [ ] 墙钟预期与用户对齐:~2650 步为 WS 的 13×,旧 7B run 曾在 ~237 步得监控读数 48.6;
      全程一次 val(2048×3)≈ 15 分钟级

## 4. 终评规格(主表数字的产出方式)

主口径 = 训练内 val 曲线的 avg@3@0.6(mean@3),主表取**终点步**读数;另需离线终评批补:
1. ALF:by-family(Pick/Look/Clean/Heat/Cool/Pick2)分项 + 6-family macro-avg,seen 与 unseen
   双表(与 SEED Table 1/Table 7 同构;我们 val 上报的是 micro,macro 须由轨迹重聚合)。
2. SQA:训练内 val = **val_eq320**(七源等额分层:每源 320、bamboogle 全量 125,共 2045;
   2026-08-23 用户裁定取代旧 val_shuf2048——旧集按比例抽,bamboogle 仅 4 条/musique 108,
   小源分项曲线纯噪声,且比例抽样与 per-source 等权 macro 口径失配;生成脚本
   scripts/classic/make_sqa_val7.py,seed42)。verl 按 data_source 自动分列七源曲线,
   macro=等权均值。**主表数字 = 终评跑 test.parquet 全量 51,713 条七源分项**(SEED 同口径),
   训练内 val_eq320 只作监控曲线。
3. WS:Score 与 Succ 双指标;另跑 SEED 同款 128-task 子集读数(供脚注直接可比)。
4. 附带 SEED 口径组(temp0.4 单次)作对拍参考,成本低(离线)。
终评脚本:待第一格跑完前落地(evalm/ 入口扩分项聚合;本文件先立规格)。

## 4.9 GPU 与宿主方案(2026-08-26 用户裁定,后续所有格适用)

**一律按 WS7B 这一格的用法**:训练在 **162 卡 0-3**,35B researcher 留 **A800**
(三端点 8880-8882),162 经 A800 发起的反向隧道以同地址访问 —— yaml 零改。
理由:A800 当前无可用训练卡(卡 1-3 = 我方 35B 服务;卡 0 与 4-7 被同学的
27B 服务与 VAGEN 训练占用,协调无期),162 四卡(2×A800-80G + 2×A100-80G)
是唯一确定可用的训练资源。

**共卡**:SQA 两格的 retriever 与训练同占 162 四卡(retriever fp16 分片 ~10G/卡)。
① retriever **必须先起**;② 发车带 `GPU_BASELINE_MIB=12000` 放行(默认 500);
③ `rollout_gpu_mem` 不进指纹,余量不足时可降。

**162 资产状态**(2026-08-26 落地):ALFWorld 数据 2.1G + worker venv(py3.10.20 /
alfworld 0.4.2,与 A800 逐包同版本,真实 episode 冒烟已过)✓;SQA
train_trainer + val_eq320(md5 与 A800 一致)✓;WebShop 服务本地 ✓;
Qwen2.5-**3B/7B 均在 `/data3/wushang/model/Qwen/`**(3B 首分片 sha256 与 A800 一致;
`/data4/zxx/` 只有 7B 且是他人目录不可写)。

## 5. 资源排程(事件驱动;2026-08-23 时点)

现况:162 卡0-3 = WS7B(在跑,投影 1.5-2 天);A800 卡0-3 = 35B×3 端点;A800 卡4-7 = 同学 VAGEN 占用。

| 触发事件 | 动作 |
|---|---|
| A800 卡4-7 归还 | 发 ALF3B(或 ALF7B)@A800 卡4-7(ALF 资产全在 A800,零依赖) |
| WS7B 跑完,162 释放 | 优先 WS3B@162(webshop 服务本地;需派生 webshop_3b_pgs_162.yaml + 确认 162 有 3B 模型);ALF 若尚有格未发亦可上 162(需搬 ALF 数据与 worker venv,成本更高,权衡) |
| ALF/WS 全清,A800 整机可用 | SQA7B→SQA3B @A800(researcher 本机直连);retriever 回 162 四卡 |
| 每格跑完 | 归档 val 全量扫描(probe/val_scan_all.py)+ 终评批(§4)+ 更新本表状态 |

## 6. 变更台账

- 2026-08-23(三):SQA val 抽样方案重做 —— val_eq320 七源等额(动机与规则见 §4.2);
  指纹不变(val_files 不进指纹,实测);A800 已生成并 DRY 验证(retriever 预检如设计拦截,
  服务未启);162 yaml 已同步、数据未生成(届时脚本一键复现)。
- 2026-08-23(二):宿主无关化 —— 6 格 yaml 路径参数化(${TRACE_MODELS}/${TRACE_DATA},
  双语境指纹逐位复现:A800 全 6 格不变、162 复现 f2bb626e6f2d);host_{a800,162}.env 宿主档案;
  launch_classic.sh 统一发射器(吸收三域全部实战预检);watchdog_single.sh 宿主化且重启改走
  统一发射器;旧发射器 5 件入 attic;webshop_7b_pgs_162.yaml 标 DEPRECATED。双机 DRY_RUN 验收绿。
- 2026-08-23:创建;SQA3B yaml 新建;ALF7B 修 avg@3(漏改)与 max_ckpt 3→1(违纪);
  全线 epsilon→0.0;A800 版 WS 重试窗 8/15s(429 事故同步);端点统一 127.0.0.1×3;
  发射器 6 件收编 scripts/classic/(SWANLAB key 脱敏)。6 格指纹经 load_config 复核与本表一致。
