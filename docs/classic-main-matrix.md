# 论文主实验总控:经典三域 × 双模型 × PGS(单臂)

> 2026-08-23 用户三裁定:①主实验 = **Qwen2.5-3B/7B × ALFWorld/SearchQA/WebShop 共 6 格,只跑 PGS 机制臂**;
> GRPO 及其他 baseline **直接引用 SEED 论文报告数字**(arXiv 2607.14777,本项目立项即以其为对拍目标,
> setting 同源);②**维持现训练协议**(ALF 444 步 16×8 / WS 200 步 32×16 / SQA ~2650 步 128×8),
> 主表脚注与 SEED 的预算差异;③主表评测口径 = **我方 avg@3 @ temp0.6**,脚注 SEED 口径差异。
> 全 6 格按统一协议复跑(旧 run 全部退火 bug 版,降级为辅助证据)。

## 1. 六格矩阵与状态(2026-08-23)

| 格 | 配置(configs/classic/) | 指纹 | 协议 | 状态 |
|---|---|---|---|---|
| ALF3B | alfworld_3b_pgs.yaml | 88a3c7172848 | 16×8,444 步(2ep),decay470 | 待发 |
| ALF7B | alfworld_7b_pgs.yaml | e914221e53da | 同上 | 待发 |
| SQA3B | searchqa_3b_pgs.yaml | de57cfcca310 | 128×8,~2650 步(2ep),decay2800 | 待发(格新建 2026-08-23) |
| SQA7B | searchqa_7b_pgs.yaml | 47c6734f849b | 同上 | 待发 |
| WS3B | webshop_3b_pgs.yaml | 0a2c352725c3 | 32×16,200 步(1ep),decay210 | 待发 |
| WS7B | webshop_7b_pgs.yaml(162 变体 `_162`) | 463b32932ef9(162: f2bb626e6f2d,仅宿主路径异) | 同上 | **在跑**@162 卡0-3(2026-08-23 发车,即本格正式 run) |

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

## 3. 发车手册(A800 通用)

发射器(库内归档 `scripts/classic/`,部署副本在 A800 `$W/probe/`;W=TRACE-OPD-TERM 树):
```
# ALFWorld(含 ALFWORLD_DATA/worker python 资产 export 与存在性预检)
probe/launch_single_alf.sh classic_alf3b_pgs alfworld_3b pgs <CVD> <PORT> alf3b
probe/launch_single_alf.sh classic_alf7b_pgs alfworld_7b pgs <CVD> <PORT> alf7b
# SearchQA(发车前先过 §3.1 拓扑检查)
probe/launch_single.sh classic_sqa3b_pgs searchqa_3b pgs <CVD> <PORT> sqa3b
probe/launch_single.sh classic_sqa7b_pgs searchqa_7b pgs <CVD> <PORT> sqa7b
# WebShop(内置 webshop 分片 /meta 预检)
probe/launch_single_ws.sh classic_ws3b_pgs webshop_3b pgs <CVD> <PORT> ws3b
# 干跑(只算指纹与命令,不写盘):DRY_RUN=1 <同上>
# 看护:probe/watchdog_single.sh(参数化);162 变体见 launch_ws7b_pgs_162.sh/watchdog_ws7b_162.sh
```

发车前检查清单(每格必过):
- [ ] GPU 清场(发射器内置 >500MiB 拦截;残留时先按 exe 路径清理,严禁 pkill 模式匹配)
- [ ] 35B 三端点真实往返(curl completion ×3 端口;162 宿主还须验反向隧道 + A800 tunnel_guard 在位)
- [ ] 域服务:WS=webshop 32 分片 /meta 全绿(池 64);SQA=retriever :8000 真实查询往返(§3.1)
- [ ] 数据/模型路径按宿主核对(A800 = /fs/fast;162 = /data4;162 需 `_162` 变体 yaml)
- [ ] `SWANLAB_API_KEY` 已 export(库内脚本已脱敏,fail-loud)
- [ ] DRY_RUN=1 干跑,确认指纹与 §1 表一致
- [ ] resume 语义:重启用同 run_name(resume_mode auto + 指纹守门;遇指纹不合必须停手)
- [ ] watchdog 布防 + 本地 Monitor 巡检挂载

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
2. SQA:val_shuf2048 按 data_source 拆七集分项 + macro-avg(SEED 同构);混采 micro 同时保留。
3. WS:Score 与 Succ 双指标;另跑 SEED 同款 128-task 子集读数(供脚注直接可比)。
4. 附带 SEED 口径组(temp0.4 单次)作对拍参考,成本低(离线)。
终评脚本:待第一格跑完前落地(evalm/ 入口扩分项聚合;本文件先立规格)。

## 5. 资源排程(事件驱动;2026-08-23 时点)

现况:162 卡0-3 = WS7B(在跑,投影 1.5-2 天);A800 卡0-3 = 35B×3 端点;A800 卡4-7 = 同学 VAGEN 占用。

| 触发事件 | 动作 |
|---|---|
| A800 卡4-7 归还 | 发 ALF3B(或 ALF7B)@A800 卡4-7(ALF 资产全在 A800,零依赖) |
| WS7B 跑完,162 释放 | 优先 WS3B@162(webshop 服务本地;需派生 webshop_3b_pgs_162.yaml + 确认 162 有 3B 模型);ALF 若尚有格未发亦可上 162(需搬 ALF 数据与 worker venv,成本更高,权衡) |
| ALF/WS 全清,A800 整机可用 | SQA7B→SQA3B @A800(researcher 本机直连);retriever 回 162 四卡 |
| 每格跑完 | 归档 val 全量扫描(probe/val_scan_all.py)+ 终评批(§4)+ 更新本表状态 |

## 6. 变更台账

- 2026-08-23:创建;SQA3B yaml 新建;ALF7B 修 avg@3(漏改)与 max_ckpt 3→1(违纪);
  全线 epsilon→0.0;A800 版 WS 重试窗 8/15s(429 事故同步);端点统一 127.0.0.1×3;
  发射器 6 件收编 scripts/classic/(SWANLAB key 脱敏)。6 格指纹经 load_config 复核与本表一致。
