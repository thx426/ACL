# 蒸馏显微镜:分布观察包 × 工作台(microscope)

> 2026-07-28,CD-005。用户需求:深入直观观察 OPD 在每条训练样本上的实际情况——
> 训练时保存学生/教师在学生轨迹上的输出分布,事后可还原一切指标与直观信息,
> 配交互式前后端系统人工分析找 insight。三轮探讨定稿(用户两次关键修正:
> ①存"最小完备源"、一切视角实时派生(否决落盘侧 top-M 预筛——预判会过时,
> 原始数据不会);②student top-K 与 teacher 对称存 64(熵膨胀时截断 16 的信息
> 损失恰发生在最想观察的时刻))。

## 三层架构

```
交互层  工作台四视图(样本/轨迹显微镜/检索台/演化剖面)— static/ 单页零依赖
派生层  一切指标实时计算(derive.py 纯函数)+ 进程内 memo(可丢弃,重启重算;
        新分析视角 = 改 derive + 重启,历史 run 立即获得)
原始层  观察包(不可变、自包含):verl 训练侧采集,npz per (step, rank)
```

## 原始层:最小完备源(判据=不可再生性)

每训练样本每 response 位置存**四件套**(K=64 对称,fp16,clamp 前 raw):
teacher top-K(ids+lps,数据面转存)/ student **自身** top-K(ids 本为 overlap
诊断已算,lps 唯一新增 gather)/ student 在 teacher 支撑上的 lps(KL 中间量)/
realized 双侧 lp(student 侧转存 old_log_probs;teacher 侧支撑外为 NaN——缺席
本身是信号)。**标量一律不存**(全部可派生),唯一例外 = pre-outer-clamp
per-token 训练 loss 作**审计真值锚**。自包含元数据:token ids、turn 结构
(extra_fields 随行)、task/won/reward/role、loss_mode 与 clamp 参数、sp 段信息。

- 采集点:两个 topk loss 函数内(分布唯一在场处);分发层做 THD 打包流→行
  response 窗(no_padding_2_padding 同窗口)→ ulysses 段交集的坐标学。
- 落盘:缓冲按 step 攒,下一 step 出现即后台 flush(npz 压缩,原子重命名);
  **best-effort 契约**:任何失败自禁 + 一次警告,训练零感知。
- 开关:`train.microscope_distributions: true`(默认关;观察/消融 run 显式开)
  + `microscope_dist_every`。成本 ~465MB/step 原始(4k 均值×128 行,K64 对称),
  压缩 ~300MB;100 step ≈ 30GB——观察 run 量级可承受,生产臂默认关。
- 边界(如实):estimator 模式(k3/gated)student logits 不过 processor,无包;
  K=64 截断外的全词表尾部不可还原(训练本身发生在截断内,"训练做了什么"
  100% 可还原;全词表视角走 P2 checkpoint 重放)。SP>1 按段落盘、读层重组
  (单段有测试;162 SP=2 首用时验证多段)。

## 派生层与审计

`derive.py`:forward/reverse 截断 KL(**与 vendored 训练函数逐位对拍钉死**,
tests/test_microscope.py 的端到端真理链:采集→npz→读取→派生 == vendored loss
== 真值锚)、双侧 mass/熵、overlap、rank 位移、gap、质量外流、分歧类型标签、
重复 n-gram 起点(thought 重复率的 token 级下沉)。离线审计:
`python -m trace_opd.microscope audit --obs-dir <run>/obs`。

## 交互层:工作台

`python -m trace_opd.microscope serve --obs-dir <run>/obs --tokenizer <model>`
(零依赖 http,SSH 隧道访问,默认端口 8643)。四视图:
1. **样本**:step 内样本表,列排序 + role/won 过滤,KL/gap/overlap 聚合列;
2. **轨迹显微镜**:token 流着色(7 种主轨道可切)+ canvas 多轨道 minimap
   (5 轨道并排,IGV 范式,点击跳转)+ turn 分隔 + obs 灰显 + 未覆盖淡显;
   键盘 ←→ 逐 token、n/p 在 95 分位高 KL 位置间跳;顶栏显示 audit 状态;
3. **分布对检查器**(点击任意 token):两侧 top-64 并集合并条形图(teacher 蓝
   / student 橙、realized 金框、rank 标注)、质量外流条、KL 分解与训练 loss
   并显、分歧类型标签、±30 token 上下文;
4. **检索台**:预置四查询(指标 Top / 判决冲突 / 重复 n-gram 起点 / teacher
   弃权)× step 范围 → 位置表 → 一键跳到轨迹对应位置;
5. **演化剖面**:任务级 turn×step 热图(崩塌的"地质剖面"),点格子跳轨迹。
   全视图 URL hash 可寻址(deep-link)。

## 挂账(P2)

checkpoint 重放(全词表视角,限 checkpoint step);持久派生缓存(SQLite,
大 run 首扫加速);token 类型透镜的环境插件化;与 viz 系统互相 deep-link。
