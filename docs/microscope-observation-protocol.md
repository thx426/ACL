# 4B「thinking 膨胀」现象 · 显微镜观察协议

> 状态:协议定稿(2026-07-28,与用户对齐)。观察对象:CD-006 4B 实验中
> "action KL 消减+成功率跃升 → KL 散落 thinking + thinking 显著变长、成功率停滞"
> 的质变现象。工具:蒸馏显微镜(`docs/microscope-design.md`)+ 本次新增的段解析层
> (`src/trace_opd/microscope/segments.py`、`/api/segment_report`)。

## 0. 待判假设(观察前先声明,避免看图说话)

- **H-A(施教面错误论,主假设,用户裁定)**:thinking 段的 KL 是**不可约的**
  (teacher 条件于 doc 的思考措辞,student 原则上不可预测),全轨迹逐 token 施教
  从第一步起就在压这部分噪声;action 信号大时被掩盖,action 对齐后噪声主导化。
  推论:thinking 段 KL 绝对值应**从早期就高位存在**,质变步改变的是份额不是量级。
- **H-B(附加动力学,辅助假设)**:除不可约噪声外,存在随训练**增长**的成分
  (熵膨胀正反馈/结束标记被压)。推论:thinking 段 KL 绝对值、student 熵、长度
  随 step 上升。
- 两者不互斥;方案含义一致(新机制均不再对 thinking 施教),判别价值在于
  理解旧机制现象与预估崩塌时间线。

## 1. 采集前置

- 观察 run 开 `microscope_distributions: true`;质变窗附近 `microscope_dist_every: 1`,
  远离窗口可 10(runbook §11.1 开销账:~300MB/step)。
- 仅 topk 系 loss 模式有观察包;estimator 模式无包(runbook §11 边界)。
- 服务:`PYTHONPATH=src python3 -m trace_opd.microscope serve --obs-dir <run>/obs
  --tokenizer <model_path>`。**段解析必须带 --tokenizer**(标签扫描是文本级)。

## 2. 段解析口径(segments.py)

- `<reasoning>…</reasoning>` / `<action>…</action>` 按字符流扫描映射回 token 区间,
  BPE 任意切分不影响;token 级边界 ±1 容差(远小于观察效应量)。
- 未闭合标签(生成截断)延伸到序列末,不计 closing。
- 聚合一律 `response_mask ∧ covered`;跨行聚合为**行等权均值**(避免单条膨胀行
  以长度支配 step 统计);`kl_share` = 段内 masked KL 和 / 行内 masked KL 总和。
- fp16 往返误差与 `audit` 门禁同 runbook §11.3;先过 audit 再读数。

## 3. 七项观察

主力 API:`GET /api/segment_report?step_lo=<a>&step_hi=<b>&role=bare`
(item 1/3/4/6/7 一次扫出,进程内 memo;首次大窗口查询慢属预期)。
下钻 API:`/api/sample`(现已附 `segments` 区间)、`/api/position`、`/api/leaderboard`。

### 观察 1|段 KL 预算分解(判 H-A vs H-B 的主战场)
- **读数**:每 step 的 `reasoning.kl_mean / kl_share` 与 `action.kl_mean / kl_share`。
- **判读**:
  - action.kl_mean 在质变步陡降、reasoning.kl_mean **早期即高且平稳** → H-A 独立成立;
  - reasoning.kl_mean 随 step **单调上升** → H-B 叠加存在(崩塌前驱,预警等级提高);
  - 同时记录质变步号(action share 崩落点),与成功率跃升步对照(val 波动 ±5pp 意识)。

### 观察 2|action 翻转(质变机理)
- **操作**:质变步前后各取同任务样本(`/api/samples` 按 task 过滤),对 action 段
  首 token 用 `/api/position` 看 student top-1 是否从 bare 偏好翻转为 teacher top-1;
  `/api/leaderboard?mode=verdict_conflict` 圈候选位置。
- **判读**:翻转集中发生 → 质变=离散翻转的聚合表现;无翻转但 KL 降 → 分布整体
  软化收敛(不同机理,记录之)。

### 观察 3|结束标记双侧概率(变长的直接因果路径)
- **读数**:`closing.teacher_lp_mean / student_lp_mean / teacher_outside_topk / n`。
- **判读**:
  - teacher_lp 显著低于 student_lp,或 `teacher_outside_topk` 占比上升 →
    "teacher 压结束标记 → 梯度压低 student 结束概率 → 变长"路径坐实;
  - 双侧接近而长度仍增 → 变长走熵膨胀间接路径(见观察 4)。

### 观察 4|thinking 段 student 熵(熵膨胀是否启动)
- **读数**:`reasoning.student_entropy` 跨 step 序列(对照 `action.student_entropy`)。
- **判读**:持续上升 → h3 崩塌最早前兆已现,预估剩余稳定窗口;平稳 → 膨胀
  另有解释(回观察 3/7)。

### 观察 5|高 KL 位置的 teacher 内容(不可约性的直接证据)
- **操作**:`/api/leaderboard?metric=kl_forward&step_lo=<质变后>` 取 thinking 段
  top 位置(借 `/api/sample` 的 `segments` 区间过滤 pos),`/api/position` 看
  teacher top-64 的 token 文本。
- **判读**:top 集中为 doc 措辞/引用性内容(如指导语复述)→ H-A 的"不可预测
  措辞"实锤,这些位置**不该施教**;若出现低熵、决策性的分歧(teacher 笃定指向
  某个思路词)→ thinking 内存在真信号,新机制的熵门控消融臂值得保留。

### 观察 6|测量伪影排查(先排除,再谈机理)
- **读数**:`reasoning.teacher_mass`(top-64 覆盖率)与 `reasoning.clamp_hit_rate`。
- **判读**:覆盖率低(如 <0.7)→ 部分"高 KL"是截断边缘行为,量级解释打折;
  clamp_hit_rate 高 → KL 显示高但梯度已死,解释"停滞而未即崩";两者都要在
  结论里如实折减效应量。

### 观察 7|won/lost 分层膨胀(膨胀发生在谁身上)
- **读数**:`won.resp_len_mean` vs `lost.resp_len_mean`、两组 `repeat_rate_mean`、
  两组 `reasoning.tokens_mean` 跨 step。
- **判读**:仅 lost 膨胀+repeat 上升 → 挣扎态(重复吸引子早期,与 thought 逐字
  重复率判别指标一致);won 也膨胀 → 分布级熵膨胀,危险等级更高;同时盯
  resp_len 长尾(A800 显存峰值∝最长序列,工程红线)。

## 4. 记录模板

每次观察会话产出一节记录(沉 `docs/observations/`,数字带 step 与 API 参数):

```
## <日期> obs-run=<run_name> steps=<a>-<b>
- 质变步定位:action.kl_share <x>→<y> @ step <s>;val won <…>(±5pp)
- 观察1:reasoning.kl_mean 序列 <…> → 支持 H-A / H-A+H-B
- 观察3:closing teacher−student = <…>,outside_topk <…>%
- 观察4:reasoning.student_entropy <…>
- 观察6:teacher_mass <…> clamp_hit <…> → 效应量折减说明
- 观察7:won/lost len = <…>/<…>,repeat = <…>/<…>
- 结论与下一步:<…>
```

## 5. 待办(不阻塞观察)

- 前端段染色(minimap/热图按 reasoning/action 底色区分)——API 已就绪,界面增强
  后补;当前用 `/api/sample` 的 `segments` 字段+现有热图人工对照。
- 观察 2/5 的批量化(脚本遍历 leaderboard→position)——首轮先人工下钻,固化
  查询模式后脚本化。
