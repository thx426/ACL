# 推理标签统一:全线弃用 `<think>` 改 `<reasoning>`(2026-08-16,CD-011)

**用户裁定**:为将来把分布干预的教师换成 35B(thinking-family)铺路,把所有
任务的格式协议、one-shot 示例等一切用到 `<think></think>` 的地方统一换掉。
标签选 **`<reasoning>`**(与 ALFWorld 既有协议一致)。

---

## 1. 为什么必须换(三层冲突,全部实测取证)

取证脚本:`scripts/probe_think_tags.py`、`scripts/probe_strip_scope.py`
(executor `Qwen2.5-7B-Instruct` vs teacher `Qwen3.6-35B-A3B`,A800 CPU)。

### 1.1 ★ 历史 assistant 轮的 `<think>` 块会被**静默剥离**

thinking-family 模板(Qwen3.5/3.6 全系)渲染历史 assistant 轮时,把 content
**截断到最后一个 `</think>` 之后**:

```
输入 assistant content : "<think>SECRET</think>\nact here"
35B 渲染后             : "<|im_start|>assistant\nact here<|im_end|>"
```

剥离的**边界**(决定改造范围):

| 位置 | 是否剥离 |
|---|---|
| assistant **历史**轮 | ★ 剥离 |
| assistant 末轮 / user 轮 / system 轮 | 保留 |

| 形态 | 是否剥离 |
|---|---|
| `<think>X</think>` 完整对 | ★ 剥离 |
| 单独一个 `</think>`(无开标签) | ★ 剥离 |
| 只有开标签 `<think>X` | 保留 |
| `<reasoning>` / `<thought>` | 保留 |

⇒ 多轮任务(SearchQA/ALFWorld/Math-TIR)每一轮的推理都会在下一轮消失。
**researcher 走 user 轮转述轨迹是安全的**(只剥 assistant 历史轮)。

### 1.2 non-thinking 模式 = 预填空 think 块,与协议直接打架

```
Qwen2.5-7B   enable_thinking=False/True → '<|im_start|>assistant\n'  (无 thinking 模板)
Qwen3.6-35B  enable_thinking=False      → '<|im_start|>assistant\n<think>\n\n</think>\n\n'
             enable_thinking=True       → '<|im_start|>assistant\n<think>\n'
```

non-thinking 不是"禁用思考",是**预填一个已闭合的空 think 块**。协议若要求
模型自己写 `<think>`,模型会在一个已闭合的块后再开一个。

### 1.3 同一字面在两族模型里不是同一个东西

| | Qwen2.5-7B | Qwen3.6-35B |
|---|---|---|
| `<think>` | 3 个普通碎片 `['<th','ink','>']` | **特殊 token** id=248068 |
| `<reasoning>` | 4 碎片 `['<','reason','ing','>']` | **同样** 4 碎片,逐 token 一致 |

---

## 2. 候选标签取证与选择

| tag | executor | teacher | 被剥离 |
|---|---|---|---|
| `<think>` | 3 tok | 1 tok(special) | ★ 是 |
| **`<reasoning>`** | 4 tok | 4 tok(同) | 否 |
| `<thought>` | 3 tok | 3 tok(同) | 否 |
| `<scratch>`/`<analysis>`/`<plan>`/`<work>` | 3 tok | 3 tok(同) | 否 |

选 `<reasoning>`:ALFWorld 已在用且实证过(zero-shot 0% → system 内 one-shot
99.9%),全项目统一到一个标签;两词表碎片逐 token 一致。

---

## 3. 实现:单一真源 + 可回退

**真源** `core/protocol.py`:

```python
REASONING_TAG = "reasoning"          # 模块常量,消费方运行时读
def rtag_open() / rtag_close() / reasoning_block_re()
def configure_reasoning_tag(tag)     # 入口注入,重建正则与 DEFAULT_FORMAT_REMINDER
```

**配置** `episode.reasoning_tag: str = "reasoning"`。设成 `"think"` 可回退
(复现历史 run / 与相关工作逐字对拍),会打 WARNING。

**注入点**(与 `configure_thinking` 同期,5 处):`evalm/run_eval.py`、
`train/agent_loop.py`、`train/build_datasets.py`、`envs/preflight.py`、
`envs/scienceworld/build_dataset.py`。

### ★ 一个实现陷阱:必须是调用时求值

含标签的 scaffold **不能是模块常量** —— `from … import SYSTEM_PROMPT_X` 会
把旧值绑定到导入方的名字,`configure_reasoning_tag` 之后不更新。故:

- `searchqa.SYSTEM_PROMPT_TEXT` → `system_prompt_text()`
- `math.SYSTEM_PROMPT_SINGLE` → `system_prompt_single()`
- `math._MENTOR_FMT_RE` → `mentor_format_re()`
- `math/build_dataset._SCAFFOLDS` 的值改成 callable

### 改动面(刻意最小化)

| 改 | 不改 |
|---|---|
| SearchQA text 档(`<think>`→标签) | **ALFWorld**:本就是 `<reasoning>`,与新默认值相同 ⇒ 行为逐字节一致。它**正在跑正式实验**,不为统一而动其代码路径 |
| Math single 档(`<think>`→标签) | Math text/fncall 档:无推理标签 |
| `protocol.DEFAULT_FORMAT_REMINDER`(改为按标签重建,默认值逐字节不变) | 各 benchmark 的动作语法(`<action>`/`<search>`/`<answer>`/`<information>`)—— 那是任务的,不是协议面的 |

### 指纹:按 env 的**历史**值条件追加

```python
_legacy_rtag = {"alfworld": "reasoning", "searchqa": "think", "math": "think"}
if cfg.episode.reasoning_tag != _legacy_rtag.get(cfg.environment.name, "reasoning"):
    material += f"|reasoning_tag={cfg.episode.reasoning_tag}"
```

⇒ ALFWorld 指纹逐字节不变(既有 run 可续跑);SearchQA/Math 换标签后指纹改变
——正是要的(换协议 = 新曲线,旧 ckpt 被 resume 守卫拦下)。

---

## 4. 验证

### 4.1 测试:567 passed, 6 skipped(全量)

- math 18/18(single 用例全部改为按运行时标签构造)
- **新建 `tests/test_searchqa_protocol.py`**(5 用例)—— 该 benchmark 此前
  **零测试覆盖**,改协议时全量测试仍全绿,等于无人守门。现补:scaffold /
  one-shot / thought 抽取三处共用同一标签;动作解析与标签无关;指纹按 env
  历史值追加。
- ★ 两处"切回 think 后与原文逐字一致"的断言(math↔MENTOR、
  searchqa↔Search-R1/SEED):一个断言同时锁住「除标签外没动别的」与
  「回退开关真能复现原文」。

### 4.2 真实 tokenizer 验收(`scripts/verify_no_strip.py`,输入由真实代码导出)

```
=== executor Qwen2.5-7B ===
  tag=reasoning  历史轮推理: 保留 ✓
  tag=think      历史轮推理: 保留 ✓        ← Qwen2.5 不剥,这正是旧依据的来源
=== teacher  Qwen3.6-35B ===
  tag=reasoning  历史轮推理: 保留 ✓        ← 目标达成
  tag=think      历史轮推理: ★ 推理被剥离   ← 要规避的
两档 scaffold 除标签外逐字一致: math True / searchqa True
```

---

## 5. 代价记账(与相关工作的偏离)

1. **SearchQA** 与 Search-R1 / SEED 的 prompt 不再逐字同源(其余措辞未动,
   仍是两家的 `<search>`/`<answer>`/`<information>` 协议与 one-shot 结构)。
2. **Math single** 与 MENTOR 不再逐字同源(这是继判分器/数据/基座/评测之后
   的第 5 处刻意差异)。
   ⚠ 换标签后 Math single 的 zero-shot 风险**上升**:`<reasoning>` 比
   `<think>` 更"自定义",而 MENTOR 原设计就是 zero-shot 无示例。用户裁定
   **先不补 one-shot**,拿到卡先测格式服从率。
3. 两处都可用 `episode.reasoning_tag: think` 一键回退做逐字对拍。
