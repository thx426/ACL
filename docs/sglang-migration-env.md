# SGLang 切换 · 162 环境配置需求(CD-006 执行册)

CD-010,2026-08-05。背景:用户裁定全栈 rollout 引擎切换 vLLM→SGLang(动机:
SGLang 原生支持 `logprob_start_len` tail-only input-logprobs + radix cache 前缀
复用,根治 B_∞=400 下 verify 全前缀重算的平方墙钟;vLLM 上游明确不支持且无
计划,自维护 patch 有长期风险)。代码适配由 CD-010 在分支 `pgs-sglang` 推进;
**本文档是环境侧需求,可与代码适配并行执行**。

## 0. 硬约束(先读)

1. **版本钉死:`sglang[srt,openai]==0.5.8`** —— vendored verl 的 setup.py 硬
   要求;verl 启动路径按 sglang 版本分支(<0.5.10 走旧 launch 路径),**不要
   自行升级到其他版本**。
2. **禁止升级 venv_train 的 torch / flash-attn** —— 训练面(FSDP/sp=2/FA2
   2.8.3)已定版。若 pip 解出 sglang 需要拉动 torch 升级 ⇒ 立即止损,改走
   独立 venv 路线(见 §1 方案 B)。
3. **s1 在跑期间不占卡**:§2 的门 1(import/依赖)不占卡可立即做;门 2/3
   (起服)需一张空闲 GPU —— 等 s1 停点或与用户协调卡窗口。
4. LD_PRELOAD 经验预期可迁移:sglang/flashinfer 的 .so 若报 GLIBCXX 缺失,
   沿用 `export LD_PRELOAD=/data3/wushang/libcxx_x/un/lib/libstdc++.so.6.0.34`。

## 1. 安装(两条路线,先试 A,冲突即转 B)

**方案 A(优先):现有 venv_train 内共存安装**

```bash
source /data3/wushang/workspace/TRACE-OPD/venv_train/bin/activate
# 先干跑解依赖,确认不会动 torch / flash-attn:
pip install --dry-run "sglang[srt,openai]==0.5.8" 2>&1 | grep -iE "torch|flash" 
# 若 dry-run 显示要改 torch 版本 ⇒ 停,转方案 B
pip install "sglang[srt,openai]==0.5.8"
```

**方案 B(A 冲突时):独立 venv_sglang(完整栈)**

```bash
python3 -m venv /data3/wushang/workspace/TRACE-OPD/venv_sglang
source /data3/wushang/workspace/TRACE-OPD/venv_sglang/bin/activate
pip install "sglang[srt,openai]==0.5.8"   # 让它自带匹配的 torch
# 再装训练面依赖(对照 venv_train 的 pip freeze 补齐 verl 运行所需;
# flash-attn 版本按新 torch 匹配重装——记录版本供 CD-010 核对)
```

方案 B 下两套 venv 并存:venv_train 跑 vLLM 后端(回摆通道),venv_sglang 跑
SGLang 后端。**哪条路线走通了,回报 experiment-feedback(含 pip freeze 中
sglang/torch/flashinfer/flash-attn 四行)。**

**flashinfer**:sglang 0.5.8 默认 attention backend 需要它(按 cu12x/torch
版本选预编译 wheel)。装不上不阻塞:SGLang 支持 `--attention-backend triton`
兜底(性能略降),verl 侧该参数可配。162 驱动 550/CUDA 12.4。

## 2. PoC 冒烟门(装好即做;结果数字回报 feedback)

**门 1 · import 冒烟(不占卡,立即做)**

```bash
python -c "import sglang; print(sglang.__version__)"      # 必须 0.5.8
python -c "import flashinfer" 2>&1 | tail -1              # 失败则记录,走 triton
# 报 GLIBCXX ⇒ 加 LD_PRELOAD 重试并记录
```

**门 2 · 单卡起服 + 真实往返(需 1 张空卡)**

```bash
python -m sglang.launch_server \
  --model-path $TRACE_MODEL_PATH --port 30000 \
  --mem-fraction-static 0.6 [--attention-backend triton]
curl -s http://127.0.0.1:30000/generate -H 'Content-Type: application/json' -d '{
  "text": "The capital of France is", "sampling_params": {"max_new_tokens": 8}}'
```

**门 3 · 机制关键验证(核心,三个断言 + 两个计时)**

用 ~2000 token 的长 prompt(内容任意,如重复段落),发原生 /generate 请求:

```python
import requests, time
long_text = "ALFWorld setup. " * 400          # ~2000 token 量级
def probe(logprob_start_len, tag):
    t0 = time.time()
    r = requests.post("http://127.0.0.1:30000/generate", json={
        "text": long_text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 1.0},
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
        "top_logprobs_num": 256,
    }).json()
    dt = time.time() - t0
    m = r["meta_info"]
    n_rows = len(m["input_token_logprobs"]) if m.get("input_token_logprobs") else 0
    width = len(m["input_top_logprobs"][ -1]) if m.get("input_top_logprobs") else 0
    print(f"{tag}: rows={n_rows} top_width={width} latency={dt:.3f}s")
    return r

n_prompt = None  # 先发一次全前缀拿行数 ⇒ prompt token 数
probe(0, "full-A")            # 全前缀第 1 次(无缓存)
probe(0, "full-B")            # 全前缀第 2 次
r = probe(0, "full-C")
L = len(r["meta_info"]["input_token_logprobs"])   # ≈ prompt token 数
probe(L - 8, "tail-A")        # tail-only 第 1 次(前缀应命中 radix cache)
probe(L - 8, "tail-B")        # tail-only 第 2 次
```

**断言**:① `top_width == 256` 且无报错/OOM(**大 K 支持度是全迁移的最大
未知项,一票否决**);② tail 模式 `rows == 8`(行数=请求窗口,且首行 logprob
非 null——`logprob_start_len>0` 时无 None 首项);③ **tail-A 的 latency 明显
低于 full-C**(radix 命中 + 只 extend 尾部 = KV 复用生效的直接证据;若不低,
说明 0.5.8 对带 logprob 请求也禁缓存 ⇒ 迁移收益不成立,立即反馈)。
另观察:256 宽下服务显存增长是否平稳(`nvidia-smi` 前后对比)。

**门 4 · memory-saver 起服(verl 依赖它做训练期 sleep/wake)**

```bash
python -m sglang.launch_server --model-path $TRACE_MODEL_PATH \
  --enable-memory-saver --mem-fraction-static 0.6 --port 30000
# 起服成功 + 门 2 的往返正常即过
```

## 3. 回报格式(experiment-feedback,一条即可)

- 路线 A/B、四个版本行(sglang/torch/flashinfer/flash-attn)、LD_PRELOAD 是否需要;
- 门 1–4 逐门结果;门 3 的五个 latency 数字与三断言判定;
- 任何报错原文(截 stderr 尾 20 行)。

CD-010 拿到门 3 结果后即联调代码适配(适配层的行语义断言以实测为准)。
