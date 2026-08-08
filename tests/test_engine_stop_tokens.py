"""EngineGenerate 的轮末停止符下发(2026-08-08 Qwen3.5-9B P0 事故回归)。

事故:Qwen3.5-9B 官方 repo 无 generation_config.json,config.json 只声明
<|endoftext|>;verl skip_tokenizer_init 下引擎停止集不含轮末符 <|im_end|>,
每轮生成越过轮末续写伪多轮对话(step1 实测 86.5% response token 是越界
垃圾)。修复 = 停止条件作为对话协议属性,由唯一引擎通道 EngineGenerate
随每个请求显式下发;对 Qwen3 系(generation_config 自带 im_end)是严格
no-op(请求侧 stop 与引擎侧 eos 集在 sglang check_finished 里取并)。
"""
import asyncio

from trace_opd.core.template_delta import turn_stop_token_ids
from trace_opd.train.backends import EngineGenerate


class _Server:
    def __init__(self):
        self.calls = []

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        self.calls.append(sampling_params)

        class Out:
            token_ids = [1, 2]
            extra_fields = {}
            stop_reason = "stop"

        return Out()


class _Tok:
    """Qwen3.5 形态:eos(<|im_end|>=248046)≠ config 里的 <|endoftext|>;
    tokenizer 面上 eos_token 就是 im_end,两者同 id。"""

    eos_token_id = 248046

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 248046}.get(token)

    def decode(self, ids, skip_special_tokens=True):
        return "x"


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_stop_token_ids_injected_on_every_engine_call():
    server = _Server()

    async def main():
        gen = EngineGenerate(server, _Tok())
        await gen([1], {"max_tokens": 8, "temperature": 1.0})

    _run(main())
    assert server.calls[0]["stop_token_ids"] == [248046]
    # 调用方的 dict 不被就地修改(episode 层每轮重建参数,防止串轮污染)。
    assert "stop_token_ids" not in {"max_tokens": 8}


def test_caller_explicit_stop_token_ids_win():
    server = _Server()

    async def main():
        gen = EngineGenerate(server, _Tok())
        await gen([1], {"max_tokens": 8, "stop_token_ids": [7]})

    _run(main())
    assert server.calls[0]["stop_token_ids"] == [7]


def test_qwen3_family_dedupes_to_single_id():
    # Qwen3:eos_token 与 <|im_end|> 同一 id(151645)——集合去重,下发
    # [151645];与引擎侧 generation_config 集 [151645, 151643] 取并后不变,
    # 对既有 4B/1.7B 线是严格 no-op。
    class Tok(_Tok):
        eos_token_id = 151645

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 151645}.get(token)

    assert turn_stop_token_ids(Tok()) == [151645]


def test_non_chatml_tokenizer_falls_back_to_eos_only():
    class Tok(_Tok):
        eos_token_id = 2

        def convert_tokens_to_ids(self, token):
            return None  # 模板家族没有 <|im_end|>

    assert turn_stop_token_ids(Tok()) == [2]


def test_distinct_eos_and_im_end_both_present_sorted():
    class Tok(_Tok):
        eos_token_id = 248044  # 假想:tokenizer eos 指向 <|endoftext|>

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 248046}.get(token)

    assert turn_stop_token_ids(Tok()) == [248044, 248046]
