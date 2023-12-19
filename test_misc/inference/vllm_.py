import os
import sys

sys.path.append(os.path.abspath("."))
from test_misc.inference.vllm_wrapper import vLLMWrapper


model = vLLMWrapper("/mnt/nas1/models/qwen/Qwen-7B-Chat", tensor_parallel_size=1)
# model = vLLMWrapper('Qwen/Qwen-7B-Chat-Int4', tensor_parallel_size=1, dtype="float16")

response, history = model.chat(query="你好", history=None)
print(response)
response, history = model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history = model.chat(query="给这个故事起一个标题", history=history)
print(response)
