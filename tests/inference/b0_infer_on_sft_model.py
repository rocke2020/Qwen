import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import transformers
from icecream import ic
from pandas import DataFrame
from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel
from torch import nn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.append(os.path.abspath("."))
from qwen_generation_utils import decode_tokens, make_context

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M",
    format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
)


model_max_length = 768
model_name_or_path = "/mnt/nas1/models/qwen/Qwen-7B-Chat-Int8"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=768,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eod_id
ic("loads tokenizer done")
generation_config = GenerationConfig.from_pretrained(
    model_name_or_path, trust_remote_code=True
)
ic(generation_config)

peft_model_path = "/mnt/nas1/models/qwen/Qwen-7B-Chat-int8-moss-small/checkpoint-1500"
config = PeftConfig.from_pretrained(peft_model_path)
ic(config)

load_basic_model = 1
load_in_normal_order = 0
if load_basic_model:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    ic("loads base model done")
else:
    if load_in_normal_order:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        ic("loads base model done")
        model = PeftModel.from_pretrained(model, peft_model_path)
        ic("loads peft model done")
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_path, trust_remote_code=True
        )
        ic("directly loads peft model done")
ic(model.device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ic(device)
model = model.to(device)
model.eval()
model.generation_config = generation_config

query = "重点领域产业关键与共性技术攻关工程有哪些?"
response, history = model.chat(tokenizer, query, history=None)
print(response)

# all_raw_text = ["重点领域产业关键与共性技术攻关工程有哪些"]
# batch_raw_text = []
# for q in all_raw_text:
#     raw_text, _ = make_context(
#         tokenizer,
#         q,
#         system="You are a helpful assistant.",
#         max_window_size=model.generation_config.max_window_size,
#         chat_format=model.generation_config.chat_format,
#     )
#     batch_raw_text.append(raw_text)

# batch_input_ids = tokenizer(batch_raw_text, padding="longest")
# batch_input_ids = torch.LongTensor(batch_input_ids["input_ids"]).to(model.device)
# ic(batch_input_ids.size())
# batch_out_ids = model.generate(
#     batch_input_ids,
#     return_dict_in_generate=False,
#     generation_config=model.generation_config,
# )
# padding_lens = [
#     batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item()
#     for i in range(batch_input_ids.size(0))
# ]
# ic(padding_lens)
# batch_response = [
#     decode_tokens(
#         batch_out_ids[i][padding_lens[i] :],
#         tokenizer,
#         raw_text_len=len(batch_raw_text[i]),
#         context_length=(batch_input_ids[i].size(0) - padding_lens[i]),
#         chat_format="chatml",
#         verbose=False,
#         errors="replace",
#     )
#     for i in range(len(all_raw_text))
# ]
# print(batch_response)


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
