import json
import logging
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import transformers
from icecream import ic
from pandas import DataFrame
from tqdm import tqdm

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
sys.path.append(os.path.abspath("."))
from comm_utils.file_util import file_util

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

moss_small_file = "/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-small.jsonl"
moss_small_json_file = "/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-small.json"
moss_small2_json_file = "/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-small2.json"
moss_tiny_json_file = "/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-tiny.json"
moss_orig_file = (
    "/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data.jsonl"
)
small_num = 100_000
small2_num = 200_000
tiny_num = 100


def read_moss():
    """num 1074551"""
    data = []
    count = 0
    with open(moss_orig_file, "r", encoding="utf-8") as f:
        # data = [json.loads(line) for line in f]
        for line in f:
            data.append(json.loads(line))
            # print(data[-1])
            if count == small2_num:
                break
            count += 1
            # break
    ic(len(data))

    with open(moss_small_file, "w", encoding="utf-8") as f:
        for line in data[:small_num]:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    file_util.write_json(data[:small_num], moss_small_json_file, ensure_ascii=False)
    file_util.write_json(data, moss_small2_json_file, ensure_ascii=False)

    tiny_data = data[:tiny_num] + data[-tiny_num:]
    file_util.write_json(tiny_data, moss_tiny_json_file, ensure_ascii=False)
    return data


def convert_moss_to_qwen_input(train_json):
    """
    [
        {
            "id": "identity_0",
            "conversations": [
                {
                    "from": "user",
                    "value": "你好"
                },
                {
                    "from": "assistant",
                    "value": "我是一个语言模型，我叫通义千问。"
                }
            ]
        }
    ]
    """
    for item in train_json:
        item["conversations"] = []
        for conv in item["conversation"]:
            for role, sent in conv.items():
                if role == "human":
                    role = "user"
                item["conversations"].append({"from": role, "value": sent})


def load_tokenizer():
    """ """
    cache_dir = None
    model_max_length = 512
    model_name_or_path = "/mnt/nas1/models/qwen/Qwen-7B-Chat-Int8"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    return tokenizer


def check_tokenizer():
    """ """
    tokenizer = load_tokenizer()

    data = json.load(open(moss_tiny_json_file, "r"))
    input0 = []
    for sent in data[0]["conversation"]:
        input0.append(sent["human"])
        input0.append(sent["assistant"])
    input_test = "\n".join(input0)
    ic(input_test)
    ic(len(input_test))

    result = tokenizer(input_test)
    ic(len(result.input_ids))


def check_convert():
    """ """
    data = file_util.read_json(moss_tiny_json_file)
    convert_moss_to_qwen_input(data)
    ic(data[0])


def main():
    # read_moss()
    # check_tokenizer()
    check_convert()


def main():
    # read_moss()
    # check_tokenizer()
    check_convert()

if __name__ == "__main__":
    main()
