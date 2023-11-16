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
from pandas import DataFrame
from tqdm import tqdm
from icecream import ic

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
sys.path.append(os.path.abspath('.'))
from comm_utils.file_util import file_util


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
moss_small_file = '/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-small.jsonl'
moss_small_json_file = '/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-small.json'
moss_tiny_json_file = '/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-tiny.json'
moss_orig_file = '/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data.jsonl'
small_num = 100_000
tiny_num = 100


def read_moss():
    """ num 1074551 """
    data = []
    count = 0
    with open(moss_orig_file, 'r', encoding='utf-8') as f:
        # data = [json.loads(line) for line in f]
        for line in f:
            data.append(json.loads(line))
            # print(data[-1])
            if count == small_num:
                break
            count += 1
            # break
    ic(len(data))

    with open(moss_small_file, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    file_util.write_json(data, moss_small_json_file, ensure_ascii=False)

    tiny_data = data[:tiny_num] + data[-tiny_num:]
    file_util.write_json(tiny_data, moss_tiny_json_file, ensure_ascii=False)
    return data


def convert_moss_to_qwen_input(train_json):
    """  """
    for item in train_json:
        item['conversations'] = []
        for conv in item['conversation']:
            for turn in conv:
                    item['conversations'].append(
                        {
                        'from': 'user',
                        'value': turn['human']
                        },
                    )
                    item['conversations'].append(
                        {
                        'from': 'assistant',
                        'value': turn['assistant']
                        }
                    )


def main():
    data = json.load(open(moss_tiny_json_file, "r"))
    ic(len(data))
    convert_moss_to_qwen_input(data)

if __name__ == '__main__':
    main()
    # read_moss()
