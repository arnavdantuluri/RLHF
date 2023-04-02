import os
from typing import List

import torch
import yaml
from trlx.trlx import train
from trlx.data.configs import TRLConfig

from datasets import load_from_disk, Dataset
from datasets import load_dataset
from transformers import pipeline
from datasets import load_dataset
from model_training.custom_datasets import get_one_dataset
import model_training.models.reward_model
from argparse import Namespace
from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS, format_pairs
from model_training.utils import get_dataset, get_model, read_yamls

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd
import datasets
import torch
import random
from argparse import ArgumentParser
import yaml
import pathlib
from typing import Dict, List
import json

QA_SPECIAL_TOKENS_V2_5 = {
    "prompter": "<|prompter|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
    "prefix_begin": "<|prefix_begin|>",
    "prefix_end": "<|prefix_end|>",
    "eos": "<|endoftext|>",
}

file_path = "2023-03-13_oasst_ready_labels.jsonl.gz"

rank_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-1-pythia-12b", padding_side="left")

def get_prompts_oasst_only(file_path):

    config = Namespace(
        cache_dir="../../../home/ubuntu/data_cache",
    )
    kwargs = {
        # "lang": "en,es,fr,de",
        "lang": "en",
        "top_k": 1,
        "input_file_path": file_path,
        "mode": "rl",
    }
    train, val = get_one_dataset(conf=config, dataset_name="oasst_export", **kwargs)

    return train, val

# train = get_prompts_oasst_only(file_path)
train, val = get_prompts_oasst_only(file_path)

# take the dataset as the eval prompt generation dataset
# trlx requires training data to be a list of prompts
# first element of each sample is the context and the prompt

prompts, eval_prompts = tuple(
    map(
        lambda x: [
            "".join(format_pairs(x[i][0], rank_tokenizer.eos_token, add_initial_reply_token=True))
            # print(format_pairs(x[i][0], rank_tokenizer.eos_token, add_initial_reply_token=True))
            for i in range(len(x))
        ],
        (train, val),
    )
)
print(len(prompts))
print(len(eval_prompts))
with open(r'output.txt', 'w') as fp:
    for item in eval_prompts:
        # write each item on a new line
        fp.write("Prompt For RL: %s\n" % item)
    print('Done')