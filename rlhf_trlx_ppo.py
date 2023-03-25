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

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import model_training.models.reward_model

import pandas as pd
import datasets
import torch
import random
import yaml
import pathlib
from typing import Dict, List
import json

directory = os.getcwd()
reward_name = "andreaskoepf/oasst-rm-1-pythia-1b"
sft_model_name = "OpenAssistant/oasst-sft-1-pythia-12b"
#rallio data is the same as the OIG data; most of which can be used here as well
# path_1 = directory + "/chip2_instruct_alpha_v6a_4.json" #Currently only set up to work with rallio's data format; for more info on these look here: https://github.com/LAION-AI/Open-Instruction-Generalist
#data_path = directory + "/en_100_tree.jsonl.gz" #change to wherever your code is located
#For OAsst data :)
file_path = "2023-03-13_oasst_ready_labels.jsonl.gz"
max_tokens = 400

QA_SPECIAL_TOKENS_V2_5 = {
    "prompter": "<|prompter|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
    "prefix_begin": "<|prefix_begin|>",
    "prefix_end": "<|prefix_end|>",
    "eos": "<|endoftext|>",
}

rm_model, rm_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
rm_model.eval()
rm_model.requires_grad_(False)
rm_device = torch.cuda.device_count() - 1
rm_model = rm_model.half().to(rm_device)

#Function to add the needed tokens in front of and behind the given string
def format_string(prompt):
    return f"{QA_SPECIAL_TOKENS_V2_5['prompter']}{prompt}{QA_SPECIAL_TOKENS_V2_5['eos']}{QA_SPECIAL_TOKENS_V2_5['assistant']}"

#This is only to be used if you are attempting do run both SFT and RLHF in the same script as it loads the dataset with the tokenizer which is quite memory intensive
#Currently only set up to work with rallio's data format; for more info on these look here: https://github.com/LAION-AI/Open-Instruction-Generalist
#assuming no prefix will be fed during RL or SFT training
def get_prompts_and_dataset(dataset_name, tokenizer): #Gives you prompts, train_dataset, eval_dataset NOT advised to use unless you have a lot of GPU hours. 
    with open(dataset_name) as my_file:
        data = my_file.read()
    entries=data.split("<|endoftext|>")
    count=0
    fixed=[]
    for i in entries:
        new_line=""
        if i[-1]=="\n" and i[0] =="\n":
            new_line=i[1:-1]
            count+=1
        elif i[0]=="\n":
            new_line=i[1:]
        elif i[-1] == "\n":
            new_line=i[:-1]
        if len(new_line) > 5:
            fixed.append(new_line)
        else:
            fixed.append(i)
    fixed_tokens=[]
    for i in fixed:
        line=i+"<|endoftext|><|endoftext|>"
        tokens=tokenizer.encode(line)
        fixed_tokens.append((line,tokens))
    max_length=280

    attention_mask=[]
    input_ids=[]
    labels=[]

    for i in fixed_tokens:
        length=len(i[1])
        attention=[]
        if length < max_length:
            for k in range(0,(max_length-length)):
                entry=i[1]
                entry.append(1)
            for k in range(0,(length)):
                attention.append(1)
            for k in range(0,(max_length-length)):
                attention.append(0)
            attention_mask.append(attention)
            input_ids.append(entry)
            labels.append(entry)



    df = pd.DataFrame({"attention_mask": attention_mask, "input_ids":input_ids,"labels":labels})
    new_dataset=datasets.Dataset.from_pandas(df)
    split_dataset = new_dataset.train_test_split(test_size=0.01)
    train_dataset=split_dataset['train']
    eval_dataset=split_dataset['test']

    train_dataset.save_to_disk("my_train_data")
    eval_dataset.save_to_disk("my_eval_data")
    my_train_dataset = load_from_disk("my_train_data")
    my_eval_dataset = load_from_disk("my_eval_data")

    prompts = []
    for i in range(len(my_train_dataset) - 1):
        x = tokenizer.decode(my_train_dataset[i]['input_ids'])
        y = x.split("\\n")
        z = y[0].split("User:")
        try:
            final = "User:" + z[1]
        except:
            #There is one prompt with an issue here; we simply skip that one
            continue
        prompts.append(format_string(final))
    return prompts, my_train_dataset, my_eval_dataset

#This is only to get prompts; currently supports rallio's data format and oasst data format
#assuming no prefix will be fed during RL or SFT training
def get_prompts_rallio_and_oasst(path_1=None, path_2=None):
    with open(path_1) as my_file:
        data = my_file.read()

    entries=data.split("<|endoftext|>")
    count=0
    fixed=[]
    for i in entries:
        new_line=""
        if i[-1]=="\n" and i[0] =="\n":
            new_line=i[1:-1]
            count+=1
        elif i[0]=="\n":
            new_line=i[1:]
        elif i[-1] == "\n":
            new_line=i[:-1]
        if len(new_line) > 5:
            fixed.append(new_line)
        else:
            fixed.append(i)

    prompts = []
    #Adds Rallio's prompts
    for i in range(len(fixed) - 1):
        try:
            prompts.append(format_string(fixed[i].split("\\n")[0].split("User:")[1]))
        except:
            continue

    #adds OAsst prompts from file name en_100_message.jsonl
    
    with open(path_2, 'r') as json_file:
        for _line in json_file:
            prompts.append(format_string({json.loads(_line)['prompt']['text']}))

    return prompts

#get prompts from oasst data only; sample can be found in the Open Assistant repository
#assuming no prefix will be fed during RL or SFT training
def get_prompts_oasst_only(file_path):

    config = Namespace(
        cache_dir="../../../home/ubuntu/data_cache",
    )
    kwargs = {
        "lang": "en",
        "top_k": 2,
        "input_file_path": file_path,
        "mode": "sft",
    }
    train, val = get_one_dataset(conf=config, dataset_name="oasst_export", **kwargs)

    #Need to actually convert these to prompts to be fed into TRLX
    prompts = []
    for i in train.data:
        if format_string(i[0]) not in prompts:
            prompts.append(format_string(i[0]))
    
    for i in val.data:
        if format_string(i[0]) not in prompts:
            prompts.append(format_string(i[0]))

    return prompts

prompts = get_prompts_oasst_only(file_path)
print(prompts[0])

@torch.no_grad()
def rank_model_fn(samples, **kwargs):
    inputs = rm_tokenizer(samples, return_tensors="pt", padding=True).to(rm_device)
    return rm_model(**inputs).logits[0].detach().cpu()

with open(directory + '/configs/ppo_config_summ_gptj.yaml') as f:
    default_config = yaml.safe_load(f)

trlx_config = TRLConfig.update(default_config, {})

trlx_config.tokenizer.tokenizer_path = sft_model_name
trlx_config.model.model_path = sft_model_name
trlx_config.method.gen_kwargs["max_new_tokens"] = max_tokens
trlx_config.train.batch_size = 16

trainer = train(
    sft_model_name,
    reward_fn=rank_model_fn,
    prompts=prompts,
    config=trlx_config,
)

directory = os.getcwd()
trainer.save_pretrained(directory + "/checkpoints/best_checkpoint")