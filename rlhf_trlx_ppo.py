import os
from typing import List

import torch
import yaml
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

from trlx.trlx import train
from trlx.data.configs import TRLConfig
from datasets import load_from_disk, Dataset
import pandas as pd
import datasets
import torch
import random
import yaml
from datasets import load_dataset
from transformers import pipeline
import pathlib
from typing import Dict, List
from trlx.data.configs import TRLConfig
import json

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
sft_model_name = "EleutherAI/pythia-70m"
dataset_name = "/Users/arnavdantuluri/Desktop/LLama TRLX/chip2_instruct_alpha_v6a_4.json" #Currently only set up to work with rallio's data format; for more info on these look here: https://github.com/LAION-AI/Open-Instruction-Generalist
dataset2_name = "/Users/arnavdantuluri/Desktop/LLama TRLX/en_100_tree.jsonl"
max_tokens = 400

rm_model, rm_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
# rm_model.eval()
# rm_model.requires_grad_(False)
# rm_device = torch.cuda.device_count() - 1
# rm_model = rm_model.half().to(rm_device)

#This is only to be used if you are attempting do run both SFT and RLHF in the same script as it loads the dataset with the tokenizer which is quite memory intensive
#Currently only set up to work with rallio's data format; for more info on these look here: https://github.com/LAION-AI/Open-Instruction-Generalist
def get_prompts_and_dataset(dataset_name, tokenizer):
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
        prompts.append(final)
    return prompts, my_train_dataset, my_eval_dataset

def get_prompts(dataset_name, dataset2_name=None):
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

    prompts = []
    #Adds Rallio's prompts
    for i in range(len(fixed) - 1):
        try:
            prompts.append(fixed[i].split("\\n")[0].split("User:")[1])
        except:
            continue

    #adds OAsst prompts from file name en_100_message.jsonl
    #optimize to only use 1 for loop
    
    with open(dataset2_name, 'r') as json_file:
        for _line in json_file:
            prompts.append(json.loads(_line)['prompt']['text'])

    return prompts

prompts = get_prompts(dataset_name, dataset2_name)
@torch.no_grad()
def rank_model_fn(samples, **kwargs):
    inputs = rm_tokenizer(samples, return_tensors="pt", padding=True)#.to(rm_device)
    inputs.pop("token_type_ids", None)
    return rm_model(**inputs).logits[:, 0].detach().cpu()
with open('/Users/arnavdantuluri/Desktop/LLama TRLX/configs/ppo_config_summ_gptj.yaml') as f:
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