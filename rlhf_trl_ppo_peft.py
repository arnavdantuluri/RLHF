#Need to modify to support perf
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline, AutoModelForSequenceClassification
import json
import pandas as pd

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import datasets

tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
sft_model_name = "decapoda-research/llama-7b-hf"
dataset_name = "/Users/arnavdantuluri/Desktop/LLama TRLX/chip2_instruct_alpha_v6a_4.json" #Currently only set up to work with rallio's data format; for more info on these look here: https://github.com/LAION-AI/Open-Instruction-Generalist
dataset2_name = "/Users/arnavdantuluri/Desktop/LLama TRLX/en_100_tree.jsonl"
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="edbeeching/gpt-neo-125M-imdb", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=16,
    batch_size=256,
)

config.model_name = sft_model_name

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def get_prompts(dataset_name, tokenizer, dataset2_name=None, ):
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

    max_length=400
    attention_mask=[]
    input_ids=[]
    labels=[]

    for i in prompts:
        length=len(i)
        attention=[]
        if length < max_length:
            for k in range(0,(max_length-length)):
                entry=i
            for k in range(0,(length)):
                attention.append(1)
            for k in range(0,(max_length-length)):
                attention.append(0)
            attention_mask.append(attention)
            input_ids.append(entry)
            labels.append(entry)
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["input_ids"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    df = pd.DataFrame({"attention_mask": attention_mask, "input_ids":input_ids,"labels":labels})
    new_dataset = datasets.Dataset.from_pandas(df)
    new_dataset = new_dataset.map(tokenize, batched=False)
    new_dataset.set_format(type="torch")
    return new_dataset


# Now let's build the model, the reference model, and the tokenizer.
pretrained_model = AutoModelForCausalLM.from_pretrained(config.model_name, load_in_8bit=True, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

rm_model, rm_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
# We retrieve the dataloader by calling the `build_dataset` function.
dataset = get_prompts(dataset_name=dataset_name, tokenizer=tokenizer ,dataset2_name=dataset2_name)

# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

pretrained_model = prepare_model_for_int8_training(pretrained_model)
pretrained_model = get_peft_model(pretrained_model, lora_config)

model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
print_trainable_parameters(model)
model.train()

@torch.no_grad()
def rank_model_fn(rm_tokenizer, rm_model, samples, **kwargs):
    rewards = []
    for i in samples:
        inputs = rm_tokenizer(i, return_tensors="pt", padding=True)#.to(rm_device)
        inputs.pop("token_type_ids", None)
        rewards.append(rm_model(**inputs).logits[:, 0].detach().cpu())
    return rewards
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model=None, tokenizer=tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True
    # Get response
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute score
    texts = [q + "<|endoftext|>" + z for q, z in zip(batch["query"], batch["response"])]
    rewards = rank_model_fn(rm_tokenizer, rm_model, texts)

    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False
    
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)