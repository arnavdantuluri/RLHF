# RLHF
Sample Script to perform RLHF on Rallio's Chip dataset and a sample of the OAsst data
# Dependencies
Install
 -TRLX
 -Huggingface Transformers
 -Pytorch
# How to Run
Supports the use of Hugging face accelerate. Current code hosts Reward Model on last gpu
Command: 
 -accelerate launch --config_file configs/default_accelerate_config.yaml --num_processes=7 rlhf.py
Launches script with distributed training on 7 gpus with the 8th hosting the reward model
3B can be comfortably run on 8 A100 40GB gpu; will test 7B model soon
