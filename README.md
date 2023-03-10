# RLHF
Sample Script to perform RLHF on Rallio's Chip dataset and a sample of the OAsst data
# Dependencies
Install\n
 -TRLX\n
 -Huggingface Transformers\n
 -Pytorch\n
# How to Run
Supports the use of Hugging face accelerate. Current code hosts Reward Model on last gpu\n
Command: \n
 -accelerate launch --config_file configs/default_accelerate_config.yaml --num_processes=7 rlhf.py\n
Launches script with distributed training on 7 gpus with the 8th hosting the reward model\n
3B can be comfortably run on 8 A100 40GB gpu; will test 7B model soon\n
