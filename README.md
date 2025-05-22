# How Much Backtracking is Enough? Exploring the Interplay of SFT and RL in Enhancing LLM Reasoning

This repository is based on [TinyZero](https://github.com/Jiayi-Pan/TinyZero), which is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown built upon [veRL](https://github.com/volcengine/verl).

Dataset generation is based on [Reasoning-Gym](https://github.com/open-thought/reasoning-gym).


## Installation
Due to a mismatch in python versions, it is recommended to install to two encironments: one for reasoning gym and the other one for training.

### For dataset 
requires Python >= 3.11.
```
conda create -n reason-gym python=3.11
pip install reasoning-gym 
```

### For training
```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Reasoning-Gym Datasets

**Data Preparation**
```
conda activate zero
python ./examples/data_preprocess/reasoning-gym/{task_name}.py \
--local_dir {path_to_your_dataset} \
--dataset_name {dataset_name} \
--train_size {size_of_training} \
--val_size {size_of_validation} \
--task_types {task_types} \
--seed {set_seed}
```
Currently supports tasks __Leg-Counting__, __Advanced-Geometry__, __Arc-1d__, __Base-Conversion__, __Caesar-Cipher__

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
```
