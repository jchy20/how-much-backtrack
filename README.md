# TinyZero w/ Reasoning Gym


## Installation

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
Currently supports tasks __Leg-Counting__, __Advanced-Geometry__, __Arc_1d__

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
