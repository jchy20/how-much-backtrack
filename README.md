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

Currently supports tasks __advanced_geometry__, __countdown__, __arc_1d__, __sudoku__, (the first four has backtrack sft dataset), __color_cube_rotation__, __zebra_puzzles__, __list_functions__, and __self_reference__.


**Data Preparation**
```
conda activate reason-gym
cd reasoning-gym
python3 countdown.py \
--sft_size=8000 \
--template_type=qwen-instruct \
--local_dir=~/data/countdown
```

**Config Explanation** 

- Use the exact name of the task listed above to generate corresponding dataset, the naming of the python file follows `task_name.py`
- `train_size`, `val_size`, `test_size`, `sft_size` has different seed and file name presets. Choose only one of the above to specify the size of the dataset.
- `template_type` has `base`, `qwen-instruct`, and `baseline`. Use `base` for base models.
- For other task specific arguments, please see `how-much-backtrack/reasoning-gym/GALLERY.md` for corresponding task

## RL training 
```
cd rl-scripts

export N_GPUS=4
export BASE_MODEL=path-to-your-model
export DATA_DIR=path-to-your-dataset
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=your-wandb-experiment-name
export VLLM_ATTENTION_BACKEND=XFORMERS
export MICRO_BATCH_SIZE=4
export MAX_RESPONSE_LENGTH=4096

# Run the training script
bash grpo.sh
```

To run training with GRPO algorithm, refer to the config in `grpo.sh`. To run training with PPO algorithm, refer to the config in `train_tiny_zero.sh`. To make further change, refer to the config in `how-much-backtrack/verl/trainer/config/ppo_trainer.yaml`

## SFT training
```
cd sft-scripts

export N_GPUS=4
export BASE_MODEL=path-to-your-model
export TRAIN_DATA_DIR=path-to-your-training-dataset
export VAL_DATA_DIR=path-to-your-validation-dataset
export PROMPT_KEY=prompt
export RESPONSE_KEY=completion
export MICRO_BATCH_SIZE=4
export MAX_LENGTH=6000
export PROJECT_NAME=your-wandb-project-name
export EXPERIMENT_NAME=your-wandb-experiment-name
export nproc_per_node=4

# Run the training script
bash train_sft.sh 
```

To change specific configs, please refer to `train_sft.sh` and `how-much-backtrack/verl/trainer/config/sft_trainer.yaml`.

## SFT Data Generation
**Set up vllm serving first**
```
python -m vllm.entrypoints.openai.api_server \
    --model /path-to-your-model \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000
```
**Run generation**
```
cd sft-data

python run_gen.py \
    --model_path /path-to-your-model \
    --model_type qwen-instruct \
    --eval_dataset_dir /path-to-your-dataset \
    --task_name countdown \
    --output_dir /path-to-generated-dataset \
    --batch_size 8 \
    --n 1 \
    --save_interval 10000 \
    --port 8000
```



## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{cai2025backtrackingenoughexploringinterplay,
      title={How Much Backtracking is Enough? Exploring the Interplay of SFT and RL in Enhancing LLM Reasoning}, 
      author={Hongyi James Cai and Junlin Wang and Xiaoyin Chen and Bhuwan Dhingra},
      year={2025},
      eprint={2505.24273},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.24273}, 
}
```
