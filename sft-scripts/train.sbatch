#!/bin/bash
#SBATCH --partition=h200ea
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=train_sft
#SBATCH --output=sft_log/%x-%j.out

export PYTHONPATH="{project source dir}:$PYTHONPATH"

source {path to miniconda}/etc/profile.d/conda.sh
conda activate {environment name}

# on 4xA6000
# note that when max_length is 2048, need to set MICRO_BATCH_SIZE to 8 or lower
# when max_length is 4096, need to set MICRO_BATCH_SIZE to 4 or lower

zsh -l -c '
# Set environment variables
export N_GPUS={number of processes, distributed size}
export BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export TRAIN_DATA_DIR={data_path}
export VAL_DATA_DIR=/home/users/hc387/data/sft_data/Qwen3b-instruct/correct/combined_data.parquet
export PROMPT_KEY=prompt
export RESPONSE_KEY=completion
export MICRO_BATCH_SIZE={4, 8, 16, usually depend on the gpus you are usuing}
export MAX_LENGTH=8192
export PROJECT_NAME=SFT_Cold-Start
export EXPERIMENT_NAME={experiment name, usually i do model name + data name}
export nproc_per_node={number of processes, distributed size}

# Run the training script
bash sft-scripts/train_sft.sh 
'