#!/bin/bash
#SBATCH --partition=h200ea
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=train_sft
#SBATCH --output=sft_log/llama32-3B_sudoku-optimal.out

export PYTHONPATH="/hpc/home/hc387/projects/backtrack:$PYTHONPATH"

source /hpc/group/szhoulab/hc387/miniconda3/etc/profile.d/conda.sh
conda activate zero

# on 4xA6000
# note that when max_length is 2048, need to set MICRO_BATCH_SIZE to 8 or lower
# when max_length is 4096, need to set MICRO_BATCH_SIZE to 4 or lower

zsh -l -c '
# Set environment variables
export N_GPUS=2
export BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export TRAIN_DATA_DIR=/hpc/group/szhoulab/hc387/data/backtrack/sudoku/sft_backtrack/qwen-instruct_sft_optim.parquet
export VAL_DATA_DIR=/hpc/group/szhoulab/hc387/data/backtrack/combined_data.parquet
export PROMPT_KEY=prompt
export RESPONSE_KEY=completion
export MICRO_BATCH_SIZE=16
export MAX_LENGTH=8192
export PROJECT_NAME=SFT_Cold-Start
export EXPERIMENT_NAME=llama32-3B_sudoku-optimal
export nproc_per_node=2

# Run the training script
bash sft-scripts/train_sft.sh 
'