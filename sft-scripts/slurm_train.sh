#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=countdown_sft_cold-start
#SBATCH --output=slurm_logs/countdown_sft_cold-start.out

source /home/users/hc387/miniconda3/etc/profile.d/conda.sh
conda activate zero

# on 4xA6000
# note that when max_length is 2048, need to set MICRO_BATCH_SIZE to 8 or lower
# when max_length is 4096, need to set MICRO_BATCH_SIZE to 4 or lower

zsh -l -c '
# Set environment variables
export N_GPUS=4
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export TRAIN_DATA_DIR=/home/users/hc387/data/sft_data/Qwen3b-instruct/correct/countdown.parquet
export VAL_DATA_DIR=/home/users/hc387/data/sft_data/Qwen3b-instruct/correct/combined_data.parquet
export PROMPT_KEY=prompt
export RESPONSE_KEY=completion
export MICRO_BATCH_SIZE=8
export MAX_LENGTH=2048
export PROJECT_NAME=SFT_Cold-Start
export EXPERIMENT_NAME=qwen3b-inst_countdown
export nproc_per_node=4

# Run the training script
bash train_sft.sh 
'