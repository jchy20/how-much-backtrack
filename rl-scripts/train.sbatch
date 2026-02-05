#!/bin/bash
#SBATCH --partition=h200ea
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=rl_train
#SBATCH --output=rl_log/qwen2.5-3B_sudoku_pureRL.out

export PYTHONPATH="/hpc/home/hc387/projects/backtrack:$PYTHONPATH"

source /hpc/group/szhoulab/hc387/miniconda3/etc/profile.d/conda.sh
conda activate zero

zsh -l -c '
# Set environment variables
export N_GPUS=4
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export DATA_DIR=/hpc/group/szhoulab/hc387/data/backtrack/sudoku
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=qwen2.5-3B_sudoku_pureRL
export VLLM_ATTENTION_BACKEND=XFORMERS
export MICRO_BATCH_SIZE=16
export MAX_RESPONSE_LENGTH=2048

# Run the training script
bash rl-scripts/train_tiny_zero.sh
'