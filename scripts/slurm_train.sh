#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=arc_1d-qwen3b
#SBATCH --output=slurm_logs/arc_1d-qwen3b.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

zsh -l -c '
# Set environment variables
export N_GPUS=4
export BASE_MODEL=/usr/xtmp/hc387/models/Qwen2.5-3B-Instruct
export DATA_DIR=/home/users/hc387/data/arc_1d
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=arc_1d
export VLLM_ATTENTION_BACKEND=XFORMERS

# Run the training script
bash train_tiny_zero.sh 
'