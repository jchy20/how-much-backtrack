#!/bin/bash
#SBATCH --job-name=countdown-qwen  # Job name (adjust as needed)
#SBATCH --output=countdown-qwen_%j.out  # Standard output and error log
#SBATCH --error=countdown-qwen_%j.err   # Error log
#SBATCH --nodes=1                      # Number of nodes (usually 1 if using a single machine)
#SBATCH --gres=gpu:a6000:4             # Number of GPUs per node (matches export N_GPUS=2)
#SBATCH --cpus-per-task=8              # Number of CPUs per task (adjust based on your need)
#SBATCH --partition=compsci-gpu        # Partition or queue name (adjust based on your cluster)
#SBATCH --ntasks=1                     # Number of tasks (most single-node GPU jobs use 1)

source /home/users/hc387/miniconda3/etc/profile.d/conda.sh
conda activate zero

######################
# Export variables
######################
export N_GPUS=4
export BASE_MODEL=/usr/xtmp/hc387/models/Qwen2.5-3B-Instruct
export DATA_DIR=/home/users/hc387/data/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

######################
# Run training script
######################
bash train_tiny_zero.sh
