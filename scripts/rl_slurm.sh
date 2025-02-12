#!/bin/bash
#SBATCH --job-name=qwen3b_rlgym   # Job name
#SBATCH --output=qwen3b_rlgym_%j.out  # Standard output (%j expands to jobId)
#SBATCH --error=qwen3b_rlgym_%j.err   # Standard error (%j expands to jobId)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Processes per node (usually 1 if using torchrun)
#SBATCH --gres=gpu:a6000:4              # Number of GPUs per node (example: 4)
#SBATCH --cpus-per-task=8          # CPU cores per task
#SBATCH --partition=compsci-gpu            # Partition name (change to your cluster's GPU partition)

source /home/users/hc387/miniconda3/etc/profile.d/conda.sh
conda activate zero

# Set environment variables
export N_GPUS=4
export BASE_MODEL=/usr/xtmp/hc387/models/Qwen2.5-3B-Instruct
export DATA_DIR=/home/users/hc387/data/leg_counting
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=leg-counting-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

# Run the training script
bash train_tiny_zero.sh