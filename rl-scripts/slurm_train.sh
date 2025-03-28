#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=self_reference_from_qwen3binst_sft306
#SBATCH --output=slurm_logs/self_reference_from_qwen3binst_sft306.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

zsh -l -c '
# Set environment variables
export N_GPUS=4
export BASE_MODEL=/usr/xtmp/hc387/TinyZero/qwen-3b/SFT_Cold-Start/qwen3b-inst_self_reference/global_step_306
export DATA_DIR=/home/users/hc387/data/self_reference
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=self_reference_from_qwen3binst_sft306
export VLLM_ATTENTION_BACKEND=XFORMERS

# Run the training script
bash train_tiny_zero.sh 
'