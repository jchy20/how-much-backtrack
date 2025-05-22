#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=advanced_geometry_from_qwen3binst_sft300_from_sudoku_synthetic_1_backtrack
#SBATCH --output=slurm_logs/advanced_geometry_from_qwen3binst_sft300_from_sudoku_synthetic_1_backtrack.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

zsh -l -c '
# Set environment variables
export N_GPUS=4
export BASE_MODEL=/usr/xtmp/jw834/saved/rl-reasoning/qwen-3b/SFT_Cold-Start/qwen3b-inst_from_sudoku_synthetic_1_backtrack/global_step_300
export DATA_DIR=/home/users/hc387/data/advanced_geometry
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=advanced_geometry_from_qwen3binst_sft300_from_sudoku_synthetic_1_backtrack
export VLLM_ATTENTION_BACKEND=XFORMERS
export MICRO_BATCH_SIZE=4
export MAX_RESPONSE_LENGTH=4096

# Run the training script
bash grpo.sh
'