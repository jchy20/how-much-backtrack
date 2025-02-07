#!/bin/bash
#SBATCH --job-name=bs17k_qwen3b   # Job name
#SBATCH --output=bs17k_qwen3b_%j.out  # Standard output (%j expands to jobId)
#SBATCH --error=bs17k_qwen3b_%j.err   # Standard error (%j expands to jobId)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Processes per node (usually 1 if using torchrun)
#SBATCH --gres=gpu:a6000:4              # Number of GPUs per node (example: 4)
#SBATCH --cpus-per-task=8          # CPU cores per task
#SBATCH --partition=compsci-gpu            # Partition name (change to your cluster's GPU partition)

source /home/users/hc387/miniconda3/etc/profile.d/conda.sh
conda activate zero

# === [ Optional: Set any environment variables for debugging or performance ] ===
export NCCL_DEBUG=INFO             # For debugging NCCL issues
export PYTHONFAULTHANDLER=1


nproc_per_node=4

shift 2  # shift past the first two arguments

# === [ Print debug info ] ===
echo "======================================================="
echo "Job:          $SLURM_JOB_NAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node list:    $SLURM_JOB_NODELIST"
echo "Num GPUs:     $SLURM_GPUS"
echo "======================================================="

# === [ Launch the training ] ===
# torchrun for single-node multi-GPU
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \