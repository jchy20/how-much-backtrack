#!/bin/bash
#SBATCH --job-name=hold_gpus
#SBATCH --output=hold_gpus_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compsci-gpu
#SBATCH --time=7-00:00:00  # Hold GPUs for 7 days


# Keep the job running for 7 days
sleep $((7 * 24 * 60 * 60))  # Sleep for 7 days (7 * 24 hours * 60 min * 60 sec)

echo "GPU reservation complete."
