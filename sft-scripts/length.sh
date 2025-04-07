#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a5000:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=length
#SBATCH --output=length.out

source /home/users/hc387/miniconda3/etc/profile.d/conda.sh
conda activate zero

python length.py