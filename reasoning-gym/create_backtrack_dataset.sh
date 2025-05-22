#!/bin/bash
#SBATCH --job-name=gen_test
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=0

source /home/users/hc387/miniconda3/etc/profile.d/conda.sh
conda activate reasoning_gym

python split_backtrack_data.py