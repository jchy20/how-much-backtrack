#!/bin/bash
#SBATCH --job-name=gen_test
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=0

hostname && nvidia-smi && env
source /home/users/hc387/miniconda3/etc/profile.d/conda.sh

conda activate reasoning_gym
which python

cd /usr/xtmp/hc387/reasoning-gym

# python3 advanced_geometry.py --train_size=163840 --template_type=qwen-instruct
# echo "advanced_geometry.py done"

# python3 countdown.py --sft_size=76800 --template_type=qwen-instruct --local_dir=/home/users/hc387/data/countdown/sft_backtrack
# echo "countdown.py done"

# python3 arc_1d.py --train_size=163840 --template_type=qwen-instruct
# echo "arc_1d.py done"

python3 sudoku.py --sft_size=76800 --template_type=qwen-instruct --local_dir=/home/users/hc387/data/sudoku/sft_backtrack
echo "sudoku.py done"

# python3 color_cube_rotation.py --sft_size=76800 --template_type=qwen-instruct
# echo "color_cube_rotation.py done"

# python3 zebra_puzzles.py --sft_size=76800 --template_type=qwen-instruct
# echo "zebra_puzzles.py done"

# python3 list_functions.py --sft_size=76800 --template_type=qwen-instruct
# echo "list_functions.py done"

# python3 self_reference.py --sft_size=76800 --template_type=qwen-instruct
# echo "self_reference.py done"
