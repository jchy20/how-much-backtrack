#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=eval
#SBATCH --output=slurm_logs/eval.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

zsh -l -c '
ag_path="$HOME/data/advanced_geometry/test.parquet"
countdown_path="$HOME/data/countdown/test.parquet"
arc_1d_path="$HOME/data/arc_1d/test.parquet"
sudoku_path="$HOME/data/sudoku/test.parquet"
color_cube_rotation_path="$HOME/data/color_cube_rotation/test.parquet"
zebra_puzzles_path="$HOME/data/zebra_puzzles/test.parquet"
list_functions_path="$HOME/data/list_functions/test.parquet"
self_reference_path="$HOME/data/self_reference/test.parquet"


model_dir="/usr/xtmp/hc387/models/Qwen2.5-3B-Instruct"
port=8000

python -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port $port > api_server.log 2>&1 &

API_PID=$!

echo "Waiting for API server to start..."
while ! nc -z localhost $port; do
  sleep 5
done
echo "API server is up!"

# Run evaluation in the foreground so the script waits for it to complete
python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $ag_path \
    --port $port > advanced_geometry.log 2>&1

echo "Advanced Geometry evaluation complete"

python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $countdown_path \
    --port $port > countdown.log 2>&1

echo "Countdown evaluation complete"

python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $arc_1d_path \
    --port $port > arc_1d.log 2>&1

echo "Arc 1D evaluation complete"

python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $sudoku_path \
    --port $port > sudoku.log 2>&1

echo "Sudoku evaluation complete"

python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $color_cube_rotation_path \
    --port $port > color_cube_rotation.log 2>&1

echo "Color Cube Rotation evaluation complete"  

python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $zebra_puzzles_path \
    --port $port > zebra_puzzles.log 2>&1

echo "Zebra Puzzles evaluation complete"

python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $list_functions_path \
    --port $port > list_functions.log 2>&1

echo "List Functions evaluation complete"

python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $self_reference_path \
    --port $port > self_reference.log 2>&1

echo "Self Reference evaluation complete"


echo "Shutting down API server..."
kill $API_PID
'