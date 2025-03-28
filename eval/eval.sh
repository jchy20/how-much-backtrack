#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a5000:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=qwen3b-countdown-500
#SBATCH --output=slurm_logs/qwen3b-countdown-500.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

zsh -l -c '
# Whether we are running baseline or not
baseline=False

# Define dataset paths
ag_path="/home/users/hc387/data/advanced_geometry/test.parquet"
countdown_path="/home/users/hc387/data/countdown/test.parquet"
arc_1d_path="/home/users/hc387/data/arc_1d/test.parquet"
sudoku_path="/home/users/hc387/data/sudoku/test.parquet"
color_cube_rotation_path="/home/users/hc387/data/color_cube_rotation/test.parquet"
zebra_puzzles_path="/home/users/hc387/data/zebra_puzzles/test.parquet"
list_functions_path="/home/users/hc387/data/list_functions/test.parquet"
self_reference_path="/home/users/hc387/data/self_reference/test.parquet"

# Define model and which dataset
eval_data="qwen3b_countdown_500"

mkdir -p ${eval_data}

model_dir="/usr/xtmp/hc387/TinyZero/qwen-3b/TinyZero/countdown-qwen3b-inst/actor/global_step_500"
port=8000

# Start the API server
python -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port $port > ${eval_data}/api_server.log 2>&1 &

API_PID=$!

echo "Waiting for API server to start..."
while ! nc -z localhost $port; do
  sleep 5
done
echo "API server is up!"

# Run all evaluations in parallel and capture PIDs directly
echo "Starting evaluation for $ag_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $ag_path \
    --task_name "advanced_geometry" \
    --baseline $baseline \
    --port $port > ${eval_data}/advanced_geometry.log 2>&1 &
ag_pid=$!

echo "Starting evaluation for $countdown_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $countdown_path \
    --task_name "countdown" \
    --baseline $baseline \
    --port $port > ${eval_data}/countdown.log 2>&1 &
countdown_pid=$!

echo "Starting evaluation for $arc_1d_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $arc_1d_path \
    --task_name "arc_1d" \
    --baseline $baseline \
    --port $port > ${eval_data}/arc_1d.log 2>&1 &
arc_1d_pid=$!

echo "Starting evaluation for $sudoku_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $sudoku_path \
    --task_name "sudoku" \
    --baseline $baseline \
    --port $port > ${eval_data}/sudoku.log 2>&1 &
sudoku_pid=$!

echo "Starting evaluation for $color_cube_rotation_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $color_cube_rotation_path \
    --task_name "color_cube_rotation" \
    --baseline $baseline \
    --port $port > ${eval_data}/color_cube_rotation.log 2>&1 &
color_cube_pid=$!

echo "Starting evaluation for $zebra_puzzles_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $zebra_puzzles_path \
    --task_name "zebra_puzzles" \
    --baseline $baseline \
    --port $port > ${eval_data}/zebra_puzzles.log 2>&1 &
zebra_pid=$!

echo "Starting evaluation for $list_functions_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $list_functions_path \
    --task_name "list_functions" \
    --baseline $baseline \
    --port $port > ${eval_data}/list_functions.log 2>&1 &
list_functions_pid=$!

echo "Starting evaluation for $self_reference_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $self_reference_path \
    --task_name "self_reference" \
    --baseline $baseline \
    --port $port > ${eval_data}/self_reference.log 2>&1 &
self_reference_pid=$!

# Store all PIDs in an array
eval_pids=($ag_pid $countdown_pid $arc_1d_pid $sudoku_pid $color_cube_pid $zebra_pid $list_functions_pid $self_reference_pid)

# Wait for all evaluation processes to complete
echo "Waiting for all evaluations to complete..."
for pid in ${eval_pids[@]}; do
    wait $pid
    echo "Process $pid completed"
done

echo "All evaluations completed!"

# Collect and display results
echo "=== $eval_data ==="
echo "=== $baseline ==="

echo "=== Evaluation Results ==="
for task in advanced_geometry countdown arc_1d sudoku color_cube_rotation zebra_puzzles list_functions self_reference; do
    echo "--- $task ---"
    grep "Task accuracy for" ${eval_data}/${task}.log
    echo ""
done

echo "Shutting down API server..."
kill $API_PID
'