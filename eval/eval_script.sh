#!/bin/bash
#SBATCH --partition=h200ea
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=eval
#SBATCH --output=slurm_logs/llama32-3B_arc-1d-optimal_sft400.out

source /hpc/group/szhoulab/hc387/miniconda3/etc/profile.d/conda.sh
conda activate zero

# Define model and which dataset
eval_data="llama32-3B_arc-1d-optimal_sft400"
mkdir -p ${eval_data}
mkdir -p slurm_logs

# Path to dataset and model
task_name="arc-1d"
dataset_path="/hpc/group/szhoulab/hc387/data/backtrack/arc_1d/test.parquet"
model_dir="/work/hc387/projects/backtrack/rl/llama32-3B_arc-1d-optimal_sft400/actor/global_step_200"
port=8010
baseline=False
batch_size=128

# Start the API server
python -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" \
    --tensor-parallel-size 1 \
    --guided-decoding-backend lm-format-enforcer \
    --host 0.0.0.0 --port $port > ${eval_data}/api_server_${port}.log 2>&1 &

API_PID=$!

echo "Waiting for API server to start..."
while ! nc -z localhost $port; do
  sleep 5
done
echo "API server is up!"

# Run all evaluations in parallel and capture PIDs directly
echo "Starting evaluation for $dataset_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $dataset_path \
    --task_name $task_name \
    --baseline $baseline \
    --batch_size $batch_size \
    --output_dir ${eval_data} \
    --port $port > ${eval_data}/${task_name}.log 2>&1 &
task_pid=$!

# Store all PIDs in an array
eval_pids=($task_pid)

# Wait for all evaluation processes to complete
echo "Waiting for all evaluations to complete..."
wait $task_pid
echo "All evaluations completed!"

# Collect and display results
echo "=== Evaluation Results ==="
echo "--- $task_name ---"
grep "Task accuracy for" ${eval_data}/${task_name}.log
echo ""

echo "Shutting down API server..."
kill $API_PID
