#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a5000:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=Qwen7b-instruct-sft-generation
#SBATCH --output=slurm_logs/Qwen7b-instruct-sft-generation-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

zsh -l -c '
# Define dataset paths
dataset_path1="/home/users/hc387/data/list_functions/qwen-instruct_sft.parquet"
dataset_path2="/home/users/hc387/data/self_reference/qwen-instruct_sft.parquet"
task1="list_functions"
task2="self_reference"

# Define model and which dataset
eval_data="Qwen7b-instruct"
mkdir -p ${eval_data}

model_dir="Qwen/Qwen2.5-7B-Instruct"
port=8007

# Start the API server
python -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port $port > ${eval_data}/api_server_${port}.log 2>&1 &

API_PID=$!

echo "Waiting for API server to start..."
while ! nc -z localhost $port; do
  sleep 5
done
echo "API server is up!"

# Start generation 
echo "Starting generation for $dataset_path1..."
python run_gen.py \
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path1 \
    --task_name $task1 \
    --port $port > ${eval_data}/${task1}.log 2>&1 &
dataset1_pid=$!

echo "Starting generation for $dataset_path2..."
python run_gen.py \
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path2 \
    --task_name $task2 \
    --port $port > ${eval_data}/${task2}.log 2>&1 &
dataset2_pid=$!

# Wait for all evaluation processes to complete
echo "Waiting for all evaluations to complete..."
for pid in ${dataset1_pid} ${dataset2_pid}; do
    wait $pid
    echo "Process $pid completed"
done

echo "All evaluations completed!"

# Collect and display results

echo "=== Generation Results ==="
for task in $task1 $task2; do
    echo "--- $task ---"
    grep "Correct trajectories for" ${eval_data}/${task}.log
    echo ""
done

echo "Shutting down API server..."
kill $API_PID
'