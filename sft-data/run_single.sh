#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a5000:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=Qwen3b-rl-sft-gen-zebra_puzzles
#SBATCH --output=slurm_logs/Qwen3b-rl-sft-gen-zebra_puzzles.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

zsh -l -c '
# Define dataset paths
dataset_path="/home/users/hc387/data/zebra_puzzles/qwen-instruct_sft.parquet"
task="zebra_puzzles"

# Define model and which dataset
eval_data="Qwen3b-RL"
output_dir="/home/users/hc387/data/sft_data"
mkdir -p ${eval_data}

model_dir="/usr/xtmp/hc387/TinyZero/qwen-3b/TinyZero/zebra_puzzles/actor/global_step_300"
port=8015

# Start the API server
echo "Starting API server on port $port..."
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
echo "Starting generation for $dataset_path..."
python run_gen.py \
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path \
    --task_name $task \
    --output_dir ${output_dir} \
    --port $port > ${eval_data}/${task}.log 2>&1 &
dataset1_pid=$!

# Wait for all generation processes to complete
echo "Waiting for all generations to complete..."
wait $dataset1_pid

# Collect and display results

echo "=== Generation Results ==="

echo "--- $task ---"
grep "Correct trajectories for" ${eval_data}/${task}.log
echo ""

echo "Shutting down API server..."
kill $API_PID
'