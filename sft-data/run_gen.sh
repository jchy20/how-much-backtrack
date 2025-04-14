#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a5000:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=Qwen3b-RL-sft-generation
#SBATCH --output=slurm_logs/Qwen3b-rl-sft-generation-arc1d.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

# Define dataset paths
# dataset_path1="/home/users/hc387/data/advanced_geometry/qwen-instruct_sft.parquet"
# dataset_path1="/home/users/hc387/data/countdown/qwen-instruct_sft.parquet"
dataset_path1="/home/users/hc387/data/arc_1d/qwen-instruct_sft.parquet"
# dataset_path1="/home/users/hc387/data/sudoku/qwen-instruct_sft.parquet"
# dataset_path1="/home/users/hc387/data/color_cube_rotation/qwen-instruct_sft.parquet"
# dataset_path1="/home/users/hc387/data/zebra_puzzles/qwen-instruct_sft.parquet"
# dataset_path1="/home/users/hc387/data/list_functions/qwen-instruct_sft.parquet"
# dataset_path1="/home/users/hc387/data/self_reference/qwen-instruct_sft.parquet"
# task1="advanced_geometry"
# task1="countdown"
task1="arc_1d"
# task1="sudoku"
# task1="color_cube_rotation"
# task1="zebra_puzzles"
# task1="list_functions"
# task1="self_reference"

model_path="junlinw/Qwen2.5_3B-Instruct_arc1d_step800"
model_type="Qwen3b-RL"
port=8000
output_dir="/home/users/hc387/data/sft_data"
save_interval=20000
batch_size=128
n=5

mkdir -p ${model_type}

# Start the API server
python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port $port > ${model_type}/api_server_${port}.log 2>&1 &

API_PID=$!

echo "Waiting for API server to start..."
while ! nc -z localhost $port; do
  sleep 5
done
echo "API server is up!"

# Start generation 
echo "Starting generation for $dataset_path1..."
python run_gen.py \
    --model_path $model_path \
    --model_type $model_type \
    --eval_dataset_dir $dataset_path1 \
    --task_name $task1 \
    --output_dir ${output_dir} \
    --batch_size $batch_size \
    --n $n \
    --save_interval $save_interval \
    --port $port > ${model_type}/${task1}.log 2>&1 &
dataset1_pid=$!

# Wait for all generation processes to complete
echo "Waiting for all generations to complete..."
wait $dataset1_pid
echo "Process $dataset1_pid completed"

# Collect and display results
echo "=== Generation Results ==="
echo "--- $task1 ---"
grep "Correct trajectories for" ${model_type}/${task1}.log
echo ""

echo "Shutting down API server..."
kill $API_PID