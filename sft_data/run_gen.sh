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
dataset_path1="/home/users/hc387/data/advanced_geometry/qwen-instruct_sft.parquet"
dataset_path2="/home/users/hc387/data/countdown/qwen-instruct_sft.parquet"
dataset_path3="/home/users/hc387/data/arc_1d/qwen-instruct_sft.parquet"
dataset_path4="/home/users/hc387/data/sudoku/qwen-instruct_sft.parquet"
dataset_path5="/home/users/hc387/data/color_cube_rotation/qwen-instruct_sft.parquet"
dataset_path6="/home/users/hc387/data/zebra_puzzles/qwen-instruct_sft.parquet"
dataset_path7="/home/users/hc387/data/list_functions/qwen-instruct_sft.parquet"
dataset_path8="/home/users/hc387/data/self_reference/qwen-instruct_sft.parquet"
task1="advanced_geometry"
task2="countdown"
task3="arc_1d"
task4="sudoku"
task5="color_cube_rotation"
task6="zebra_puzzles"
task7="list_functions"
task8="self_reference"

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

echo "Starting generation for $dataset_path3..."
python run_gen.py \
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path3 \
    --task_name $task3 \
    --port $port > ${eval_data}/${task3}.log 2>&1 &
dataset3_pid=$!

echo "Starting generation for $dataset_path4..."
python run_gen.py \ 
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path4 \
    --task_name $task4 \
    --port $port > ${eval_data}/${task4}.log 2>&1 &
dataset4_pid=$!

echo "Starting generation for $dataset_path5..."
python run_gen.py \
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path5 \
    --task_name $task5 \
    --port $port > ${eval_data}/${task5}.log 2>&1 &
dataset5_pid=$!

echo "Starting generation for $dataset_path6..."
python run_gen.py \
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path6 \
    --task_name $task6 \
    --port $port > ${eval_data}/${task6}.log 2>&1 &
dataset6_pid=$!

echo "Starting generation for $dataset_path7..."
python run_gen.py \
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path7 \
    --task_name $task7 \
    --port $port > ${eval_data}/${task7}.log 2>&1 &
dataset7_pid=$!

echo "Starting generation for $dataset_path8..."
python run_gen.py \
    --model_path $model_dir \
    --model_type $eval_data \
    --eval_dataset_dir $dataset_path8 \
    --task_name $task8 \
    --port $port > ${eval_data}/${task8}.log 2>&1 &
dataset8_pid=$!

# Store all PIDs in an array
eval_pids=($dataset1_pid $dataset2_pid $dataset3_pid $dataset4_pid $dataset5_pid $dataset6_pid $dataset7_pid $dataset8_pid)

# Wait for all generation processes to complete
echo "Waiting for all generations to complete..."
for pid in ${eval_pids[@]}; do
    wait $pid
    echo "Process $pid completed"
done

echo "All generations completed!"

# Collect and display results

echo "=== Generation Results ==="
for task in $task1 $task2 $task3 $task4 $task5 $task6 $task7 $task8; do
    echo "--- $task ---"
    grep "Correct trajectories for" ${eval_data}/${task}.log
    echo ""
done

echo "Shutting down API server..."
kill $API_PID
'