# Define model and which dataset
eval_data="advanced_geometry_checkpoint1_eval"
mkdir -p ${eval_data}

# Path to dataset and model
task_name="advanced_geometry"
dataset_path="~/data/advanced_geometry/test.parquet"
model_dir="~/models/checkpoint1.pt"
port=8000

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

# Run all evaluations in parallel and capture PIDs directly
echo "Starting evaluation for $dataset_path..."
python baseline_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $dataset_path \
    --task_name $task_name \
    --baseline $baseline \
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