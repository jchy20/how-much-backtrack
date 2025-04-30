#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --time=14-00:00:00
#SBATCH --gres=gpu:a5000:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=QwQ32B-sft-generation-arc_1d
#SBATCH --array=0-79%4
#SBATCH --output=slurm_logs/QwQ32B-sft-generation-arc_1d.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

### Config
DATASET_PATH="/home/users/hc387/data/arc_1d/qwen-instruct_sft.parquet"
TASK="arc_1d"
JOB_ID=$SLURM_ARRAY_TASK_ID
MODEL_PATH="Qwen/QwQ-32B"
MODEL_TYPE="QwQ-32B"
BASE_PORT=8000
OUTPUT_DIR="/home/users/hc387/data/sft_data/${TASK}"    # output directory
SAVE_INTERVAL=20000                                     # save interval
BATCH_SIZE=8                                            # batch size for generation
N=1                                                     # number of samples to generate
PORT=$((BASE_PORT + JOB_ID))                            # port number for the API server
TOTAL_SPLITS=80                                         # number of splits

mkdir -p ${MODEL_TYPE}

# Start the API server
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port $PORT > ${MODEL_TYPE}/api_server_${PORT}.log 2>&1 &

API_PID=$!

echo "Waiting for API server to start..."
sleep 10
while ! nc -z localhost $PORT; do
  sleep 5
done
echo "API server is up!"

# Start generation 
echo "Starting generation for $DATASET_PATH..."
python run_gen.py \
    --model_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --eval_dataset_dir $DATASET_PATH \
    --task_name $TASK \
    --output_dir ${OUTPUT_DIR} \
    --batch_size $BATCH_SIZE \
    --n $N \
    --save_interval $SAVE_INTERVAL \
    --job_id $JOB_ID \
    --total_splits $TOTAL_SPLITS \
    --port $PORT > ${MODEL_TYPE}/${TASK}_${JOB_ID}.log 2>&1 &
dataset1_pid=$!

# Wait for all generation processes to complete
echo "Waiting for all generations to complete..."
wait $dataset1_pid
echo "Process $dataset1_pid completed"

# Collect and display results
echo "=== Generation Results ==="
echo "--- $TASK ---"
grep "Correct trajectories for" ${MODEL_TYPE}/${TASK}_${JOB_ID}.log
echo ""

echo "Shutting down API server..."
kill $API_PID