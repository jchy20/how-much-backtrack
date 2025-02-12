# Set environment variables
export N_GPUS=4
export BASE_MODEL=/usr/xtmp/hc387/models/Qwen2.5-3B-Instruct
export DATA_DIR=/home/users/hc387/data/leg_counting
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=leg-counting-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

# Run the training script
bash train_tiny_zero.sh
