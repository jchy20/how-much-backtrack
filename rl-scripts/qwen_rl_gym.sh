# Set environment variables
export N_GPUS=4
export BASE_MODEL=/usr/xtmp/hc387/models/Qwen2.5-3B-Instruct
export DATA_DIR=/home/users/hc387/data/advanced_geometry/incircle_radius
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=advanced-geometry-incircle-radius-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

# Run the training script
bash train_tiny_zero.sh
