#!/bin/bash

eval_dataset_dir=""
model_dir="/usr/xtmp/hc387/TinyZero/qwen-3b/TinyZero/advanced-geometry-incircle-radius-qwen2.5-3b/actor/global_step_700"
port=8000

# Ensure vllm is installed correctly
# export HF_HOME="/home/users/hc387/.cache/huggingface"

python -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port $port


python vllm_eval.py \
    --model_path $model_dir \
    --eval_dataset_dir $eval_dataset_dir \
    --port $port