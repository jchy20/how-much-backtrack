torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA_DIR \
    data.val_files=$VAL_DATA_DIR \
    data.prompt_key=$PROMPT_KEY \
    data.response_key=$RESPONSE_KEY \
    data.micro_batch_size=$MICRO_BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    data.truncation=right \
    +data.apply_chat_template=False \
    model.partial_pretrain=$BASE_MODEL \
    model.enable_gradient_checkpointing=True \
    trainer.logger=['wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    +trainer.checkpoint_interval=20 \
    trainer.default_local_dir=/usr/xtmp/hc387/TinyZero/qwen-3b/${PROJECT_NAME}/${EXPERIMENT_NAME} 2>&1 | tee ${PROJECT_NAME}_${EXPERIMENT_NAME}.log