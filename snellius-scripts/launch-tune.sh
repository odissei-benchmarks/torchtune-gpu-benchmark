#!/bin/bash

# This simple launcher script sets job-specific environment variables
# and executes the provided command with python.

export WORLD_SIZE=$SLURM_NTASKS  # Note: only valid if ntasks==ngpus
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export NCCL_SOCKET_IFNAME="eth1"
export NCCL_DEBUG=INFO

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"

source snellius-scripts/setup.sh

tune run \
    --nproc_per_node=$SLURM_GPUS \
    --nnodes=$SLURM_JOB_NUM_NODES \  # May need to be GPUs per node
    lora_finetune_distributed \
    --config configs/1B_lora_distributed.yaml \
    max_steps_per_epoch=$MAX_STEPS \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
    checkpointer.output_dir=$OUTPUT_DIR \
    metric_logger.log_dir=$OUTPUT_DIR \
    metric_logger.name=$WANDB_NAME \
    metric_logger.id=$WANDB_NAME \
    batch_size=$BATCH_SIZE