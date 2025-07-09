#!/bin/bash

# This simple launcher script sets job-specific environment variables
# and executes the provided command with python.

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# eth1 for OSSC
export NCCL_SOCKET_IFNAME="eno2np0"
export NCCL_DEBUG=INFO

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"

export APPTAINER_TMPDIR=/home/mniznik/GitHub/torchtune-gpu-benchmark/snellius-scripts/tmp/
export APPTAINER_CACHEDIR=/home/mniznik/GitHub/torchtune-gpu-benchmark/snellius-scripts/cache/

source setup.sh

tune run \
    --nproc_per_node=4 \
    --nnodes=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    lora_finetune_distributed \
    --config ../configs/1B_lora_distributed.yaml \
    max_steps_per_epoch=$MAX_STEPS \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
    checkpointer.output_dir=$OUTPUT_DIR \
    metric_logger.log_dir=$OUTPUT_DIR \
    metric_logger.name=$WANDB_NAME \
    metric_logger.id=$WANDB_NAME \
    batch_size=$BATCH_SIZE