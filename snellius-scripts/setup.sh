#!/bin/bash

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

echo "Running on host: $RDZV_HOST"

MAX_STEPS=512 # 128 for testing, 512 for benchmarking
BATCH_SIZE=6

MODEL_NAME="Llama-3-2-1b-Instruct"

if [[ $RDZV_HOST == *"ossc"* ]]; then
    DATA_ROOT="/gpfs/ostor/ossc9424/homedir/torchtune/"
else
    DATA_ROOT="/projects/0/prjs1019/torchtune"
fi

MODEL_DIR="${DATA_ROOT}/models/${MODEL_NAME}"
OUTPUT_DIR="${DATA_ROOT}/outputs/${MODEL_NAME}"

WANDB_DIR="${OUTPUT_DIR}/wandb"
if [ ! -d "${WANDB_DIR}" ]; then
    mkdir -p "${WANDB_DIR}"
fi

WANDB_NAME=$SLURM_JOB_NAME




