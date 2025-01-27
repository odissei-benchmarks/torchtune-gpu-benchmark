#!/bin/bash

# ----- Params to Tweak -----
BATCH_SIZE=6
# ----- End Params to Tweak -----

MAX_STEPS=512 # 128 for testing, 512 for benchmarking

MODEL_NAME="Llama-3.2-1B-Instruct"
MODEL_NAME_SHORT="L32"
DATA_ROOT="/scratch/gpfs/${USER}/" # abstracted for any user on della

MODEL_DIR="${DATA_ROOT}/torchtune_models/${MODEL_NAME}"
OUTPUT_DIR="${DATA_ROOT}/torchtune_bench/${MODEL_NAME}/${SLURM_JOB_NAME}"

WANDB_DIR="${OUTPUT_DIR}/wandb"
if [ ! -d "${WANDB_DIR}" ]; then
    mkdir -p "${WANDB_DIR}"
fi

WANDB_NAME=$SLURM_JOB_NAME
