#!/bin/bash
#
#SBATCH --job-name=lora_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=15:00
#SBATCH --mem=80G
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out


MODEL_NAME="Llama-3-2-1b-Instruct"
DATA_ROOT="/projects/0/prjs1019"

MODEL_DIR="${DATA_ROOT}/torchtune/models/${MODEL_NAME}"
OUTPUT_DIR="${DATA_ROOT}/torchtune/outputs/${MODEL_NAME}"

WANDB_DIR="${OUTPUT_DIR}/wandb"
if [ ! -d "${WANDB_DIR}" ]; then
    mkdir -p "${WANDB_DIR}"
fi

source requirements/load_venv.sh

tune run \
    lora_finetune_single_device \
    --config configs/1B_lora_single_device.yaml \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
    checkpointer.output_dir=$OUTPUT_DIR \
    metric_logger.log_dir=$OUTPUT_DIR 


