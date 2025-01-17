#!/bin/bash
#
#SBATCH --job-name=a100_1device
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=15:00
#SBATCH --mem=80G
#SBATCH -p gpu_a100
#SBATCH --gpus 1
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

source requirements/load_venv.sh
source snellius-scripts/setup.sh

tune run \
    lora_finetune_single_device \
    --config configs/1B_lora_single_device.yaml \
    max_steps_per_epoch=$MAX_STEPS \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
    checkpointer.output_dir=$OUTPUT_DIR \
    metric_logger.log_dir=$OUTPUT_DIR \
    metric_logger.name=$WANDB_NAME \
    metric_logger.id=$WANDB_NAME \
    batch_size=$BATCH_SIZE


