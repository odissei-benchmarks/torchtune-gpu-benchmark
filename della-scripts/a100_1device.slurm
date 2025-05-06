#!/bin/bash
#SBATCH --job-name=a100_g1b6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --output=logs/%x-%j.out

module purge
module load anaconda3/2024.6
conda activate ttenv
source setup-della.sh

tune run \
    lora_finetune_single_device \
    --config ../configs/1B_lora_single_device.yaml \
    max_steps_per_epoch=$MAX_STEPS \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
    output_dir=$OUTPUT_DIR \
    metric_logger.log_dir=$OUTPUT_DIR \
    metric_logger.name=$WANDB_NAME \
    metric_logger.id=$WANDB_NAME \
    batch_size=$BATCH_SIZE


