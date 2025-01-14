#!/bin/bash
#
#SBATCH --job-name=lora_dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=15:00
#SBATCH --mem=80G
#SBATCH -p gpu_h100
#SBATCH --gpus 4
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out


source requirements/load_venv.sh

MODEL_VERSION=3-2-1b-Instruct

WANDB_MODE=offline
MODEL_DIR=/projects/0/prjs1019/torchtune/models/Llama-${MODEL_VERSION}

BATCH_SIZE=1
OUTPUT_DIR=/projects/0/prjs1019/torchtune/outputs/${MODEL_VERSION}

tune run \
    --nproc_per_node 4 \
    lora_finetune_distributed \
    --config llama3_2/1B_lora \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
    max_steps_per_epoch=512 \
    checkpointer.output_dir=$OUTPUT_DIR \
    dataset.packed=True \
    tokenizer.max_seq_len=2048 \
    seed=14 \
    batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=1 \
    metric_logger._component_=torchtune.training.metric_logging.WandBLogger \
    metric_logger.log_dir=$OUTPUT_DIR \
    metric_logger.project="llama3.2_lora" \
    metric_logger.mode="offline" \
    log_every_n_steps=5 




