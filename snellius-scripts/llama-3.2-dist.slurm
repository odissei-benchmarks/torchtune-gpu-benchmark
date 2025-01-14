#!/bin/bash
#
#SBATCH --job-name=lora_dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=15:00
#SBATCH --mem=80G
#SBATCH -p gpu_a100
#SBATCH --gpus 4
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out


MODEL_VERSION=3-2-1b-Instruct
MODEL_DIR=/projects/0/prjs1019/torchtune/models/Llama-${MODEL_VERSION}
OUTPUT_DIR=/projects/0/prjs1019/torchtune/outputs/${MODEL_VERSION}


source requirements/load_venv.sh
source snellius-scripts/common_params.sh

tune run \
    --nproc_per_node 4 \
    lora_finetune_distributed \
    --config llama3_2/1B_lora \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    checkpointer.output_dir=$OUTPUT_DIR \
    metric_logger._component_=torchtune.training.metric_logging.WandBLogger \
    metric_logger.log_dir=$OUTPUT_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
    max_steps_per_epoch=$MAX_STEPS_PER_EPOCH \
    dataset.packed=$DATASET_PACKED \
    tokenizer.max_seq_len=$MAX_SEQ_LEN \
    seed=$RANDOM_SEED \
    batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$GRAD_ACCUMUL_STEPS \
    metric_logger.project=$WANDB_PROJECT \
    metric_logger.mode=$WANDB_MODE \
    log_every_n_steps=$LOG_EVERY_N_STEPS 




