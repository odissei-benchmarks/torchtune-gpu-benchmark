#!/bin/bash
#SBATCH --job-name=l32t_w1g4     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G        # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load anaconda3/2024.6
conda activate tttest-della

WANDB_MODE=offline
MODEL_DIR=/home/niznik/scratch/torchtune_models/Llama-3.2-1B-Instruct/

BATCH_SIZE=1
OUTPUT_DIR=/home/niznik/scratch/torchtune_models/L3-test-finetune-w1g4

# Need this so that wandb doesn't try to write to a directory that doesn't exist
mkdir $OUTPUT_DIR

# TODO - abstract gpu count into a variable
tune run \
    --nproc_per_node 4 \
    lora_finetune_distributed --config llama3_2/1B_lora \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
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