#!/bin/bash
#SBATCH --job-name=a100_8device
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=18
#SBATCH --time=20:00
#SBATCH --mem=80G
#SBATCH -p gpu_a100
#SBATCH --gpus 4
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

source requirements/load_venv.sh
source snellius-scripts/setup.sh

# Grab the IP for head node:
# You may need to set this to the fully qualified domain name of your head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

srun tune run \
    --nproc_per_node=$SLURM_GPUS \
    --nnodes=$SLURM_JOB_NUM_NODES \
    lora_finetune_distributed \
    --config configs/1B_lora_distributed.yaml \
    max_steps_per_epoch=$MAX_STEPS \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    tokenizer.path=$MODEL_DIR/original/tokenizer.model \
    checkpointer.output_dir=$OUTPUT_DIR \
    metric_logger.log_dir=$OUTPUT_DIR \
    metric_logger.name=$WANDB_NAME \
    metric_logger.id=$WANDB_NAME \
    batch_size=3 \
    dataset.data_files="datasets/alpaca_data_cleaned.json" \
    dataset.source="json"


