#!/bin/bash
#
#SBATCH --job-name=h100_8device_con
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00
#SBATCH --mem=80G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=4
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"

# Grab the IP for head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

# eth1 for OSSC
export NCCL_SOCKET_IFNAME="eno2np0"
export NCCL_DEBUG=INFO

# Set distributed training env variables
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=12345
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))

srun apptainer exec \
  --bind /projects/0/prjs1019:/prjs1019 \
  --bind /tmp:/home/mniznik/GitHub/torchtune-gpu-benchmark/snellius-scripts/tmp_exec \
  --env MASTER_PORT=$MASTER_PORT \
  --env MASTER_ADDR=$MASTER_ADDR \
  --env WORLD_SIZE=$WORLD_SIZE \
  --env GPUS_PER_NODE=$GPUS_PER_NODE \
  --nv \
  /projects/0/prjs1019/torchtune/containers/megatron-torch-2.7-nvcr.25-06.sif \
  ./launch-tune.sh
