#!/bin/bash
#SBATCH --job-name=h100_1device
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00
#SBATCH --mem=80G
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:8
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

export NCCL_SOCKET_IFNAME="eth1"
export NCCL_DEBUG=INFO

# Set distributed training env variables
export MaSTER_ADDR=$head_node_ip
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS


srun apptainer exec --env MASTER_PORT=$MASTER_PORT --env MASTER_ADDR=$MASTER_ADDR --nv megatron-torch-2.7-nvcr.25-05.sif ./launch.sh script.py

