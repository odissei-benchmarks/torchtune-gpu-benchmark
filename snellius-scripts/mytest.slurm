#!/bin/bash
#SBATCH --job-name=a100_8device
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00
#SBATCH --mem=80G
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=4
#SBATCH -e logs/%x-%j-%n.err
#SBATCH -o logs/%x-%j-%n.out

echo "Running on node $SLURM_NODEID"

source requirements/load_venv.sh
source snellius-scripts/setup.sh

# Grab the IP for head node:
# You may need to set this to the fully qualified domain name of your head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

ip address

