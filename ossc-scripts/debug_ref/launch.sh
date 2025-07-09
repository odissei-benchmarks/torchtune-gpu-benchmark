#!/bin/bash

# This simple launcher script sets job-specific environment variables
# and executes the provided command with python.

export WORLD_SIZE=$SLURM_NTASKS  # Note: only valid if ntasks==ngpus
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export NCCL_SOCKET_IFNAME="eth1"
export NCCL_DEBUG=INFO

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"

python "$@"