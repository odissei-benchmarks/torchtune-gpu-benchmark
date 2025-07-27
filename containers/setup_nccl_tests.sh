#!/bin/bash

module purge
module load 2024
#module load MPICH/4.2.2-GCC-13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0

PROJECT_SPACE=/projects/0/prjs1589/torchtune/
CONTAINER_NAME="megatron-torch-25.06-v3-nccl-mpi.sif"
IMAGE=$PROJECT_SPACE/containers/$CONTAINER_NAME

export NCCL_DEBUG=INFO

MLX_LIBS=$(find /usr/lib64 -name "libmlx*.so" -o -name "libibverbs.so*" | xargs -I{} dirname {} | sort | uniq | paste -sd "," -)
IB_LIBS=$(find /usr/lib64 -path "*/libibverbs/*" | xargs -I{} dirname {} | sort | uniq | paste -sd "," -)
