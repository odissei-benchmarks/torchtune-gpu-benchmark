#!/bin/bash
#SBATCH --job-name=mpi-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00
#SBATCH --mem=80G
#SBATCH -p rome
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0

export MPI_DIR="/sw/arch/RHEL9/EB_production/2024/modulefiles/mpi/OpenMPI/5.0.3-GCC-13.3.0.lua"


srun -N 2 -n 8 apptainer exec \
       	--bind "$MPI_DIR" \
	$HOME/data/torchtune/containers/ubuntu-mpi.sif \
	/opt/mpitest

