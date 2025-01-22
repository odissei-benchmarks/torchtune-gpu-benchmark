#!/bin/bash

# Submit all jobs
# invoke from repo root with source snellius-scripts/run_jobs.sh

SCRIPT_DIR="snellius-scripts"

rm -rf logs/*

sbatch "${SCRIPT_DIR}/a100_1device.slurm"
sbatch "${SCRIPT_DIR}/a100_2device.slurm"
sbatch "${SCRIPT_DIR}/a100_4device.slurm"

sbatch "${SCRIPT_DIR}/h100_1device.slurm"
sbatch "${SCRIPT_DIR}/h100_2device.slurm"
sbatch "${SCRIPT_DIR}/h100_4device.slurm"


