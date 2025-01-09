#!/bin/bash
#
#SBATCH --job-name=tune_lora
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --nodes=1
#SBATCH --time=15:00
#SBATCH --mem=80G
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out


source requirements/load_venv.sh


#tune run full_finetune_single_device --config full_low_memory.yaml
tune run lora_finetune_single_device --config configs/lora_single.yaml


