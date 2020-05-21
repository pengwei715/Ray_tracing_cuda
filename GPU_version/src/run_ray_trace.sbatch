#!/bin/bash
#SBATCH --job-name=ray
#SBATCH --output=hist_gpu.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

module load cuda/10.0

./ray
