#!/usr/bin/env bash
#SBATCH --job-name=nexa-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out

module load cuda/12.1
module load python/3.11

source /path/to/venv/bin/activate

srun python scripts/cli.py train --config configs/distributed.yaml
