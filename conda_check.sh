#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH -t 00:02:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=logs/conda-check-%j.out
#SBATCH --error=logs/conda-check-%j.err

module purge
module add anaconda/2024.02
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda not found"; exit 1
fi
conda activate ll-sft
python -c "import transformers, datasets, trl; print('OK imports'); import torch; print('CUDA:', torch.cuda.is_available())"
