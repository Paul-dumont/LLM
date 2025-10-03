#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH -t 00:20:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --job-name=qwen-1_5b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/12.2
module add anaconda/2024.02

# activer conda proprement
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda introuvable après 'module add anaconda/2024.02'"; exit 1
fi
conda activate ll-sft

# caches / réseau
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_MODE=disabled

# mini check
python - << 'PY'
import sys, torch
print("Python:", sys.version)
try:
    import transformers, datasets, trl
    print("transformers:", transformers.__version__)
    print("datasets:", datasets.__version__)
    print("trl:", trl.__version__)
except Exception as e:
    print("Import error:", e)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU Memory:", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    except Exception as ee:
        print("GPU info error:", ee)
PY

# lance TON script (même dossier)
python "/nas/longleaf/home/dumontp/LONGLEAF/scripts/quick/Qwen2.5-1.5B.py"
