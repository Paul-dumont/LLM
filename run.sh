#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH -t 00:20:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --job-name=sft-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Sécurité + revenir dans le dossier de soumission
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Assurer l'existence du dossier logs
mkdir -p logs

############################
# 1) Modules
############################
module purge
module load cuda/12.2
module add anaconda/2024.02

############################
# 2) Activer conda (hook)
############################
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda introuvable après 'module add anaconda/2024.02'"; exit 1
fi
conda activate ll-sft

############################
# 3) Caches / Réseau HF
############################
export HF_HOME="$HOME/.cache/huggingface"   # cache unique et persistant
# Ne plus définir TRANSFORMERS_CACHE (déprécié dans v5) ; HF lit HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1          # fast downloader (tu as installé hf_transfer)
# (optionnel) éviter tout logging wandb involontaire
export WANDB_MODE=disabled

############################
# 4) Petit check (stdout)
############################
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
    except Exception as ee:
        print("GPU name error:", ee)
PY

############################
# 5) Lancer ton script
############################
python quick_run.py
