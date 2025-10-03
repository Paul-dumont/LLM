#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH -t 00:30:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --job-name=qwen-7b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "🚀 QUICK TEST: Qwen2.5-7B-Instruct"
echo "================================================="

# 1) Modules
module purge
module load cuda/12.2
module load anaconda/2024.02

# 2) Activer conda (hook)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda introuvable après 'module add anaconda/2024.02'"; exit 1
fi
conda activate ll-sft

# 3) Caches / Réseau HF
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_MODE=disabled

# 4) Quick check
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
        print("GPU Memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as ee:
        print("GPU info error:", ee)
PY

# 5) Lancement du script quick dédié 7B
QUICK_SCRIPT="scripts/quick/Qwen2.5-7B-Instruct-quick.py"
echo -e "\n🎯 Launching: $QUICK_SCRIPT"
echo -e "⏰ Expected runtime: ~5-10 minutes\n"

start_time=$(date +%s)
export QUICK_MAX_INPUTS=${QUICK_MAX_INPUTS:-50}
export QUICK_MAX_PREDICTIONS=${QUICK_MAX_PREDICTIONS:-20}
export QUICK_PRED_SOURCE=${QUICK_PRED_SOURCE:-all}
python "$QUICK_SCRIPT"
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo -e "\n✅ Terminé en ${runtime}s ($(($runtime / 60))min $(($runtime % 60))s)"

echo -e "\n🏁 Quick test Qwen2.5-7B-Instruct terminé!"
