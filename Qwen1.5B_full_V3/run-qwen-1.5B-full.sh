#!/bin/bash
#
# ==================== SLURM JOB CONFIG ====================
#SBATCH --job-name=qwenV3
#SBATCH --output=Qwen1.5B_full_V3/logs/qwen-1.5b-full-%j.out
#SBATCH --error=Qwen1.5B_full_V3/logs/qwen-1.5b-full-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# --- Choisis l'une des deux lignes suivantes selon Longleaf ---
#SBATCH --partition=l40-gpu            # Partition dédiée L40/L40s

# Nombre et type de GPU (si requis par la conf SLURM)
#SBATCH --gres=gpu:1

# QoS si exigé par ta politique locale (à commenter sinon)
#SBATCH --qos=gpu_access

# Optionnel : exclure un noeud instable
# #SBATCH --exclude=g181005

echo "================================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "CWD:   $(pwd)"
echo "================================================="

# ==================== MODULES / ENV ====================
module purge
module load cuda/12.2
module load anaconda/2024.02

# Activer conda
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda introuvable après 'module load anaconda'"; exit 1
fi

# Active ton env qui contient torch/transformers/trl/datasets
source ~/.bashrc
conda activate ll-sft || { echo "conda env ll-sft introuvable"; exit 1; }

# ==================== RÉGLAGES RUNTIME ====================
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN

echo ""
echo "🔍 Environment check"
python - <<'PY'
import torch, transformers, datasets, trl
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| is_available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("trl:", trl.__version__)
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY

# ==================== CRÉATION DU DOSSIER DE LOGS ====================
mkdir -p Qwen1.5B_full/logs

if command -v nvidia-smi &> /dev/null; then
  echo ""; echo "🖥️  GPU:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# ==================== LANCEMENT ====================
# Adapte le chemin si ton fichier est ailleurs
SCRIPT="Qwen1.5B_full_V3/Qwen2.5-1.5B-full.py"

echo ""
echo "🚀 Launching: $SCRIPT"
python "$SCRIPT"
code=$?

echo ""
echo "================================================="
echo "End: $(date)"
echo "Exit code: $code"
exit $code
