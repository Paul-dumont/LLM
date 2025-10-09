#!/bin/bash

# ==================== SLURM JOB CONFIGURATION ====================
#SBATCH --job-name=qwen-7b-full
#SBATCH --output=logs/qwen-7b-full-%j.out
#SBATCH --error=logs/qwen-7b-full-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# ==================== EXCLUSION DE NŒUDS PROBLÉMATIQUES ====================
# Exclure les nœuds connus pour avoir des problèmes GPU
#SBATCH --exclude=g181005

# ==================== VARIABLES D'ENVIRONNEMENT ====================
# Configuration de l'entraînement complet
export FULL_MAX_INPUTS=${FULL_MAX_INPUTS:-0}           # 0 = utiliser toutes les données
export FULL_EVAL_RATIO=${FULL_EVAL_RATIO:-0.2}         # 20% pour l'évaluation
export FULL_USE_WANDB=${FULL_USE_WANDB:-0}              # Weights & Biases logging
export FULL_SAVE_FINAL=${FULL_SAVE_FINAL:-1}           # Sauvegarder le modèle final
export FULL_MAX_PREDICTIONS=${FULL_MAX_PREDICTIONS:-5}  # Prédictions test après entraînement
export FULL_SEED=${FULL_SEED:-42}                      # Seed pour reproductibilité

# ==================== INFORMATION SYSTÈME ====================
echo "🚀 FULL TRAINING: Qwen2.5-7B-Instruct"
echo "================================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Python: $(python --version 2>&1)"

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

# ==================== ACTIVATION DE L'ENVIRONNEMENT ====================
echo ""
echo "🐍 Activating conda environment..."
source ~/.bashrc
conda activate ll-sft

# ==================== VÉRIFICATION DE L'ENVIRONNEMENT ====================
echo ""
echo "🔍 Environment check:"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "datasets: $(python -c 'import datasets; print(datasets.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "trl: $(python -c 'import trl; print(trl.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'NO TORCH')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️  GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
else
    echo "❌ nvidia-smi not available"
fi

# ==================== CRÉATION DU DOSSIER DE LOGS ====================
mkdir -p logs

# ==================== CONFIGURATION DE L'ENTRAÎNEMENT ====================
echo ""
echo "⚙️  Training Configuration:"
echo "Max inputs: ${FULL_MAX_INPUTS:-ALL}"
echo "Eval ratio: ${FULL_EVAL_RATIO:-0.2}"
echo "Use W&B: ${FULL_USE_WANDB:-0}"
echo "Save final: ${FULL_SAVE_FINAL:-1}"
echo "Test predictions: ${FULL_MAX_PREDICTIONS:-5}"
echo "Seed: ${FULL_SEED:-42}"
echo ""

# ==================== ESTIMATION DU TEMPS ====================
echo "⏰ Estimated runtime: 3-4 hours (500 files, 3000 tokens, 95.4% coverage, full fine-tuning)"
echo "🎯 Starting full training script..."
echo ""

# ==================== LANCEMENT DU SCRIPT PRINCIPAL ====================
python scripts/Qwen2.5-7B-train-full.py

# ==================== CAPTURE DU CODE DE RETOUR ====================
exit_code=$?

# ==================== INFORMATION DE FIN ====================
echo ""
echo "================================================="
echo "End time: $(date)"
echo "Exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "✅ Full training completed successfully!"
    
    # Afficher la taille du dossier de sauvegarde si il existe
    if [ -d "runs" ]; then
        echo ""
        echo "📁 Training outputs:"
        find runs -name "Qwen2.5-7B-full_*" -type d -exec sh -c 'echo "  $(du -sh "$1" | cut -f1) - $1"' _ {} \; | tail -5
    fi
    
else
    echo "❌ Full training failed with exit code: $exit_code"
fi

echo ""
echo "🏁 Job completed on node: $(hostname)"

exit $exit_code
