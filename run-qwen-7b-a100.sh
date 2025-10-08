#!/bin/bash

# ==================== SLURM JOB CONFIGURATION ====================
#SBATCH --job-name=qwen-7b-a100
#SBATCH --output=logs/qwen-7b-a100-%j.out
#SBATCH --error=logs/qwen-7b-a100-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# ==================== EXCLUSION DE NŒUDS PROBLÉMATIQUES ====================
#SBATCH --exclude=g181005

# ==================== INFORMATION SYSTÈME ====================

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
source ~/.bashrc
conda activate ll-sft

# ==================== VÉRIFICATION DE L'ENVIRONNEMENT ====================

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
fi

# ==================== CRÉATION DU DOSSIER DE LOGS ====================
mkdir -p logs

# ==================== CONFIGURATION DE L'ENTRAÎNEMENT ====================
echo "Training Configuration:"
echo "Model: Qwen/Qwen2.5-7B-Instruct (full fine-tuning, no LoRA/quantization)"
echo "Prompt: Simple extraction prompt"
echo "Dataset: 500 medical notes"
echo "Output dir: runs/Qwen2.5-7B-full_*/"

# ==================== ESTIMATION DU TEMPS ====================
echo "Estimated runtime: ~8-12 hours (full training)"

# ==================== LANCEMENT DU SCRIPT PRINCIPAL ====================
python scripts/Qwen2.5-7B-a100.py

# ==================== CAPTURE DU CODE DE RETOUR ====================
exit_code=$?

# ==================== INFORMATION DE FIN ====================

if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Afficher les dossiers créés
    ls -d runs/Qwen2.5-7B-full_* 2>/dev/null | head -1
    
else
    echo "Training failed with exit code: $exit_code"
fi
