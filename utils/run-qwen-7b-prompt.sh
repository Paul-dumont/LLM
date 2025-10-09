#!/bin/bash

# ==================== SLURM JOB CONFIGURATION ====================
#SBATCH --job-name=qwen-7b-prompt
#SBATCH --output=logs/qwen-7b-prompt-%j.out
#SBATCH --error=logs/qwen-7b-prompt-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=l40-gpu
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

python scripts/Qwen2.5-7B-prompt.py

exit_code=$?

# ==================== INFORMATION DE FIN ====================

if [ $exit_code -eq 0 ]; then
    
    # Afficher le nombre de fichiers générés
    if [ -d "Data_predict/Qwen2.5-7B-base_predict" ]; then
        ls Data_predict/Qwen2.5-7B-base_predict/*.txt | wc -l
    fi
    
else
    echo "Inference failed with exit code: $exit_code"
fi

exit $exit_code