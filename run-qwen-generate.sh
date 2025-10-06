#!/bin/bash

# ==================== SLURM JOB CONFIGURATION ====================
#SBATCH --job-name=qwen-generate
#SBATCH --output=logs/qwen-generate-%j.out
#SBATCH --error=logs/qwen-generate-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# ==================== EXCLUSION DE NŒUDS PROBLÉMATIQUES ====================
# Exclure les nœuds connus pour avoir des problèmes GPU
#SBATCH --exclude=g181005

# ==================== VARIABLES D'ENVIRONNEMENT ====================
# Configuration de la génération
export PRED_MAX_NEW_TOKENS=${PRED_MAX_NEW_TOKENS:-500}  # Tokens de sortie (500 vs 200 original)

# ==================== INFORMATION SYSTÈME ====================
echo "🧪 GENERATION: Qwen2.5-7B-Instruct (500 tokens)"
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

# ==================== CONFIGURATION DE LA GÉNÉRATION ====================
echo ""
echo "⚙️  Generation Configuration:"
echo "Max new tokens: ${PRED_MAX_NEW_TOKENS:-500}"
echo "Model source: Auto-detect latest run"
echo "Input source: Data_predict/Qwen2.5-7B-full_predict"
echo "Output target: Data_predict/Qwen2.5-7B-full_predict_500"
echo ""

# ==================== STATUS AVANT GÉNÉRATION ====================
echo "📊 Current prediction status:"
echo "Original predictions (200 tokens): $(ls Data_predict/Qwen2.5-7B-full_predict/ 2>/dev/null | wc -l || echo '0')"
echo "New predictions (500 tokens): $(ls Data_predict/Qwen2.5-7B-full_predict_500/ 2>/dev/null | wc -l || echo '0')"
echo ""
echo "📁 Available model runs:"
ls -1t runs/ | head -3
echo ""

# ==================== ESTIMATION DU TEMPS ====================
echo "⏰ Estimated runtime: 10-15 minutes (5 predictions, inference only)"
echo "🎯 Starting generation script..."
echo ""

# ==================== LANCEMENT DU SCRIPT PRINCIPAL ====================
python scripts/generate_qwen_full_predictions.py

# ==================== CAPTURE DU CODE DE RETOUR ====================
exit_code=$?

# ==================== INFORMATION DE FIN ====================
echo ""
echo "================================================="
echo "End time: $(date)"
echo "Exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "✅ Generation completed successfully!"
    
    # Afficher les résultats
    if [ -d "Data_predict/Qwen2.5-7B-full_predict_500" ]; then
        echo ""
        echo "📁 Generated predictions:"
        ls -la Data_predict/Qwen2.5-7B-full_predict_500/
        echo ""
        echo "📝 Preview of first prediction:"
        head -5 Data_predict/Qwen2.5-7B-full_predict_500/*.txt | head -10
    fi
    
else
    echo "❌ Generation failed with exit code: $exit_code"
fi

echo ""
echo "🏁 Job completed on node: $(hostname)"

exit $exit_code
