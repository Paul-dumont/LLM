#!/bin/bash
# Script de vérification rapide à lancer sur longleaf (CPU, pas besoin de GPU)
# Usage: ./check_env.sh

echo "🔍 Vérification environnement LLM sur Longleaf"
echo "=============================================="

# Chargement des modules nécessaires
echo "📦 Chargement des modules..."
module purge
module load cuda/12.2
module add anaconda/2024.02

# Activation conda
echo "🐍 Activation environnement conda..."
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ll-sft
  echo "✅ Environnement ll-sft activé"
else
  echo "❌ Conda introuvable"
  exit 1
fi

# Variables d'environnement recommandées
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_MODE=disabled

echo ""
echo "🚀 Lancement de la vérification complète..."
echo "=============================================="

# Lancer le script Python de vérification
python check_environment.py

# Conserver le code de sortie
exit_code=$?

echo ""
echo "=============================================="
if [ $exit_code -eq 0 ]; then
    echo "🎯 SUCCÈS: Environnement prêt pour l'entraînement!"
    echo "Vous pouvez maintenant soumettre vos jobs SLURM:"
    echo "  MODEL=Qwen-0.5B sbatch run.sh"
    echo "  MODEL=Llama-8B sbatch run.sh"
else
    echo "⚠️  ATTENTION: Problèmes détectés"
    echo "Corrigez les erreurs avant de soumettre des jobs."
fi

exit $exit_code
