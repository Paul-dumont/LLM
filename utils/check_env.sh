#!/bin/bash
# Script de vérification à lancer sur longleaf (CPU, pas besoin de GPU)
# ⚠️  LECTURE SEULE - Ne modifie pas votre environnement existant
# Usage: 
#   ./check_env.sh           -> Vérification complète
#   ./check_env.sh quick     -> Diagnostic rapide (30 sec)

MODE=${1:-complete}

if [[ "$MODE" == "quick" ]]; then
    echo "⚡ DIAGNOSTIC ULTRA-RAPIDE (30 sec)"
else
    echo "🔍 Vérification environnement LLM complète"
fi
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
if [[ "$MODE" == "quick" ]]; then
    echo "🚀 Lancement du diagnostic rapide..."
    echo "=============================================="
    # Lancer le diagnostic rapide
    python quick_check.py
else
    echo "🚀 Lancement de la vérification complète..."
    echo "=============================================="
    # Lancer le script Python de vérification complète
    python check_environment.py
fi

# Conserver le code de sortie
exit_code=$?

echo ""
echo "=============================================="
if [ $exit_code -eq 0 ]; then
    echo "🎯 SUCCÈS: Environnement prêt pour l'entraînement!"
    echo "Vous pouvez maintenant soumettre vos jobs SLURM:"
    echo "  MODEL=Qwen-0.5B sbatch run.sh"
    echo "  MODEL=Llama-8B sbatch run.sh"
    if [[ "$MODE" == "quick" ]]; then
        echo ""
        echo "💡 Pour une vérification complète: ./check_env.sh"
    fi
else
    echo "⚠️  ATTENTION: Problèmes détectés"
    echo "Corrigez les erreurs avant de soumettre des jobs."
    if [[ "$MODE" == "quick" ]]; then
        echo "💡 Pour plus de détails: ./check_env.sh"
    fi
fi

exit $exit_code
