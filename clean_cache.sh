#!/bin/bash
# Script de nettoyage du cache Hugging Face pour libérer de l'espace

echo "🧹 NETTOYAGE CACHE HUGGING FACE"
echo "==============================="

cd ~/.cache/huggingface/hub

echo "📊 ESPACE AVANT NETTOYAGE:"
du -sh ~/.cache/huggingface

echo ""
echo "🔍 MODÈLES EN CACHE:"
du -sh * 2>/dev/null | sort -hr

echo ""
echo "⚠️  SUPPRESSION DES MODÈLES NON-QWEN (on garde uniquement 'models--Qwen--*')"

shopt -s nullglob
for model in models--*; do
    if [[ "$model" == models--Qwen--* ]]; then
        echo "✅ Gardé (Qwen): $model"
    else
        echo "🗑️  Suppression (non-Qwen): $model"
        rm -rf -- "$model"
    fi
done
shopt -u nullglob

echo ""
echo "🧹 NETTOYAGE CACHE PIP:"
pip cache purge || true

echo ""
echo "📊 ESPACE APRÈS NETTOYAGE:"
du -sh ~/.cache/huggingface || true
du -sh ~/ || true

echo ""
echo "🎯 NETTOYAGE TERMINÉ!"
