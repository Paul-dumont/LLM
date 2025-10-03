#!/bin/bash
# Script de nettoyage du cache Hugging Face pour libérer de l'espace

echo "🧹 NETTOYAGE CACHE HUGGING FACE"
echo "==============================="

cd ~/.cache/huggingface/hub

echo "📊 ESPACE AVANT NETTOYAGE:"
du -sh ~/.cache/huggingface

echo ""
echo "🔍 MODÈLES EN CACHE:"
du -sh * | sort -hr

echo ""
echo "⚠️  SUPPRESSION DES ANCIENS MODÈLES (garde les 4 principaux):"

# Garder seulement les modèles actuellement utilisés
KEEP_MODELS=(
    "models--HuggingFaceH4--zephyr-7b-beta"
    "models--microsoft--Phi-3.5-mini-instruct" 
    "models--Qwen--Qwen2.5-7B-Instruct"
    "models--NousResearch--Llama-2-7b-chat-hf"
)

# Supprimer les autres modèles
for model in models--*; do
    if [[ ! " ${KEEP_MODELS[@]} " =~ " ${model} " ]]; then
        echo "🗑️  Suppression: $model"
        rm -rf "$model"
    else
        echo "✅ Gardé: $model"
    fi
done

echo ""
echo "🧹 NETTOYAGE CACHE PIP:"
pip cache purge

echo ""
echo "📊 ESPACE APRÈS NETTOYAGE:"
du -sh ~/.cache/huggingface
du -sh ~/

echo ""
echo "🎯 NETTOYAGE TERMINÉ!"
