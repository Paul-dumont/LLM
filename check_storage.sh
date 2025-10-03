#!/bin/bash
# Script de monitoring de l'espace disque sur Longleaf

echo "📊 MONITORING ESPACE DISQUE LONGLEAF"
echo "===================================="
echo ""

# Espace total du home directory
HOME_SIZE=$(du -sh ~ 2>/dev/null | cut -f1)
HOME_SIZE_BYTES=$(du -sb ~ 2>/dev/null | cut -f1)

# Estimation de la limite (à ajuster selon les vraies limites Longleaf)
LIMIT_GB=70
LIMIT_BYTES=$((LIMIT_GB * 1024 * 1024 * 1024))

# Calculs
HOME_GB=$(echo "scale=2; $HOME_SIZE_BYTES / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "N/A")
USED_PERCENT=$(echo "scale=1; $HOME_GB * 100 / $LIMIT_GB" | bc -l 2>/dev/null || echo "N/A")
REMAINING_GB=$(echo "scale=2; $LIMIT_GB - $HOME_GB" | bc -l 2>/dev/null || echo "N/A")

echo "🏠 RÉPERTOIRE HOME: $(pwd | sed 's|/[^/]*$||')"
echo "📁 Espace utilisé: $HOME_SIZE (${HOME_GB}GB)"
echo "📊 Limite estimée: ${LIMIT_GB}GB"
echo "📈 Utilisation: ${USED_PERCENT}%"
echo "💾 Espace restant: ${REMAINING_GB}GB"
echo ""

# Barre de progression
if command -v bc >/dev/null 2>&1 && [[ "$USED_PERCENT" != "N/A" ]]; then
    BARS=$(echo "scale=0; $USED_PERCENT / 5" | bc -l)
    BARS=${BARS%.*}  # Enlever les décimales
    printf "["
    for ((i=1; i<=20; i++)); do
        if [[ $i -le $BARS ]]; then
            if [[ $BARS -lt 12 ]]; then
                printf "█"  # Vert
            elif [[ $BARS -lt 16 ]]; then
                printf "█"  # Jaune
            else
                printf "█"  # Rouge
            fi
        else
            printf "░"
        fi
    done
    printf "] ${USED_PERCENT}%%\n"
fi

echo ""
echo "📂 RÉPARTITION PAR DOSSIER:"
echo "=========================="
cd ~
du -sh .[^.]* * 2>/dev/null | sort -hr | head -10

echo ""
echo "🔍 DÉTAIL PROJET LLM:"
echo "==================="
if [[ -d "$HOME/LONGLEAF" ]]; then
    cd "$HOME/LONGLEAF"
    du -sh . 2>/dev/null | awk '{print "Total projet: " $1}'
    echo "Répartition:"
    du -sh * 2>/dev/null | sort -hr
fi

echo ""
echo "⚠️  NOTE: Limite de 70GB estimée - à vérifier avec 'srun quotas' si disponible"
