#!/usr/bin/env python3
"""
Script de validation pour vérifier l'extraction des 56 indicateurs cliniques
"""

import re
from pathlib import Path

# Liste des 56 indicateurs requis
REQUIRED_INDICATORS = [
    "patient_id", "patient_age", "headache_intensity", "headache_frequency", 
    "headache_location", "migraine_history", "migraine_frequency", 
    "average_daily_pain_intensity", "diet_score", "tmj_pain_rating", 
    "disability_rating", "jaw_function_score", "jaw_clicking", "jaw_crepitus", 
    "jaw_locking", "maximum_opening", "maximum_opening_without_pain", 
    "disc_displacement", "muscle_pain_score", "muscle_pain_location", 
    "muscle_spasm_present", "muscle_tenderness_present", "muscle_stiffness_present", 
    "muscle_soreness_present", "joint_pain_areas", "joint_arthritis_location", 
    "neck_pain_present", "back_pain_present", "earache_present", "tinnitus_present", 
    "vertigo_present", "hearing_loss_present", "hearing_sensitivity_present", 
    "sleep_apnea_diagnosed", "sleep_disorder_type", "airway_obstruction_present", 
    "anxiety_present", "depression_present", "stress_present", "autoimmune_condition", 
    "fibromyalgia_present", "current_medications", "previous_medications", 
    "adverse_reactions", "appliance_history", "current_appliance", "cpap_used", 
    "apap_used", "bipap_used", "physical_therapy_status", "pain_onset_date", 
    "pain_duration", "pain_frequency", "onset_triggers", "pain_relieving_factors", 
    "pain_aggravating_factors"
]

def validate_prediction_file(file_path):
    """Valide qu'un fichier de prédiction contient tous les indicateurs requis"""
    
    if not Path(file_path).exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        content = Path(file_path).read_text(encoding='utf-8')
    except Exception as e:
        return {"error": f"Cannot read file: {e}"}
    
    # Extraire tous les indicateurs trouvés (format: indicateur=valeur)
    found_indicators = {}
    for line in content.split('\n'):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            found_indicators[key] = value
    
    # Vérifier la couverture
    missing = set(REQUIRED_INDICATORS) - set(found_indicators.keys())
    extra = set(found_indicators.keys()) - set(REQUIRED_INDICATORS)
    
    # Analyser les valeurs
    unknown_count = sum(1 for v in found_indicators.values() if v.lower() in ['unknown', '', 'n/a'])
    
    return {
        "file": str(file_path),
        "total_found": len(found_indicators),
        "required": len(REQUIRED_INDICATORS),
        "coverage": len(found_indicators) / len(REQUIRED_INDICATORS) * 100,
        "missing": list(missing),
        "extra": list(extra),
        "unknown_count": unknown_count,
        "indicators": found_indicators
    }

def validate_all_predictions(predictions_dir):
    """Valide tous les fichiers de prédictions dans un dossier"""
    
    pred_dir = Path(predictions_dir)
    if not pred_dir.exists():
        print(f"❌ Directory not found: {predictions_dir}")
        return
    
    pred_files = list(pred_dir.glob("*_pred.txt"))
    if not pred_files:
        print(f"❌ No prediction files found in: {predictions_dir}")
        return
    
    print(f"🔍 Validating {len(pred_files)} prediction files...")
    print("=" * 80)
    
    total_coverage = 0
    total_unknown = 0
    all_missing = set()
    
    for pred_file in sorted(pred_files):
        result = validate_prediction_file(pred_file)
        
        if "error" in result:
            print(f"❌ {pred_file.name}: {result['error']}")
            continue
        
        coverage = result['coverage']
        unknown_pct = result['unknown_count'] / result['required'] * 100
        
        total_coverage += coverage
        total_unknown += unknown_pct
        all_missing.update(result['missing'])
        
        # Status emoji
        if coverage == 100:
            status = "✅"
        elif coverage >= 90:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"{status} {pred_file.name}: {coverage:.1f}% coverage, {unknown_pct:.1f}% unknown")
        
        if result['missing']:
            print(f"    Missing: {', '.join(result['missing'][:5])}{'...' if len(result['missing']) > 5 else ''}")
    
    # Résumé global
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   Files analyzed: {len(pred_files)}")
    print(f"   Average coverage: {total_coverage/len(pred_files):.1f}%")
    print(f"   Average unknown: {total_unknown/len(pred_files):.1f}%")
    
    if all_missing:
        print(f"   Most common missing indicators:")
        from collections import Counter
        missing_counts = Counter()
        for pred_file in pred_files:
            result = validate_prediction_file(pred_file)
            if "error" not in result:
                missing_counts.update(result['missing'])
        
        for indicator, count in missing_counts.most_common(10):
            pct = count / len(pred_files) * 100
            print(f"     {indicator}: missing in {count} files ({pct:.1f}%)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        predictions_dir = sys.argv[1]
    else:
        # Par défaut, chercher les prédictions du modèle 7B full
        predictions_dir = "Data_predict/Qwen2.5-7B-full_predict"
    
    print(f"🎯 Validating extraction of 56 clinical indicators")
    print(f"📁 Predictions directory: {predictions_dir}")
    print()
    
    validate_all_predictions(predictions_dir)
