#!/usr/bin/env python3
"""
DIAGNOSTIC ULTRA-RAPIDE - 30 secondes max
Usage: python quick_check.py
"""

import sys
from pathlib import Path

def quick_check():
    print("🚀 DIAGNOSTIC ULTRA-RAPIDE")
    print("=" * 30)
    
    issues = []
    
    # 1. Python
    if sys.version_info < (3, 8):
        issues.append("Python trop ancien")
    else:
        print("✅ Python OK")
    
    # 2. Imports critiques
    critical = ['torch', 'transformers', 'datasets', 'trl']
    for lib in critical:
        try:
            __import__(lib)
            print(f"✅ {lib}")
        except ImportError:
            print(f"❌ {lib} manquant")
            issues.append(f"{lib} manquant")
    
    # 3. Données
    if Path("Data_input").exists() and Path("Data_output").exists():
        input_files = list(Path("Data_input").glob("*.txt"))
        output_files = list(Path("Data_output").glob("*.txt"))
        print(f"✅ Données: {len(input_files)} inputs, {len(output_files)} outputs")
    else:
        print("❌ Dossiers Data_input/Data_output manquants")
        issues.append("Données manquantes")
    
    # 4. CUDA (si disponible)
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA disponible")
        else:
            print("⚠️  CUDA non disponible (normal sur CPU)")
    except:
        pass
    
    print("=" * 30)
    if not issues:
        print("🎯 PRÊT POUR L'ENTRAÎNEMENT!")
        return 0
    else:
        print("⚠️  PROBLÈMES:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

if __name__ == "__main__":
    sys.exit(quick_check())
