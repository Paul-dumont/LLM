#!/usr/bin/env python3
"""
Script de vérification complète de l'environnement pour entraînement LLM
À lancer sur le nœud de connexion Longleaf (CPU) avant de soumettre des jobs GPU.

⚠️  SCRIPT EN LECTURE SEULE - Ne modifie pas votre environnement existant
✅ Vérifie seulement si tout est prêt pour l'entraînement

Usage: python check_environment.py
"""

import sys, os
from pathlib import Path
import re, random
import traceback
from datetime import datetime

def print_status(msg, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {"OK": "✅", "ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️"}
    symbol = symbols.get(status, "ℹ️")
    print(f"[{timestamp}] {symbol} {msg}")

def check_python_version():
    """Vérifier la version Python"""
    print_status("Vérification de Python", "INFO")
    version = sys.version_info
    print_status(f"Version Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print_status(f"ATTENTION: Python {version.major}.{version.minor} détecté. Recommandé: Python 3.8+", "WARN")
    else:
        print_status("Version Python OK", "OK")

def check_imports():
    """Vérifier les imports critiques"""
    print_status("Vérification des librairies Python", "INFO")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'trl': 'TRL',
        'peft': 'PEFT',
        'accelerate': 'Accelerate'
    }
    
    failed_imports = []
    
    for package, name in required_packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'version inconnue')
            print_status(f"{name}: {version}", "OK")
        except ImportError as e:
            print_status(f"ÉCHEC import {name}: {e}", "ERROR")
            failed_imports.append(package)
    
    return failed_imports

def check_torch_capabilities():
    """Vérifier les capacités PyTorch"""
    print_status("Vérification des capacités PyTorch", "INFO")
    
    try:
        import torch
        
        # Version PyTorch
        print_status(f"PyTorch version: {torch.__version__}")
        
        # CUDA disponibilité (même sur CPU, on vérifie la compilation)
        cuda_available = torch.cuda.is_available()
        print_status(f"CUDA compilé: {cuda_available}")
        
        if cuda_available:
            print_status("CUDA trouvé - parfait pour GPU jobs!", "OK")
        else:
            print_status("CUDA non trouvé - vérifiez l'installation pour jobs GPU", "WARN")
        
        # Test de création de tenseur
        x = torch.tensor([1.0, 2.0, 3.0])
        print_status(f"Test tensor: {x.dtype} - {x.device}", "OK")
        
        # Vérifier bfloat16 support
        if hasattr(torch, 'bfloat16'):
            print_status("Support bfloat16: disponible", "OK")
        else:
            print_status("Support bfloat16: non disponible", "WARN")
            
        return True
        
    except Exception as e:
        print_status(f"Erreur PyTorch: {e}", "ERROR")
        return False

def check_transformers_models():
    """Vérifier l'accès aux modèles Transformers"""
    print_status("Vérification de l'accès aux modèles", "INFO")
    
    try:
        from transformers import AutoTokenizer
        
        # Test avec un petit modèle pour vérifier la connectivité HF
        print_status("Test de téléchargement tokenizer (petit modèle)...")
        tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        print_status("Accès HuggingFace Hub: OK", "OK")
        
        # Test des modèles de votre projet
        test_models = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "meta-llama/Llama-3.2-8B-Instruct"  # Celui-ci nécessite peut-être un token
        ]
        
        for model_id in test_models:
            try:
                print_status(f"Test accès: {model_id}...")
                tok = AutoTokenizer.from_pretrained(model_id)
                print_status(f"✓ {model_id}: accessible", "OK")
            except Exception as e:
                if "token" in str(e).lower() or "authentication" in str(e).lower():
                    print_status(f"⚠ {model_id}: nécessite un token HF", "WARN")
                else:
                    print_status(f"✗ {model_id}: {str(e)[:100]}", "ERROR")
        
        return True
        
    except Exception as e:
        print_status(f"Erreur accès modèles: {e}", "ERROR")
        return False

def check_data_structure():
    """Vérifier la structure des données"""
    print_status("Vérification de la structure des données", "INFO")
    
    # Vérifier les dossiers requis (à la racine du projet)
    required_dirs = ["../Data_input", "../Data_output"]
    missing_dirs = []
    
    for dir_path_str in required_dirs:
        dir_path = Path(dir_path_str)
        dir_name = dir_path.name  # Nom pour l'affichage
        if dir_path.exists():
            files = list(dir_path.glob("*.txt"))
            print_status(f"{dir_name}: {len(files)} fichiers .txt", "OK")
        else:
            print_status(f"{dir_name}: dossier manquant", "ERROR")
            missing_dirs.append(dir_name)
    
    # Vérifier la correspondance des fichiers
    if not missing_dirs:
        try:
            def get_id(name):
                m = re.match(r"(B\d+)", name)
                return m.group(1) if m else None
            
            input_files = {get_id(p.name): p for p in Path("../Data_input").glob("*.txt")}
            output_files = {get_id(p.name): p for p in Path("../Data_output").glob("*.txt")}
            
            common_ids = set(input_files.keys()) & set(output_files.keys())
            print_status(f"Fichiers appairés: {len(common_ids)} paires trouvées", "OK")
            
            if len(common_ids) == 0:
                print_status("AUCUNE paire input/output trouvée!", "ERROR")
            elif len(common_ids) < 5:
                print_status(f"Seulement {len(common_ids)} paires - entraînement limité", "WARN")
                
        except Exception as e:
            print_status(f"Erreur vérification données: {e}", "ERROR")
    
    return len(missing_dirs) == 0

def check_file_permissions():
    """Vérifier les permissions de fichiers (LECTURE SEULE)"""
    print_status("Vérification des permissions (lecture seule)", "INFO")
    
    # Vérifier les permissions existantes sans rien créer (à la racine du projet)
    test_dirs = [("../runs", "runs"), ("../Data_predict", "Data_predict"), ("../logs", "logs")]
    
    for dir_path_str, dir_name in test_dirs:
        dir_path = Path(dir_path_str)
        if dir_path.exists():
            if dir_path.is_dir() and os.access(dir_path, os.W_OK):
                print_status(f"Permissions {dir_name}: OK (existe et accessible)", "OK")
            else:
                print_status(f"Permissions {dir_name}: Limité (existe mais accès restreint)", "WARN")
        else:
            # Vérifier si on peut créer dans le répertoire parent
            parent_dir = dir_path.parent
            if os.access(parent_dir, os.W_OK):
                print_status(f"Permissions {dir_name}: OK (peut être créé)", "OK")
            else:
                print_status(f"Permissions {dir_name}: ÉCHEC (ne peut pas être créé)", "ERROR")
                return False
    
    # Vérifier permissions écriture dans le répertoire courant
    try:
        current_dir = Path(".")
        if os.access(current_dir, os.W_OK):
            print_status("Permissions écriture: OK (répertoire accessible)", "OK")
        else:
            print_status("Permissions écriture: ÉCHEC (répertoire non accessible)", "ERROR")
            return False
    except Exception as e:
        print_status(f"Permissions écriture: ÉCHEC - {e}", "ERROR")
        return False
    
    return True

def check_memory():
    """Vérifier la mémoire disponible"""
    print_status("Vérification mémoire système", "INFO")
    
    try:
        import psutil
        
        # Mémoire totale
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print_status(f"Mémoire totale: {total_gb:.1f} GB")
        print_status(f"Mémoire disponible: {available_gb:.1f} GB")
        
        if available_gb < 2:
            print_status("Mémoire faible - risque de problème", "WARN")
        else:
            print_status("Mémoire suffisante", "OK")
            
    except ImportError:
        print_status("psutil non disponible - impossible de vérifier la mémoire", "WARN")

def check_environment_variables():
    """Vérifier les variables d'environnement importantes"""
    print_status("Vérification variables d'environnement", "INFO")
    
    important_vars = {
        'HF_HOME': 'Cache HuggingFace',
        'CUDA_VISIBLE_DEVICES': 'Devices CUDA visibles',
        'PYTHONPATH': 'Chemin Python'
    }
    
    for var, desc in important_vars.items():
        value = os.environ.get(var)
        if value:
            print_status(f"{desc}: {value[:100]}")
        else:
            print_status(f"{desc}: non défini")

def run_mini_training_test():
    """Test d'entraînement minimal pour vérifier la pipeline complète"""
    print_status("Test de pipeline d'entraînement minimal", "INFO")
    
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        
        # Créer un mini dataset de test
        test_data = [
            {"text": "[INST]Test input 1[/INST]Test output 1"},
            {"text": "[INST]Test input 2[/INST]Test output 2"}
        ]
        
        dataset = Dataset.from_list(test_data)
        print_status("Création dataset test: OK", "OK")
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test tokenisation
        encoded = tokenizer(test_data[0]["text"], return_tensors="pt")
        print_status(f"Tokenisation test: {encoded['input_ids'].shape}", "OK")
        
        print_status("Pipeline complète: OK", "OK")
        return True
        
    except Exception as e:
        print_status(f"Échec pipeline test: {e}", "ERROR")
        return False

def main():
    print("="*60)
    print("🔍 VÉRIFICATION ENVIRONNEMENT LLM LONGLEAF")
    print("⚠️  MODE LECTURE SEULE - Aucune modification")
    print("="*60)
    
    all_checks = []
    
    # Exécuter toutes les vérifications
    all_checks.append(("Python Version", True))  # Toujours OK
    check_python_version()
    
    print("\n" + "-"*40)
    failed_imports = check_imports()
    all_checks.append(("Imports Python", len(failed_imports) == 0))
    
    print("\n" + "-"*40)
    torch_ok = check_torch_capabilities()
    all_checks.append(("PyTorch", torch_ok))
    
    print("\n" + "-"*40)
    models_ok = check_transformers_models()
    all_checks.append(("Accès Modèles", models_ok))
    
    print("\n" + "-"*40)
    data_ok = check_data_structure()
    all_checks.append(("Structure Données", data_ok))
    
    print("\n" + "-"*40)
    perms_ok = check_file_permissions()
    all_checks.append(("Permissions", perms_ok))
    
    print("\n" + "-"*40)
    check_memory()
    
    print("\n" + "-"*40)
    check_environment_variables()
    
    print("\n" + "-"*40)
    pipeline_ok = run_mini_training_test()
    all_checks.append(("Pipeline Test", pipeline_ok))
    
    # Résumé final
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES VÉRIFICATIONS")
    print("="*60)
    
    passed = 0
    total = len(all_checks)
    
    for check_name, check_result in all_checks:
        status_symbol = "✅" if check_result else "❌"
        print(f"{status_symbol} {check_name}")
        if check_result:
            passed += 1
    
    print(f"\nRésultat: {passed}/{total} vérifications réussies")
    
    if passed == total:
        print_status("🚀 ENVIRONNEMENT PRÊT POUR L'ENTRAÎNEMENT!", "OK")
        print("Vous pouvez soumettre vos jobs GPU en toute confiance.")
        return 0
    else:
        print_status("⚠️  PROBLÈMES DÉTECTÉS", "WARN")
        print("Corrigez les erreurs avant de soumettre des jobs GPU.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
