# Utils - Scripts de vérification environnement

## 🎯 Scripts disponibles

### 1. `check_env.sh` - Script principal (recommandé)
Script bash qui configure l'environnement et lance les vérifications.

```bash
# Vérification complète (2-3 minutes)
./check_env.sh

# Diagnostic rapide (30 secondes)
./check_env.sh quick
```

### 2. `check_environment.py` - Vérification complète
Script Python détaillé qui vérifie tout l'environnement.

```bash
python check_environment.py
```

### 3. `quick_check.py` - Diagnostic ultra-rapide
Script Python minimal pour un diagnostic en 30 secondes.

```bash
python quick_check.py
```

## 🚀 Usage recommandé

**Avant chaque session d'entraînement :**

```bash
# Aller dans le dossier utils
cd utils

# Diagnostic rapide pour vérifier que tout va bien
./check_env.sh quick

# Si problèmes détectés, vérification complète
./check_env.sh

# Retourner à la racine pour lancer l'entraînement
cd ..
MODEL=Qwen-0.5B sbatch run.sh
```

## 🔍 Ce qui est vérifié

- ✅ Environnement Python et librairies
- ✅ Modules Longleaf (CUDA, anaconda)
- ✅ Accès aux modèles HuggingFace
- ✅ Structure des données d'entraînement
- ✅ Permissions de fichiers
- ✅ Pipeline d'entraînement complète

## 📁 Structure recommandée

```
LONGLEAF/
├── utils/                      # ← Scripts de vérification
│   ├── check_env.sh           # Script principal
│   ├── check_environment.py   # Vérification complète
│   ├── quick_check.py         # Diagnostic rapide
│   └── README.md              # Cette documentation
├── scripts/                   # Scripts d'entraînement
├── Data_input/               # Données d'entrée
├── Data_output/              # Labels attendus
├── run.sh                    # Script SLURM
└── ...
```
