#!/usr/bin/env python3
"""
Script to generate 500 predictions from the saved full fine-tuned Qwen 7B model.
Loads the latest saved model and generates predictions for all available input notes.
"""

from pathlib import Path
import re, random, time, os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==================== CONFIGURATION ====================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SHORT = "qwen_full"
MAX_PREDICTIONS = 500

# ==================== PROMPT SYSTÈME ====================
INSTRUCTION = """Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:
"""

def get_id(name):
    """Extrait l'identifiant Bxxx du nom de fichier"""
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

# ==================== REPRODUCTIBILITÉ ====================
SEED = int(os.getenv("FULL_SEED", "42"))
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================== CONFIGURATION DES RÉPERTOIRES ====================
BASE_DIR = Path(__file__).parent.parent
DATA_INPUT = BASE_DIR / "DATA_TRAINING" / "Data_input"
RUNS_DIR = BASE_DIR / "Qwen1.5B_full" / "model"

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")

# Auto-detect latest saved full model
model_dirs = []
for p in RUNS_DIR.iterdir():
    if p.is_dir() and MODEL_SHORT in p.name:
        fm = p / "final_model"
        if fm.exists() and fm.is_dir():
            model_dirs.append(fm)

if not model_dirs:
    print("❌ No saved full model found in runs/")
    exit(1)

model_dir = sorted(model_dirs, key=lambda d: d.parent.stat().st_mtime)[-1]
print(f"🤖 Using model: {model_dir}")

# ==================== CHARGEMENT DU MODÈLE ET TOKENIZER ====================
print("🔧 Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'right'

print("🤖 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    use_cache=True,
)
model.config.pad_token_id = tok.pad_token_id
model.config.eos_token_id = tok.eos_token_id
model.eval()

# ==================== PIPELINE DE GÉNÉRATION ====================
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    return_full_text=False,
    pad_token_id=tok.pad_token_id,
    torch_dtype=torch.bfloat16,
)

# ==================== CHARGEMENT DES DONNÉES ====================
print("📂 Loading all input notes...")
pairs = []
input_files = sorted(DATA_INPUT.glob("*.txt"))
for inp_file in input_files:
    cid = get_id(inp_file.name)
    if not cid:
        continue
    try:
        note = inp_file.read_text(encoding='utf-8').strip()
        if note:
            pairs.append({"cid": cid, "note": note})
    except:
        continue

if not pairs:
    print("❌ No input notes found")
    exit(1)

# Prendre les 500 premières (ou toutes si moins)
eval_pairs = pairs[:MAX_PREDICTIONS]
print(f"📊 Will generate predictions for {len(eval_pairs)} notes")

# ==================== GÉNÉRATION DES PRÉDICTIONS ====================
PREDICT_DIR = Path(__file__).parent.parent / "Qwen1.5B_full" / "predictions_500"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

# Nettoyage des prédictions précédentes
for old_pred in PREDICT_DIR.glob("*.txt"):
    old_pred.unlink()

print(f"🧪 Generating {len(eval_pairs)} predictions...")

for i, pair in enumerate(eval_pairs):
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": pair["note"]}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    try:
        result = gen_pipeline(
            prompt,
            max_new_tokens=500,
            do_sample=False
        )
        generated = result[0]["generated_text"].strip()
        
        # Sauvegarder la prédiction
        pred_file = PREDICT_DIR / f"{pair['cid']}_pred.txt"
        pred_file.write_text(generated, encoding='utf-8')
        
        print(f"✅ Prediction {i+1}/{len(eval_pairs)} ({pair['cid']}) saved")
        
    except Exception as e:
        print(f"❌ Prediction {i+1} ({pair['cid']}): {e}")

print(f"\n📁 Predictions saved in: {PREDICT_DIR}")
print("🎯 GENERATION COMPLETED!")
