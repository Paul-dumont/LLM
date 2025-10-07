#!/usr/bin/env python3
"""
COPIE EXACTE de la section génération de Qwen2.5-7B-Instruct-full.py qui marche
Seule différence: charge le modèle sauvé au lieu d'utiliser trainer.model
"""

from pathlib import Path
import re, random, time, os, json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TrainingArguments, pipeline
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# ==================== CONFIGURATION IDENTIQUE ====================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "Qwen2.5-7B-full"
MAX_PREDICTIONS = 500

# ==================== PROMPT SYSTÈME IDENTIQUE ====================
INSTRUCTION = """You are a medical data extraction specialist. Extract the following 56 specific clinical indicators from this medical note and format as key=value pairs. Use "unknown" if information is not available.

REQUIRED INDICATORS:
patient_id=
patient_age=
headache_intensity=
headache_frequency=
headache_location=
migraine_history=
migraine_frequency=
average_daily_pain_intensity=
diet_score=
tmj_pain_rating=
disability_rating=
jaw_function_score=
jaw_clicking=
jaw_crepitus=
jaw_locking=
maximum_opening=
maximum_opening_without_pain=
disc_displacement=
muscle_pain_score=
muscle_pain_location=
muscle_spasm_present=
muscle_tenderness_present=
muscle_stiffness_present=
muscle_soreness_present=
joint_pain_areas=
joint_arthritis_location=
neck_pain_present=
back_pain_present=
earache_present=
tinnitus_present=
vertigo_present=
hearing_loss_present=
hearing_sensitivity_present=
sleep_apnea_diagnosed=
sleep_disorder_type=
airway_obstruction_present=
anxiety_present=
depression_present=
stress_present=
autoimmune_condition=
fibromyalgia_present=
current_medications=
previous_medications=
adverse_reactions=
appliance_history=
current_appliance=
cpap_used=
apap_used=
bipap_used=
physical_therapy_status=
pain_onset_date=
pain_duration=
pain_frequency=
onset_triggers=
pain_relieving_factors=
pain_aggravating_factors=

EXTRACTION RULES:
- Use exact indicator names as shown above
- For boolean indicators: use "true", "false", or "unknown"
- For numeric values: extract numbers and units (e.g., "25 years", "7/10", "45mm")
- For text values: be concise but specific
- If multiple values exist, use semicolon separation
- Always include all 56 indicators, even if "unknown"
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
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT = BASE_DIR / "Data_output"

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")

# Auto-detect latest saved model
runs_dir = BASE_DIR / "runs"
model_dirs = []
for p in runs_dir.iterdir():
    if p.is_dir() and p.name.startswith(MODEL_SHORT):
        fm = p / "final_model"
        if fm.exists() and fm.is_dir():
            model_dirs.append(fm)

if not model_dirs:
    print("❌ No saved model found")
    exit(1)

model_dir = sorted(model_dirs, key=lambda d: d.parent.stat().st_mtime)[-1]
print(f"🤖 Using model: {model_dir}")

print("📂 Loading all input notes for generation...")
pairs = []
input_files = sorted(DATA_INPUT.glob("*.txt"))
for inp_file in input_files:
    cid = get_id(inp_file.name)
    if not cid:
        continue
    try:
        note = inp_file.read_text(encoding='utf-8').strip()
        if note:
            messages = [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": note}
            ]
            pairs.append({"cid": cid, "messages": messages, "note_only": note})
    except:
        continue

# Prendre tous les pairs d'input
eval_pairs = pairs[:MAX_PREDICTIONS]
print(f"📊 Eval pairs ready: {len(eval_pairs)} (from Data_input)")
    
# ==================== CONFIGURATION DU TOKENIZER ====================
print("🔧 Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'right'

def render(messages, add_generation_prompt=True):
    """Convertit les messages au format chat en texte selon le template Qwen"""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

# ==================== CONFIGURATION DE LA QUANTIZATION ====================
print("⚙️ Configuring 4-bit quantization...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ==================== CHARGEMENT DU MODÈLE ====================
print("🤖 Loading model with quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,
    attn_implementation="eager",
)

model.config.pad_token_id = tok.pad_token_id
model.config.eos_token_id = tok.eos_token_id
model.config.use_cache = False

print("✅ Model loaded successfully")

# ==================== CONFIGURATION LORA ÉTENDUE ====================
print("🔧 Configuring LoRA...")
peft_cfg = LoraConfig(
    r=32,                                           
    lora_alpha=64,                                  
    lora_dropout=0.1,                               
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[                                
        "q_proj", "k_proj", "v_proj", "o_proj",    
        "gate_proj", "up_proj", "down_proj"         
    ]
)

# ==================== CRÉATION DU TRAINER SIMULÉ ====================
print("🏋️ Creating simulated trainer...")
# On crée un trainer minimal juste pour avoir trainer.model comme dans l'original
from trl import SFTTrainer, SFTConfig

# Dataset minimal pour le trainer
dummy_ds = Dataset.from_list([{"text": "dummy"}])
dummy_dset = DatasetDict({"train": dummy_ds, "eval": dummy_ds})

cfg = SFTConfig(
    output_dir=str(BASE_DIR / "temp"),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    max_seq_length=100,
    dataset_text_field="text",
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=dummy_ds,
    eval_dataset=dummy_ds,
    tokenizer=tok,
    peft_config=peft_cfg,
)

# Charger les poids LoRA sauvés
trainer.model.load_adapter(str(model_dir), adapter_name="default", is_trainable=False)

print("✅ Trainer ready with loaded adapter")
    
# ==================== GÉNÉRATION DE PRÉDICTIONS TEST IDENTIQUE ====================
if MAX_PREDICTIONS > 0:
    print(f"\n🧪 Generating {MAX_PREDICTIONS} test predictions...")
    
    # Créer le dossier de prédictions
    PREDICT_DIR = BASE_DIR / "Data_predict" / f"{MODEL_SHORT}_predict_500_fixed"
    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Nettoyage optionnel des prédictions précédentes
    for old_pred in PREDICT_DIR.glob("*.txt"):
        old_pred.unlink()
    
    # Pipeline de génération (IDENTIQUE au smoke test qui fonctionnait)
    print("🔧 Getting trained model for testing...")
    trained_model = trainer.model  # EXACTEMENT comme dans l'original
    trained_model.eval()
    
    gen = pipeline(
        "text-generation",
        model=trained_model,
        tokenizer=tok,
        return_full_text=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id
    )
    
    # Prendre quelques exemples de validation pour tester
    test_pairs = eval_pairs[:MAX_PREDICTIONS]
    
    for i, pair in enumerate(test_pairs):
        test_messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": pair["note_only"]}
        ]
        test_prompt = render(test_messages, add_generation_prompt=True)
        
        try:
            result = gen(
                test_prompt,
                max_new_tokens=500,  # Optimisé : 500 tokens au lieu de 200
                do_sample=False
            )
            generated = result[0]["generated_text"].strip()
            
            # Sauvegarder la prédiction (même format que smoke test)
            pred_file = PREDICT_DIR / f"{pair['cid']}_pred.txt"
            pred_file.write_text(generated, encoding='utf-8')
            
            print(f"✅ Test {i+1} ({pair['cid']}) - Pipeline OK")
            print(f"   Prediction saved: {pred_file}")
            print(f"   Preview: {generated[:100]}...")
            
        except Exception as e:
            print(f"❌ Test {i+1}: {e}")
    
    print(f"📁 Predictions saved in: {PREDICT_DIR}")

print("\n🎯 GENERATION PIPELINE COMPLETED!")
print(f"🧪 Test predictions available in: Data_predict/{MODEL_SHORT}_predict_500_fixed/")
