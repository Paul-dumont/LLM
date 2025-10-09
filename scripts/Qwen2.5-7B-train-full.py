# ==================== QWEN 2.5-7B INSTRUCT - ENTRAÎNEMENT COMPLET ====================
# Script d'entraînement complet pour le fine-tuning du modèle Qwen2.5-7B-Instruct
# Utilise toutes les données disponibles avec des paramètres optimisés pour la production

# ==================== IMPORTS ET CONFIGURATION ====================
from pathlib import Path
import re, random, time, os, json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
# ==================== CONFIGURATION DU MODÈLE ====================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "Qwen2.5-7B-train-full"

# ==================== HYPERPARAMÈTRES D'ENTRAÎNEMENT COMPLET ====================
MAX_SEQ_LEN = 3000         # Compromis performance optimal: couvre 95.4% des notes (477/500) - très rapide
NUM_EPOCHS = 3             # 3 époques pour un apprentissage approfondi
BATCH_SIZE = 4             # Batch size optimisé pour A100 (plus de VRAM)
GRAD_ACCUM = 2             # Accumulation pour batch effectif = 8 (performance optimale)
LR = 2e-4                  # Learning rate optimal pour fine-tuning complet
WEIGHT_DECAY = 0.01        # Régularisation L2 pour éviter l'overfitting
WARMUP_RATIO = 0.05        # 5% de warmup (plus conservateur)
SAVE_STEPS = 25            # Sauvegarder toutes les 25 steps (plus fréquent pour 150 steps total)
EVAL_STEPS = 25            # Évaluer toutes les 25 steps (6 évaluations au total)
LOGGING_STEPS = 10         # Log toutes les 10 steps

# ==================== CONTRÔLES PAR VARIABLES D'ENVIRONNEMENT ====================
MAX_INPUTS = int(os.getenv("FULL_MAX_INPUTS", "0"))              # 0 = utiliser toutes les données
EVAL_RATIO = float(os.getenv("FULL_EVAL_RATIO", "0.2"))          # 20% pour l'évaluation
USE_WANDB = os.getenv("FULL_USE_WANDB", "0") == "1"              # Logging W&B optionnel
SAVE_FINAL = os.getenv("FULL_SAVE_FINAL", "1") == "1"            # Sauvegarder le modèle final
MAX_PREDICTIONS = int(os.getenv("FULL_MAX_PREDICTIONS", "5"))     # Nombre de prédictions test après entraînement

# ==================== REPRODUCTIBILITÉ ====================
SEED = int(os.getenv("FULL_SEED", "42"))
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================== CONFIGURATION DES RÉPERTOIRES ====================
BASE_DIR = Path(__file__).parent.parent
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT = BASE_DIR / "Data_output2"
SAVE_DIR = BASE_DIR / "runs" / f"{MODEL_SHORT}_{int(time.time())}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")
print(f"📤 DATA_OUTPUT: {DATA_OUTPUT}")
print(f"💾 SAVE_DIR: {SAVE_DIR}")

# ==================== INITIALISATION W&B (OPTIONNEL) ====================
if USE_WANDB:
    wandb.init(
        project="qwen-7b-clinical",
        name=f"{MODEL_SHORT}_{int(time.time())}",
        config={
            "model_id": MODEL_ID,
            "max_seq_len": MAX_SEQ_LEN,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
        }
    )

# ==================== PROMPT SYSTÈME ====================
INSTRUCTION = """Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:"""

# ==================== FONCTION UTILITAIRE ====================
def get_id(name):
    """Extrait l'identifiant Bxxx du nom de fichier"""
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

# ==================== CHARGEMENT COMPLET DES DONNÉES ====================
print("📂 Loading all available data...")
pairs = []
input_files = sorted(DATA_INPUT.glob("*.txt"))
output_files = sorted(DATA_OUTPUT.glob("*.txt"))

print(f"📁 Found {len(input_files)} input files and {len(output_files)} output files")

# Utiliser toutes les données disponibles ou limiter selon MAX_INPUTS
files_to_process = input_files if MAX_INPUTS == 0 else input_files[:MAX_INPUTS]

for inp_file in files_to_process:
    cid = get_id(inp_file.name)
    if not cid:
        print(f"❌ No ID found for: {inp_file.name}")
        continue

    # Chercher le fichier de sortie correspondant
    out_file = None
    for of in output_files:
        if get_id(of.name) == cid:
            out_file = of
            break

    if not out_file or not out_file.exists():
        print(f"❌ No output file found for {cid}")
        continue

    try:
        note = inp_file.read_text(encoding='utf-8').strip()
        lab = out_file.read_text(encoding='utf-8').strip()
    except Exception as e:
        print(f"❌ Error reading {cid}: {e}")
        continue

    if not note or not lab:
        print(f"❌ Empty content for {cid}")
        continue

    # Pas de troncature pour l'entraînement complet - garder le contenu intégral
    print(f"✅ Loaded {cid}: note={len(note)}chars, lab={len(lab)}chars")
    
    # Vérification de la taille estimée en tokens (approximation: 1 token ≈ 4.5 chars)
    estimated_tokens = len(note) / 4.5
    if estimated_tokens > MAX_SEQ_LEN - 200:  # Garder une marge pour le système prompt
        print(f"⚠️  {cid}: {estimated_tokens:.0f} tokens estimés → SERA TRONQUÉE (limite {MAX_SEQ_LEN})")

    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": note},
        {"role": "assistant", "content": lab},
    ]
    pairs.append({"cid": cid, "messages": messages, "note_only": note})

print(f"📊 Total loaded pairs: {len(pairs)}")

if len(pairs) < 10:
    raise RuntimeError(f"Pas assez de données pour un entraînement complet (n={len(pairs)})")

# ==================== DIVISION TRAIN/EVAL ====================
random.shuffle(pairs)
n = len(pairs)
n_eval = max(5, int(EVAL_RATIO * n))  # Au moins 5 exemples pour l'évaluation
train_pairs = pairs[:-n_eval]
eval_pairs = pairs[-n_eval:]

print(f"🚀 FULL TRAINING {MODEL_SHORT}")
print(f"📊 Total: {n} | Train: {len(train_pairs)} | Eval: {len(eval_pairs)}")

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

# ==================== CRÉATION DES DATASETS ====================
print("📊 Creating datasets...")
train_texts = [render(p["messages"], add_generation_prompt=False) for p in train_pairs]
eval_texts = [render(p["messages"], add_generation_prompt=False) for p in eval_pairs]

train_ds = Dataset.from_list([{"text": t} for t in train_texts])
eval_ds = Dataset.from_list([{"text": t} for t in eval_texts])
dset = DatasetDict({"train": train_ds, "eval": eval_ds})

print(f"📊 Datasets ready - Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ==================== CHARGEMENT DU MODÈLE ====================
print("🤖 Loading model without quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,
    attn_implementation="eager",
)

model.config.pad_token_id = tok.pad_token_id
model.config.eos_token_id = tok.eos_token_id
model.config.use_cache = False

print("✅ Model loaded successfully")

# ==================== CONFIGURATION D'ENTRAÎNEMENT COMPLÈTE ====================
print("⚙️ Setting up training configuration...")

report_to_list = ["wandb"] if USE_WANDB else []

cfg = SFTConfig(
    # ==================== RÉPERTOIRES ET SAUVEGARDE ====================
    output_dir=str(SAVE_DIR),
    logging_dir=str(SAVE_DIR / "logs"),
    
    # ==================== PARAMÈTRES D'ENTRAÎNEMENT ====================
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    # ==================== OPTIMISATION ====================
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",                     # Scheduler cosine pour meilleure convergence
    
    # ==================== ÉVALUATION ET LOGGING ====================
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    
    # ==================== SAUVEGARDE ====================
    save_strategy="steps" if SAVE_FINAL else "no",
    save_steps=SAVE_STEPS if SAVE_FINAL else 10000,
    save_total_limit=3,                             # Garder seulement les 3 derniers checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # ==================== TECHNIQUES D'OPTIMISATION ====================
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
    
    # ==================== CONFIGURATION SFT ====================
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    dataset_text_field="text",
    
    # ==================== REPORTING ====================
    report_to=report_to_list,
    run_name=f"{MODEL_SHORT}_{int(time.time())}" if USE_WANDB else None,
    
    # ==================== EARLY STOPPING ====================
    # Note: Early stopping géré par load_best_model_at_end et metric_for_best_model
)

# ==================== CRÉATION DU TRAINER ====================
print("🏋️ Creating trainer...")
trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=dset["train"],
    eval_dataset=dset["eval"],
    tokenizer=tok,
)

# ==================== SAUVEGARDE DU CONFIG ====================
config_info = {
    "model_id": MODEL_ID,
    "training_params": {
        "max_seq_len": MAX_SEQ_LEN,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
    },
    "data_info": {
        "total_pairs": len(pairs),
        "train_size": len(train_pairs),
        "eval_size": len(eval_pairs),
        "eval_ratio": EVAL_RATIO,
    }
}

with open(SAVE_DIR / "training_config.json", "w") as f:
    json.dump(config_info, f, indent=2)

print(f"📋 Training config saved to: {SAVE_DIR / 'training_config.json'}")

# ==================== LANCEMENT DE L'ENTRAÎNEMENT ====================
print("\n" + "="*80)
print("🚀 STARTING FULL TRAINING")
print("="*80)

start_time = time.time()

try:
    # Entraînement complet avec évaluation
    train_result = trainer.train()
    
    train_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"⏱️  Total training time: {train_time:.1f}s ({train_time/60:.1f} minutes)")
    print(f"📉 Final training loss: {train_result.training_loss:.4f}")
    
    # ==================== SAUVEGARDE FINALE ====================
    if SAVE_FINAL:
        print("💾 Saving final model...")
        
        # Sauvegarder le modèle
        trainer.save_model(str(SAVE_DIR / "final_model"))
        
        # Sauvegarder le tokenizer
        tok.save_pretrained(str(SAVE_DIR / "final_model"))
        
        # Sauvegarder les résultats d'entraînement
        results = {
            "training_loss": train_result.training_loss,
            "training_time": train_time,
            "total_steps": train_result.global_step,
            "epochs_completed": NUM_EPOCHS,
        }
        
        with open(SAVE_DIR / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Model saved to: {SAVE_DIR / 'final_model'}")
        print(f"📊 Results saved to: {SAVE_DIR / 'training_results.json'}")
    
    # ==================== ÉVALUATION FINALE ====================
    print("\n🧪 Running final evaluation...")
    eval_result = trainer.evaluate()
    print(f"📊 Final evaluation loss: {eval_result['eval_loss']:.4f}")
    
    if USE_WANDB:
        wandb.log({
            "final_train_loss": train_result.training_loss,
            "final_eval_loss": eval_result["eval_loss"],
            "training_time": train_time
        })
        wandb.finish()

    # ==================== GÉNÉRATION DE PRÉDICTIONS TEST ====================
    if MAX_PREDICTIONS > 0:
        print(f"\n🧪 Generating {MAX_PREDICTIONS} test predictions...")
        
        # Créer le dossier de prédictions
        PREDICT_DIR = BASE_DIR / "Data_predict" / f"{MODEL_SHORT}_predict"
        PREDICT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Nettoyage optionnel des prédictions précédentes
        for old_pred in PREDICT_DIR.glob("*.txt"):
            old_pred.unlink()
        
        # Pipeline de génération (IDENTIQUE au smoke test qui fonctionnait)
        print("🔧 Getting trained model for testing...")
        trained_model = trainer.model
        trained_model.eval()
        
        from transformers import pipeline
        device_id = 0 if torch.cuda.is_available() else -1
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
                    max_new_tokens=500,  # Optimisé : couvre 375 tokens moyens + marge sécurité
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

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    if USE_WANDB:
        wandb.finish(exit_code=1)
    raise

print("\n🎯 FULL TRAINING PIPELINE COMPLETED!")
print(f"📁 All outputs saved in: {SAVE_DIR}")
if MAX_PREDICTIONS > 0:
    print(f"🧪 Test predictions available in: Data_predict/{MODEL_SHORT}_predict/")
