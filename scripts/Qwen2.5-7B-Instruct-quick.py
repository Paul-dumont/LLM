from pathlib import Path
import re, random, time, os
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel

# -------------------- PIPELINE TEST - Paramètres minimaux --------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "Qwen2.5-7B-quick"
MAX_SEQ_LEN = 512
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.1

# Contrôles rapides par variables d'environnement
MAX_INPUTS = int(os.getenv("QUICK_MAX_INPUTS", "10"))             # nombre de paires chargées
MAX_PREDICTIONS = int(os.getenv("QUICK_MAX_PREDICTIONS", "10"))    # nombre de prédictions à écrire
PRED_SOURCE = os.getenv("QUICK_PRED_SOURCE", "val").lower()        # val | train | all

# Optionnel: seed pour reproductibilité
random.seed(42)
torch.manual_seed(42)

# Répertoires (script situé dans scripts/ désormais)
BASE_DIR = Path(__file__).parent.parent
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT = BASE_DIR / "Data_output"
SAVE_DIR = BASE_DIR / "runs" / f"{MODEL_SHORT}_{int(time.time())}"
PREDICT_DIR = BASE_DIR / "Data_predict" / f"{MODEL_SHORT}_predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

# Optionnel: nettoyage des prédictions précédentes
if os.getenv("QUICK_CLEAN_PRED", "0") == "1":
    removed = 0
    for p in PREDICT_DIR.glob("*.txt"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    print(f"🧹 Clean predictions: removed {removed} files from {PREDICT_DIR}")

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")
print(f"📤 DATA_OUTPUT: {DATA_OUTPUT}")

INSTRUCTION = "Extract key information from this clinical note as key=value pairs. Keep it concise."

def get_id(name):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

# --- Charger un sous-ensemble (par défaut 10) pour test pipeline
pairs = []
input_files = sorted(DATA_INPUT.glob("*.txt"))
output_files = sorted(DATA_OUTPUT.glob("*.txt"))

print(f"📁 Found {len(input_files)} input files and {len(output_files)} output files")

for inp_file in input_files[:MAX_INPUTS]:
    cid = get_id(inp_file.name)
    if not cid:
        print(f"❌ No ID found for: {inp_file.name}")
        continue

    out_file = None
    for of in output_files:
        if get_id(of.name) == cid:
            out_file = of
            break

    if not out_file or not out_file.exists():
        print(f"❌ No output file found for {cid} (input: {inp_file.name})")
        continue

    try:
        note = inp_file.read_text(encoding='utf-8').strip()
        lab = out_file.read_text(encoding='utf-8').strip()
    except Exception as e:
        print(f"❌ Error reading {cid}: {e}")
        continue

    if not note or not lab:
        print(f"❌ Empty content for {cid} (note:{len(note) if note else 0}, lab:{len(lab) if lab else 0})")
        continue

    print(f"✅ Loaded {cid}: note={len(note)}chars, lab={len(lab)}chars")

    # Tronquer pour aller vite (facultatif)
    note = note[:2000] if len(note) > 2000 else note
    lab = lab[:500] if len(lab) > 500 else lab

    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": note},
        {"role": "assistant", "content": lab},
    ]
    pairs.append({"cid": cid, "messages": messages, "note_only": note})

random.shuffle(pairs)
n = len(pairs)
if n < 2:
    raise RuntimeError(f"Pas assez d’exemples appariés pour un test (n={n}). Vérifie les IDs Bxxxx.")

n_val = max(1, int(0.2 * n))
train_pairs = pairs[:-n_val]
val_pairs = pairs[-n_val:]

print(f"🚀 PIPELINE TEST {MODEL_SHORT} | Total={n} | Train={len(train_pairs)} | Val={len(val_pairs)}")

# --- Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'right'

def render(messages, add_generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

# --- Dataset
train_texts = [render(p["messages"], add_generation_prompt=False) for p in train_pairs]
val_texts = [render(p["messages"], add_generation_prompt=False) for p in val_pairs]

train_ds = Dataset.from_list([{"text": t} for t in train_texts])
val_ds = Dataset.from_list([{"text": t} for t in val_texts])
dset = DatasetDict({"train": train_ds, "validation": val_ds})

print(f"📊 Dataset ready - Train: {len(train_ds)} | Val: {len(val_ds)}")

# --- Quantization (fp16 pour éviter le mismatch BF16 vs Float)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # <<< switch en fp16
    bnb_4bit_use_double_quant=True,
)

# --- Modèle (laisser bitsandbytes gérer les dtypes)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,
    attn_implementation="eager",
)
print("✅ Model loaded")

# Aligner pad/eos sur le modèle pour la génération
model.config.pad_token_id = tok.pad_token_id
model.config.eos_token_id = tok.eos_token_id

# --- LoRA (ton réglage minimal OK; on peut élargir plus tard si besoin)
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# --- Training config (fp16 aligné)
cfg = SFTConfig(
    output_dir=str(SAVE_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=2,
    eval_strategy="no",
    save_strategy="no",
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    max_seq_length=MAX_SEQ_LEN,
    bf16=False,      # <<< off
    fp16=True,       # <<< on
    gradient_checkpointing=True,
    packing=False,
    dataset_text_field="text",
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=dset["train"],
    eval_dataset=dset["validation"],
    tokenizer=tok,
    peft_config=peft_cfg,
)

print(">> 🚀 QUICK TRAINING START (PIPELINE TEST ONLY)")
start_time = time.time()
train_res = trainer.train()
train_time = time.time() - start_time
print(f"✅ Training completed in {train_time:.1f}s! Loss: {train_res.training_loss:.4f}")

print("🔧 Getting trained model for testing (no save)...")
trained_model = trainer.model
trained_model.eval()

print(f"🧪 Quick test - predictions (max={MAX_PREDICTIONS})...")

# Forcer l'usage du GPU 0 pour éviter un cast CPU->Float32 (ne pas passer device à pipeline quand accelerate est utilisé)
device_id = 0 if torch.cuda.is_available() else -1
gen = pipeline(
    "text-generation",
    model=trained_model,
    tokenizer=tok,
    return_full_text=False,
    pad_token_id=tok.pad_token_id or tok.eos_token_id
)

# Sélection des paires pour prédiction
if PRED_SOURCE == "all":
    pred_pairs = train_pairs + val_pairs
elif PRED_SOURCE == "train":
    pred_pairs = train_pairs
else:
    pred_pairs = val_pairs

print(f"🧪 Quick test - predictions: source={PRED_SOURCE} | max={MAX_PREDICTIONS}")

for i, pair in enumerate(pred_pairs[:MAX_PREDICTIONS]):
    test_messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": pair["note_only"]}
    ]
    test_prompt = render(test_messages, add_generation_prompt=True)

    try:
        result = gen(
            test_prompt,
            max_new_tokens=200,
            do_sample=False
        )
        generated = result[0]["generated_text"].strip()

        # Sauvegarder les prédictions
        pred_file = PREDICT_DIR / f"{pair['cid']}_pred.txt"
        pred_file.write_text(generated, encoding='utf-8')

        print(f"✅ Test {i+1} ({pair['cid']}) - Pipeline OK")
        print(f"   Prediction saved: {pred_file}")
        print(f"   Preview: {generated[:100]}...")
    except Exception as e:
        print(f"❌ Test {i+1}: {e}")

total_time = time.time() - start_time
print(f"🎯 PIPELINE TEST COMPLETE! Total time: {total_time:.1f}s")
print(f"✅ Le pipeline fonctionne correctement!")
print(f"Prêt pour l'entraînement complet")
