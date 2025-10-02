from pathlib import Path
import re, random, time, os
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,# --- MODE TEST: Générer seulement 10 prédictions pour validation
print("🧪 TEST MODE: Generating 10 predictions for format validation...")
pipe = pipeline("text-generation", 
                model=trainer.model, 
                tokenizer=tok,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                trust_remote_code=True)

# Prendre les 10 premiers exemples de tous les pairs pour test complet
test_pairs = pairs[:10]
print(f"🔬 Testing on {len(test_pairs)} examples...")

for i, pair in enumerate(test_pairs):dBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel

# -------------------- Réglages optimisés pour Qwen2.5-7B --------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"           # Qwen2.5 7B Instruct (très performant)
MODEL_SHORT = "Qwen2.5-7B"                      # Pour runs/
MAX_SEQ_LEN = 4096                               # 4K tokens (contexte large disponible mais pas nécessaire)
NUM_EPOCHS = 1                                   # 1 epoch suffisant pour 7B
BATCH_SIZE = 1                                   # Batch réduit (7B plus gros)
GRAD_ACCUM = 16                                  # => batch effectif = 16
LR = 2e-4                                        # LR standard pour 7B
WEIGHT_DECAY = 0.01                              # Régularisation
WARMUP_RATIO = 0.05                              # 5% warmup

# Répertoires
BASE_DIR = Path(__file__).parent.parent
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT = BASE_DIR / "Data_output"
SAVE_DIR = BASE_DIR / "runs" / f"{MODEL_SHORT}_{int(time.time())}"
PREDICT_DIR = BASE_DIR / "Data_predict" / f"{MODEL_SHORT}_predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = (
    "From the clinical note, produce exactly the expected output as key=value lines. "
    "Return ONLY key=value pairs (one per line). If a value is unknown, write 'NA'. "
    "NO extra commentary, NO free text, NO JSON."
)

def get_id(name):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

# --- Charger toutes les paires (≈ 500)
pairs = []
input_files = sorted(DATA_INPUT.glob("*.txt"))
output_files = sorted(DATA_OUTPUT.glob("*.txt"))

for inp_file in input_files:
    cid = get_id(inp_file.name)
    if not cid:
        continue
    
    # Chercher le fichier de sortie correspondant
    out_file = None
    for of in output_files:
        if get_id(of.name) == cid:
            out_file = of
            break
    
    if not out_file or not out_file.exists():
        continue
    
    note = inp_file.read_text(encoding='utf-8').strip()
    lab = out_file.read_text(encoding='utf-8').strip()
    
    if not note or not lab:
        continue
    
    # Format de conversation Qwen2.5
    messages = [
        {"role": "system",    "content": INSTRUCTION},
        {"role": "user",      "content": note},
        {"role": "assistant", "content": lab},
    ]
    pairs.append({"cid": cid, "messages": messages, "note_only": note})

random.shuffle(pairs)

# --- Split 90/10 (train/val) pour ≈500 ex
n = len(pairs)
n_val = max(1, int(0.1 * n))
train_pairs = pairs[:-n_val]
val_pairs   = pairs[-n_val:]

print(f"🚀 Training {MODEL_SHORT} | Total={n} | Train={len(train_pairs)} | Val={len(val_pairs)}")

# --- Tokenizer + chat template (Qwen2.5 a son propre template)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Fix padding side for SFTTrainer
tok.padding_side = 'right'
print(f"✅ Tokenizer loaded - Vocab size: {len(tok)}")

def render(messages, add_generation_prompt=True):
    return tok.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=add_generation_prompt
    )

# --- Convertir en texte brut
train_texts = [render(p["messages"], add_generation_prompt=False) for p in train_pairs]
val_texts   = [render(p["messages"], add_generation_prompt=False) for p in val_pairs]

train_ds = Dataset.from_dict({"text": train_texts})
val_ds   = Dataset.from_dict({"text": val_texts})
dset = DatasetDict({"train": train_ds, "validation": val_ds})

print(f"📊 Dataset ready - Train: {len(train_ds)} | Val: {len(val_ds)}")

# --- 4-bit quantization config pour gérer le 7B
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Charger le modèle avec quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print(f"✅ Model loaded with 4-bit quantization")

# --- LoRA config pour Qwen2.5 (cibles spécifiques)
peft_cfg = LoraConfig(
    r=16,                       # Rang LoRA
    lora_alpha=32,              # Alpha scaling
    lora_dropout=0.1,           # Dropout
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

# --- SFT config optimisé pour 7B
cfg = SFTConfig(
    output_dir=str(SAVE_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=max(20, len(train_ds)//10),
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    max_seq_length=MAX_SEQ_LEN,
    bf16=True,
    fp16=False,
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

print(">> TRAIN START")
train_res = trainer.train()
print(f"✅ Training completed! Loss: {train_res.training_loss:.4f}")

# --- Sauvegarder le modèle final
final_dir = SAVE_DIR / "final"
trainer.save_model(str(final_dir))
tok.save_pretrained(str(final_dir))
print(f"💾 Model saved to: {final_dir}")

# --- Tester sur quelques exemples
print("🧪 Testing model on validation examples...")
pipe = pipeline("text-generation", 
                model=trainer.model, 
                tokenizer=tok,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                trust_remote_code=True)

for i, pair in enumerate(val_pairs[:3]):
    test_prompt = render([
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": pair["note_only"]}
    ], add_generation_prompt=True)
    
    result = pipe(test_prompt)
    generated = result[0]["generated_text"][len(test_prompt):].strip()
    
    # Sauvegarder la prédiction
    pred_file = PREDICT_DIR / f"{pair['cid']}_pred.txt"
    pred_file.write_text(generated, encoding='utf-8')
    
    print(f"\n--- Test {i+1} ({pair['cid']}) ---")
    print(f"Generated: {generated[:200]}...")

print(f"🎯 Training complete! Model: {MODEL_SHORT}")
print(f"📁 Saved to: {SAVE_DIR}")
print(f"🔮 Predictions: {PREDICT_DIR}")
