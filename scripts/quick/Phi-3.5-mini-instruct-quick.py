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

# -------------------- PIPELINE TEST PHI - Paramètres ultra conservateurs --------------------
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MODEL_SHORT = "Phi-3.5-mini-PIPELINE"
MAX_SEQ_LEN = 512                                # ✅ Plus court pour éviter problèmes
NUM_EPOCHS = 1
BATCH_SIZE = 1                                   
GRAD_ACCUM = 2                                   # ✅ Plus petit pour test
LR = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03

# Répertoires - remonter au répertoire racine du projet
BASE_DIR = Path(__file__).parent.parent.parent  # scripts/quick/ -> scripts/ -> LONGLEAF/
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT = BASE_DIR / "Data_output"
SAVE_DIR = BASE_DIR / "runs" / f"{MODEL_SHORT}_{int(time.time())}"
PREDICT_DIR = BASE_DIR / "Data_predict" / f"{MODEL_SHORT}_predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")
print(f"📤 DATA_OUTPUT: {DATA_OUTPUT}")

INSTRUCTION = "Extract key information from this clinical note as key=value pairs. Keep it concise."

def get_id(name):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

# --- Charger SEULEMENT 10 exemples pour test pipeline
pairs = []
input_files = sorted(DATA_INPUT.glob("*.txt"))
output_files = sorted(DATA_OUTPUT.glob("*.txt"))

print(f"📁 Found {len(input_files)} input files and {len(output_files)} output files")

for inp_file in input_files[:10]:
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
    
    # Tronquer agressivement pour éviter les problèmes de tokenisation
    note = note[:1000] if len(note) > 1000 else note  # Plus court
    lab = lab[:200] if len(lab) > 200 else lab          # Plus court
    
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": note},
        {"role": "assistant", "content": lab},
    ]
    pairs.append({"cid": cid, "messages": messages, "note_only": note})

random.shuffle(pairs)
n = len(pairs)
n_val = max(1, int(0.2 * n))
train_pairs = pairs[:-n_val]
val_pairs = pairs[-n_val:]

print(f"🚀 QUICK TEST {MODEL_SHORT} | Total={n} | Train={len(train_pairs)} | Val={len(val_pairs)}")

# --- Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'right'

def render(messages, add_generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

# --- Dataset avec vérification de longueur
train_texts = []
val_texts = []

MAX_TEXT_LEN = 2000  # Longueur max en caractères pour éviter les problèmes

for p in train_pairs:
    text = render(p["messages"], add_generation_prompt=False)
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]  # Tronquer si trop long
    train_texts.append(text)

for p in val_pairs:
    text = render(p["messages"], add_generation_prompt=False)
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]  # Tronquer si trop long
    val_texts.append(text)

train_ds = Dataset.from_dict({"text": train_texts})
val_ds = Dataset.from_dict({"text": val_texts})
dset = DatasetDict({"train": train_ds, "validation": val_ds})

print(f"📊 Dataset ready - Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"📏 Max text length: {max([len(t) for t in train_texts + val_texts])} chars")

# --- Quantization
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Modèle
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    use_cache=False,
    attn_implementation="eager",
)

print(f"✅ Model loaded")

# --- LoRA simple et rapide pour Phi-3.5
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Modules corrects pour Phi-3.5
)

# --- Training config rapide
cfg = SFTConfig(
    output_dir=str(SAVE_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=2,
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    max_seq_length=MAX_SEQ_LEN,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    packing=False,
    dataset_text_field="text",
    report_to=[],
    remove_unused_columns=False,  # Fix pour l'erreur de colonnes dataset
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=dset["train"],
    tokenizer=tok,
    peft_config=peft_cfg,
)

print(">> 🚀 QUICK TRAINING START")
start_time = time.time()
train_res = trainer.train()
train_time = time.time() - start_time
print(f"✅ Training completed in {train_time:.1f}s! Loss: {train_res.training_loss:.4f}")

# --- Test
final_dir = SAVE_DIR / "final"
trainer.save_model(str(final_dir))
tok.save_pretrained(str(final_dir))

print("🔧 Merging model...")
merged = trainer.model.merge_and_unload()

print("🧪 Quick test - 3 predictions...")
gen = pipeline("text-generation", model=merged, tokenizer=tok, return_full_text=False, device_map="auto")

for i, pair in enumerate(val_pairs[:3]):
    test_messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": pair["note_only"]}
    ]
    test_prompt = render(test_messages, add_generation_prompt=True)
    
    try:
        result = gen(test_prompt, max_new_tokens=200, do_sample=False)
        generated = result[0]["generated_text"].strip()
        
        pred_file = PREDICT_DIR / f"{pair['cid']}_pred.txt"
        pred_file.write_text(generated, encoding='utf-8')
        
        print(f"✅ Test {i+1} ({pair['cid']}) - OK")
    except Exception as e:
        print(f"❌ Test {i+1}: {e}")

total_time = time.time() - start_time
print(f"🎯 QUICK TEST COMPLETE! Total time: {total_time:.1f}s")
print(f"📁 Saved: {SAVE_DIR}")
print(f"🔮 Predictions: {PREDICT_DIR}")
