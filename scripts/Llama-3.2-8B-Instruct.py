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

# -------------------- Réglages simples en haut --------------------
MODEL_ID = "meta-llama/Llama-3.2-8B-Instruct"   # Llama 3.2 8B Instruct
MODEL_SHORT = "Llama-8B"                         # Pour runs/
MAX_SEQ_LEN = 1024                               # 1024 tokens pour tenir en 1×L40
NUM_EPOCHS = 1                                   # mets 2 si tu veux plus de perf (plus long)
BATCH_SIZE = 1                                   # micro-batch
GRAD_ACCUM = 16                                  # => batch effectif = 16
LR = 2e-4                                        # QLoRA SFT: 1e-4 ~ 2e-4 souvent ok
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
SAVE_DIR = Path(f"runs/{MODEL_SHORT}")
ADAPTER_OUT = SAVE_DIR / "adapter"               # dossier des poids LoRA
MERGED_OUT  = SAVE_DIR / "merged_model"          # dossier du modèle fusionné final
# ------------------------------------------------------------------

INPUT_DIR  = Path("Data_input")
OUTPUT_DIR = Path("Data_output")
PRED_DIR   = Path("Data_predict") / f"{MODEL_SHORT}_predict"

# Nettoyer complètement le dossier de prédiction du modèle à chaque run
if PRED_DIR.exists():
    import shutil
    shutil.rmtree(PRED_DIR)
PRED_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
_start_time = time.time()

INSTRUCTION = (
    "From the clinical note, produce exactly the expected output as key=value lines. "
    "Return ONLY key=value pairs (one per line). If a value is unknown, write 'NA'. "
    "NO extra commentary, NO free text, NO JSON."
)

def get_id(name):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

# --- Charger toutes les paires (≈ 500)
notes  = {get_id(p.name): p for p in INPUT_DIR.glob("*.txt")}
outs   = {get_id(p.name): p for p in OUTPUT_DIR.glob("*.txt")}
common = sorted(set(notes) & set(outs))
assert len(common) > 0, "Aucune paire trouvée (vérifie Data_input/Data_output)!"

pairs = []
for cid in common:
    note = notes[cid].read_text(encoding="utf-8").strip()
    lab  = outs[cid].read_text(encoding="utf-8").strip()
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

# --- Tokenizer + chat template
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def render(messages, add_generation_prompt: bool):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )

train_texts = [render(p["messages"], add_generation_prompt=False) for p in train_pairs]
val_texts   = [render(p["messages"], add_generation_prompt=False) for p in val_pairs]

train_ds = Dataset.from_list([{"text": t} for t in train_texts])
val_ds   = Dataset.from_list([{"text": t} for t in val_texts])
dset = DatasetDict({"train": train_ds, "validation": val_ds})

# --- QLoRA 4-bit
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto"
)

# --- LoRA (ciblage Llama standard)
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
)

# --- SFT config
cfg = SFTConfig(
    output_dir=str(SAVE_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=max(20, len(train_ds)//10),  # une éval ~10x/epoch
    save_strategy="epoch",                   # on sauvegarde à la fin de chaque epoch
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
    peft_config=peft_cfg,
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=dset["train"],
    eval_dataset=dset["validation"],
    tokenizer=tok,
)

print(">> TRAIN START")
train_res = trainer.train()
print(">> TRAIN DONE:", train_res)

# --- Sauvegarder l'adapter LoRA (léger)
ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(ADAPTER_OUT))   # adapter_config.json + adapter_model.bin
tok.save_pretrained(ADAPTER_OUT)       # sauver aussi le tokenizer à côté

# --- (Option) Fusionner les poids LoRA + base et sauvegarder un modèle complet
print(">> MERGE LoRA into base (this loads base model in bf16)...")
base_full = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
peft = PeftModel.from_pretrained(base_full, ADAPTER_OUT)
merged = peft.merge_and_unload()       # fusion des poids dans le modèle de base

MERGED_OUT.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(MERGED_OUT, safe_serialization=True)
tok.save_pretrained(MERGED_OUT)
print(f">> MERGED MODEL saved to: {MERGED_OUT}")

# --- Génération: prédire TOUTES les notes (train+val) avec le modèle fusionné
print(">> GENERATION on all notes (merged model)...")
gen = pipeline(
    "text-generation",
    model=merged,
    tokenizer=tok,
    return_full_text=False
)

for p in pairs:
    gen_messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user",   "content": p["note_only"]},
    ]
    prompt = render(gen_messages, add_generation_prompt=True)
    out = gen(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]
    (PRED_DIR / f"{p['cid']}_pred.txt").write_text(out, encoding="utf-8")

print(f"✅ {MODEL_SHORT} - Predictions saved in {PRED_DIR}")
print(f"⏱️  Execution time: {time.time() - _start_time:.2f}s")
