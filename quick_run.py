# ssh dumontp@longleaf.unc.edu 
# GtX6RNe4*AFgA6D

from pathlib import Path
import re, random
import time
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import SFTTrainer, SFTConfig

INPUT_DIR  = Path("Data_input")
OUTPUT_DIR = Path("Data_output")
PRED_DIR   = Path("Data_predict"); PRED_DIR.mkdir(parents=True, exist_ok=True)

# start timer
_start_time = time.time()

# ++ Instruction un peu plus stricte sur le format
INSTRUCTION = (
    "From the clinical note, produce exactly the expected output text. "
    "RETURN ONLY a structured text file with 56 key=value pairs (one per line), "
    "without introduction sentences or free text."
)

# --- associer fichiers par ID ---
def get_id(name):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

notes  = {get_id(p.name): p for p in INPUT_DIR.glob("*.txt")}
outs   = {get_id(p.name): p for p in OUTPUT_DIR.glob("*.txt")}
common = sorted(set(notes) & set(outs))

pairs = []
for cid in common:
    note = notes[cid].read_text(encoding="utf-8").strip()
    lab  = outs[cid].read_text(encoding="utf-8").strip()
    prompt = f"[INST]{INSTRUCTION}\n\n{note}[/INST]\n"
    pairs.append({"cid": cid, "text": prompt + lab, "note_only": note})

# --- mélange puis limite (pour échantillonner au hasard) ---
random.seed(42); random.shuffle(pairs)
pairs = pairs[:10]   # ~8 train / 2 val avec le split 80/20

n = len(pairs)
n_val = max(1, n // 5)
train_pairs = pairs[:-n_val] if n > 1 else pairs
val_pairs   = pairs[-n_val:] if n > 1 else pairs

train_ds = Dataset.from_list([{"text": p["text"]} for p in train_pairs])
val_ds   = Dataset.from_list([{"text": p["text"]} for p in val_pairs])
dset = DatasetDict({"train": train_ds, "validation": val_ds})

# --- petit modèle pour tester ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # rapide pour smoke test GPU
# (alternative un peu plus costaude) # MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

cfg = SFTConfig(
    output_dir="runs/local_train",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    evaluation_strategy="steps",   # (corrige eval_strategy)
    eval_steps=10,
    max_seq_length=1024,           # (corrige max_length) 1024 pour aller vite
    fp16=False,
    bf16=True,                     # bfloat16 sur L40/A100
    # no_cuda=True                 # ne pas mettre pour utiliser le GPU
)

trainer = SFTTrainer(
    model=model, args=cfg,
    train_dataset=dset["train"], eval_dataset=dset["validation"],
    tokenizer=tok,                 # <= (au lieu de processing_class)
    dataset_text_field="text",     # <= précise où est le texte
    formatting_func=None
)

print(f"Train {len(train_pairs)} / Val {len(val_pairs)}")
trainer.train()
trainer.save_model("runs/local_test")  # ok de garder si tu veux la trace sur disque

# --- génération : réutiliser le modèle en mémoire pour rester sur GPU ---
gen = pipeline(
    "text-generation",
    model=trainer.model,
    tokenizer=tok,
    return_full_text=False         # on ne veut pas que le prompt soit recopié
)

for p in pairs:
    prompt = f"[INST]{INSTRUCTION}\n\n{p['note_only']}[/INST]\n"
    out = gen(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    (PRED_DIR / f"{p['cid']}_pred.txt").write_text(out, encoding="utf-8")

print(f"Sorties générées dans {PRED_DIR}")

# affiche le temps d'exécution total
_elapsed = time.time() - _start_time
print(f"Temps d'exécution (wall-clock): {_elapsed:.2f} s")
