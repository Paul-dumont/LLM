# --- DEBUG & sécurité ---
import os, sys, time, torch
from pathlib import Path

t0 = time.time()
print(">>> START Qwen2.5-1.5B.py")
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("GPU name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("GPU name: <unknown>", e)

# Caches HF (au cas où)
os.environ.setdefault("HF_HOME", str(Path.home()/".cache/huggingface"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","1")

# --- chemins ---
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR  = BASE_DIR / "Data_input"
OUTPUT_DIR = BASE_DIR / "Data_output"
PRED_DIR   = BASE_DIR / "Data_predict" / "Qwen2.5-1.5B_predict"; PRED_DIR.mkdir(parents=True, exist_ok=True)

assert INPUT_DIR.exists(),  f"Missing folder: {INPUT_DIR}"
assert OUTPUT_DIR.exists(), f"Missing folder: {OUTPUT_DIR}"
print("Folders OK:",
      "Data_input=", INPUT_DIR.resolve(),
      "Data_output=", OUTPUT_DIR.resolve(),
      "Data_predict=", PRED_DIR.resolve())

import re, random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import SFTTrainer, SFTConfig

INSTRUCTION = (
    "From the clinical note, produce exactly the expected output text. "
    "RETURN ONLY a structured text file with 56 key=value pairs (one per line), "
    "without introduction sentences or free text."
)

def get_id(name):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

notes  = {get_id(p.name): p for p in INPUT_DIR.glob("*.txt")}
outs   = {get_id(p.name): p for p in OUTPUT_DIR.glob("*.txt")}
common = sorted(set(notes) & set(outs))
print(f">>> Found {len(notes)} notes, {len(outs)} labels, {len(common)} paired IDs")
if not common:
    raise RuntimeError("No paired files (Bxxxx) found between Data_input and Data_output.")

pairs = []
for cid in common:
    note = notes[cid].read_text(encoding="utf-8").strip()
    lab  = outs[cid].read_text(encoding="utf-8").strip()
    prompt = f"[INST]{INSTRUCTION}\n\n{note}[/INST]\n"
    pairs.append({"cid": cid, "text": prompt + lab, "note_only": note})

# smoke test : 10 paires
random.seed(42); random.shuffle(pairs)
pairs = pairs[:10]
print(f">>> Using {len(pairs)} pairs for smoke test")

n = len(pairs)
n_val = max(1, n // 5)
train_pairs = pairs[:-n_val] if n > 1 else pairs
val_pairs   = pairs[-n_val:] if n > 1 else pairs
print(f">>> Split: train={len(train_pairs)} val={len(val_pairs)}")

train_ds = Dataset.from_list([{"text": p["text"]} for p in train_pairs])
val_ds   = Dataset.from_list([{"text": p["text"]} for p in val_pairs])
dset = DatasetDict({"train": train_ds, "validation": val_ds})

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f">>> Loading tokenizer/model: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", trust_remote_code=True)
print(">>> Model loaded in %.1fs" % (time.time()-t0))

cfg = SFTConfig(
    output_dir="runs/local_train",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    eval_strategy="no",
    save_strategy="no",
    max_seq_length=1024,
    fp16=False,
    bf16=True
)
trainer = SFTTrainer(
    model=model, args=cfg,
    train_dataset=dset["train"], eval_dataset=dset["validation"],
    tokenizer=tok,
    dataset_text_field="text",
)

print(f">>> Train {len(train_pairs)} / Val {len(val_pairs)} — starting training…")
t1 = time.time()
trainer.train()
print(">>> Training done in %.1fs" % (time.time()-t1))

# génération propre (pas de warnings)
print(">>> Starting generation…")
gen = pipeline("text-generation", model=trainer.model, tokenizer=tok, return_full_text=False)
gen_kwargs = dict(max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None)

ok = 0
for p in pairs:
    prompt = f"[INST]{INSTRUCTION}\n\n{p['note_only']}[/INST]\n"
    out = gen(prompt, **gen_kwargs)[0]["generated_text"].strip()
    (PRED_DIR / f"{p['cid']}_pred.txt").write_text(out + "\n", encoding="utf-8")
    ok += 1

print(f">>> Wrote {ok} prediction files to {PRED_DIR.resolve()}")
print(">>> DONE in %.1fs" % (time.time()-t0))
