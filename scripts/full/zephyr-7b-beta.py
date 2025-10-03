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

# -------------------- Réglages optimisés pour Zephyr-7B (CHAMPION) --------------------
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"       # Zephyr 7B DPO fine-tuned (SOTA)
MODEL_SHORT = "Zephyr-7B"                       # Pour runs/
MAX_SEQ_LEN = 3072                               # 3K tokens (vos fichiers font max ~3.3K)
NUM_EPOCHS = 1                                   # 1 epoch suffisant pour modèle déjà fine-tuné
BATCH_SIZE = 2                                   # Batch optimal pour 7B sur L40
GRAD_ACCUM = 8                                   # => batch effectif = 16
LR = 1e-4                                        # LR réduit car déjà fine-tuné DPO
WEIGHT_DECAY = 0.01                              # Régularisation légère
WARMUP_RATIO = 0.03                              # Warmup minimal (3%)

# Répertoires
BASE_DIR = Path(__file__).parent.parent
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT = BASE_DIR / "Data_output"
SAVE_DIR = BASE_DIR / "runs" / f"{MODEL_SHORT}_{int(time.time())}"
PREDICT_DIR = BASE_DIR / "Data_predict" / f"{MODEL_SHORT}_predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = (
    "CRITICAL: You must extract information from the clinical note and format it as EXACT key=value pairs.\n\n"
    "RULES:\n"
    "1. Use ONLY this format: key_name=value\n"
    "2. One key=value pair per line\n"
    "3. NO headers, NO sections, NO ### markup\n"
    "4. If unknown, use: key_name=NA\n"
    "5. NO extra text, NO explanations, NO JSON\n\n"
    "EXAMPLE OUTPUT:\n"
    "patient_id=B123\n"
    "patient_age=45\n"
    "pain_location=left jaw\n"
    "pain_intensity=7\n\n"
    "START YOUR RESPONSE WITH THE FIRST key=value PAIR:"
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
    
    # Format de conversation Zephyr (compatible ChatML)
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

print(f"🚀 Training {MODEL_SHORT} (DPO SOTA) | Total={n} | Train={len(train_pairs)} | Val={len(val_pairs)}")

# --- Tokenizer + chat template (Zephyr utilise ChatML)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Fix padding side for SFTTrainer
tok.padding_side = 'right'

# Zephyr utilise le format ChatML, vérifier si template existe
if tok.chat_template is None:
    # Template ChatML standard pour Zephyr si absent
    tok.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    print("✅ ChatML template configured for Zephyr")

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

# --- 4-bit quantization config optimisé pour Zephyr-7B
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Charger le modèle Zephyr avec quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,  # Fixed: torch_dtype -> dtype
    use_cache=False,  # Fixed: Éviter les problèmes de cache
    attn_implementation="eager",  # Fixed: Use eager instead of flash_attention_2
)

print(f"✅ Zephyr-7B loaded with 4-bit quantization")

# --- LoRA config optimisé pour Zephyr (Mistral architecture)
peft_cfg = LoraConfig(
    r=16,                       # Rang LoRA standard
    lora_alpha=32,              # Alpha scaling
    lora_dropout=0.05,          # Dropout réduit (modèle déjà stable)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        # Modules Mistral-7B (base de Zephyr)
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

# --- SFT config optimisé pour Zephyr fine-tuning
cfg = SFTConfig(
    output_dir=str(SAVE_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=5,                             # Log plus fréquent
    eval_strategy="steps",
    eval_steps=max(10, len(train_ds)//20),       # Éval plus fréquente
    save_strategy="epoch",
    save_total_limit=2,
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
    # Fixed: Remove problematic parameter
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=dset["train"],
    eval_dataset=dset["validation"],
    tokenizer=tok,  # Fixed: Back to tokenizer for TRL 0.9.6
    peft_config=peft_cfg,
)

print(">> 🚀 ZEPHYR TRAINING START (DPO-enhanced)")
train_res = trainer.train()
print(f"✅ Training completed! Loss: {train_res.training_loss:.4f}")

# --- Sauvegarder le modèle final
final_dir = SAVE_DIR / "final"
trainer.save_model(str(final_dir))
tok.save_pretrained(str(final_dir))
print(f"💾 Zephyr model saved to: {final_dir}")

# --- MODE TEST: Fusionner et générer (approche Llama qui fonctionne)
print("🔧 Merging LoRA adapters with base model...")
merged = trainer.model.merge_and_unload()

print("🧪 TEST MODE: Generating 10 predictions for format validation...")
gen = pipeline(
    "text-generation",
    model=merged,
    tokenizer=tok,
    return_full_text=False,
    device_map="auto"
)

# Prendre les 10 premiers exemples de tous les pairs (pas juste validation)
test_pairs = pairs[:10]  # 10 premiers pour test complet
print(f"🔬 Testing on {len(test_pairs)} examples...")

for i, pair in enumerate(test_pairs):
    test_messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": pair["note_only"]}
    ]
    test_prompt = render(test_messages, add_generation_prompt=True)
    
    try:
        result = gen(test_prompt, max_new_tokens=512, do_sample=False)
        generated = result[0]["generated_text"].strip()
        
        # Nettoyer les tokens spéciaux Zephyr
        if "<|im_end|>" in generated:
            generated = generated.split("<|im_end|>")[0].strip()
            
    except Exception as e:
        print(f"❌ Erreur génération exemple {i+1}: {e}")
        generated = "ERREUR_GENERATION"
    
    # Nettoyer la sortie (enlever tokens spéciaux si présents)
    if "<|im_end|>" in generated:
        generated = generated.split("<|im_end|>")[0].strip()
    
    # Sauvegarder la prédiction
    pred_file = PREDICT_DIR / f"{pair['cid']}_pred.txt"
    pred_file.write_text(generated, encoding='utf-8')
    
    print(f"\n--- Zephyr Test {i+1} ({pair['cid']}) ---")
    print(f"Generated: {generated[:300]}...")

print(f"\n🎯 ZEPHYR TRAINING COMPLETE!")
print(f"📁 Model saved: {SAVE_DIR}")
print(f"🔮 Predictions: {PREDICT_DIR}")
print(f"🏆 Quality: DPO fine-tuned for superior instruction following")
print(f"⚡ Performance: 7B params optimized for medical extraction")
