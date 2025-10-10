# ==================== QWEN 2.5-7B INSTRUCT — QLoRA SFT ====================
from pathlib import Path
import re, random, os, time
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from contextlib import nullcontext

# -------------------- PARAMS --------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 3072
NUM_EPOCHS = 3
BATCH_SIZE = 1                     # micro-batch = 1 (VRAM safe)
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 16                    # batch effectif via accumulation
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.15
SAVE_STEPS = 25
EVAL_STEPS = 25
LOGGING_STEPS = 10
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

BASE_DIR = Path(__file__).parent.parent
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT = BASE_DIR / "Data_output2"
SAVE_DIR = BASE_DIR / "runs" / f"train_{int(time.time())}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:"

# -------------------- HELPERS --------------------
def get_id(name: str):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

def setup_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    def render(messages, add_generation_prompt=True):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tok, render

def load_data(tok):
    pairs = []
    input_files = sorted(DATA_INPUT.glob("*.txt"))
    output_files = sorted(DATA_OUTPUT.glob("*.txt"))
    if not input_files or not output_files:
        raise RuntimeError("DATA_INPUT ou DATA_OUTPUT est vide.")
    for inp in input_files:
        cid = get_id(inp.name)
        if not cid:
            continue
        out = next((of for of in output_files if get_id(of.name) == cid), None)
        if not out:
            continue
        note = inp.read_text(encoding="utf-8").strip()
        lab = out.read_text(encoding="utf-8").strip()
        if not note or not lab:
            continue
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": note},
            {"role": "assistant", "content": lab},
        ]
        pairs.append({"cid": cid, "messages": messages})

    if not pairs:
        raise RuntimeError("Aucune paire appariée note/label trouvée.")

    random.shuffle(pairs)
    # pairs = pairs[:50]  # limiter pour test rapide

    n_eval = int(0.2 * len(pairs))
    if n_eval == 0:
        train_pairs, eval_pairs = pairs, []
    else:
        train_pairs = pairs[:-n_eval] if n_eval < len(pairs) else []
        eval_pairs = pairs[-n_eval:] if n_eval < len(pairs) else pairs
        if not train_pairs:  # garde un minimum pour train
            train_pairs, eval_pairs = pairs, []

    return train_pairs, eval_pairs

def create_datasets(train_pairs, eval_pairs, render, tok):
    train_texts = [render(p["messages"], add_generation_prompt=False) for p in train_pairs]
    eval_texts  = [render(p["messages"], add_generation_prompt=False) for p in eval_pairs]
    if not train_texts:
        raise RuntimeError("Dataset d'entraînement vide.")
    dset = DatasetDict({
        "train": Dataset.from_list([{"text": t} for t in train_texts]),
        "eval":  Dataset.from_list([{"text": t} for t in eval_texts]) if eval_texts else Dataset.from_list([{"text": ""}]),
    })
    return dset

def setup_model(tok):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,  # off pendant train
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    # Si Flash-Attention 2 dispo, active-la
    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass
    return model

def setup_lora():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

def setup_training_config(has_eval=True):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return SFTConfig(
        output_dir=str(SAVE_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,            # 1
        per_device_eval_batch_size=EVAL_BATCH_SIZE,        # 1
        gradient_accumulation_steps=GRAD_ACCUM,            # 16
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=EVAL_STEPS if has_eval else None,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=(not use_bf16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
        optim="adamw_bnb_8bit",                           # clé VRAM
    )

def main():
    tok, render = setup_tokenizer()
    train_pairs, eval_pairs = load_data(tok)
    dset = create_datasets(train_pairs, eval_pairs, render, tok)
    model = setup_model(tok)
    peft_cfg = setup_lora()
    cfg = setup_training_config(has_eval=(len(eval_pairs) > 0))

    # Sanity check QLoRA
    print("is_loaded_in_4bit:", getattr(model, "is_loaded_in_4bit", False))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dset["train"],
        eval_dataset=(dset["eval"] if len(eval_pairs) > 0 else None),
        tokenizer=tok,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.model.save_pretrained(str(SAVE_DIR / "final_model"))
    tok.save_pretrained(str(SAVE_DIR / "final_model"))

    # -------------------- PREDICTIONS (5 CAS) --------------------
    PRED_DIR = BASE_DIR / "Data_predict" / "qlora"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.config.use_cache = True

    # Harmoniser la tête en BF16 (évite conflit BF16/Float)
    if hasattr(model, "lm_head"):
        try:
            model.lm_head = model.lm_head.to(torch.bfloat16)
        except Exception:
            pass

    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()

    for i, pair in enumerate(eval_pairs[:5]):
        prompt = render([
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": pair["messages"][1]["content"]},
        ], add_generation_prompt=True)

        inputs = tok(prompt, return_tensors="pt")  # ne pas .to() directement si device_map="auto"
        # Route les tensors vers le même device que le 1er paramètre du modèle
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with amp_ctx:
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )

        gen = outputs[0, inputs["input_ids"].shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True).strip()

        cid = pair.get("cid", f"case{i+1}")
        out_path = PRED_DIR / f"{cid}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"✅ Saved prediction {i+1} → {out_path}")

if __name__ == "__main__":
    main()
