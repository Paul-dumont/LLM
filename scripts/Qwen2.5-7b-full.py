# qwen_full_ft.py
# ==================== QWEN 2.5-7B INSTRUCT - FULL FINETUNE (A100, NO LoRA/NO QUANT) ====================
from pathlib import Path
import os, re, time, json, random
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# ==================== PARAMS ====================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "qwen7b_full"

MAX_SEQ_LEN  = int(os.getenv("MAX_SEQ_LEN", 4096))
NUM_EPOCHS   = int(os.getenv("NUM_EPOCHS", 3))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 1))       # per-device
GRAD_ACCUM   = int(os.getenv("GRAD_ACCUM", 8))
LR           = float(os.getenv("LR", "2e-5"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.10"))
SAVE_STEPS   = int(os.getenv("SAVE_STEPS", 200))
EVAL_STEPS   = int(os.getenv("EVAL_STEPS", 200))
LOG_STEPS    = int(os.getenv("LOG_STEPS", 50))
USE_WANDB    = os.getenv("USE_WANDB", "0") == "1"
EVAL_RATIO   = float(os.getenv("EVAL_RATIO", "0.2"))
MAX_INPUTS   = int(os.getenv("MAX_INPUTS", "0"))     # 0 = all
MAX_PRED     = int(os.getenv("MAX_PRED", "5"))
SEED         = int(os.getenv("SEED", "42"))

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR   = Path(__file__).parent
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT= BASE_DIR / "Data_output2"
RUN_DIR    = BASE_DIR / "runs" / f"{MODEL_SHORT}_{int(time.time())}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:"

# ==================== UTILS ====================
def get_id(name: str):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

def setup_tokenizer():
    print("🔧 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    def render(messages, add_generation_prompt=True):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tok, render

def load_data(tok):
    print("📂 Loading data...")
    pairs = []
    input_files  = sorted(DATA_INPUT.glob("*.txt"))
    output_files = sorted(DATA_OUTPUT.glob("*.txt"))
    files_to_process = input_files if MAX_INPUTS == 0 else input_files[:MAX_INPUTS]

    id2out = {}
    for of in output_files:
        cid = get_id(of.name)
        if cid: id2out[cid] = of

    for inp in files_to_process:
        cid = get_id(inp.name)
        if not cid or cid not in id2out: 
            print(f"❌ Missing match for {inp.name}")
            continue
        try:
            note = inp.read_text(encoding="utf-8").strip()
            lab  = id2out[cid].read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"❌ Read error {cid}: {e}"); continue
        if not note or not lab: 
            print(f"❌ Empty content {cid}"); continue

        if len(tok.encode(note)) > MAX_SEQ_LEN - 200:
            print(f"⚠️  {cid}: input will be truncated at {MAX_SEQ_LEN}")

        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user",   "content": note},
            {"role": "assistant","content": lab},
        ]
        pairs.append({"cid": cid, "messages": messages, "note_only": note})

    if len(pairs) < 10:
        raise RuntimeError(f"Not enough training pairs (n={len(pairs)})")

    random.shuffle(pairs)
    n = len(pairs)
    n_eval = max(5, int(EVAL_RATIO * n))
    train_pairs, eval_pairs = pairs[:-n_eval], pairs[-n_eval:]
    print(f"📊 Total: {n} | Train: {len(train_pairs)} | Eval: {len(eval_pairs)}")
    return pairs, train_pairs, eval_pairs

def make_datasets(train_pairs, eval_pairs, render):
    train_texts = [render(p["messages"], add_generation_prompt=False) for p in train_pairs]
    eval_texts  = [render(p["messages"], add_generation_prompt=False) for p in eval_pairs]
    dset = DatasetDict({
        "train": Dataset.from_list([{"text": t} for t in train_texts]),
        "eval":  Dataset.from_list([{"text": t} for t in eval_texts]),
    })
    print(f"✅ Datasets ready: train={len(dset['train'])} eval={len(dset['eval'])}")
    return dset

def load_model(tok):
    print("🤖 Loading base model for full finetune (bf16)…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",      # works out-of-the-box on A100 with PyTorch SDPA
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False  # disable during training
    # memory helpers
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model

def sft_config():
    report_to = ["wandb"] if USE_WANDB else []
    return SFTConfig(
        output_dir=str(RUN_DIR),
        logging_dir=str(RUN_DIR / "logs"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(1, min(2, BATCH_SIZE)),
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,                    # A100 → bf16
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
        report_to=report_to,
        run_name=f"{MODEL_SHORT}_{int(time.time())}" if USE_WANDB else None,
    )

def train_and_eval(model, tok, dset, cfg):
    print("🏋️ Training…")
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dset["train"],
        eval_dataset=dset["eval"],
        tokenizer=tok,
    )
    tr = trainer.train()
    print("✅ Training finished")
    results = {
        "training_loss": float(tr.training_loss) if tr.training_loss is not None else None,
        "global_step": tr.global_step,
        "epochs": NUM_EPOCHS,
    }
    (RUN_DIR / "training_results.json").write_text(json.dumps(results, indent=2))
    print(f"📄 Saved: {RUN_DIR/'training_results.json'}")

    print("🧪 Evaluating best model…")
    ev = trainer.evaluate()
    (RUN_DIR / "eval_results.json").write_text(json.dumps(ev, indent=2))
    print(f"📄 Saved: {RUN_DIR/'eval_results.json'}")
    return trainer

def save_final(trainer, tok):
    out = RUN_DIR / "final_model"
    out.mkdir(exist_ok=True, parents=True)
    print("💾 Saving final model/tokenizer…")
    trainer.model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"✅ Saved to {out}")

def predict_samples(trainer, tok, eval_pairs, render):
    if MAX_PRED <= 0: return
    from transformers import pipeline
    pred_dir = BASE_DIR / "Data_predict" / f"{MODEL_SHORT}_predict"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for f in pred_dir.glob("*.txt"): f.unlink()

    gen = pipeline(
        task="text-generation",
        model=trainer.model.eval(),
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        torch_dtype=torch.bfloat16,
    )

    for i, pair in enumerate(eval_pairs[:MAX_PRED], 1):
        prompt = render([
            {"role":"system","content":INSTRUCTION},
            {"role":"user","content":pair["note_only"]}
        ], add_generation_prompt=True)
        out = gen(prompt, max_new_tokens=500, do_sample=False)
        txt = out[0]["generated_text"].strip()
        (pred_dir / f"{pair['cid']}_pred.txt").write_text(txt, encoding="utf-8")
        print(f"✅ Pred {i}/{MAX_PRED} → {pair['cid']}")

def main():
    print(f"🏠 RUN_DIR: {RUN_DIR}")
    tok, render = setup_tokenizer()
    pairs, train_pairs, eval_pairs = load_data(tok)
    dset = make_datasets(train_pairs, eval_pairs, render)
    model = load_model(tok)
    cfg = sft_config()

    # persist config
    (RUN_DIR / "training_config.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "max_seq_len": MAX_SEQ_LEN,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "sizes": {"total_pairs": len(pairs), "train": len(train_pairs), "eval": len(eval_pairs)},
    }, indent=2))

    trainer = train_and_eval(model, tok, dset, cfg)
    save_final(trainer, tok)
    predict_samples(trainer, tok, eval_pairs, render)
    print("\n🎯 FULL TRAINING PIPELINE COMPLETED")

if __name__ == "__main__":
    main()
