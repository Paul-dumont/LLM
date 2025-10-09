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
    BitsAndBytesConfig, TrainingArguments,  # TrainingArguments pas directement utilisé, mais ok à garder
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model  # get_peft_model non utilisé, mais import sûr

# ==================== CONFIGURATION DU MODÈLE ====================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "train"

# ==================== HYPERPARAMÈTRES D'ENTRAÎNEMENT COMPLET ====================
MAX_SEQ_LEN = 4096         # Couvrir la plupart des inputs (max ~6-7k pour Qwen chat template)
NUM_EPOCHS  = 3            # 3 époques pour un apprentissage approfondi
BATCH_SIZE  = 2            # Batch size par device
GRAD_ACCUM  = 4            # Batch effectif = 8
LR          = 2e-5         # LR réduit pour stabilité
WEIGHT_DECAY = 0.01        # Régularisation L2
WARMUP_RATIO = 0.15        # Warmup plus long pour la stabilité
SAVE_STEPS   = 25          # Checkpointing fréquent
EVAL_STEPS   = 25
LOGGING_STEPS = 10

# ==================== CONTRÔLES PAR VARIABLES D'ENVIRONNEMENT ====================
MAX_INPUTS       = int(os.getenv("FULL_MAX_INPUTS", "0"))          # 0 = toutes les données
EVAL_RATIO       = float(os.getenv("FULL_EVAL_RATIO", "0.2"))      # 20% pour l'éval
USE_WANDB        = os.getenv("FULL_USE_WANDB", "0") == "1"         # Logging W&B optionnel
SAVE_FINAL       = os.getenv("FULL_SAVE_FINAL", "1") == "1"        # Sauvegarder modèle final
MAX_PREDICTIONS  = int(os.getenv("FULL_MAX_PREDICTIONS", "5"))     # # de prédictions de test
SEED             = int(os.getenv("FULL_SEED", "42"))

if USE_WANDB:
    import wandb

# ==================== REPRODUCTIBILITÉ ====================
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================== CONFIGURATION DES RÉPERTOIRES ====================
def _safe_basedir():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        # Fallback pour notebook/REPL
        return Path.cwd()

BASE_DIR   = _safe_basedir()
DATA_INPUT = BASE_DIR / "Data_input"
DATA_OUTPUT = BASE_DIR / "Data_output2"
SAVE_DIR   = BASE_DIR / "runs" / f"{MODEL_SHORT}_{int(time.time())}"

# Créer les dossiers si besoin
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_INPUT.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")
print(f"📤 DATA_OUTPUT: {DATA_OUTPUT}")
print(f"💾 SAVE_DIR: {SAVE_DIR}")

# ==================== INITIALISATION W&B (OPTIONNEL) ====================
def maybe_init_wandb():
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
                "warmup_ratio": WARMUP_RATIO,
            }
        )

# ==================== PROMPT SYSTÈME ====================
INSTRUCTION = """Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:"""

# ==================== FONCTION UTILITAIRE ====================
def get_id(name: str):
    """Extrait l'identifiant Bxxx du nom de fichier"""
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

# ==================== FONCTIONS PRINCIPALES ====================
def setup_tokenizer():
    """Configure le tokenizer"""
    print("🔧 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = 'right'

    def render(messages, add_generation_prompt=True):
        """Convertit les messages au format chat en texte selon le template Qwen"""
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    return tok, render

def load_data(tok):
    """Charge et prépare les données d'entraînement"""
    print("📂 Loading all available data...")
    pairs = []
    input_files = sorted(DATA_INPUT.glob("*.txt"))
    output_files = sorted(DATA_OUTPUT.glob("*.txt"))

    print(f"📁 Found {len(input_files)} input files and {len(output_files)} output files")

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
            lab  = out_file.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"❌ Error reading {cid}: {e}")
            continue

        if not note or not lab:
            print(f"❌ Empty content for {cid}")
            continue

        # Calcul de la longueur en tokens (approximation rapide)
        note_tokens = len(tok.encode(note))
        if note_tokens > MAX_SEQ_LEN - 200:  # marge pour le system prompt et la réponse
            print(f"⚠️  {cid}: {note_tokens} tokens → SERA TRONQUÉE (limite {MAX_SEQ_LEN})")

        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": note},
            {"role": "assistant", "content": lab},
        ]
        pairs.append({"cid": cid, "messages": messages, "note_only": note})

    print(f"📊 Total loaded pairs: {len(pairs)}")

    if len(pairs) < 10:
        raise RuntimeError(f"Pas assez de données pour un entraînement complet (n={len(pairs)})")

    # Division train/eval
    random.shuffle(pairs)
    n = len(pairs)
    n_eval = max(5, int(EVAL_RATIO * n))  # au moins 5 exemples pour l'évaluation
    train_pairs = pairs[:-n_eval]
    eval_pairs  = pairs[-n_eval:]

    print(f"🚀 FULL TRAINING {MODEL_SHORT}")
    print(f"📊 Total: {n} | Train: {len(train_pairs)} | Eval: {len(eval_pairs)}")

    return pairs, train_pairs, eval_pairs

def create_datasets(train_pairs, eval_pairs, render):
    """Crée les datasets Hugging Face"""
    print("📊 Creating datasets...")
    train_texts = [render(p["messages"], add_generation_prompt=False) for p in train_pairs]
    eval_texts  = [render(p["messages"], add_generation_prompt=False) for p in eval_pairs]

    train_ds = Dataset.from_list([{"text": t} for t in train_texts])
    eval_ds  = Dataset.from_list([{"text": t} for t in eval_texts])
    dset     = DatasetDict({"train": train_ds, "eval": eval_ds})

    print(f"📊 Datasets ready - Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    return dset

def setup_model(tok):
    """Configure et charge le modèle avec quantization 4-bit"""
    print("⚙️ Configuring 4-bit quantization...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Préfère bfloat16 si dispo
        bnb_4bit_use_double_quant=True,
    )

    print("🤖 Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="eager",  # peut être "sdpa" selon la stack
    )

    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False

    print("✅ Model loaded successfully")
    return model

def setup_lora():
    """Configure LoRA"""
    print("🔧 Configuring LoRA...")
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
            "gate_proj", "up_proj", "down_proj"       # MLP
        ]
    )
    return peft_cfg

def _bf16_available():
    # Détermine si bf16 est supporté (cartes récentes type A100/H100/L40…)
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except AttributeError:
        # Anciennes versions de PyTorch
        return False

def setup_training_config():
    """Configure les paramètres d'entraînement"""
    print("⚙️ Setting up training configuration...")

    bf16_ok = _bf16_available()
    fp16_ok = torch.cuda.is_available() and not bf16_ok

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
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        optim="paged_adamw_8bit",

        # ==================== ÉVALUATION ET LOGGING ====================
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,

        # ==================== SAUVEGARDE ====================
        save_strategy="steps" if SAVE_FINAL else "no",
        save_steps=SAVE_STEPS if SAVE_FINAL else 10000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ==================== TYPES ====================
        fp16=fp16_ok,
        bf16=bf16_ok,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,

        # ==================== CONFIGURATION SFT ====================
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",

        # ==================== REPORTING ====================
        report_to=report_to_list,
        run_name=f"{MODEL_SHORT}_{int(time.time())}" if USE_WANDB else None,
    )
    return cfg

def create_trainer(model, cfg, dset, tok, peft_cfg):
    """Crée le trainer SFT"""
    print("🏋️ Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dset["train"],
        eval_dataset=dset["eval"],
        tokenizer=tok,
        peft_config=peft_cfg,
    )
    return trainer

def save_config(pairs, train_pairs, eval_pairs, peft_cfg):
    """Sauvegarde la configuration d'entraînement"""
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
        },
        "lora_config": {
            "r": peft_cfg.r,
            "alpha": peft_cfg.lora_alpha,
            "dropout": peft_cfg.lora_dropout,
            "target_modules": list(peft_cfg.target_modules)
                if hasattr(peft_cfg.target_modules, '__iter__') else peft_cfg.target_modules,
        }
    }

    with open(SAVE_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    print(f"📋 Training config saved to: {SAVE_DIR / 'training_config.json'}")

def train_model(trainer):
    """Lance l'entraînement"""
    print("\n" + "="*80)
    print("🚀 STARTING FULL TRAINING")
    print("="*80)

    start_time = time.time()

    try:
        train_result = trainer.train()
        train_time = time.time() - start_time

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏱️  Total training time: {train_time:.1f}s ({train_time/60:.1f} minutes)")

        training_loss = getattr(train_result, "training_loss", None)
        if training_loss is not None and not (training_loss != training_loss):  # non-NaN
            print(f"📉 Final training loss: {training_loss:.4f}")
        else:
            print("📉 Final training loss: NaN (training failed)")
            training_loss = float('nan')

        return train_result, training_loss, train_time

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if USE_WANDB:
            wandb.finish(exit_code=1)
        raise

def save_model(trainer, tok, training_loss, train_time, train_result):
    """Sauvegarde le modèle et les résultats"""
    if SAVE_FINAL:
        print("💾 Saving final model...")
        # Sauvegarder l'adaptateur LoRA (et poids associés) via save_pretrained
        trainer.model.save_pretrained(str(SAVE_DIR / "final_model"))
        tok.save_pretrained(str(SAVE_DIR / "final_model"))

        results = {
            "training_loss": training_loss,
            "training_time": train_time,
            "total_steps": getattr(train_result, "global_step", None),
            "epochs_completed": NUM_EPOCHS,
        }
        with open(SAVE_DIR / "training_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ Model saved to: {SAVE_DIR / 'final_model'}")
        print(f"📊 Results saved to: {SAVE_DIR / 'training_results.json'}")

def evaluate_model(trainer):
    """Évalue le modèle final"""
    print("\n🧪 Running final evaluation...")
    eval_result = trainer.evaluate()
    eval_loss = eval_result.get('eval_loss', float('nan'))
    if eval_loss is not None and not (eval_loss != eval_loss):
        print(f"📊 Final evaluation loss: {eval_loss:.4f}")
    else:
        print("📊 Final evaluation loss: NaN (possible training issue)")
        eval_loss = float('nan')
    return eval_loss

def generate_predictions(trainer, tok, eval_pairs, render):
    """Génère des prédictions de test (pipeline identique au smoke test)"""
    if MAX_PREDICTIONS <= 0:
        return

    print(f"\n🧪 Generating {MAX_PREDICTIONS} test predictions...")
    predict_dir = BASE_DIR / "Data_predict" / f"{MODEL_SHORT}_predict"
    predict_dir.mkdir(parents=True, exist_ok=True)

    # Nettoyage des anciennes prédictions
    for old_pred in predict_dir.glob("*.txt"):
        try:
            old_pred.unlink()
        except Exception:
            pass

    from transformers import pipeline
    device_id = 0 if torch.cuda.is_available() else -1

    trained_model = trainer.model
    trained_model.eval()

    gen = pipeline(
        "text-generation",
        model=trained_model,
        tokenizer=tok,
        device=device_id if device_id >= 0 else None,
        return_full_text=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id
    )

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
                max_new_tokens=500,  # marge suffisante
                do_sample=False
            )
            generated = result[0]["generated_text"].strip()

            pred_file = predict_dir / f"{pair['cid']}_pred.txt"
            pred_file.write_text(generated, encoding='utf-8')

            print(f"✅ Test {i+1} ({pair['cid']}) - Prediction saved: {pred_file}")
            print(f"   Preview: {generated[:100]}...")

        except Exception as e:
            print(f"❌ Test {i+1} ({pair['cid']}): {e}")

    print(f"📁 Predictions saved in: {predict_dir}")

# ==================== CODE PRINCIPAL ====================
if __name__ == "__main__":
    # 1) Tokenizer d'abord
    tok, render = setup_tokenizer()

    # 2) (Optionnel) W&B
    maybe_init_wandb()

    # 3) Données
    pairs, train_pairs, eval_pairs = load_data(tok)

    # 4) Datasets
    dset = create_datasets(train_pairs, eval_pairs, render)

    # 5) Modèle + LoRA + config d'entraînement
    model    = setup_model(tok)
    peft_cfg = setup_lora()
    cfg      = setup_training_config()

    # 6) Trainer
    trainer = create_trainer(model, cfg, dset, tok, peft_cfg)

    # 7) Sauvegarde de la config
    save_config(pairs, train_pairs, eval_pairs, peft_cfg)

    # 8) Entraînement
    train_result, training_loss, train_time = train_model(trainer)

    # 9) Sauvegarde du modèle
    save_model(trainer, tok, training_loss, train_time, train_result)

    # 10) Évaluation finale
    eval_loss = evaluate_model(trainer)

    # 11) Logging W&B si activé
    if USE_WANDB:
        wandb.log({
            "final_train_loss": float(training_loss) if training_loss == training_loss else None,
            "final_eval_loss": float(eval_loss) if eval_loss == eval_loss else None,
            "training_time": float(train_time),
        })
        wandb.finish()

    # 12) Génération de prédictions
    generate_predictions(trainer, tok, eval_pairs, render)

    print("\n🎯 FULL TRAINING PIPELINE COMPLETED!")
    print(f"📁 All outputs saved in: {SAVE_DIR}")
    if MAX_PREDICTIONS > 0:
        print(f"🧪 Test predictions available in: Data_predict/{MODEL_SHORT}_predict/")
