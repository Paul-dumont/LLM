#!/usr/bin/env python3
"""
Regenerate predictions for Qwen2.5-7B-Instruct (full) with longer outputs (max_new_tokens=500)
using the already trained LoRA adapter saved in runs/.../final_model.

This script:
- Auto-detects the latest run folder for Qwen2.5-7B-full unless --model-dir is provided
- Reads the same IDs from Data_predict/Qwen2.5-7B-full_predict (the 5 prior predictions)
- Loads the saved tokenizer and LoRA model
- Generates new predictions with max_new_tokens=500
- Saves to Data_predict/Qwen2.5-7B-full_predict_500

No retraining performed.
"""

from pathlib import Path
import argparse
import re
import sys
import os
import json
import torch

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

try:
    # Prefer AutoPeftModelForCausalLM to load base + adapter in one call
    from peft import AutoPeftModelForCausalLM
    HAVE_AUTO_PEFT = True
except Exception:
    HAVE_AUTO_PEFT = False


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "Qwen2.5-7B-full"


INSTRUCTION = """You are a medical data extraction specialist. Extract the following 56 specific clinical indicators from this medical note and format as key=value pairs. Use "unknown" if information is not available.

REQUIRED INDICATORS:
patient_id=
patient_age=
headache_intensity=
headache_frequency=
headache_location=
migraine_history=
migraine_frequency=
average_daily_pain_intensity=
diet_score=
tmj_pain_rating=
disability_rating=
jaw_function_score=
jaw_clicking=
jaw_crepitus=
jaw_locking=
maximum_opening=
maximum_opening_without_pain=
disc_displacement=
muscle_pain_score=
muscle_pain_location=
muscle_spasm_present=
muscle_tenderness_present=
muscle_stiffness_present=
muscle_soreness_present=
joint_pain_areas=
joint_arthritis_location=
neck_pain_present=
back_pain_present=
earache_present=
tinnitus_present=
vertigo_present=
hearing_loss_present=
hearing_sensitivity_present=
sleep_apnea_diagnosed=
sleep_disorder_type=
airway_obstruction_present=
anxiety_present=
depression_present=
stress_present=
autoimmune_condition=
fibromyalgia_present=
current_medications=
previous_medications=
adverse_reactions=
appliance_history=
current_appliance=
cpap_used=
apap_used=
bipap_used=
physical_therapy_status=
pain_onset_date=
pain_duration=
pain_frequency=
onset_triggers=
pain_relieving_factors=
pain_aggravating_factors=

EXTRACTION RULES:
- Use exact indicator names as shown above
- For boolean indicators: use "true", "false", or "unknown"
- For numeric values: extract numbers and units (e.g., "25 years", "7/10", "45mm")
- For text values: be concise but specific
- If multiple values exist, use semicolon separation
- Always include all 56 indicators, even if "unknown"
"""


def get_id(name: str):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None


def detect_latest_model_dir(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    # Find subfolders starting with MODEL_SHORT and having a 'final_model' directory
    candidates = []
    for p in runs_dir.iterdir():
        if p.is_dir() and p.name.startswith(MODEL_SHORT):
            fm = p / "final_model"
            if fm.exists() and fm.is_dir():
                candidates.append(fm)
    if not candidates:
        return None
    # Sort by parent mtime (most recent run)
    candidates.sort(key=lambda d: d.parent.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_input_file(data_input: Path, cid: str) -> Path | None:
    # Prefer the canonical naming present in the dataset
    candidate = data_input / f"{cid}_Word_text.txt"
    if candidate.exists():
        return candidate
    # Fallback: any txt starting with cid
    matches = sorted(data_input.glob(f"{cid}_*.txt"))
    if matches:
        return matches[0]
    # Last chance: any txt containing cid
    matches = sorted(x for x in data_input.glob("*.txt") if cid in x.name)
    return matches[0] if matches else None


def build_pipeline(model_dir: Path, use_4bit: bool = True):
    print(f"🔧 Loading tokenizer from: {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    print("🤖 Loading model (with adapter) for generation...")
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": "eager",  # match original script
    }
    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb_cfg
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    if HAVE_AUTO_PEFT:
        # Load base model and attach LoRA from the adapter directory
        model = AutoPeftModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    else:
        # Fallback: require internet/cache to fetch base model
        from transformers import AutoModelForCausalLM
        print("⚠️ peft.AutoPeftModelForCausalLM not available; loading base model then adapter may fail.")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(model_dir))
        except Exception as e:
            print(f"❌ Failed to attach adapter: {e}")
            raise

    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    # Match original script behavior
    model.config.use_cache = False

    print("✅ Model ready")
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        return_full_text=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok, gen


def render(tok, messages, add_generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def main():
    parser = argparse.ArgumentParser(description="Regenerate Qwen full predictions with longer outputs (500 tokens)")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to runs/.../final_model. If omitted, auto-detect latest.")
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("PRED_MAX_NEW_TOKENS", "500")), help="Max new tokens for generation (default 500)")
    parser.add_argument("--predict-src", type=str, default=None, help="Path to previous prediction folder. Defaults to Data_predict/Qwen2.5-7B-full_predict")
    parser.add_argument("--predict-dst", type=str, default=None, help="Output folder. Defaults to Data_predict/Qwen2.5-7B-full_predict_500")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading if causing issues")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_input = base_dir / "Data_input"
    runs_dir = base_dir / "runs"
    default_src = base_dir / "Data_predict" / f"{MODEL_SHORT}_predict"
    default_dst = base_dir / "Data_predict" / f"{MODEL_SHORT}_predict_500"

    model_dir = Path(args.model_dir) if args.model_dir else detect_latest_model_dir(runs_dir)
    if not model_dir or not model_dir.exists():
        print("❌ Could not locate the saved model directory. Provide --model-dir pointing to runs/.../final_model")
        sys.exit(1)

    predict_src = Path(args.predict_src) if args.predict_src else default_src
    predict_dst = Path(args.predict_dst) if args.predict_dst else default_dst

    predict_dst.mkdir(parents=True, exist_ok=True)

    print(f"🏠 BASE_DIR: {base_dir}")
    print(f"📥 DATA_INPUT: {data_input}")
    print(f"🤖 MODEL_DIR: {model_dir}")
    print(f"📂 SRC PRED: {predict_src}")
    print(f"📤 DST PRED: {predict_dst}")

    # Determine which IDs to regenerate based on previous prediction files
    if not predict_src.exists():
        print(f"❌ Source prediction folder not found: {predict_src}")
        sys.exit(1)

    prev_preds = sorted([p for p in predict_src.glob("*.txt") if re.match(r"^B\d+_pred\.txt$", p.name)])
    if not prev_preds:
        print("❌ No previous predictions found matching 'Bxxx_pred.txt'")
        sys.exit(1)

    ids = [get_id(p.name) for p in prev_preds if get_id(p.name)]
    print(f"📊 Found {len(ids)} IDs to regenerate: {', '.join(ids)}")

    # Load tokenizer and model/pipeline
    tok, gen = build_pipeline(model_dir=model_dir, use_4bit=not args.no_4bit)

    regenerated = 0
    for cid in ids:
        inp_file = resolve_input_file(data_input, cid)
        if not inp_file:
            print(f"⚠️  {cid}: input file not found, skipping")
            continue
        try:
            note = inp_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"⚠️  {cid}: failed reading input: {e}")
            continue

        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": note},
        ]
        prompt = render(tok, messages, add_generation_prompt=True)

        try:
            out = gen(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            generated = out[0]["generated_text"].strip()
            out_path = predict_dst / f"{cid}_pred.txt"
            out_path.write_text(generated, encoding="utf-8")
            print(f"✅ {cid}: saved → {out_path}")
            print(f"   Preview: {generated[:120]}...")
            regenerated += 1
        except Exception as e:
            print(f"❌ {cid}: generation failed: {e}")

    print(f"\n🎯 Done. Regenerated {regenerated}/{len(ids)} predictions to: {predict_dst}")


if __name__ == "__main__":
    main()
