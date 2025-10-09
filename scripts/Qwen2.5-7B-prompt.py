#!/usr/bin/env python3

import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

# Configuration
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_DIR = Path(__file__).parent.parent
DATA_INPUT = BASE_DIR / "Data_input"
PREDICT_DIR = BASE_DIR / "Data_predict" / "Qwen2.5-7B-prompt"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

# Prompt détaillé pour l'extraction
INSTRUCTION = """You are a medical data extraction specialist. Extract the following 56 specific clinical indicators from this medical note 
and format as key: value pairs. Only extract indicators that have actual information available in the text.

REQUIRED INDICATORS TO LOOK FOR:
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
- For boolean indicators: use "True" or "False" only if explicitly stated
- For numeric values: extract numbers and units (e.g., "25 years", "7/10", "45mm")
- For text values: be concise but specific
- If multiple values exist, use semicolon separation
- CRITICAL: Do NOT try to include all 56 indicators. Only extract indicators that are EXPLICITLY MENTIONED in the note.
- It is PERFECTLY FINE to have only 10-20 indicators if that's all the information available.
- Do not invent or assume values for indicators not present in the text.
- Do not output any indicator with "unknown" or empty values.
- If the note contains information about different aspects or time periods, separate them with "----------------------------------------------------------------------------------------------------"
- If a section has no relevant information, end it with "NoInfo"
- Extract as many relevant indicators as possible, but ONLY those with concrete information from the note.
- Format: key: value (one per line)"""

print("🔧 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configuration quantization
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)

# Pipeline de génération
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
)

print("📂 Processing input files...")
input_files = list(DATA_INPUT.glob("*.txt"))
input_files = input_files[:5]  # Limit to 5 predictions for testing
print(f"Found {len(input_files)} input files")

for i, input_file in enumerate(input_files):
    cid_match = re.match(r"(B\d+)", input_file.stem)
    if not cid_match:
        continue
    cid = cid_match.group(1)

    print(f"Processing {cid} ({i+1}/{len(input_files)})...")

    # Charger le texte d'entrée
    note_text = input_file.read_text(encoding='utf-8').strip()

    # Créer le prompt
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": note_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Générer la prédiction
    try:
        result = gen(
            prompt,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.0
        )
        prediction = result[0]["generated_text"].strip()

        # Sauvegarder
        pred_file = PREDICT_DIR / f"{cid}_pred.txt"
        pred_file.write_text(prediction, encoding='utf-8')

        print(f"  ✅ Saved to {pred_file}")

    except Exception as e:
        print(f"  ❌ Error for {cid}: {e}")

print("🎯 Inference completed!")
print(f"Predictions saved in: {PREDICT_DIR}")
