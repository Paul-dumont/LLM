import os
import json
from rouge_score import rouge_scorer, scoring

# Paths
manual_folder = "/nas/longleaf/home/dumontp/LONGLEAF/DATA_TRAINING/Data_output2"
llm_folder = "/nas/longleaf/home/dumontp/LONGLEAF/Qwen1.5B_full/predictions_500"

if not os.path.exists(manual_folder) or not os.path.exists(llm_folder):
    print("Error: Folders not found.")
    exit(1)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
aggregator = scoring.BootstrapAggregator()

llm_files = [f for f in os.listdir(llm_folder) if f.endswith('_pred.txt')]

for llm_file in llm_files:
    patient_id = llm_file.replace('_pred.txt', '')
    manual_file = f"{patient_id}_summary.txt"
    
    manual_path = os.path.join(manual_folder, manual_file)
    llm_path = os.path.join(llm_folder, llm_file)
    
    if os.path.exists(manual_path):
        with open(manual_path, 'r') as f:
            ref = f.read().strip()
        with open(llm_path, 'r') as f:
            pred = f.read().strip()
        
        scores = scorer.score(ref, pred)
        aggregator.add_scores(scores)

result = aggregator.aggregate()

rouge_scores = {
    'rouge1': result['rouge1'].mid.fmeasure,
    'rouge2': result['rouge2'].mid.fmeasure,
    'rougeL': result['rougeL'].mid.fmeasure,
    'rougeLsum': result['rougeLsum'].mid.fmeasure
}

print("ROUGE Scores (mid F1 with confidence intervals):")
for key, value in rouge_scores.items():
    print(f"{key}: {value:.4f}")

# Save to file
os.makedirs("Qwen1.5B_full/metrics", exist_ok=True)
with open("Qwen1.5B_full/metrics/rouge.json", "w") as f:
    json.dump(rouge_scores, f, indent=4)

print("Scores saved to Qwen1.5B_full/metrics/rouge.json")

