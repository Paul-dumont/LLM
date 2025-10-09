import os
import re
from pathlib import Path
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score


# Dossiers de prédiction et de vérité terrain structurée
PRED_DIR = "Data_predict/QWEN/"
GT_DIR = "Data_output/"

# Indexer les fichiers de vérité terrain structurée
gt_files = {re.match(r"(B\d+)", f.stem).group(1): f for f in Path(GT_DIR).glob("*.txt") if re.match(r"(B\d+)", f.stem)}

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge1_scores, rougeL_scores = [], []
f1_scores = []

def extract_indicators(text):
    return {k: v for k, v in re.findall(r"(\w+(?:_\w+)*)\s*[=:]\s*([^\n\r]+)", text)}

for pred_file in Path(PRED_DIR).glob("*.txt"):
    cid_match = re.match(r"(B\d+)", pred_file.stem)
    if not cid_match: continue
    cid = cid_match.group(1)
    gt_file = gt_files.get(cid)
    if not gt_file: continue

    pred_text = pred_file.read_text(encoding="utf-8").strip()
    gt_text = gt_file.read_text(encoding="utf-8").strip()

    print(f"\n=== {cid} ===")
    print("Prediction:\n", pred_text)
    print("Ground Truth:\n", gt_text)

    # ROUGE
    scores = scorer.score(gt_text, pred_text)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

    # F1 (sur indicateurs extraits)
    pred_ind = extract_indicators(pred_text)
    gt_ind = extract_indicators(gt_text)
    common_keys = set(pred_ind.keys()) & set(gt_ind.keys())
    y_true = [gt_ind[k] for k in common_keys]
    y_pred = [pred_ind[k] for k in common_keys]
    if y_true and y_pred:
        f1_scores.append(f1_score(y_true, y_pred, average='macro'))

print("\n=== GLOBAL SCORES ===")
print("ROUGE-1 (moyenne):", sum(rouge1_scores)/len(rouge1_scores) if rouge1_scores else 0)
print("ROUGE-L (moyenne):", sum(rougeL_scores)/len(rougeL_scores) if rougeL_scores else 0)
print("F1 (moyenne):", sum(f1_scores)/len(f1_scores) if f1_scores else 0)