import os
import json
from collections import defaultdict

# Define the 56 criteria
CRITERIA = [
    "patient_id",
    "patient_age",
    "headache_intensity",
    "headache_frequency",
    "headache_location",
    "migraine_history",
    "migraine_frequency",
    "average_daily_pain_intensity",
    "diet_score",
    "tmj_pain_rating",
    "disability_rating",
    "jaw_function_score",
    "jaw_clicking",
    "jaw_crepitus",
    "jaw_locking", 
    "maximum_opening", 
    "maximum_opening_without_pain",
    "disc_displacement", 
    "muscle_pain_score", 
    "muscle_pain_location",
    "muscle_spasm_present", 
    "muscle_tenderness_present", 
    "muscle_stiffness_present",
    "muscle_soreness_present", 
    "joint_pain_areas", 
    "joint_arthritis_location",
    "neck_pain_present", 
    "back_pain_present", 
    "earache_present", 
    "tinnitus_present",
    "vertigo_present", 
    "hearing_loss_present", 
    "hearing_sensitivity_present",
    "sleep_apnea_diagnosed",
    "sleep_disorder_type",
    "airway_obstruction_present",
    "anxiety_present",
    "depression_present",
    "stress_present",
    "autoimmune_condition",
    "fibromyalgia_present", 
    "current_medications", 
    "previous_medications",
    "adverse_reactions",
    "appliance_history",
    "current_appliance",
    "cpap_used",
    "apap_used", 
    "bipap_used",
    "physical_therapy_status", 
    "pain_onset_date",
    "pain_duration", 
    "pain_frequency", 
    "onset_triggers", 
    "pain_relieving_factors",
    "pain_aggravating_factors"
]

def parse_summary_file(file_path):
    """Parse a summary file into a dictionary of key-value pairs."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
            elif ':' in line:
                key, value = line.split(':', 1)
            else:
                continue
            key = key.strip()
            value = value.strip()
            if key in CRITERIA:
                data[key] = value
    return data

def calculate_metrics(manual_data, llm_data):
    """Return raw TP, FP, FN counts per field."""
    metrics = {criterion: {'TP': 0, 'FP': 0, 'FN': 0} for criterion in CRITERIA}
    
    for criterion in CRITERIA:
        manual_value = manual_data.get(criterion, None)
        llm_value = llm_data.get(criterion, None)
        
        if manual_value is not None and llm_value is not None:
            if manual_value == llm_value:
                metrics[criterion]['TP'] += 1
            else:
                metrics[criterion]['FP'] += 1
                metrics[criterion]['FN'] += 1
        elif manual_value is not None:
            metrics[criterion]['FN'] += 1
        elif llm_value is not None:
            metrics[criterion]['FP'] += 1
    
    return metrics

def evaluate_folders(manual_folder, llm_folder):
    """Aggregate TP, FP, FN over all patients and compute final metrics."""
    all_results = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    
    llm_files = [f for f in os.listdir(llm_folder) if f.endswith('_pred.txt')]

    for llm_file in llm_files:
        patient_id = llm_file.replace('_pred.txt', '')
        manual_file = f"{patient_id}_summary.txt"
        
        manual_path = os.path.join(manual_folder, manual_file)
        llm_path = os.path.join(llm_folder, llm_file)
        
        if os.path.exists(manual_path):
            manual_data = parse_summary_file(manual_path)
            llm_data = parse_summary_file(llm_path)
            
            patient_metrics = calculate_metrics(manual_data, llm_data)
            
            for criterion, counts in patient_metrics.items():
                all_results[criterion]['TP'] += counts['TP']
                all_results[criterion]['FP'] += counts['FP']
                all_results[criterion]['FN'] += counts['FN']
    
    # Compute metrics per field from accumulated TP, FP, FN
    avg_results = {}
    for criterion in CRITERIA:
        TP = all_results[criterion]['TP']
        FP = all_results[criterion]['FP']
        FN = all_results[criterion]['FN']
        
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        Accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

        avg_results[criterion] = {
            'Precision': round(Precision, 4),
            'Recall': round(Recall, 4),
            'Accuracy': round(Accuracy, 4),
            'F1': round(F1, 4)
        }
    
    return avg_results

if __name__ == "__main__":
    manual_folder = "/nas/longleaf/home/dumontp/LONGLEAF/Data_output2"
    llm_folder = "/nas/longleaf/home/dumontp/LONGLEAF/Data_predict/qwen_full_predict_3072"
    
    if not os.path.exists(manual_folder) or not os.path.exists(llm_folder):
        print("Error: Folders not found.")
    else:
        avg_metrics = evaluate_folders(manual_folder, llm_folder)
        
        with open("extraction_metrics_full.json", "w") as f:
            json.dump(avg_metrics, f, indent=4)
        
        print("Metrics saved to extraction_metrics_full.json")
        
        # Calculate and print F1 statistics
        f1_scores = [v['F1'] for v in avg_metrics.values()]
        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            sorted_f1 = sorted(f1_scores)
            median_f1 = (sorted_f1[len(f1_scores)//2 - 1] + sorted_f1[len(f1_scores)//2]) / 2 if len(f1_scores) % 2 == 0 else sorted_f1[len(f1_scores)//2]
            min_f1 = min(f1_scores)
            max_f1 = max(f1_scores)
            std_f1 = (sum((x - avg_f1)**2 for x in f1_scores) / len(f1_scores))**0.5
            
            print("\nStatistiques F1")
            print(f"Average F1: {avg_f1:.4f} (moyenne globale)")
            print(f"Median F1: {median_f1:.4f} (médiane)")
            print(f"Min F1: {min_f1:.4f} (minimum, ex: certains champs non extraits)")
            print(f"Max F1: {max_f1:.4f} (maximum, ex: patient_id parfait)")
            print(f"Std Dev F1: {std_f1:.4f} (dispersion élevée, certains champs excellents, d'autres nuls)")
            
            # All F1 scores sorted by F1 descending
            all_sorted = sorted(avg_metrics.items(), key=lambda x: x[1]['F1'], reverse=True)
            
            print("\nAll F1 Scores (sorted by F1 descending)")
            for k, v in all_sorted:
                print(f"{k}: {v['F1']:.4f}")
        else:
            print("No F1 scores to analyze.")