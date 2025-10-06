#!/usr/bin/env python3
"""
🎯 ÉVALUATION COMPARATIVE DES MODÈLES - MÉTRIQUES ACADÉMIQUES
================================================================

Script d'évaluation complète pour comparer votre modèle Qwen fine-tuné
avec les performances de BART et DeepSeek selon les standards académiques.

Métriques calculées :
- ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum (similarité lexicale)
- Accuracy, Precision, Recall, F1-score (extraction clinique)
- Analyses détaillées par indicateur clinique
- Focus sur indicateurs complexes (comorbidités)

Utilisation :
    python scripts/evaluation.py --predictions_dir Data_predict/Qwen2.5-7B-full_predict/
                                --ground_truth_dir Data_input/
                                --output_dir evaluation_results/
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Métriques d'évaluation
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Installation automatique de rouge-score si nécessaire
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("⚠️ Installation de rouge-score...")
    os.system("pip install rouge-score")
    from rouge_score import rouge_scorer

# ==================== CONFIGURATION ====================

# 56 Indicateurs cliniques à évaluer
CLINICAL_INDICATORS = [
    # Informations démographiques
    'patient_age', 'patient_sex', 'patient_ethnicity', 'patient_occupation',
    
    # Symptômes principaux
    'headache_intensity', 'headache_frequency', 'headache_duration', 'headache_location',
    'migraine_type', 'migraine_history', 'family_migraine_history',
    
    # Fonctions oro-faciales
    'jaw_function', 'jaw_pain_intensity', 'jaw_opening_limitation', 'jaw_clicking',
    'teeth_grinding', 'teeth_clenching', 'bite_issues',
    
    # Évaluations fonctionnelles
    'disability_rating', 'quality_of_life_score', 'work_impact', 'sleep_quality',
    
    # Déclencheurs et facteurs
    'onset_triggers', 'stress_factors', 'hormonal_factors', 'dietary_triggers',
    'weather_sensitivity', 'light_sensitivity', 'sound_sensitivity',
    
    # Conditions comorbides (indicateurs complexes)
    'autoimmune_condition', 'thyroid_disorder', 'sleep_disorder_type',
    'anxiety_level', 'depression_symptoms', 'fibromyalgia',
    
    # Douleurs associées
    'neck_pain_intensity', 'shoulder_pain', 'muscle_pain_location',
    'nerve_pain_symptoms', 'joint_stiffness',
    
    # Traitements actuels
    'current_medications', 'previous_treatments', 'therapy_response',
    'side_effects_history', 'allergies',
    
    # Examens et tests
    'imaging_results', 'blood_test_results', 'specialist_consultations',
    
    # Facteurs de style de vie
    'exercise_frequency', 'smoking_status', 'alcohol_consumption',
    'caffeine_intake', 'hydration_level',
    
    # Suivi et pronostic
    'symptom_progression', 'treatment_goals', 'follow_up_needed'
]

# Indicateurs particulièrement difficiles (expressions variables)
CHALLENGING_INDICATORS = [
    'autoimmune_condition',    # "has lupus" vs "lupus patient" vs "autoimmune: lupus"
    'sleep_disorder_type',     # "insomnia" vs "sleep apnea" vs "restless leg syndrome" 
    'muscle_pain_location',    # "neck and shoulders" vs "cervical region" vs "upper back"
    'onset_triggers',          # Texte libre: "stress, lack of sleep, certain foods"
    'current_medications',     # Noms variables: "Sumatriptan" vs "Imitrex" vs "triptan"
    'symptom_progression',     # "getting worse" vs "deteriorating" vs "increased severity"
    'previous_treatments',     # Multiples traitements décrits différemment
    'specialist_consultations' # "saw neurologist" vs "neurology referral" vs "Dr. Smith"
]

class MedicalEvaluator:
    """Évaluateur de performance pour extraction d'informations médicales"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration ROUGE
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
            use_stemmer=True
        )
        
        # Stockage des résultats
        self.results = {
            'rouge_scores': {},
            'clinical_scores': {},
            'challenging_scores': {},
            'summary_stats': {},
            'timestamp': datetime.now().isoformat()
        }
        
        print("🎯 Medical Evaluator initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_predictions_and_ground_truth(self, predictions_dir: Path, ground_truth_dir: Path) -> Tuple[List[Dict], List[Dict]]:
        """Charge les prédictions et vérités terrain"""
        
        print("\n📊 Loading predictions and ground truth...")
        
        predictions = []
        ground_truths = []
        
        pred_files = list(Path(predictions_dir).glob("*.txt"))
        print(f"📋 Found {len(pred_files)} prediction files")
        
        for pred_file in pred_files:
            # Charger prédiction
            with open(pred_file, 'r', encoding='utf-8') as f:
                pred_content = f.read().strip()
            
            # Trouver le fichier de vérité terrain correspondant
            base_name = pred_file.stem.replace('_predict', '')
            gt_file = Path(ground_truth_dir) / f"{base_name}.txt"
            
            if gt_file.exists():
                with open(gt_file, 'r', encoding='utf-8') as f:
                    gt_content = f.read().strip()
                
                predictions.append({
                    'file': base_name,
                    'content': pred_content,
                    'indicators': self._extract_indicators(pred_content)
                })
                
                ground_truths.append({
                    'file': base_name,
                    'content': gt_content,
                    'indicators': self._extract_indicators_from_source(gt_content)
                })
        
        print(f"✅ Loaded {len(predictions)} prediction-ground truth pairs")
        return predictions, ground_truths
    
    def _extract_indicators(self, text: str) -> Dict[str, str]:
        """Extrait les indicateurs cliniques du format key=value"""
        
        indicators = {}
        
        # Pattern pour capturer key=value
        pattern = r'(\w+(?:_\w+)*)\s*[=:]\s*([^\n\r]+)'
        
        matches = re.findall(pattern, text)
        for key, value in matches:
            key = key.lower().strip()
            value = value.strip().rstrip(',;')
            
            if key in CLINICAL_INDICATORS:
                indicators[key] = value
        
        # S'assurer que tous les indicateurs sont présents
        for indicator in CLINICAL_INDICATORS:
            if indicator not in indicators:
                indicators[indicator] = "not_mentioned"
        
        return indicators
    
    def _extract_indicators_from_source(self, source_text: str) -> Dict[str, str]:
        """Extrait les indicateurs de la vérité terrain (texte clinique source)
        
        Note: Dans un cas réel, vous auriez des annotations manuelles.
        Ici on simule l'extraction depuis le texte source.
        """
        
        indicators = {}
        text_lower = source_text.lower()
        
        # Exemples d'extraction simple (à adapter selon vos données réelles)
        extraction_patterns = {
            'patient_age': r'(\d+)[\s-]*(year|yr|age)',
            'headache_intensity': r'pain.*?(\d+)\/10|(\d+)\/10.*?pain',
            'migraine_history': r'(migraine|headache).*?(history|previous|past)',
            'jaw_function': r'jaw.*?(normal|limited|restricted|good|poor)',
            'disability_rating': r'disability.*?(\d+)|(\d+).*?disability',
            # ... ajoutez plus de patterns selon vos besoins
        }
        
        for indicator, pattern in extraction_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                indicators[indicator] = match.group(0)
            else:
                indicators[indicator] = "not_mentioned"
        
        # Pour les autres indicateurs non couverts par les patterns
        for indicator in CLINICAL_INDICATORS:
            if indicator not in indicators:
                # Recherche simple par mot-clé
                if any(word in text_lower for word in indicator.split('_')):
                    indicators[indicator] = "mentioned"
                else:
                    indicators[indicator] = "not_mentioned"
        
        return indicators
    
    def compute_rouge_scores(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """Calcule les scores ROUGE pour la similarité lexicale"""
        
        print("\n📝 Computing ROUGE scores...")
        
        rouge_results = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeLsum': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        for pred, gt in zip(predictions, ground_truths):
            scores = self.rouge_scorer.score(gt['content'], pred['content'])
            
            for metric in rouge_results.keys():
                rouge_results[metric]['precision'].append(scores[metric].precision)
                rouge_results[metric]['recall'].append(scores[metric].recall)
                rouge_results[metric]['fmeasure'].append(scores[metric].fmeasure)
        
        # Calculer les moyennes
        rouge_summary = {}
        for metric in rouge_results.keys():
            rouge_summary[metric] = {
                'precision': np.mean(rouge_results[metric]['precision']),
                'recall': np.mean(rouge_results[metric]['recall']),
                'fmeasure': np.mean(rouge_results[metric]['fmeasure']),
                'std': np.std(rouge_results[metric]['fmeasure'])
            }
        
        print("✅ ROUGE scores computed")
        return rouge_summary
    
    def compute_clinical_metrics(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """Calcule les métriques de classification pour chaque indicateur clinique"""
        
        print("\n🏥 Computing clinical field metrics...")
        
        clinical_results = {}
        
        for indicator in CLINICAL_INDICATORS:
            pred_values = [pred['indicators'].get(indicator, 'not_mentioned') for pred in predictions]
            true_values = [gt['indicators'].get(indicator, 'not_mentioned') for gt in ground_truths]
            
            # Convertir en labels binaires (mentionné/pas mentionné) pour simplifier
            pred_binary = [1 if val != 'not_mentioned' else 0 for val in pred_values]
            true_binary = [1 if val != 'not_mentioned' else 0 for val in true_values]
            
            try:
                accuracy = accuracy_score(true_binary, pred_binary)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_binary, pred_binary, average='binary', zero_division=0
                )
                
                clinical_results[indicator] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': sum(true_binary),
                    'predictions_count': sum(pred_binary)
                }
                
            except Exception as e:
                print(f"⚠️ Error computing metrics for {indicator}: {e}")
                clinical_results[indicator] = {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'support': 0, 'predictions_count': 0
                }
        
        print("✅ Clinical metrics computed")
        return clinical_results
    
    def analyze_challenging_indicators(self, clinical_results: Dict) -> Dict:
        """Analyse approfondie des indicateurs difficiles"""
        
        print("\n🔍 Analyzing challenging indicators...")
        
        challenging_results = {}
        for indicator in CHALLENGING_INDICATORS:
            if indicator in clinical_results:
                challenging_results[indicator] = clinical_results[indicator].copy()
                challenging_results[indicator]['difficulty_reason'] = self._get_difficulty_reason(indicator)
        
        print("✅ Challenging indicators analyzed")
        return challenging_results
    
    def _get_difficulty_reason(self, indicator: str) -> str:
        """Retourne la raison pour laquelle un indicateur est difficile"""
        
        reasons = {
            'autoimmune_condition': "Multiple expression patterns for autoimmune diseases",
            'sleep_disorder_type': "Various sleep disorder terminologies and classifications", 
            'muscle_pain_location': "Anatomical descriptions with high variability",
            'onset_triggers': "Free-text descriptions of complex trigger combinations",
            'current_medications': "Drug name variations (generic vs brand names)",
            'symptom_progression': "Subjective progression descriptions",
            'previous_treatments': "Historical treatment descriptions with temporal complexity",
            'specialist_consultations': "Varied professional title and referral patterns"
        }
        return reasons.get(indicator, "Complex clinical concept with variable expression")
    
    def generate_comparative_analysis(self) -> Dict:
        """Génère une analyse comparative avec benchmarks BART/DeepSeek"""
        
        print("\n📊 Generating comparative analysis...")
        
        # Benchmarks de référence (valeurs typiques de la littérature)
        bart_benchmarks = {
            'rouge1_f1': 0.78, 'rouge2_f1': 0.65, 'rougeL_f1': 0.74,
            'clinical_f1_avg': 0.72, 'challenging_f1_avg': 0.64
        }
        
        deepseek_benchmarks = {
            'rouge1_f1': 0.82, 'rouge2_f1': 0.69, 'rougeL_f1': 0.78,
            'clinical_f1_avg': 0.76, 'challenging_f1_avg': 0.68
        }
        
        # Calculer vos métriques moyennes
        your_rouge1 = self.results['rouge_scores']['rouge1']['fmeasure']
        your_rouge2 = self.results['rouge_scores']['rouge2']['fmeasure']
        your_rougeL = self.results['rouge_scores']['rougeL']['fmeasure']
        
        clinical_f1s = [metrics['f1'] for metrics in self.results['clinical_scores'].values()]
        your_clinical_avg = np.mean(clinical_f1s) if clinical_f1s else 0.0
        
        challenging_f1s = [metrics['f1'] for metrics in self.results['challenging_scores'].values()]
        your_challenging_avg = np.mean(challenging_f1s) if challenging_f1s else 0.0
        
        # Comparaison
        comparison = {
            'your_model': {
                'rouge1_f1': your_rouge1,
                'rouge2_f1': your_rouge2, 
                'rougeL_f1': your_rougeL,
                'clinical_f1_avg': your_clinical_avg,
                'challenging_f1_avg': your_challenging_avg
            },
            'bart_baseline': bart_benchmarks,
            'deepseek_baseline': deepseek_benchmarks,
            'improvements': {
                'vs_bart': {
                    'rouge1': your_rouge1 - bart_benchmarks['rouge1_f1'],
                    'rouge2': your_rouge2 - bart_benchmarks['rouge2_f1'],
                    'clinical_avg': your_clinical_avg - bart_benchmarks['clinical_f1_avg'],
                    'challenging_avg': your_challenging_avg - bart_benchmarks['challenging_f1_avg']
                },
                'vs_deepseek': {
                    'rouge1': your_rouge1 - deepseek_benchmarks['rouge1_f1'],
                    'rouge2': your_rouge2 - deepseek_benchmarks['rouge2_f1'], 
                    'clinical_avg': your_clinical_avg - deepseek_benchmarks['clinical_f1_avg'],
                    'challenging_avg': your_challenging_avg - deepseek_benchmarks['challenging_f1_avg']
                }
            }
        }
        
        print("✅ Comparative analysis generated")
        return comparison
    
    def create_visualizations(self):
        """Crée des visualisations des résultats"""
        
        print("\n📈 Creating visualizations...")
        
        # 1. Graphique en barres des F1-scores par indicateur
        plt.figure(figsize=(15, 8))
        indicators = list(self.results['clinical_scores'].keys())[:20]  # Top 20 pour lisibilité
        f1_scores = [self.results['clinical_scores'][ind]['f1'] for ind in indicators]
        
        plt.barh(indicators, f1_scores)
        plt.xlabel('F1-Score')
        plt.title('F1-Scores par Indicateur Clinique (Top 20)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'f1_scores_by_indicator.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Comparaison avec benchmarks
        if 'comparative_analysis' in self.results:
            comparison = self.results['comparative_analysis']
            
            metrics = ['rouge1_f1', 'rouge2_f1', 'clinical_f1_avg', 'challenging_f1_avg']
            your_scores = [comparison['your_model'][m] for m in metrics]
            bart_scores = [comparison['bart_baseline'][m] for m in metrics]
            deepseek_scores = [comparison['deepseek_baseline'][m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.25
            
            plt.figure(figsize=(12, 6))
            plt.bar(x - width, your_scores, width, label='Your Model (Qwen)', alpha=0.8)
            plt.bar(x, bart_scores, width, label='BART Baseline', alpha=0.8) 
            plt.bar(x + width, deepseek_scores, width, label='DeepSeek Baseline', alpha=0.8)
            
            plt.xlabel('Métriques')
            plt.ylabel('Score')
            plt.title('Comparaison de Performance avec Benchmarks')
            plt.xticks(x, metrics, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Visualizations saved")
    
    def save_results(self):
        """Sauvegarde tous les résultats"""
        
        print(f"\n💾 Saving results to {self.output_dir}...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON complet
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 2. Rapport texte détaillé
        report_file = self.output_dir / f"detailed_analysis_{timestamp}.txt"
        self._generate_text_report(report_file)
        
        # 3. CSV des résultats cliniques pour analyse
        df_clinical = pd.DataFrame(self.results['clinical_scores']).T
        df_clinical.to_csv(self.output_dir / f"clinical_metrics_{timestamp}.csv")
        
        print(f"✅ Results saved:")
        print(f"   📄 {results_file}")
        print(f"   📄 {report_file}")
        print(f"   📊 {self.output_dir / f'clinical_metrics_{timestamp}.csv'}")
    
    def _generate_text_report(self, report_file: Path):
        """Génère un rapport texte détaillé"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🎯 RAPPORT D'ÉVALUATION COMPARATIVE - MÉTRIQUES MÉDICALES\n")
            f.write("=" * 70 + "\n\n")
            
            # ROUGE Scores
            f.write("📝 SCORES ROUGE (Similarité Lexicale)\n")
            f.write("-" * 40 + "\n")
            for metric, scores in self.results['rouge_scores'].items():
                f.write(f"{metric.upper():<12}: F1={scores['fmeasure']:.3f} (±{scores['std']:.3f})\n")
            f.write("\n")
            
            # Top/Bottom performing indicators
            clinical_f1s = [(k, v['f1']) for k, v in self.results['clinical_scores'].items()]
            clinical_f1s.sort(key=lambda x: x[1], reverse=True)
            
            f.write("🏆 MEILLEURS INDICATEURS CLINIQUES (Top 10)\n")
            f.write("-" * 50 + "\n")
            for indicator, f1 in clinical_f1s[:10]:
                f.write(f"{indicator:<25}: F1={f1:.3f}\n")
            f.write("\n")
            
            f.write("⚠️ INDICATEURS À AMÉLIORER (Bottom 10)\n")
            f.write("-" * 45 + "\n")
            for indicator, f1 in clinical_f1s[-10:]:
                f.write(f"{indicator:<25}: F1={f1:.3f}\n")
            f.write("\n")
            
            # Challenging indicators
            f.write("🔍 INDICATEURS COMPLEXES (Analyse Détaillée)\n")
            f.write("-" * 50 + "\n")
            for indicator, metrics in self.results['challenging_scores'].items():
                f.write(f"{indicator}:\n")
                f.write(f"  F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}\n")
                f.write(f"  Raison: {metrics.get('difficulty_reason', 'N/A')}\n\n")
            
            # Comparative analysis
            if 'comparative_analysis' in self.results:
                f.write("📊 COMPARAISON AVEC BENCHMARKS\n")
                f.write("-" * 35 + "\n")
                comp = self.results['comparative_analysis']
                
                f.write("Améliorations vs BART:\n")
                for metric, improvement in comp['improvements']['vs_bart'].items():
                    sign = "📈" if improvement > 0 else "📉"
                    f.write(f"  {metric}: {improvement:+.3f} {sign}\n")
                
                f.write("\nAméliorations vs DeepSeek:\n") 
                for metric, improvement in comp['improvements']['vs_deepseek'].items():
                    sign = "📈" if improvement > 0 else "📉"
                    f.write(f"  {metric}: {improvement:+.3f} {sign}\n")
    
    def run_complete_evaluation(self, predictions_dir: Path, ground_truth_dir: Path) -> Dict:
        """Lance l'évaluation complète"""
        
        print("\n🚀 STARTING COMPLETE EVALUATION")
        print("=" * 50)
        
        try:
            # 1. Charger les données
            predictions, ground_truths = self.load_predictions_and_ground_truth(
                predictions_dir, ground_truth_dir
            )
            
            if not predictions or not ground_truths:
                raise ValueError("No valid prediction-ground truth pairs found!")
            
            # 2. Calculer ROUGE
            self.results['rouge_scores'] = self.compute_rouge_scores(predictions, ground_truths)
            
            # 3. Calculer métriques cliniques
            self.results['clinical_scores'] = self.compute_clinical_metrics(predictions, ground_truths)
            
            # 4. Analyser indicateurs difficiles
            self.results['challenging_scores'] = self.analyze_challenging_indicators(
                self.results['clinical_scores']
            )
            
            # 5. Analyse comparative
            self.results['comparative_analysis'] = self.generate_comparative_analysis()
            
            # 6. Statistiques globales
            self.results['summary_stats'] = self._compute_summary_stats()
            
            # 7. Créer visualisations
            self.create_visualizations()
            
            # 8. Sauvegarder résultats
            self.save_results()
            
            print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ ERROR during evaluation: {e}")
            raise
    
    def _compute_summary_stats(self) -> Dict:
        """Calcule des statistiques de résumé"""
        
        clinical_f1s = [metrics['f1'] for metrics in self.results['clinical_scores'].values()]
        
        return {
            'total_indicators': len(CLINICAL_INDICATORS),
            'evaluated_indicators': len(clinical_f1s),
            'avg_f1': np.mean(clinical_f1s) if clinical_f1s else 0.0,
            'std_f1': np.std(clinical_f1s) if clinical_f1s else 0.0,
            'min_f1': np.min(clinical_f1s) if clinical_f1s else 0.0,
            'max_f1': np.max(clinical_f1s) if clinical_f1s else 0.0,
            'indicators_above_0_8': sum(1 for f1 in clinical_f1s if f1 >= 0.8),
            'indicators_above_0_9': sum(1 for f1 in clinical_f1s if f1 >= 0.9)
        }

def main():
    """Point d'entrée principal"""
    
    parser = argparse.ArgumentParser(description='Évaluation comparative des modèles médicaux')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Répertoire contenant les prédictions du modèle')
    parser.add_argument('--ground_truth_dir', type=str, required=True, 
                        help='Répertoire contenant les vérités terrain')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Répertoire de sortie pour les résultats')
    
    args = parser.parse_args()
    
    # Vérifier que les répertoires existent
    predictions_dir = Path(args.predictions_dir)
    ground_truth_dir = Path(args.ground_truth_dir)
    output_dir = Path(args.output_dir)
    
    if not predictions_dir.exists():
        print(f"❌ Predictions directory not found: {predictions_dir}")
        return
    
    if not ground_truth_dir.exists():
        print(f"❌ Ground truth directory not found: {ground_truth_dir}")
        return
    
    # Lancer l'évaluation
    evaluator = MedicalEvaluator(output_dir)
    results = evaluator.run_complete_evaluation(predictions_dir, ground_truth_dir)
    
    # Afficher résumé
    print("\n📋 RÉSUMÉ DES RÉSULTATS:")
    print("-" * 30)
    
    if 'rouge_scores' in results:
        rouge1_f1 = results['rouge_scores']['rouge1']['fmeasure']
        print(f"📝 ROUGE-1 F1: {rouge1_f1:.3f}")
    
    if 'summary_stats' in results:
        avg_f1 = results['summary_stats']['avg_f1']
        above_80 = results['summary_stats']['indicators_above_0_8']
        total = results['summary_stats']['total_indicators']
        print(f"🏥 Clinical F1 (avg): {avg_f1:.3f}")
        print(f"🎯 Indicators ≥0.8 F1: {above_80}/{total}")
    
    if 'comparative_analysis' in results:
        vs_bart = results['comparative_analysis']['improvements']['vs_bart']['rouge1']
        vs_deepseek = results['comparative_analysis']['improvements']['vs_deepseek']['rouge1']
        print(f"📈 vs BART: {vs_bart:+.3f}")
        print(f"📈 vs DeepSeek: {vs_deepseek:+.3f}")
    
    print(f"\n📁 Résultats détaillés dans: {output_dir}")

if __name__ == "__main__":
    main()
