"""
model_evaluation.py - Reproducible & Leakage-Resistant Evaluation Engine
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Executes and aggregates rigorous, leakage-resistant evaluations across all models:
- Model A: Force-Regression (Synthetic Mechanics)
- Model B: Tactile Slip Classification (IntelliMan PVDF & Bristol TacTip)
- Model C: Multi-Class Contact & Grasp Phase Classification
- Model D: sEMG Grasp Intention Classification

Maintains strict separation between:
1. Synthetic data evaluation
2. Real-world physical research data evaluation
3. Baseline heuristic/dummy comparisons
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional

METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation", "metrics"))
os.makedirs(METRICS_DIR, exist_ok=True)


class ComprehensiveEvaluator:
    """Orchestrates evaluation across all task-specific models and generates unified benchmarks."""

    def __init__(self):
        self.metrics_dir = METRICS_DIR

    def get_all_evaluation_summaries(self) -> list[dict]:
        """Loads and formats all available metric JSON summaries."""
        summaries = []
        if not os.path.exists(self.metrics_dir):
            return summaries

        for fname in sorted(os.listdir(self.metrics_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(self.metrics_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Distinguish regression vs classification
                    if "r2_score" in data or "r2" in data:
                        summaries.append({
                            "Model ID": fname.replace(".json", ""),
                            "Model Task": "Force Regression (Model A)",
                            "Dataset Source": data.get("dataset_name", "Synthetic GripWise Demonstration"),
                            "Data Nature": "Synthetic Demonstration Data",
                            "Primary Metric": f"R² = {data.get('r2_score', data.get('r2', 0.0)):.4f}",
                            "Secondary Metric": f"MAE = {data.get('mae', 0.0):.3f} N",
                            "Leakage Protection": "Held-out test set (20% split, scale-invariant features)",
                            "File": fname
                        })
                    elif "gestures" in data:
                        summaries.append({
                            "Model ID": fname.replace(".json", ""),
                            "Model Task": "sEMG Grasp Intent (Model D)",
                            "Dataset Source": data.get("dataset_name", "NinaPro sEMG Database"),
                            "Data Nature": "Real-World Physical Research Data",
                            "Primary Metric": f"Accuracy = {data.get('accuracy', 0.0)*100:.1f}%",
                            "Secondary Metric": f"Macro F1 = {data.get('macro_f1', 0.0):.3f}",
                            "Leakage Protection": "Subject-independent held-out subject cross-validation",
                            "File": fname
                        })
                    elif "class_names" in data:
                        summaries.append({
                            "Model ID": fname.replace(".json", ""),
                            "Model Task": "Contact & Grasp Phase (Model C)",
                            "Dataset Source": data.get("dataset_name", "IntelliMan ICECS 2025"),
                            "Data Nature": "Real-World Physical Research Data",
                            "Primary Metric": f"Accuracy = {data.get('accuracy', 0.0)*100:.1f}%",
                            "Secondary Metric": f"Macro F1 = {data.get('macro_f1', 0.0):.3f}",
                            "Leakage Protection": "Recording-level partition (unseen physical trial files)",
                            "File": fname
                        })
                    else:
                        summaries.append({
                            "Model ID": fname.replace(".json", ""),
                            "Model Task": "Tactile Slip Classifier (Model B)",
                            "Dataset Source": data.get("dataset_name", "Public Tactile Research Data"),
                            "Data Nature": "Real-World Physical Research Data",
                            "Primary Metric": f"F1-Score = {data.get('f1_score', 0.0):.3f}",
                            "Secondary Metric": f"ROC-AUC = {data.get('roc_auc', 0.0):.3f} (Acc: {data.get('accuracy', 0.0)*100:.1f}%)",
                            "Leakage Protection": data.get("leakage_protection", "Trial-level independent split"),
                            "File": fname
                        })
                except Exception:
                    continue

        return summaries

    def evaluate_model_a_force(self) -> dict:
        """Evaluates Model A (Force Regression) and writes evaluation/metrics/model_a_force_metrics.json."""
        from prediction import ForcePredictor, MODEL_FILE
        predictor = ForcePredictor()
        
        metrics = {
            "dataset_name": "GripWise Synthetic Force-Regression Benchmark",
            "model_name": "RandomForestRegressor (Min-Safe-Force Estimator)",
            "data_nature": "Synthetic Demonstration Data",
            "r2_score": predictor.r2,
            "mae": predictor.mae,
            "training_samples": 1200,
            "test_samples": 300,
            "features": predictor.FEATURE_NAMES,
            "leakage_protection": "Strict train/test split on 1,500 independent samples. No test data contamination.",
            "disclaimer": "Trained exclusively on synthetic mechanics heuristics. Not certified for clinical prosthesis."
        }
        out_path = os.path.join(self.metrics_dir, "model_a_force_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def run_full_evaluation_pipeline(self) -> dict:
        """Executes training and evaluation across all task models."""
        results = {}

        # 1. Model A: Force
        results["model_a"] = self.evaluate_model_a_force()

        # 2. Model B1: IntelliMan Slip
        from slip_model import IntelliManPVDFSlipClassifier
        clf_i = IntelliManPVDFSlipClassifier()
        if not clf_i.metrics:
            clf_i.train_and_evaluate()
        results["model_b_intelliman"] = clf_i.metrics

        # 3. Model B2: Bristol TacTip Slip
        from slip_model import BristolTacTipSlipClassifier
        clf_b = BristolTacTipSlipClassifier()
        if not clf_b.metrics:
            clf_b.train_and_evaluate()
        results["model_b_bristol"] = clf_b.metrics

        # 4. Model C: Contact State
        from grasp_model import ContactGraspStateClassifier
        clf_c = ContactGraspStateClassifier()
        if not clf_c.metrics:
            clf_c.train_and_evaluate()
        results["model_c_contact"] = clf_c.metrics

        # 5. Model D: sEMG Intent
        from emg_model import EMGIntentionClassifier
        clf_d = EMGIntentionClassifier()
        if not clf_d.metrics:
            clf_d.train_and_evaluate()
        results["model_d_emg"] = clf_d.metrics

        return results

