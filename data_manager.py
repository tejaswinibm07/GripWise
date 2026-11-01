"""
data_manager.py - High-Level Data & ML Pipeline Coordinator
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Coordinates dataset registration, automated downloading, preprocessing,
model training, and evaluation caching for the Streamlit Data & AI Lab.
"""

import os
import pandas as pd
from typing import Optional, Callable

from dataset_registry import DatasetRegistry, DatasetMetadata
from dataset_download import download_dataset
from dataset_preprocessing import IntelliManSlipAdapter, BristolSlipAdapter, NinaProEMGAdapter
from slip_model import IntelliManPVDFSlipClassifier, BristolTacTipSlipClassifier
from grasp_model import ContactGraspStateClassifier
from emg_model import EMGIntentionClassifier
from model_evaluation import ComprehensiveEvaluator


class DataManager:
    """Singleton-style coordinator for datasets and ML models in GripWise."""

    def __init__(self):
        self.registry = DatasetRegistry()
        self.evaluator = ComprehensiveEvaluator()

    def get_dataset_metadata(self, key: str) -> Optional[DatasetMetadata]:
        return self.registry.get_dataset(key)

    def get_all_datasets(self) -> list[DatasetMetadata]:
        return self.registry.list_datasets()

    def download(self, key: str, progress_callback=None, force=False) -> tuple[bool, str]:
        return download_dataset(key, registry=self.registry, progress_callback=progress_callback, force_redownload=force)

    def preprocess(self, key: str, progress_callback=None) -> tuple[bool, str]:
        """Runs the specific preprocessing adapter for the dataset."""
        try:
            if key == "intelliman_slip":
                adapter = IntelliManSlipAdapter(self.registry)
                df_tr, df_te = adapter.process(progress_callback=progress_callback)
                return True, f"Preprocessed IntelliMan: {len(df_tr)} train windows, {len(df_te)} test windows."

            elif key == "bristol_slip":
                adapter = BristolSlipAdapter(self.registry)
                df_tr, df_te = adapter.process(progress_callback=progress_callback)
                return True, f"Preprocessed Bristol TacTip: {len(df_tr)} train windows, {len(df_te)} test windows."

            elif key == "ninapro_emg":
                adapter = NinaProEMGAdapter(self.registry)
                df_tr, df_te = adapter.process()
                return True, f"Preprocessed NinaPro sEMG: {len(df_tr)} train trials, {len(df_te)} test trials."

            elif key == "gripwise_force_synthetic":
                from prediction import ForcePredictor
                ForcePredictor(force_retrain=True)
                return True, "Preprocessed and trained GripWise Synthetic Force Regression model."

            else:
                return False, f"Preprocessing adapter for '{key}' is not implemented."
        except Exception as e:
            return False, f"Preprocessing failed for '{key}': {e}"

    def train_model(self, model_key: str) -> tuple[bool, str]:
        """Trains or retrains a task-specific model."""
        try:
            if model_key == "intelliman_slip":
                clf = IntelliManPVDFSlipClassifier()
                metrics = clf.train_and_evaluate()
                return True, f"Trained IntelliMan Slip Classifier: F1 = {metrics.f1_score:.3f}, ROC-AUC = {metrics.roc_auc:.3f}."

            elif model_key == "bristol_slip":
                clf = BristolTacTipSlipClassifier()
                metrics = clf.train_and_evaluate()
                return True, f"Trained Bristol TacTip Classifier: F1 = {metrics.f1_score:.3f}, ROC-AUC = {metrics.roc_auc:.3f}."

            elif model_key == "contact_state":
                clf = ContactGraspStateClassifier()
                metrics = clf.train_and_evaluate()
                return True, f"Trained Contact & Grasp-State Classifier: Macro F1 = {metrics.macro_f1:.3f}."

            elif model_key == "ninapro_emg":
                clf = EMGIntentionClassifier()
                metrics = clf.train_and_evaluate()
                return True, f"Trained sEMG Grasp Intention Classifier: Accuracy = {metrics.accuracy*100:.1f}%."

            elif model_key == "gripwise_force":
                from prediction import ForcePredictor
                p = ForcePredictor(force_retrain=True)
                return True, f"Retrained Synthetic Force Regressor: R² = {p.r2:.4f}, MAE = {p.mae:.3f}N."

            else:
                return False, f"Unknown model key '{model_key}'."
        except Exception as e:
            return False, f"Model training error for '{model_key}': {e}"

    def get_evaluation_summaries(self) -> list[dict]:
        return self.evaluator.get_all_evaluation_summaries()

    def load_sample_dataset(self, key: str, max_rows: int = 100) -> Optional[pd.DataFrame]:
        """Loads a preview DataFrame for Dataset Explorer."""
        meta = self.registry.get_dataset(key)
        if not meta or not meta.is_processed():
            # Check if synthetic CSV exists
            if key == "gripwise_force_synthetic":
                from prediction import DATASET_CSV
                if os.path.exists(DATASET_CSV):
                    return pd.read_csv(DATASET_CSV).head(max_rows)
            return None

        # Look in processed directory for train or test CSV
        pdir = meta.local_processed_dir
        csv_files = [f for f in os.listdir(pdir) if f.endswith(".csv")]
        if csv_files:
            target_csv = os.path.join(pdir, csv_files[0])
            try:
                return pd.read_csv(target_csv).head(max_rows)
            except Exception:
                return None
        return None

