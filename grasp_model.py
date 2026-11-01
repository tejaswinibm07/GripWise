"""
grasp_model.py - Contact and Grasp-State Multi-Class Classification Model
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Implements Model C: Contact and Grasp-State Classifier.
Classifies tactile signals into three physical grasp phases:
- Class 0: Rest / Non-Contact (Sensor detached)
- Class 1: Static Stable Contact (Active holding)
- Class 2: Dynamic Slippage (Tangential shear slip)

Trained on ground-truth phase annotations from IntelliMan ICECS 2025 dataset.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "saved"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation", "metrics"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


@dataclass
class MulticlassMetrics:
    dataset_name: str
    model_name: str
    class_names: list[str]
    train_samples: int
    test_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    class_f1_scores: dict[str, float]
    confusion_matrix: list[list[int]]
    feature_names: list[str]
    feature_importances: dict[str, float]
    disclaimer: str


class ContactGraspStateClassifier:
    """
    Model C: Three-State Tactile Phase Classifier:
    0 = Rest / Separation
    1 = Static Contact (Hold)
    2 = Dynamic Slippage
    """

    MODEL_PATH = os.path.join(MODEL_DIR, "intelliman_contact_state_model.joblib")
    METRICS_PATH = os.path.join(METRICS_DIR, "intelliman_contact_state_metrics.json")
    CLASS_NAMES = ["0: Rest / Detached", "1: Static Contact", "2: Dynamic Slippage"]

    FEATURE_COLS = [
        "global_vibration_mean_std", "global_vibration_max_std",
        "global_vibration_energy", "global_signal_ptp"
    ] + [f"ch{ch}_std" for ch in range(1, 9)] + [f"ch{ch}_mean" for ch in range(1, 9)]

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.metrics: MulticlassMetrics | None = None
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.METRICS_PATH):
            try:
                bundle = joblib.load(self.MODEL_PATH)
                self.model = bundle["model"]
                self.scaler = bundle["scaler"]
                with open(self.METRICS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.metrics = MulticlassMetrics(**data)
                return True
            except Exception:
                pass
        return False

    def train_and_evaluate(
        self,
        train_csv: str = "data/processed/intelliman_slip/intelliman_slip_train.csv",
        test_csv: str = "data/processed/intelliman_slip/intelliman_slip_test.csv",
        random_seed: int = 42
    ) -> MulticlassMetrics:
        train_path = os.path.abspath(train_csv)
        test_path = os.path.abspath(test_csv)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            from dataset_preprocessing import IntelliManSlipAdapter
            adapter = IntelliManSlipAdapter()
            df_train, df_test = adapter.process()
        else:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

        avail_cols = [c for c in self.FEATURE_COLS if c in df_train.columns]
        X_train = df_train[avail_cols].values
        y_train = df_train["contact_state"].values

        X_test = df_test[avail_cols].values
        y_test = df_test["contact_state"].values

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=55,
            max_depth=8,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        self.model = rf

        preds = rf.predict(X_test_scaled)

        acc = float(round(accuracy_score(y_test, preds), 4))
        macro_f1 = float(round(f1_score(y_test, preds, average="macro", zero_division=0), 4))
        weighted_f1 = float(round(f1_score(y_test, preds, average="weighted", zero_division=0), 4))
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2]).tolist()

        per_class_f1 = f1_score(y_test, preds, average=None, labels=[0, 1, 2], zero_division=0)
        class_f1_dict = {
            self.CLASS_NAMES[i]: float(round(per_class_f1[i], 4)) for i in range(len(self.CLASS_NAMES))
        }

        importances = dict(zip(avail_cols, [round(float(v), 4) for v in rf.feature_importances_]))

        metrics = MulticlassMetrics(
            dataset_name="IntelliMan Contact and Slippage (ICECS 2025)",
            model_name="Multi-Class Grasp Phase Random Forest",
            class_names=self.CLASS_NAMES,
            train_samples=len(X_train),
            test_samples=len(X_test),
            accuracy=acc,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            class_f1_scores=class_f1_dict,
            confusion_matrix=cm,
            feature_names=avail_cols,
            feature_importances=importances,
            disclaimer="Classifies physical PVDF sensor windows into Rest (0), Contact (1), and Slip (2)."
        )
        self.metrics = metrics

        joblib.dump({"model": rf, "scaler": self.scaler, "feature_names": avail_cols}, self.MODEL_PATH)
        with open(self.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        return metrics

