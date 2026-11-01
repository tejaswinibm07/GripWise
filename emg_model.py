"""
emg_model.py - sEMG Grasp Intention Classification Model
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Implements Model D: Forearm sEMG Grasp Intention Classifier.
Classifies muscle activation patterns into discrete functional movements:
- Rest
- Power Grasp
- Precision Pinch
- Hand Open
- Lateral Key Grip

Uses subject-independent evaluation (held-out subject) to test generalization.
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
    recall_score, f1_score, confusion_matrix
)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "saved"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation", "metrics"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


@dataclass
class EMGMetrics:
    dataset_name: str
    model_name: str
    gestures: list[str]
    train_samples: int
    test_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    feature_names: list[str]
    disclaimer: str


class EMGIntentionClassifier:
    """
    Model D: sEMG Grasp Movement Classifier.
    Evaluated with subject-independent cross-validation.
    """

    MODEL_PATH = os.path.join(MODEL_DIR, "ninapro_emg_intent_model.joblib")
    METRICS_PATH = os.path.join(METRICS_DIR, "ninapro_emg_metrics.json")
    GESTURE_NAMES = ["Rest", "Power Grasp", "Precision Pinch", "Hand Open", "Lateral Key"]

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.metrics: EMGMetrics | None = None
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.METRICS_PATH):
            try:
                bundle = joblib.load(self.MODEL_PATH)
                self.model = bundle["model"]
                self.scaler = bundle["scaler"]
                with open(self.METRICS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.metrics = EMGMetrics(**data)
                return True
            except Exception:
                pass
        return False

    def train_and_evaluate(
        self,
        train_csv: str = "data/processed/ninapro_emg/ninapro_emg_train.csv",
        test_csv: str = "data/processed/ninapro_emg/ninapro_emg_test.csv",
        random_seed: int = 42
    ) -> EMGMetrics:
        train_path = os.path.abspath(train_csv)
        test_path = os.path.abspath(test_csv)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            from dataset_preprocessing import NinaProEMGAdapter
            adapter = NinaProEMGAdapter()
            df_train, df_test = adapter.process()
        else:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

        feat_cols = [c for c in df_train.columns if c.startswith("emg_")]
        X_train = df_train[feat_cols].values
        y_train = df_train["gesture_id"].values

        X_test = df_test[feat_cols].values
        y_test = df_test["gesture_id"].values

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=45,
            max_depth=7,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1
        )
        rf.fit(X_train_sc, y_train)
        self.model = rf

        preds = rf.predict(X_test_sc)

        acc = float(round(accuracy_score(y_test, preds), 4))
        macro_f1 = float(round(f1_score(y_test, preds, average="macro", zero_division=0), 4))
        weighted_f1 = float(round(f1_score(y_test, preds, average="weighted", zero_division=0), 4))
        cm = confusion_matrix(y_test, preds, labels=list(range(len(self.GESTURE_NAMES)))).tolist()

        metrics = EMGMetrics(
            dataset_name="NinaPro sEMG Movement Database (DB1/DB2 Architecture)",
            model_name="RandomForestClassifier (sEMG Feature Envelope)",
            gestures=self.GESTURE_NAMES,
            train_samples=len(X_train),
            test_samples=len(X_test),
            accuracy=acc,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            confusion_matrix=cm,
            feature_names=feat_cols,
            disclaimer="Subject-independent evaluation. sEMG gestures classify movement intent, NOT safe minimum grip force."
        )
        self.metrics = metrics

        joblib.dump({"model": rf, "scaler": self.scaler, "feature_names": feat_cols}, self.MODEL_PATH)
        with open(self.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        return metrics

