"""
slip_model.py - Dedicated Tactile Slip Classification Models
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Implements dedicated machine learning classifiers trained on real-world
physical tactile sensor datasets:
1. IntelliManPVDFSlipClassifier (8-channel PVDF piezoelectric dynamic vibration)
2. BristolTacTipSlipClassifier (127-pin biomimetic optical tactile kinematics)

Strictly separates model architectures, sensors, and evaluation pipelines.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score
)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "saved"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation", "metrics"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


@dataclass
class ClassificationMetrics:
    dataset_name: str
    model_name: str
    sensor_type: str
    train_samples: int
    test_samples: int
    train_class_distribution: dict
    test_class_distribution: dict
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    feature_names: list[str]
    feature_importances: dict[str, float]
    leakage_protection: str
    disclaimer: str


class IntelliManPVDFSlipClassifier:
    """
    Dedicated Slip Classifier trained on real physical 8-channel PVDF tactile signals
    from the IntelliMan ICECS 2025 dataset.
    Uses recording-level train/test separation (unseen trials) to prevent leakage.
    """

    MODEL_PATH = os.path.join(MODEL_DIR, "intelliman_pvdf_slip_classifier.joblib")
    METRICS_PATH = os.path.join(METRICS_DIR, "intelliman_pvdf_slip_metrics.json")

    FEATURE_COLS = [
        "global_vibration_mean_std", "global_vibration_max_std",
        "global_vibration_energy", "global_signal_ptp"
    ] + [f"ch{ch}_std" for ch in range(1, 9)] + [f"ch{ch}_ptp" for ch in range(1, 9)]

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.baseline_model: DummyClassifier | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.metrics: ClassificationMetrics | None = None
        self.baseline_f1: float = 0.0
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.METRICS_PATH):
            try:
                bundle = joblib.load(self.MODEL_PATH)
                self.model = bundle["model"]
                self.scaler = bundle["scaler"]
                self.baseline_f1 = bundle.get("baseline_f1", 0.0)
                with open(self.METRICS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.metrics = ClassificationMetrics(**data)
                return True
            except Exception:
                pass
        return False

    def train_and_evaluate(
        self,
        train_csv: str = "data/processed/intelliman_slip/intelliman_slip_train.csv",
        test_csv: str = "data/processed/intelliman_slip/intelliman_slip_test.csv",
        random_seed: int = 42
    ) -> ClassificationMetrics:
        """
        Trains Random Forest and Baseline on training trial recordings,
        and evaluates strictly on held-out test trial recordings.
        """
        train_path = os.path.abspath(train_csv)
        test_path = os.path.abspath(test_csv)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            from dataset_preprocessing import IntelliManSlipAdapter
            adapter = IntelliManSlipAdapter()
            df_train, df_test = adapter.process()
        else:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

        available_features = [c for c in self.FEATURE_COLS if c in df_train.columns]
        X_train = df_train[available_features].values
        y_train = df_train["is_slip"].values

        X_test = df_test[available_features].values
        y_test = df_test["is_slip"].values

        # Scaling fit strictly on training set
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1. Baseline Model (Most Frequent Class)
        baseline = DummyClassifier(strategy="most_frequent", random_state=random_seed)
        baseline.fit(X_train_scaled, y_train)
        b_preds = baseline.predict(X_test_scaled)
        self.baseline_f1 = float(round(f1_score(y_test, b_preds, zero_division=0), 4))
        self.baseline_model = baseline

        # 2. Random Forest Model
        rf = RandomForestClassifier(
            n_estimators=65,
            max_depth=8,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        self.model = rf

        # Predictions on held-out test recordings
        preds = rf.predict(X_test_scaled)
        probs = rf.predict_proba(X_test_scaled)[:, 1]

        acc = float(round(accuracy_score(y_test, preds), 4))
        b_acc = float(round(balanced_accuracy_score(y_test, preds), 4))
        prec = float(round(precision_score(y_test, preds, zero_division=0), 4))
        rec = float(round(recall_score(y_test, preds, zero_division=0), 4))
        f1 = float(round(f1_score(y_test, preds, zero_division=0), 4))
        auc = float(round(roc_auc_score(y_test, probs), 4))
        cm = confusion_matrix(y_test, preds).tolist()

        importances = dict(zip(available_features, [round(float(v), 4) for v in rf.feature_importances_]))

        metrics = ClassificationMetrics(
            dataset_name="IntelliMan Contact and Slippage (ICECS 2025)",
            model_name="RandomForestClassifier (PVDF Dynamic Vibration)",
            sensor_type="8-Channel PVDF Piezoelectric Film",
            train_samples=len(X_train),
            test_samples=len(X_test),
            train_class_distribution={int(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
            test_class_distribution={int(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
            accuracy=acc,
            balanced_accuracy=b_acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            roc_auc=auc,
            confusion_matrix=cm,
            feature_names=available_features,
            feature_importances=importances,
            leakage_protection="Trial/recording-level partition: entire trial runs held out. Zero window overlap.",
            disclaimer="Trained on real public research PVDF tactile data. Evaluated on unseen physical trial recordings."
        )
        self.metrics = metrics

        # Persist model bundle and metrics
        joblib.dump({
            "model": rf,
            "scaler": self.scaler,
            "feature_names": available_features,
            "baseline_f1": self.baseline_f1
        }, self.MODEL_PATH)

        with open(self.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        return metrics

    def predict(self, feature_vector: np.ndarray) -> tuple[int, float]:
        """
        Classifies whether input feature vector indicates dynamic slip.
        Returns (predicted_class: 0 or 1, slip_probability: float).
        """
        if self.model is None:
            return 0, 0.0
        X_vec = np.array(feature_vector).reshape(1, -1)
        X_sc = self.scaler.transform(X_vec)
        pred = int(self.model.predict(X_sc)[0])
        prob = float(self.model.predict_proba(X_sc)[0, 1])
        return pred, round(prob, 3)


class BristolTacTipSlipClassifier:
    """
    Dedicated Slip Classifier trained on the Bristol Biomimetic TacTip optical tactile dataset.
    Inputs: Pin velocity magnitude, standard deviation, spatial directional coherence, max pin speed.
    """

    MODEL_PATH = os.path.join(MODEL_DIR, "bristol_tactip_slip_classifier.joblib")
    METRICS_PATH = os.path.join(METRICS_DIR, "bristol_tactip_slip_metrics.json")

    FEATURE_COLS = [
        "mean_pin_speed", "std_pin_speed", "max_pin_speed",
        "min_pin_speed", "spatial_coherence", "coherence_std"
    ]

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.metrics: ClassificationMetrics | None = None
        self.baseline_f1: float = 0.0
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.METRICS_PATH):
            try:
                bundle = joblib.load(self.MODEL_PATH)
                self.model = bundle["model"]
                self.baseline_f1 = bundle.get("baseline_f1", 0.0)
                with open(self.METRICS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.metrics = ClassificationMetrics(**data)
                return True
            except Exception:
                pass
        return False

    def train_and_evaluate(
        self,
        train_csv: str = "data/processed/bristol_slip/bristol_slip_train.csv",
        test_csv: str = "data/processed/bristol_slip/bristol_slip_test.csv",
        random_seed: int = 42
    ) -> ClassificationMetrics:
        train_path = os.path.abspath(train_csv)
        test_path = os.path.abspath(test_csv)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            from dataset_preprocessing import BristolSlipAdapter
            adapter = BristolSlipAdapter()
            df_train, df_test = adapter.process()
        else:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

        X_train = df_train[self.FEATURE_COLS].values
        y_train = df_train["is_slip"].values
        X_test = df_test[self.FEATURE_COLS].values
        y_test = df_test["is_slip"].values

        # Baseline
        baseline = DummyClassifier(strategy="most_frequent", random_state=random_seed)
        baseline.fit(X_train, y_train)
        b_preds = baseline.predict(X_test)
        self.baseline_f1 = float(round(f1_score(y_test, b_preds, zero_division=0), 4))

        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=7,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.model = rf

        preds = rf.predict(X_test)
        probs = rf.predict_proba(X_test)[:, 1]

        acc = float(round(accuracy_score(y_test, preds), 4))
        b_acc = float(round(balanced_accuracy_score(y_test, preds), 4))
        prec = float(round(precision_score(y_test, preds, zero_division=0), 4))
        rec = float(round(recall_score(y_test, preds, zero_division=0), 4))
        f1 = float(round(f1_score(y_test, preds, zero_division=0), 4))
        auc = float(round(roc_auc_score(y_test, probs), 4))
        cm = confusion_matrix(y_test, preds).tolist()

        importances = dict(zip(self.FEATURE_COLS, [round(float(v), 4) for v in rf.feature_importances_]))

        metrics = ClassificationMetrics(
            dataset_name="Bristol Biomimetic Tactile Slip Dataset",
            model_name="RandomForestClassifier (TacTip Optical Kinematics)",
            sensor_type="TacTip Biomimetic Optical Array (127 pins)",
            train_samples=len(X_train),
            test_samples=len(X_test),
            train_class_distribution={int(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
            test_class_distribution={int(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
            accuracy=acc,
            balanced_accuracy=b_acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            roc_auc=auc,
            confusion_matrix=cm,
            feature_names=self.FEATURE_COLS,
            feature_importances=importances,
            leakage_protection="Preserves Bristol official split: 20 train runs vs 20 independent test runs.",
            disclaimer="Trained strictly on Bristol physical optical tactile recordings."
        )
        self.metrics = metrics

        joblib.dump({"model": rf, "feature_names": self.FEATURE_COLS, "baseline_f1": self.baseline_f1}, self.MODEL_PATH)
        with open(self.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        return metrics

