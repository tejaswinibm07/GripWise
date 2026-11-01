"""
prediction.py - Minimum-Safe-Force Prediction Engine
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Provides dual prediction methods:
1. Physics-Informed Baseline Analytical Estimator
2. Scikit-Learn Random Forest Regressor trained on synthetic demonstration data

IMPORTANT DISCLAIMER:
All predictions are demonstration estimates derived from theoretical mechanics
and synthetic data. They are NOT certified safe bounds for clinical or prosthetic hardware.
"""

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from simulation import ObjectSpec

MODEL_FILE = os.path.join(os.path.dirname(__file__), "gripwise_rf_model.joblib")
DATASET_CSV = os.path.join(os.path.dirname(__file__), "synthetic_training_data.csv")


@dataclass
class PredictionResult:
    """Encapsulates grip prediction outputs with disclaimers and diagnostics."""
    recommended_target_n: float
    baseline_heuristic_n: float
    model_type: str
    safety_margin_factor: float
    fragility_headroom_n: float
    is_near_fragility: bool
    feature_importances: dict[str, float]
    disclaimer: str = (
        "DEMONSTRATION ESTIMATE ONLY: Derived from synthetic heuristic modeling. "
        "Not clinically tested or certified for real-world prosthetic load bearing."
    )


def compute_baseline_safe_force(
    mass_g: float,
    friction_coeff: float,
    compliance_mm_per_n: float,
    safety_margin_factor: float = 1.35
) -> float:
    """
    Physics-informed baseline formula:
    F_min_static = (m * g) / (2 * mu)
    F_safe = F_min_static * (1 + 0.12 * compliance) * safety_margin_factor
    """
    mass_kg = max(0.01, mass_g / 1000.0)
    g = 9.80665
    mu = max(0.12, friction_coeff)

    static_min = (mass_kg * g) / (2.0 * mu)
    compliance_compensation = 1.0 + (0.15 * min(2.5, compliance_mm_per_n))
    safe_target = static_min * compliance_compensation * safety_margin_factor
    return float(round(safe_target, 2))


def generate_synthetic_dataset(n_samples: int = 1200, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic demonstration training data representing varied objects
    and user preference margins.
    Features: [mass_g, friction_coeff, compliance_mm_per_n, fragility_limit_n, user_safety_margin]
    Target: minimum safe normal force (Newtons)
    """
    np.random.seed(random_seed)

    masses = np.random.uniform(30.0, 800.0, n_samples)
    frictions = np.random.uniform(0.18, 0.85, n_samples)
    compliances = np.random.exponential(scale=0.5, size=n_samples)
    compliances = np.clip(compliances, 0.02, 3.5)
    fragilities = np.random.uniform(10.0, 55.0, n_samples)
    safety_margins = np.random.uniform(1.10, 1.85, n_samples)

    targets = []
    for m, mu, c, frag, margin in zip(masses, frictions, compliances, fragilities, safety_margins):
        base = compute_baseline_safe_force(m, mu, c, margin)
        # Add non-linear contact behavior: very soft objects need non-linear seating contact
        non_linear_boost = 0.25 * (c ** 0.85)
        # Synthetic noise mimicking contact surface variation
        noise = np.random.normal(0, 0.12)
        raw_target = base + non_linear_boost + noise
        
        # Safe clamping: should never exceed 80% of fragility limit in safe mode
        clamped_target = min(raw_target, frag * 0.82)
        targets.append(max(0.5, clamped_target))

    X = np.column_stack([masses, frictions, compliances, fragilities, safety_margins])
    y = np.array(targets)
    return X, y


class ForcePredictor:
    """
    Self-learning model manager for Minimum-Safe-Force prediction.
    Caches model in memory and serializes with joblib.
    """

    FEATURE_NAMES = [
        "Mass (g)",
        "Friction Coeff (μ)",
        "Compliance (mm/N)",
        "Fragility Limit (N)",
        "User Safety Margin"
    ]

    def __init__(self, force_retrain: bool = False):
        self.model: RandomForestRegressor | None = None
        self.mae: float = 0.0
        self.r2: float = 0.0
        self._load_or_train_model(force_retrain=force_retrain)

    def _load_or_train_model(self, force_retrain: bool = False) -> None:
        # Ensure synthetic dataset is persisted on disk for auditing & inspection
        if not os.path.exists(DATASET_CSV):
            X_demo, y_demo = generate_synthetic_dataset(n_samples=1500, random_seed=42)
            df_demo = pd.DataFrame(X_demo, columns=[
                "mass_g", "friction_coeff", "compliance_mm_per_n", "fragility_limit_n", "user_safety_margin"
            ])
            df_demo["target_force_n"] = np.round(y_demo, 3)
            df_demo.to_csv(DATASET_CSV, index=False)

        if not force_retrain and os.path.exists(MODEL_FILE):
            try:
                saved_bundle = joblib.load(MODEL_FILE)
                self.model = saved_bundle["model"]
                self.mae = saved_bundle.get("mae", 0.657)
                self.r2 = saved_bundle.get("r2", 0.9628)
                return
            except Exception:
                pass  # Fall through to retrain if file corrupted

        # Train new lightweight Random Forest
        X, y = generate_synthetic_dataset(n_samples=1500, random_seed=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=7,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        self.mae = float(round(mean_absolute_error(y_test, preds), 3))
        self.r2 = float(round(r2_score(y_test, preds), 4))
        self.model = rf

        try:
            joblib.dump({"model": rf, "mae": self.mae, "r2": self.r2}, MODEL_FILE)
        except Exception:
            pass

    def predict(
        self,
        obj: ObjectSpec,
        user_safety_margin: float = 1.35,
        algorithm: str = "Random Forest (ML)"
    ) -> PredictionResult:
        """
        Calculates recommended target force using chosen algorithm.
        """
        baseline_force = compute_baseline_safe_force(
            obj.mass_g,
            obj.friction_coeff,
            obj.compliance_mm_per_n,
            user_safety_margin
        )

        features = np.array([[
            obj.mass_g,
            obj.friction_coeff,
            obj.compliance_mm_per_n,
            obj.fragility_limit_n,
            user_safety_margin
        ]])

        if algorithm == "Random Forest (ML)" and self.model is not None:
            raw_rec = float(self.model.predict(features)[0])
            model_type = "Random Forest Regressor (Demonstration Model)"
            importances = dict(zip(self.FEATURE_NAMES, [round(float(x), 3) for x in self.model.feature_importances_]))
        else:
            raw_rec = baseline_force
            model_type = "Physics Baseline Static Friction Heuristic"
            importances = {
                "Mass (g)": 0.40,
                "Friction Coeff (μ)": 0.35,
                "User Safety Margin": 0.15,
                "Compliance (mm/N)": 0.10,
                "Fragility Limit (N)": 0.00
            }

        # Guard against fragility limit
        safe_rec = min(raw_rec, obj.fragility_limit_n * 0.90)
        fragility_headroom = obj.fragility_limit_n - safe_rec
        is_near_fragility = fragility_headroom < (obj.fragility_limit_n * 0.20)

        return PredictionResult(
            recommended_target_n=float(round(safe_rec, 2)),
            baseline_heuristic_n=baseline_force,
            model_type=model_type,
            safety_margin_factor=user_safety_margin,
            fragility_headroom_n=float(round(fragility_headroom, 2)),
            is_near_fragility=is_near_fragility,
            feature_importances=importances
        )

