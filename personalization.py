"""
personalization.py - User Profiles, Personalization, and Experiment Logging
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Enables personalized grip preferences (Delicate, Standard, Firm, Custom),
accumulates self-learning feedback biases from simulated trials,
maintains experiment trial logging with CSV/JSON export/import,
and provides robust backup and restore capabilities.
"""

import os
import json
import time
import shutil
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import pandas as pd

PROFILES_FILE = os.path.join(os.path.dirname(__file__), "user_profiles.json")
LOG_FILE = os.path.join(os.path.dirname(__file__), "experiment_log.json")
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "data", "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)


@dataclass
class UserProfile:
    """User personalization configuration."""
    name: str = "Default User"
    preset_mode: str = "Standard"  # Delicate, Standard, Firm, Custom
    base_safety_margin: float = 1.35
    learned_adaptation_bias: float = 0.0
    slip_sensitivity: float = 0.8
    total_trials: int = 0
    successful_trials: int = 0
    created_at: str = ""

    @property
    def effective_safety_margin(self) -> float:
        """Effective safety margin factoring in learned adaptation bias."""
        eff = self.base_safety_margin + self.learned_adaptation_bias
        return float(round(max(1.05, min(2.50, eff)), 2))


@dataclass
class TrialRecord:
    """Telemetry and outcome record for a completed trial."""
    trial_id: str
    timestamp: str
    object_name: str
    user_profile: str
    preset_mode: str
    initial_target_f: float
    final_target_f: float
    peak_measured_f: float
    slip_events: int
    outcome: str  # STABLE_SUCCESS, ADAPTED_SUCCESS, SLIP_FAILED, CRUSH_RISK, MANUAL_CHECK
    feedback_rating: str  # "Ideal", "Too Loose (Slipped)", "Too Tight", "Unrated"
    notes: str


DEFAULT_PRESETS = {
    "Delicate": {
        "description": "Minimized clamp force for deformable, brittle, or soft items. Lower safety margin (1.15x).",
        "safety_margin": 1.15,
        "slip_sensitivity": 0.95
    },
    "Standard": {
        "description": "Balanced compromise between grip security and energy conservation (1.35x).",
        "safety_margin": 1.35,
        "slip_sensitivity": 0.80
    },
    "Firm": {
        "description": "Higher clamp force for heavy, slippery, or dynamic activities (1.65x).",
        "safety_margin": 1.65,
        "slip_sensitivity": 0.65
    },
    "Custom": {
        "description": "User-defined manual parameters.",
        "safety_margin": 1.40,
        "slip_sensitivity": 0.80
    }
}


class PersonalizationManager:
    """Manages profile persistence, trial history logging, import/export, and backups."""

    def __init__(self):
        self.profiles: dict[str, UserProfile] = {}
        self.trials: list[TrialRecord] = []
        self._load_profiles()
        self._load_trials()

    def _load_profiles(self) -> None:
        if os.path.exists(PROFILES_FILE):
            try:
                with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.profiles[k] = UserProfile(**v)
                return
            except Exception:
                # Backup corrupt file and re-seed
                corrupt_backup = f"{PROFILES_FILE}.corrupt_{int(time.time())}.bak"
                try:
                    shutil.copy(PROFILES_FILE, corrupt_backup)
                except Exception:
                    pass

        # Populate defaults if missing or corrupted
        self.profiles["Alex (Standard)"] = UserProfile(
            name="Alex (Standard)",
            preset_mode="Standard",
            base_safety_margin=1.35,
            learned_adaptation_bias=0.05,
            created_at=time.strftime("%Y-%m-%d %H:%M")
        )
        self.profiles["Sam (Delicate Focus)"] = UserProfile(
            name="Sam (Delicate Focus)",
            preset_mode="Delicate",
            base_safety_margin=1.15,
            learned_adaptation_bias=-0.02,
            created_at=time.strftime("%Y-%m-%d %H:%M")
        )
        self.profiles["Jordan (Firm Grip)"] = UserProfile(
            name="Jordan (Firm Grip)",
            preset_mode="Firm",
            base_safety_margin=1.65,
            learned_adaptation_bias=0.10,
            created_at=time.strftime("%Y-%m-%d %H:%M")
        )
        self.save_profiles()

    def save_profiles(self) -> None:
        try:
            data = {k: asdict(v) for k, v in self.profiles.items()}
            with open(PROFILES_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def create_profile(self, name: str, preset_mode: str = "Standard", base_margin: Optional[float] = None) -> Tuple[bool, str]:
        """Creates a new user profile with validation."""
        clean_name = name.strip()
        if not clean_name:
            return False, "Profile name cannot be empty."
        if clean_name in self.profiles:
            return False, f"Profile '{clean_name}' already exists."

        margin = base_margin if base_margin is not None else DEFAULT_PRESETS.get(preset_mode, {}).get("safety_margin", 1.35)
        prof = UserProfile(
            name=clean_name,
            preset_mode=preset_mode,
            base_safety_margin=float(round(margin, 2)),
            learned_adaptation_bias=0.0,
            created_at=time.strftime("%Y-%m-%d %H:%M")
        )
        self.profiles[clean_name] = prof
        self.save_profiles()
        return True, f"Created profile '{clean_name}'."

    def delete_profile(self, name: str) -> Tuple[bool, str]:
        if name not in self.profiles:
            return False, f"Profile '{name}' not found."
        if len(self.profiles) <= 1:
            return False, "Cannot delete the sole remaining profile."
        del self.profiles[name]
        self.save_profiles()
        return True, f"Deleted profile '{name}'."

    def _load_trials(self) -> None:
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.trials = [TrialRecord(**r) for r in raw]
                return
            except Exception:
                corrupt_backup = f"{LOG_FILE}.corrupt_{int(time.time())}.bak"
                try:
                    shutil.copy(LOG_FILE, corrupt_backup)
                except Exception:
                    pass

        # Seed sample demonstration history
        self.trials = [
            TrialRecord(
                trial_id="TR-001",
                timestamp=time.strftime("%Y-%m-%d 19:15:22"),
                object_name="Ceramic Cup",
                user_profile="Alex (Standard)",
                preset_mode="Standard",
                initial_target_f=3.85,
                final_target_f=3.85,
                peak_measured_f=3.92,
                slip_events=0,
                outcome="STABLE_SUCCESS",
                feedback_rating="Ideal",
                notes="Stable lift with no micro-slip detected."
            ),
            TrialRecord(
                trial_id="TR-002",
                timestamp=time.strftime("%Y-%m-%d 19:18:45"),
                object_name="Plastic Bottle",
                user_profile="Alex (Standard)",
                preset_mode="Standard",
                initial_target_f=4.20,
                final_target_f=4.85,
                peak_measured_f=4.91,
                slip_events=1,
                outcome="ADAPTED_SUCCESS",
                feedback_rating="Ideal",
                notes="Recovered from injected micro-slip via adaptive boost (+0.65N)."
            ),
            TrialRecord(
                trial_id="TR-003",
                timestamp=time.strftime("%Y-%m-%d 19:22:10"),
                object_name="Polyurethane Sponge",
                user_profile="Sam (Delicate Focus)",
                preset_mode="Delicate",
                initial_target_f=0.85,
                final_target_f=0.85,
                peak_measured_f=0.89,
                slip_events=0,
                outcome="STABLE_SUCCESS",
                feedback_rating="Ideal",
                notes="Soft grasp preserved open-cell structure."
            )
        ]
        self.save_trials()

    def save_trials(self) -> None:
        try:
            raw = [asdict(t) for t in self.trials]
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2)
        except Exception:
            pass

    def log_trial(self, trial: TrialRecord) -> None:
        """Appends a new completed trial and updates profile stats."""
        self.trials.insert(0, trial)
        self.save_trials()

        if trial.user_profile in self.profiles:
            prof = self.profiles[trial.user_profile]
            prof.total_trials += 1
            if "SUCCESS" in trial.outcome:
                prof.successful_trials += 1
            self.save_profiles()

    def apply_feedback_learning(self, profile_name: str, feedback: str) -> float:
        """Adjusts the user's learned adaptation bias based on post-trial feedback."""
        if profile_name not in self.profiles:
            return 0.0

        prof = self.profiles[profile_name]
        old_bias = prof.learned_adaptation_bias

        if "Loose" in feedback or "Slipped" in feedback:
            prof.learned_adaptation_bias += 0.05
        elif "Tight" in feedback or "Crush" in feedback:
            prof.learned_adaptation_bias -= 0.04
        elif "Ideal" in feedback:
            prof.learned_adaptation_bias = round(prof.learned_adaptation_bias * 0.98, 3)

        prof.learned_adaptation_bias = float(round(max(-0.30, min(0.60, prof.learned_adaptation_bias)), 3))
        self.save_profiles()
        return prof.learned_adaptation_bias - old_bias

    def filter_trials(
        self,
        object_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        outcome_filter: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> list[TrialRecord]:
        """Filters trial records by query criteria."""
        results = []
        for t in self.trials:
            if object_name and object_name != "All Objects" and t.object_name != object_name:
                continue
            if profile_name and profile_name != "All Profiles" and t.user_profile != profile_name:
                continue
            if outcome_filter and outcome_filter != "All Outcomes" and outcome_filter not in t.outcome:
                continue
            if search_query:
                q = search_query.lower()
                matches = (
                    q in t.trial_id.lower() or
                    q in t.object_name.lower() or
                    q in t.user_profile.lower() or
                    q in t.notes.lower()
                )
                if not matches:
                    continue
            results.append(t)
        return results

    def get_trials_dataframe(self, filtered_trials: Optional[list[TrialRecord]] = None) -> pd.DataFrame:
        """Returns trial history formatted as a pandas DataFrame."""
        source_trials = filtered_trials if filtered_trials is not None else self.trials
        if not source_trials:
            return pd.DataFrame(columns=[
                "Trial ID", "Timestamp", "Object", "Profile", "Preset",
                "Init Target (N)", "Final Target (N)", "Peak Measured (N)",
                "Slips", "Outcome", "Feedback", "Notes"
            ])

        records = []
        for t in source_trials:
            records.append({
                "Trial ID": t.trial_id,
                "Timestamp": t.timestamp,
                "Object": t.object_name,
                "Profile": t.user_profile,
                "Preset": t.preset_mode,
                "Init Target (N)": t.initial_target_f,
                "Final Target (N)": t.final_target_f,
                "Peak Measured (N)": t.peak_measured_f,
                "Slips": t.slip_events,
                "Outcome": t.outcome,
                "Feedback": t.feedback_rating,
                "Notes": t.notes
            })
        return pd.DataFrame(records)

    # --------------------------------------------------------------------------
    # IMPORT & EXPORT
    # --------------------------------------------------------------------------
    def export_trials_json(self) -> str:
        """Serializes trial history as a formatted JSON string."""
        raw = [asdict(t) for t in self.trials]
        return json.dumps(raw, indent=2)

    def import_trials_json(self, json_string: str, overwrite: bool = False) -> Tuple[int, list[str]]:
        """
        Imports trial records from a JSON string with strict schema validation.
        Returns (num_imported: int, error_log: list[str]).
        """
        errors = []
        try:
            parsed = json.loads(json_string)
            if not isinstance(parsed, list):
                return 0, ["Invalid format: JSON root must be an array of trial records."]
        except Exception as e:
            return 0, [f"JSON Parse Error: {str(e)}"]

        valid_records = []
        required_fields = ["trial_id", "timestamp", "object_name", "user_profile", "initial_target_f", "final_target_f"]

        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                errors.append(f"Row {idx}: Item is not an object")
                continue
            missing = [f for f in required_fields if f not in item]
            if missing:
                errors.append(f"Row {idx} missing required fields: {missing}")
                continue
            try:
                rec = TrialRecord(
                    trial_id=str(item.get("trial_id", f"IMP-{idx}")),
                    timestamp=str(item.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))),
                    object_name=str(item.get("object_name", "Unknown")),
                    user_profile=str(item.get("user_profile", "Default User")),
                    preset_mode=str(item.get("preset_mode", "Standard")),
                    initial_target_f=float(item.get("initial_target_f", 0.0)),
                    final_target_f=float(item.get("final_target_f", 0.0)),
                    peak_measured_f=float(item.get("peak_measured_f", 0.0)),
                    slip_events=int(item.get("slip_events", 0)),
                    outcome=str(item.get("outcome", "IMPORTED")),
                    feedback_rating=str(item.get("feedback_rating", "Unrated")),
                    notes=str(item.get("notes", "Imported experiment record"))
                )
                valid_records.append(rec)
            except Exception as e:
                errors.append(f"Row {idx} conversion error: {str(e)}")

        if valid_records:
            if overwrite:
                self.trials = valid_records
            else:
                existing_ids = {t.trial_id for t in self.trials}
                for r in valid_records:
                    if r.trial_id in existing_ids:
                        r.trial_id = f"{r.trial_id}-imp"
                    self.trials.insert(0, r)
            self.save_trials()

        return len(valid_records), errors

    # --------------------------------------------------------------------------
    # BACKUP & RESTORE
    # --------------------------------------------------------------------------
    def create_backup(self) -> str:
        """Creates a timestamped JSON backup snapshot in data/backups/."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"gripwise_backup_{timestamp}.json"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)

        payload = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profiles": {k: asdict(v) for k, v in self.profiles.items()},
            "trials": [asdict(t) for t in self.trials]
        }
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return backup_filename

    def list_backups(self) -> list[dict]:
        """Lists available backup snapshots."""
        backups = []
        if os.path.exists(BACKUP_DIR):
            for f in sorted(os.listdir(BACKUP_DIR), reverse=True):
                if f.endswith(".json"):
                    fpath = os.path.join(BACKUP_DIR, f)
                    backups.append({
                        "filename": f,
                        "size_kb": round(os.path.getsize(fpath) / 1024.0, 2),
                        "timestamp": time.ctime(os.path.getmtime(fpath))
                    })
        return backups

    def restore_backup(self, backup_filename: str) -> Tuple[bool, str]:
        """Restores profiles and trials from a backup snapshot."""
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        if not os.path.exists(backup_path):
            return False, f"Backup file '{backup_filename}' does not exist."

        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "profiles" in data:
                self.profiles = {k: UserProfile(**v) for k, v in data["profiles"].items()}
                self.save_profiles()

            if "trials" in data:
                self.trials = [TrialRecord(**t) for t in data["trials"]]
                self.save_trials()

            return True, f"Successfully restored data from '{backup_filename}'."
        except Exception as e:
            return False, f"Restore failed: {str(e)}"
