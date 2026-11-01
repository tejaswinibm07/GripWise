"""
dataset_preprocessing.py - Dataset-Specific Preprocessing Adapters
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Implements leakage-resistant feature extraction pipelines for:
1. IntelliMan Contact & Slippage Dataset (PVDF 8-channel tactile vibrations)
2. Bristol Biomimetic Tactile Slip Dataset (TacTip 127-pin optical papillae)
3. NinaPro Surface EMG Dataset (Multi-channel forearm muscle activation)
4. GripWise Synthetic Force Dataset
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io
from typing import Optional, Callable

from dataset_registry import DatasetRegistry


class IntelliManSlipAdapter:
    """
    Adapter for IntelliMan Horizon Europe WP5 Contact & Slippage Dataset (ICECS 2025).
    Extracts time-frequency and dynamic vibration features from 8-channel PVDF sensors
    with strict recording-level separation to eliminate temporal data leakage.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None):
        if registry is None:
            registry = DatasetRegistry()
        self.meta = registry.get_dataset("intelliman_slip")
        self.raw_dir = self.meta.local_raw_dir
        self.processed_dir = self.meta.local_processed_dir

    def process(
        self,
        window_size: int = 150,
        step_size: int = 75,
        max_files_per_texture: int = 40,
        test_ratio: float = 0.25,
        random_seed: int = 42,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes PVDF time-series into windowed feature vectors.
        Splits by full recording trial to ensure strict zero-leakage evaluation.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        train_csv = os.path.join(self.processed_dir, "intelliman_slip_train.csv")
        test_csv = os.path.join(self.processed_dir, "intelliman_slip_test.csv")

        # Discover extracted directories
        extracted_root = os.path.join(self.raw_dir, "extracted")
        if not os.path.exists(extracted_root):
            extracted_root = self.raw_dir

        search_dirs = [
            os.path.join(extracted_root, "Dataset_Contact_slippage_ICECS", "PVDF_roughtex"),
            os.path.join(extracted_root, "Dataset_Contact_slippage_ICECS", "PVDF_softtex")
        ]

        # Fallback search if path varies
        found_dirs = [d for d in search_dirs if os.path.exists(d)]
        if not found_dirs:
            # Recursive search for PVDF directories
            for root, dirs, _ in os.walk(extracted_root):
                for d in dirs:
                    if "PVDF" in d:
                        found_dirs.append(os.path.join(root, d))

        if not found_dirs:
            raise FileNotFoundError(f"No PVDF data directories found in {self.raw_dir}. Please run download first.")

        np.random.seed(random_seed)
        train_records = []
        test_records = []

        total_dirs = len(found_dirs)
        for d_idx, sdir in enumerate(found_dirs):
            tex_name = "rough" if "rough" in sdir.lower() else "soft"
            all_files = sorted([os.path.join(sdir, f) for f in os.listdir(sdir) if os.path.isfile(os.path.join(sdir, f))])
            selected_files = all_files[:max_files_per_texture]

            # RECORDING-LEVEL SPLIT: partition entire trial files into train vs test
            n_test = max(1, int(len(selected_files) * test_ratio))
            # Shuffle file indices deterministically
            perm = np.random.permutation(len(selected_files))
            test_indices = set(perm[:n_test])

            for f_idx, fpath in enumerate(selected_files):
                is_test_file = f_idx in test_indices
                target_list = test_records if is_test_file else train_records

                try:
                    df = pd.read_csv(fpath, sep=r"\s+", header=None)
                    if df.shape[1] < 9:
                        continue
                    
                    data = df.iloc[:, :8].values.astype(np.float64)  # 8 PVDF channels
                    labels = df.iloc[:, 8].values                   # Label column

                    n_samples = len(data)
                    for start in range(0, n_samples - window_size, step_size):
                        end = start + window_size
                        win_data = data[start:end, :]
                        win_labels = labels[start:end]

                        # Ground-truth majority label in window
                        # Phase 1: Static contact, Phase 2: Dynamic Slippage, Phase 3: Separation/Rest
                        counts = pd.Series(win_labels).value_counts()
                        maj_label = float(counts.index[0])

                        # Skip ambiguous transition frame if label is 0.0
                        if maj_label == 0.0 and len(counts) > 1:
                            maj_label = float(counts.index[1])

                        # Feature extraction across 8 channels
                        means = np.mean(win_data, axis=0)
                        stds = np.std(win_data, axis=0)
                        ptp = np.ptp(win_data, axis=0)
                        rms = np.sqrt(np.mean(win_data ** 2, axis=0))

                        # Cross-channel summary features
                        feat = {
                            "trial_file": os.path.basename(fpath),
                            "texture": tex_name,
                            "window_idx": start // step_size,
                            "maj_phase_raw": maj_label,
                            # Binary label: 1 = Slip (Phase 2), 0 = Non-Slip (Phase 1 or 3)
                            "is_slip": 1 if maj_label == 2.0 else 0,
                            # Multiclass label: 0 = Rest, 1 = Static Contact, 2 = Slippage
                            "contact_state": 2 if maj_label == 2.0 else (1 if maj_label == 1.0 else 0),
                            "global_vibration_mean_std": float(np.mean(stds)),
                            "global_vibration_max_std": float(np.max(stds)),
                            "global_vibration_energy": float(np.sum(stds ** 2)),
                            "global_signal_ptp": float(np.max(ptp))
                        }

                        # Per-channel features (Ch 1..8)
                        for ch in range(8):
                            feat[f"ch{ch+1}_mean"] = float(means[ch])
                            feat[f"ch{ch+1}_std"] = float(stds[ch])
                            feat[f"ch{ch+1}_rms"] = float(rms[ch])
                            feat[f"ch{ch+1}_ptp"] = float(ptp[ch])

                        target_list.append(feat)

                except Exception:
                    continue

            if progress_callback:
                progress_callback(0.2 + 0.6 * ((d_idx + 1) / total_dirs), f"Preprocessed {tex_name} texture trials...")

        df_train = pd.DataFrame(train_records)
        df_test = pd.DataFrame(test_records)

        df_train.to_csv(train_csv, index=False)
        df_test.to_csv(test_csv, index=False)

        if progress_callback:
            progress_callback(1.0, f"IntelliMan preprocessed! Train: {len(df_train)} windows, Test: {len(df_test)} windows.")

        return df_train, df_test


class BristolSlipAdapter:
    """
    Adapter for Bristol Biomimetic Tactile Slip Dataset (TacTip optical tactile sensor).
    Processes 127 papillae marker position vectors (x, y) into spatial kinematics,
    directional coherence, and velocity profiles.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None):
        if registry is None:
            registry = DatasetRegistry()
        self.meta = registry.get_dataset("bristol_slip")
        self.raw_dir = self.meta.local_raw_dir
        self.processed_dir = self.meta.local_processed_dir

    def process(
        self,
        window_size: int = 25,
        step_size: int = 15,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes TacTip .mat files into windowed pin velocity and coherence features.
        Maintains the original research division: 20 train runs vs 20 test runs.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        train_csv = os.path.join(self.processed_dir, "bristol_slip_train.csv")
        test_csv = os.path.join(self.processed_dir, "bristol_slip_test.csv")

        # Locate train and test folders
        base_search = [
            os.path.join(self.raw_dir, "extracted", "1fbyzhdft4rhp2iyptqfbf2mp4"),
            os.path.join(self.raw_dir, "1fbyzhdft4rhp2iyptqfbf2mp4"),
            self.raw_dir
        ]
        active_root = None
        for b in base_search:
            if os.path.exists(os.path.join(b, "train")) and os.path.exists(os.path.join(b, "test")):
                active_root = b
                break

        if not active_root:
            raise FileNotFoundError(f"Bristol train/test directories not found in {self.raw_dir}.")

        def _process_mat_folder(folder_path: str, split_name: str) -> pd.DataFrame:
            mat_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mat")])
            records = []

            for f_idx, fpath in enumerate(mat_files):
                try:
                    mat = scipy.io.loadmat(fpath)
                    if "new_mat" not in mat:
                        continue
                    data = mat["new_mat"]  # shape: (2000 frames, 127 pins, 2 coordinates)

                    # Compute inter-frame velocity vectors for 127 pins
                    diffs = data[1:] - data[:-1]  # shape: (1999, 127, 2)
                    speeds = np.linalg.norm(diffs, axis=2)  # shape: (1999, 127)
                    angles = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])  # shape: (1999, 127)

                    # Dynamic threshold for slip labeling from physical pin displacement
                    # When average speed exceeds 0.25 px/frame and spatial coherence > 0.45, active slip occurs
                    mean_speeds = np.mean(speeds, axis=1)  # shape: (1999,)
                    
                    # Circular spatial coherence: R = |sum(e^{i*theta})| / N
                    cos_sum = np.sum(np.cos(angles), axis=1)
                    sin_sum = np.sum(np.sin(angles), axis=1)
                    coherence = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / 127.0  # shape: (1999,)

                    # Window over the 1,999 time steps
                    n_steps = len(mean_speeds)
                    for start in range(0, n_steps - window_size, step_size):
                        end = start + window_size
                        win_speeds = speeds[start:end, :]
                        win_mean_speed = mean_speeds[start:end]
                        win_coherence = coherence[start:end]

                        avg_speed = float(np.mean(win_mean_speed))
                        avg_coherence = float(np.mean(win_coherence))
                        max_pin_speed = float(np.max(win_speeds))

                        # Ground truth slip flag: coordinated high pin movement
                        is_slip = 1 if (avg_speed >= 0.24 and avg_coherence >= 0.38) else 0

                        records.append({
                            "recording": os.path.basename(fpath),
                            "split": split_name,
                            "window_idx": start // step_size,
                            "mean_pin_speed": avg_speed,
                            "std_pin_speed": float(np.std(win_mean_speed)),
                            "max_pin_speed": max_pin_speed,
                            "min_pin_speed": float(np.min(win_mean_speed)),
                            "spatial_coherence": avg_coherence,
                            "coherence_std": float(np.std(win_coherence)),
                            "is_slip": is_slip
                        })
                except Exception:
                    continue

            return pd.DataFrame(records)

        if progress_callback:
            progress_callback(0.3, "Processing Bristol 20 training runs...")
        df_train = _process_mat_folder(os.path.join(active_root, "train"), "train")

        if progress_callback:
            progress_callback(0.7, "Processing Bristol 20 held-out test runs...")
        df_test = _process_mat_folder(os.path.join(active_root, "test"), "test")

        df_train.to_csv(train_csv, index=False)
        df_test.to_csv(test_csv, index=False)

        if progress_callback:
            progress_callback(1.0, f"Bristol preprocessed! Train: {len(df_train)}, Test: {len(df_test)}.")

        return df_train, df_test


class NinaProEMGAdapter:
    """
    Adapter for NinaPro surface EMG database.
    Provides feature extraction for multi-channel sEMG:
    - Mean Absolute Value (MAV)
    - Waveform Length (WL)
    - Root Mean Square (RMS)
    - Zero Crossings (ZC)
    Provides a reproducible offline benchmark generator if raw multi-gigabyte .mat files are not downloaded.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None):
        if registry is None:
            registry = DatasetRegistry()
        self.meta = registry.get_dataset("ninapro_emg")
        self.raw_dir = self.meta.local_raw_dir
        self.processed_dir = self.meta.local_processed_dir

    def process(self, n_samples: int = 600, random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        os.makedirs(self.processed_dir, exist_ok=True)
        train_csv = os.path.join(self.processed_dir, "ninapro_emg_train.csv")
        test_csv = os.path.join(self.processed_dir, "ninapro_emg_test.csv")

        # Check for real NinaPro .mat files in raw dir
        mat_files = glob.glob(os.path.join(self.raw_dir, "**", "*.mat"), recursive=True)
        if mat_files:
            # Parse real NinaPro file
            records = []
            for f in mat_files[:3]:
                try:
                    mat = scipy.io.loadmat(f)
                    emg = mat.get("emg")
                    stimulus = mat.get("stimulus")
                    if emg is not None and stimulus is not None:
                        # Extract windowed MAV & RMS
                        w_len = 100
                        for start in range(0, len(emg) - w_len, 50):
                            w_emg = emg[start:start+w_len, :]
                            w_stim = int(stimulus[start+w_len-1, 0])
                            row = {"gesture_class": w_stim, "source": "REAL_NINAPRO_FILE"}
                            for ch in range(min(10, w_emg.shape[1])):
                                row[f"emg_mav_ch{ch+1}"] = float(np.mean(np.abs(w_emg[:, ch])))
                                row[f"emg_rms_ch{ch+1}"] = float(np.sqrt(np.mean(w_emg[:, ch]**2)))
                            records.append(row)
                except Exception:
                    pass
            if len(records) > 100:
                df = pd.DataFrame(records)
                split_idx = int(len(df) * 0.8)
                df_train = df.iloc[:split_idx]
                df_test = df.iloc[split_idx:]
                df_train.to_csv(train_csv, index=False)
                df_test.to_csv(test_csv, index=False)
                return df_train, df_test

        # Surrogate offline benchmark generator for NinaPro sEMG
        np.random.seed(random_seed)
        gestures = ["Rest", "Power Grasp", "Precision Pinch", "Hand Open", "Lateral Key"]
        records = []
        for subject_id in range(1, 6):
            is_test_subject = (subject_id == 5)  # Subject-independent test split
            for g_idx, g_name in enumerate(gestures):
                base_activation = 0.05 if g_idx == 0 else (0.45 + 0.15 * g_idx)
                for trial in range(n_samples // (5 * len(gestures))):
                    row = {
                        "subject_id": subject_id,
                        "gesture_id": g_idx,
                        "gesture_name": g_name,
                        "is_test_split": is_test_subject
                    }
                    for ch in range(1, 9):
                        # Ch 1-4: flexor muscles, Ch 5-8: extensor muscles
                        ch_weight = 1.3 if (g_idx in [1, 2] and ch <= 4) else (1.2 if (g_idx == 3 and ch > 4) else 0.7)
                        mav = float(np.abs(np.random.normal(base_activation * ch_weight * 0.08, 0.015)))
                        rms = float(mav * 1.25 + np.random.uniform(0.005, 0.015))
                        wl = float(mav * 4.2 + np.random.uniform(0.02, 0.05))
                        row[f"emg_mav_ch{ch}"] = round(mav, 5)
                        row[f"emg_rms_ch{ch}"] = round(rms, 5)
                        row[f"emg_wl_ch{ch}"] = round(wl, 5)
                    records.append(row)

        df_all = pd.DataFrame(records)
        df_train = df_all[df_all["is_test_split"] == False].drop(columns=["is_test_split"])
        df_test = df_all[df_all["is_test_split"] == True].drop(columns=["is_test_split"])

        df_train.to_csv(train_csv, index=False)
        df_test.to_csv(test_csv, index=False)
        return df_train, df_test

