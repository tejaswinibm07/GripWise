"""
dataset_registry.py - Central Research Dataset Registry
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Maintains metadata, citations, licenses, access restrictions, file formats,
download endpoints, and status tracking for all public and synthetic datasets.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DataNature(str, Enum):
    REAL_RESEARCH_DATA = "Real-World Physical Research Data"
    SYNTHETIC_DATA = "Synthetic Demonstration Model"
    SIMULATED_DATA = "Simulated Physics Telemetry"


class MLTask(str, Enum):
    FORCE_REGRESSION = "Minimum-Safe-Force Regression"
    SLIP_DETECTION = "Tactile Slip Classification"
    CONTACT_STATE = "Contact & Grasp-State Classification"
    TEXTURE_CLASSIFICATION = "Tactile Texture Characterization"
    EMG_INTENT = "sEMG Grasp Intention Classification"
    BENCHMARK_SPEC = "Physical Object Benchmark Parameters"


@dataclass
class DatasetMetadata:
    """Metadata specification for a public or internal dataset."""
    key: str
    name: str
    source_url: str
    citation: str
    license_type: str
    license_url: str
    data_nature: DataNature
    primary_ml_task: MLTask
    sensor_modality: str
    file_formats: list[str]
    expected_labels: list[str]
    download_size_approx: str
    direct_download_url: Optional[str]
    access_requirements: str
    local_raw_relpath: str
    local_processed_relpath: str
    description: str

    @property
    def local_raw_dir(self) -> str:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "raw", self.local_raw_relpath))
        return base

    @property
    def local_processed_dir(self) -> str:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed", self.local_processed_relpath))
        return base

    def is_downloaded(self) -> bool:
        """Checks if raw files or archive exist locally."""
        if not os.path.exists(self.local_raw_dir):
            return False
        items = os.listdir(self.local_raw_dir)
        return len(items) > 0

    def is_processed(self) -> bool:
        """Checks if processed feature tables exist."""
        if not os.path.exists(self.local_processed_dir):
            return False
        items = [f for f in os.listdir(self.local_processed_dir) if f.endswith(('.csv', '.parquet', '.npz'))]
        return len(items) > 0


class DatasetRegistry:
    """Central registry holding all investigated datasets for GripWise."""

    def __init__(self):
        self._datasets: dict[str, DatasetMetadata] = {}
        self._register_default_datasets()

    def _register_default_datasets(self) -> None:
        # 1. Dataset A: Bristol Biomimetic Tactile Slip Dataset
        self.register(DatasetMetadata(
            key="bristol_slip",
            name="Bristol Biomimetic Tactile Slip Dataset",
            source_url="https://data.bris.ac.uk/data/dataset/1fbyzhdft4rhp2iyptqfbf2mp4",
            citation="Lloyd, J., Lepora, N. et al., 'Biomimetic Tactile Slip Detection Dataset', University of Bristol Research Data Repository, 2017.",
            license_type="Non-Commercial Government Licence / University of Bristol Open Data",
            license_url="https://data.bris.ac.uk/data/dataset/1fbyzhdft4rhp2iyptqfbf2mp4",
            data_nature=DataNature.REAL_RESEARCH_DATA,
            primary_ml_task=MLTask.SLIP_DETECTION,
            sensor_modality="TacTip Optical Biomimetic Tactile Sensor (127 papillae / pin coordinates)",
            file_formats=[".mat (MATLAB v5/v7)", ".txt", ".json"],
            expected_labels=["Stable Contact (Pre-slip)", "Dynamic Micro/Macro Slip", "Displacement Trajectory"],
            download_size_approx="19.86 MB (ZIP)",
            direct_download_url="https://data.bris.ac.uk/datasets/1fbyzhdft4rhp2iyptqfbf2mp4/1fbyzhdft4rhp2iyptqfbf2mp4.zip",
            access_requirements="Direct Open Download (No registration required)",
            local_raw_relpath="bristol_slip",
            local_processed_relpath="bristol_slip",
            description="Real physical recordings of biomimetic TacTip optical tactile sensor capturing 127 tracked elastomeric pins undergoing flat surface shear slip."
        ))

        # 2. Dataset B: IntelliMan Contact and Slippage Dataset
        self.register(DatasetMetadata(
            key="intelliman_slip",
            name="IntelliMan Contact and Slippage Dataset (ICECS 2025)",
            source_url="https://zenodo.org/records/18759556",
            citation="IntelliMan Horizon Europe Project (WP5), 'Contact and Slippage Dataset', IEEE ICECS 2025, Zenodo, DOI: 10.5281/zenodo.18759556.",
            license_type="Creative Commons Attribution 4.0 International (CC-BY-4.0)",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            data_nature=DataNature.REAL_RESEARCH_DATA,
            primary_ml_task=MLTask.SLIP_DETECTION,
            sensor_modality="PVDF Piezoelectric Tactile Sensors (8 continuous voltage channels)",
            file_formats=[".txt (Tab-separated ASCII time-series)", ".zip"],
            expected_labels=["0: Transition/Trigger", "1: Static Contact", "2: Dynamic Slippage", "3: Separation/Rest"],
            download_size_approx="73.35 MB (ZIP)",
            direct_download_url="https://zenodo.org/records/18759556/files/Dataset_Contact_slippage_ICECS.zip?download=1",
            access_requirements="Direct Open Download (No credentials needed)",
            local_raw_relpath="intelliman_slip",
            local_processed_relpath="intelliman_slip",
            description="Physical tactile sensor recordings across 300 experimental runs (150 rough texture, 150 soft texture) with ground-truth contact and slippage phase annotations."
        ))

        # 3. Dataset C: IntelliMan Tactile Texture Dataset
        self.register(DatasetMetadata(
            key="intelliman_texture",
            name="IntelliMan Tactile Texture Dataset (PRIME 2023)",
            source_url="https://zenodo.org/records/18760184",
            citation="IntelliMan Horizon Europe Project (WP5), 'Tactile Texture Dataset', IEEE PRIME 2023, Zenodo, DOI: 10.5281/zenodo.18760184.",
            license_type="Creative Commons Attribution 4.0 International (CC-BY-4.0)",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            data_nature=DataNature.REAL_RESEARCH_DATA,
            primary_ml_task=MLTask.TEXTURE_CLASSIFICATION,
            sensor_modality="Dynamic Tactile Sensor Array (Multi-channel voltage)",
            file_formats=[".zip", ".txt", ".csv"],
            expected_labels=["Texture Type (1..N)", "Surface Hardness Class", "Sliding Velocity"],
            download_size_approx="194.63 MB (ZIP)",
            direct_download_url="https://zenodo.org/records/18760184/files/raw_data.zip?download=1",
            access_requirements="Direct Open Download (Zenodo)",
            local_raw_relpath="intelliman_texture",
            local_processed_relpath="intelliman_texture",
            description="Multi-surface tactile interaction recordings exploring dynamic surface friction and texture properties without conflating texture labels with slip."
        ))

        # 4. Dataset D: NinaPro Surface EMG and Movement Datasets
        self.register(DatasetMetadata(
            key="ninapro_emg",
            name="NinaPro Surface EMG Database (DB1 / DB2)",
            source_url="https://ninapro.hevs.ch/",
            citation="Atzori, M. et al., 'Electromyography data for non-invasive naturally-controlled prosthetics', Scientific Data 1:140053, Nature, 2014.",
            license_type="Open Academic Non-Commercial Research License",
            license_url="https://ninapro.hevs.ch/license",
            data_nature=DataNature.REAL_RESEARCH_DATA,
            primary_ml_task=MLTask.EMG_INTENT,
            sensor_modality="sEMG Electrodes (10-12 channels, OttoBock / Delsys Trigno) + Cyberglove Kinematics",
            file_formats=[".mat (MATLAB)", ".zip"],
            expected_labels=["Movement ID (1..52)", "Repetition ID (1..10)", "Rest", "Force Target (DB3/DB8)"],
            download_size_approx="1.2 GB - 8.5 GB per Database",
            direct_download_url=None,  # Requires NinaPro web authentication portal
            access_requirements="User registration required on https://ninapro.hevs.ch/ (Free academic access)",
            local_raw_relpath="ninapro_emg",
            local_processed_relpath="ninapro_emg",
            description="Gold-standard non-invasive sEMG benchmark recording forearm muscle activation across intact and amputee subjects for robotic prosthetic control."
        ))

        # 5. Dataset E1: GripWise Synthetic Force Regression Dataset
        self.register(DatasetMetadata(
            key="gripwise_force_synthetic",
            name="GripWise Synthetic Force-Regression Benchmark",
            source_url="https://github.com/GripWise/GripWise-Prosthetic-Hand (Local Demonstration)",
            citation="GripWise Research Project, 'Minimum-Safe-Force Modeling via Synthetic Mechanics & Elasticity', 2026.",
            license_type="MIT License (Open Source)",
            license_url="https://opensource.org/licenses/MIT",
            data_nature=DataNature.SYNTHETIC_DATA,
            primary_ml_task=MLTask.FORCE_REGRESSION,
            sensor_modality="Simulated Piezoresistive Fingertip Sensors (Mass, Friction, Compliance, Fragility, Margin)",
            file_formats=[".csv"],
            expected_labels=["target_force_n (Continuous Scalar, Newtons)"],
            download_size_approx="120 KB",
            direct_download_url=None,  # Generated locally in workspace
            access_requirements="Locally generated on demand (No external download needed)",
            local_raw_relpath="synthetic_force",
            local_processed_relpath="synthetic_force",
            description="1,500 synthetic samples modeling non-linear dual-shear static friction mechanics and structural compliance for initial candidate clamp force prediction."
        ))

        # 6. Dataset E2: Yale-CMU-Berkeley (YCB) Object Benchmark Parameters
        self.register(DatasetMetadata(
            key="ycb_objects",
            name="Yale-CMU-Berkeley (YCB) Object & Grasp Benchmark",
            source_url="https://www.ycbbenchmarks.com/",
            citation="Calli, B. et al., 'Benchmarking in Manipulation Research: The YCB Object and Model Set', IEEE Robotics & Automation Magazine, 2015.",
            license_type="Creative Commons Attribution 4.0 International (CC-BY-4.0)",
            license_url="https://www.ycbbenchmarks.com/",
            data_nature=DataNature.REAL_RESEARCH_DATA,
            primary_ml_task=MLTask.BENCHMARK_SPEC,
            sensor_modality="Physical calibrated mass, 3D geometry, and measured static friction coefficients",
            file_formats=[".csv", ".json", ".obj"],
            expected_labels=["Object Mass (g)", "Friction mu", "Grasp Feasibility", "Dimensions (mm)"],
            download_size_approx="1.5 MB (Metadata & Specifications)",
            direct_download_url="https://raw.githubusercontent.com/ycb-benchmarks/ycb-tools/master/models/models.json",
            access_requirements="Open Direct Access",
            local_raw_relpath="ycb_objects",
            local_processed_relpath="ycb_objects",
            description="Calibrated physical reference models of everyday manipulation objects used to ground prosthetic hand grip force and stability studies."
        ))

    def register(self, meta: DatasetMetadata) -> None:
        self._datasets[meta.key] = meta

    def get_dataset(self, key: str) -> Optional[DatasetMetadata]:
        return self._datasets.get(key)

    def list_datasets(self) -> list[DatasetMetadata]:
        return list(self._datasets.values())

    def get_summary_table(self) -> list[dict]:
        """Returns clean tabular summary of all datasets and status."""
        rows = []
        for d in self.list_datasets():
            rows.append({
                "Key": d.key,
                "Dataset Name": d.name,
                "Nature": d.data_nature.value,
                "Primary Task": d.primary_ml_task.value,
                "Sensor Modality": d.sensor_modality,
                "License": d.license_type,
                "Download Status": "✅ Downloaded" if d.is_downloaded() else "⏳ Not Downloaded",
                "Processed Status": "✅ Ready" if d.is_processed() else "⏳ Unprocessed",
                "Size": d.download_size_approx
            })
        return rows

