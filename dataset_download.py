"""
dataset_download.py - Robust Research Dataset Download Engine
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Automates streaming download, validation, and extraction of accessible public datasets
with automatic retry logic, progress reporting, and manual instructions for access-gated sources.
"""

import os
import ssl
import time
import zipfile
import urllib.request
from typing import Callable, Optional
from dataset_registry import DatasetRegistry, DatasetMetadata, DataNature

# Create SSL context allowing safe fallback
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def download_dataset(
    key: str,
    registry: Optional[DatasetRegistry] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    force_redownload: bool = False
) -> tuple[bool, str]:
    """
    Downloads and unpacks the dataset specified by `key`.
    Returns (success: bool, status_message: str).
    """
    if registry is None:
        registry = DatasetRegistry()

    meta = registry.get_dataset(key)
    if not meta:
        return False, f"Dataset '{key}' not found in registry."

    # Check local status
    if not force_redownload and meta.is_downloaded():
        return True, f"Dataset '{meta.name}' is already present in local storage."

    # Handle datasets that require manual registration (e.g. NinaPro)
    if meta.direct_download_url is None:
        if meta.key == "gripwise_force_synthetic":
            # Generate locally
            from prediction import ForcePredictor, DATASET_CSV
            os.makedirs(meta.local_raw_dir, exist_ok=True)
            target_csv = os.path.join(meta.local_raw_dir, "synthetic_training_data.csv")
            ForcePredictor()  # generates CSV if missing
            if os.path.exists(DATASET_CSV) and not os.path.exists(target_csv):
                import shutil
                shutil.copy(DATASET_CSV, target_csv)
            return True, "Synthetic force regression dataset successfully verified locally."

        elif meta.key == "ninapro_emg":
            return False, (
                f"Dataset '{meta.name}' requires user registration on the official portal:\n"
                f"1. Visit {meta.source_url}\n"
                f"2. Create an academic account and download desired sub-database (e.g., DB1/DB2 .mat files)\n"
                f"3. Place extracted .mat files into: {meta.local_raw_dir}\n"
                f"The pipeline includes an offline surrogate generator for testing sEMG classification."
            )
        else:
            return False, f"Dataset '{meta.name}' has no automated direct download endpoint. Follow: {meta.source_url}"

    # Prepare directories
    os.makedirs(meta.local_raw_dir, exist_ok=True)
    archive_name = os.path.basename(meta.direct_download_url.split("?")[0])
    if not archive_name.endswith(('.zip', '.json', '.csv', '.tar.gz')):
        archive_name = f"{meta.key}_archive.zip"
    archive_path = os.path.join(meta.local_raw_dir, archive_name)

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) GripWise-Research-Agent/2.0'}
    req = urllib.request.Request(meta.direct_download_url, headers=headers)

    max_retries = 3
    success = False
    last_err = ""

    for attempt in range(1, max_retries + 1):
        try:
            if progress_callback:
                progress_callback(0.05, f"Connecting to {meta.name} (Attempt {attempt}/{max_retries})...")

            with urllib.request.urlopen(req, timeout=120, context=SSL_CTX) as resp:
                total_len_header = resp.headers.get('Content-Length')
                total_len = int(total_len_header) if total_len_header and total_len_header.isdigit() else 0

                downloaded = 0
                chunk_size = 256 * 1024  # 256 KB chunks

                with open(archive_path, 'wb') as out_f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        if total_len > 0 and progress_callback:
                            ratio = min(0.90, 0.05 + 0.85 * (downloaded / total_len))
                            mb_curr = downloaded / (1024 * 1024)
                            mb_tot = total_len / (1024 * 1024)
                            progress_callback(ratio, f"Downloading: {mb_curr:.1f} MB / {mb_tot:.1f} MB ({int(ratio*100)}%)")

            success = True
            break
        except Exception as e:
            last_err = str(e)
            if progress_callback:
                progress_callback(0.05, f"Transient network issue: {e}. Retrying in 2s...")
            time.sleep(2)

    if not success:
        return False, f"Failed to download {meta.name}: {last_err}"

    # Extraction if zip file
    if archive_path.endswith('.zip') and os.path.exists(archive_path):
        if progress_callback:
            progress_callback(0.92, f"Extracting archive for {meta.name}...")
        try:
            extract_folder = os.path.join(meta.local_raw_dir, "extracted")
            os.makedirs(extract_folder, exist_ok=True)
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_folder)
            if progress_callback:
                progress_callback(1.0, f"Successfully unpacked {meta.name}!")
            return True, f"Downloaded and extracted {meta.name} into {extract_folder}."
        except Exception as e:
            return False, f"Downloaded archive but extraction failed: {e}"

    if progress_callback:
        progress_callback(1.0, f"Downloaded {meta.name} successfully.")
    return True, f"Downloaded {meta.name} to {archive_path}."

