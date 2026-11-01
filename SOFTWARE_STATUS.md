# GripWise Software Maturity & Implementation Status Report

> **Academic & Experimental Disclaimer**  
> *GripWise is an open academic software research prototype designed to explore adaptive prosthetic grasp strategies, ML-assisted force estimation, and tactile/sEMG signal processing.*  
> **No clinical validation, patient trials, or physical hardware tests have been conducted.** All sensor readings, force dynamics, and tactile feedback in this prototype are generated via physics simulations, deterministic replay buffers, mock stream generators, or processed from publicly available benchmark datasets. Real-time physical sensing, biological muscle signals, and physical ESP32 microcontrollers are strictly simulated or staged for future cyber-physical integration.

---

## Software Maturity Categorization

To maintain scientific integrity and prevent conflating running software algorithms with physically measured hardware behavior, the software deliverables are categorized into five distinct readiness levels:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Category 1: Fully Implemented in Software                                  │
│ Category 2: Tested Using Synthetic or Replayed Data                         │
│ Category 3: Dependent on Public Datasets                                    │
│ Category 4: Ready for Future Hardware Integration (Not Physically Tested)   │
│ Category 5: Requires Physical Hardware or Experimental Validation           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Category 1: Fully Implemented in Software

These components are 100% written, functionally self-contained, unit-tested, and execute without requiring external hardware or public dataset files:

1. **Digital Twin 5-Finger SVG Articulation Renderer (`virtual_hand.py`)**:
   - Parametric vector graphics rendering 5 distinct anatomical fingers (Thumb, Index, Middle, Ring, Pinky), palm, and wrist anchor.
   - Dynamic joint bending, tendon flex lines, fingertip contact pads, and color-coded slip warnings across 6 operational states (`OPEN`, `CLOSING`, `GRASPING`, `STABLE_GRIP`, `SLIPPING`, `RELEASING`).
   - Pure SVG output embedded safely via HTML encapsulation in Streamlit.

2. **Adaptive Grip State Machine & Controller (`controller.py`)**:
   - Strict transition table validation (`LEGAL_TRANSITIONS` graph) preventing illicit jumps between non-adjacent operational states.
   - Closed-loop proportional adaptation with slip increment boost (+0.35 N nominal, capped at object fragility headroom).
   - Dynamic grasp infeasibility detector calculating theoretical slip threshold $F_{\text{min\_slip}} = \frac{m \cdot g}{2\mu}$; flags a critical warning if $F_{\text{min\_slip}} > F_{\text{fragility}}$.
   - Software watchdog timer that monitors stale telemetry packets and halts actuation if 5 consecutive stalls occur.

3. **User Personalization & Experiment Profile Manager (`personalization.py`)**:
   - Complete Profile CRUD (create, read, update, delete) supporting 4 preset grip personas (`Delicate`, `Standard`, `Firm`, `Custom`).
   - Self-learning adaptation bias updating incrementally based on post-trial qualitative feedback.
   - Experiment trial logging with multi-parameter filtering (by Object, Profile, Outcome, Keyword search).
   - Bidirectional JSON import/export with schema validation and error reporting.
   - Snapshot backup and restore subsystem storing timestamped archives in `data/backups/`.

4. **Multi-View Streamlit Research Application (`app.py`, `hardware_view.py`, `offline_inference_view.py`, `model_comparison_view.py`, `data_lab_view.py`)**:
   - Cohesive 5-view navigation deck:
     - `Digital Twin`: Interactive grasp execution, live force telemetry, SVG hand, manual slip injection, and trial logger.
     - `Hardware Integration & Test Bench`: Actuator power gating (Arm/Disarm/Emergency Stop), operating mode tabs (Mock Serial, Telemetry Replay, Physics Simulation, ESP32 Serial), packet inspector with raw ASCII display and per-finger force bars.
     - `Offline Research Inference`: Public dataset inspector for held-out sample inference on Models B1, B2, C, and D.
     - `Model Comparison`: Side-by-side benchmark comparison (Model A vs Physics Baseline; Models B1/B2 vs Static Threshold).
     - `Data & AI Lab`: Dataset registry audit, preprocessing pipeline triggers, and training summaries.

5. **End-to-End Automated Demonstration Engine (`demo_scenario.py`)**:
   - 8-stage deterministic demonstration executing: Object Selection → Model Inference → Pre-Grasp Flexion → Tactile Grasp → Stable Hold → Micro-Slip Injection → Adaptive Force Recovery → Release & Trial Logging.

---

### Category 2: Tested Using Synthetic or Replayed Data

These modules operate deterministically in software and have been verified against synthetic models, simulated physics, or pre-recorded telemetry files:

1. **Minimum-Safe-Force Regression Model A (`prediction.py`, `gripwise_rf_model.joblib`)**:
   - Scikit-Learn `RandomForestRegressor` trained on 1,200 synthetic parameter combinations.
   - Verified on a 20% held-out test split achieving $R^2 \approx 0.963$ and $\text{MAE} \approx 0.163\text{ N}$.
   - Evaluated alongside analytical physics baseline $F_{\text{baseline}} = \frac{m \cdot g}{2\mu} \times S$.

2. **Prosthetic Hand Physics Simulation (`simulation.py`, `hardware.py`)**:
   - `SimulatedSensorInterface`: First-order exponential force response modeling compliance ($\alpha \approx 0.35$), Gaussian measurement noise ($\sigma = 0.04\text{ N}$), mechanical drift, and micro-slip force drops.
   - 5 standard object specifications (`Cup`, `Bottle`, `Ball`, `Box`, `Sponge`) with distinct friction, mass, compliance, and fragility limits.

3. **Telemetry Replay Engine (`RecordedReplaySensorInterface` in `hardware.py`)**:
   - Replays pre-recorded CSV telemetry traces (`data/telemetry_recordings/demo_slip_recording.csv`) frame-by-frame.
   - Provides deterministic reproduction of micro-slip events, motor current draw, and per-finger force distributions without live sensor hardware.

4. **Hardware-Free Mock Serial Stream with Fault Injection (`MockSerialSensorInterface` in `hardware.py`)**:
   - Emulates microcontroller UART byte streaming at 115200 baud.
   - Injects controllable cyber-physical faults: corrupt checksum/bytes, UART stream freeze/stall, high-amplitude white noise, and thermal sensor drift.

---

### Category 3: Dependent on Public Datasets

These machine learning models and data preprocessing pipelines are implemented and functional, but their training and evaluation depend on the presence of public research datasets:

1. **Bristol TacTip Optical Tactile Slip Classifier - Model B1 (`slip_model.py`, `dataset_preprocessing.py`)**:
   - **Source**: University of Bristol Biomimetic Tactile Slip Dataset (Open Data Bristol).
   - **Features**: 252 features per sample (mean, std, max, ptp, energy, skew, kurtosis across 36 centroid-tracked tactile pins).
   - **Split**: Recording-level split (0 recording overlap between training and held-out test partitions).
   - **Status**: Preprocessing pipeline and classifier implemented. Operates in synthetic demonstration mode if raw `.mat` files are absent; executes full training and test evaluation when downloaded.

2. **IntelliMan PVDF Dynamic Contact & Slip Classifier - Model B2 (`slip_model.py`, `dataset_preprocessing.py`)**:
   - **Source**: Zenodo Record 18759556 (Creative Commons Attribution 4.0).
   - **Features**: 40 features (10 time-domain and frequency-domain statistics per PVDF dynamic piezopolymer tactile channel).
   - **Split**: 80/20 train/test split.
   - **Status**: Implemented and verified on held-out test splits.

3. **IntelliMan Contact-State Classifier - Model C (`grasp_model.py`, `dataset_preprocessing.py`)**:
   - **Source**: Zenodo Record 18760184.
   - **Features**: 3-state grasp phase prediction (`NO_CONTACT`, `STABLE_CONTACT`, `DYNAMIC_SLIP`).
   - **Status**: Preprocessing adapter and Random Forest classifier fully functional.

4. **NinaPro sEMG Gesture & Intention Classifier - Model D (`emg_model.py`, `dataset_preprocessing.py`)**:
   - **Source**: NinaPro DB1 / DB2 Benchmark (HEVS / University of Applied Sciences Western Switzerland).
   - **Features**: 40 temporal-spatial features extracted across 8-channel surface electromyography arrays (MAV, RMS, WL, VAR, SSI).
   - **Split**: Subject-independent holdout (training on Subject 1, testing on Subject 2) ensuring realistic cross-user generalization testing.
   - **Status**: Adapter and classifier implemented and evaluated in offline research view.

---

### Category 4: Ready for Future Hardware Integration (Not Physically Tested)

These software layers are completely implemented and wire-protocol verified, but await connection to a physical microcontroller and sensors:

1. **Standard Microcontroller ASCII Wire Protocol (`PacketParser` in `hardware.py`)**:
   - Parses newline-terminated comma-delimited ASCII strings:
     $$\langle F_{\text{total}}\rangle,\langle \text{slip}\rangle,\langle F_{\text{th}}\rangle,\langle F_{\text{idx}}\rangle,\langle F_{\text{mid}}\rangle,\langle F_{\text{rng}}\rangle,\langle F_{\text{pnk}}\rangle,\langle I_{\text{motor}}\rangle\backslash n$$
   - Rigorously tested against corrupted frames, truncated fields, negative forces, out-of-range readings ($>65.0\text{ N}$), NaN/Inf floats, and invalid slip flags.

2. **PySerial Microcontroller UART Driver (`SerialSensorInterface` in `hardware.py`)**:
   - Thread-safe serial port discovery, baud configuration (115200 baud default), background reading, and connection recovery.
   - Graceful fallback: If PySerial is not installed or no COM port is present, the app flags an informative diagnostic instead of crashing.

3. **Actuator Safety Arming & Software Emergency Stop**:
   - Actuators remain **disarmed by default** upon system boot.
   - Positive target forces are rejected by the driver until an explicit `ARM` command is sent.
   - Software limit clamps all commanded forces to a maximum configurable threshold ($35.0\text{ N}$).
   - Immediate Software Emergency Stop (`EMERGENCY_STOP`) instantly drops target force to $0.0\text{ N}$, latches disarmed state, and opens prosthetic fingers.

---

### Category 5: Requires Physical Hardware or Experimental Validation

These aspects **cannot** be completed in software and strictly require procurement of physical parts, 3D printing, bench instruments, and approved human subject protocols:

1. **Physical Sensor Calibration & Real Noise Characterization**:
   - Real piezoresistive FSR sensors exhibit non-linear resistance vs. force curves, hysteresis, and thermal drift that cannot be measured without physical hardware.
   - True slip detection requires physical high-frequency vibration sensing (e.g., PVDF dynamic films or optical TacTip silicone skin).

2. **Mechanical Compliances & Actuator Dynamics**:
   - Physical cable/tendon friction, gear backlash, servo stall currents, and 3D-printed flexure hinge fatigue can only be tuned through empirical bench testing.

3. **Biomechanical & Clinical Efficacy**:
   - Real surface EMG electrodes face skin impedance variability, electrode shifts, perspiration artifacts, and anatomical differences across amputee users.
   - Clinical grasp safety, slip prevention efficacy, and user comfort require rigorous physical trial protocols approved by institutional review boards (IRB).

---

## Component Traceability Matrix

| Component | File Location | Readiness Category | Hardware Required? | Test Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Digital Twin Hand SVG** | `virtual_hand.py` | Category 1 | No | Automated Unit Test & Visual |
| **Adaptive Controller** | `controller.py` | Category 1 | No | Automated Unit Test |
| **Infeasibility Detector** | `controller.py` | Category 1 | No | Automated Unit Test |
| **Personalization Profiles** | `personalization.py` | Category 1 | No | Automated Unit Test |
| **Trial Logging & Filter** | `personalization.py` | Category 1 | No | Automated Unit Test |
| **JSON Export / Import** | `personalization.py` | Category 1 | No | Automated Unit Test |
| **Snapshot Backup/Restore**| `personalization.py` | Category 1 | No | Automated Unit Test |
| **End-to-End Demo** | `demo_scenario.py` | Category 1 & 2 | No | Automated Unit Test |
| **Force Model A (RF)** | `prediction.py` | Category 2 | No | 20% Held-Out Split ($R^2=0.96$) |
| **Simulated Physics Engine**| `simulation.py`, `hardware.py` | Category 2 | No | Automated Unit Test |
| **Recorded Telemetry Replay**| `hardware.py` | Category 2 | No | Automated Unit Test |
| **Mock Stream & Faults** | `hardware.py` | Category 2 | No | Automated Unit Test |
| **Bristol TacTip Model B1** | `slip_model.py` | Category 3 | No (Public Dataset) | Recording-Level Split |
| **IntelliMan PVDF Model B2**| `slip_model.py` | Category 3 | No (Public Dataset) | Held-Out Split |
| **Contact Classifier Model C**| `grasp_model.py` | Category 3 | No (Public Dataset) | Held-Out Split |
| **NinaPro sEMG Model D** | `emg_model.py` | Category 3 | No (Public Dataset) | Subject-Independent Split |
| **Wire Protocol Parser** | `hardware.py` | Category 4 | No (Tested with Mock) | 100% Negative/Positive Tests |
| **Actuator Arming & E-Stop**| `hardware.py` | Category 4 | No (Tested with Mock) | Automated Unit Test |
| **PySerial UART Driver** | `hardware.py` | Category 4 | Yes (ESP32 for live) | Driver written & guarded |
| **Physical Sensor Bench** | Hardware Lab | Category 5 | Yes (Sensors & ESP32)| Awaiting Hardware Procurement |
| **Clinical Grasp Safety** | Clinical Lab | Category 5 | Yes (Prosthetic & Amputee)| Awaiting Clinical Protocols |

