# 🦾 GripWise: Low-Cost Self-Learning Prosthetic Hand
## Multi-Dataset Research Platform, Cyber-Physical Digital Twin & AI Demonstration Suite

> **Personalized Grip Strategies, Minimum-Safe-Force Prediction, and Multimodal Tactile Slip Detection**

---

## 📌 Executive Summary

**GripWise** is an open research software platform, cyber-physical Digital Twin, and AI demonstration suite for an intelligent 5-finger articulated prosthetic hand. Designed for assistive robotics research, bio-mechatronics education, and rapid algorithmic prototyping, GripWise bridges the gap between simulated mechanical grip dynamics and real-world tactile/electromyographic research datasets.

The platform provides five fully integrated operating decks switchable via the sidebar navigation:

1. **🦾 Digital Twin Simulation**: An interactive cyber-physical simulator featuring real-time parametric SVG hand kinematics, contact force simulation, autonomous micro-slip recovery control, personalization preference learning, live telemetry plotting, and the **End-to-End AI Demonstration Scenario Runner**.
2. **🔌 Hardware Integration & Test Bench**: A comprehensive hardware test bench featuring:
   - **Actuator Power & Safety Watchdog**: Master Arm / Disarm controls, Software Emergency Stop, and maximum clamp limit setting ($35.0\,\text{N}$).
   - **4 Operating Modes**: Hardware-Free Mock Serial Stream (with packet corruption, stream freeze, noise, and drift fault injection), Telemetry Replay Engine (deterministic frame playback from CSV), Physics Simulation, and ESP32 PySerial connection.
   - **Real-Time Microcontroller Packet Inspector**: Displays raw incoming ASCII string, live checksum status, latency, and per-finger force distribution.
   - **Microcontroller Firmware Specification & Wire Protocol Reference**.
3. **🧪 Offline Research Inference Studio**: An interactive inference workbench that evaluates trained machine learning models on **verified held-out physical research datasets** (IntelliMan PVDF dynamic tactile skin & Bristol TacTip biomimetic optical array), displaying model provenance, extracted features, ground-truth labels, and confidence distributions without cross-modality contamination.
4. **⚖️ Model vs. Baseline Comparison Deck**: An empirical comparative evaluation contrasting **Model A (Random Forest Regressor)** against the **Physics Analytical Baseline**, and **Models B1 & B2 (Tactile Slip Classifiers)** against the **Classical Normal-Force Threshold Detector**.
5. **🔬 Data & AI Research Lab**: A reproducible research workbench integrating public benchmark datasets, automated chunked downloading, leak-free feature extraction pipelines, modular machine learning training, and cross-validated benchmark evaluations.

```
       ┌────────────────────────────────────────────────────────────────────────┐
       │                       GripWise Research Suite                          │
       └───────────────────────────────────┬────────────────────────────────────┘
                                           │
         ┌───────────────────┬─────────────┴─────────────┬────────────────────┐
         ▼                   ▼                           ▼                    ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐┌──────────────────┐
│  🦾 Digital Twin ││ 🧪 Offline Studio││ ⚖️ Model vs Base ││ 🔬 Data & AI Lab │
│  - 5-Finger SVG  ││  - Real PVDF Test││  - Force Regressor││  - Registry      │
│  - End-to-End AI ││  - TacTip Optical││    vs Heuristic  ││  - Downloader    │
│  - Adaptive Boost││  - sEMG Gestures ││  - Slip RF vs    ││  - Preprocessor  │
│  - CSV Export    ││  - Feature Schema││    Threshold Det ││  - Evaluator     │
└──────────────────┘└──────────────────┘└──────────────────┘└──────────────────┘
```

---

## 🚀 End-to-End AI Demonstration Scenario

The system includes a fully automated, reproducible 8-stage demonstration scenario implemented in [`demo_scenario.py`](demo_scenario.py):

### The 8-Stage Sequence
1. **Object & User Preference Selection**: Loads test object parameters (e.g. Ceramic Cup, mass: $240\,\text{g}$, $\mu=0.42$, compliance: $0.04\,\text{mm/N}$, fragility: $38\,\text{N}$) and the active user personalization profile (e.g. Alex, $1.35\times$ safety margin).
2. **Force Prediction (Model A)**: Evaluates mechanical features `[mass_g, friction_coeff, compliance_mm_per_n, fragility_limit_n, user_safety_margin]` through the trained Random Forest regressor to compute the candidate minimum-safe-force target ($F_{\text{target}} \approx 3.85\,\text{N}$).
3. **Fingers Articulation (Approach)**: Actuator drives 5-finger kinematics in anatomical cascade (Thumb leads, fingers follow) from `OPEN` ($0\%$) to `CLOSING` ($45\%$).
4. **Contact & Grasp Stabilization**: Piezoresistive tactile pads detect contact ($F \ge 0.40\,\text{N}$), ramp to the predicted target, and stabilize inside the safe operating corridor (`STABLE_GRIP`).
5. **Micro-Slip Shear Injected**: An external tangential perturbation is simulated (`SLIPPING`), causing a tactile shear alert.
6. **Adaptive Closed-Loop Recovery**: Controller detects slip and dynamically steps up the commanded normal force by $+\Delta F$ ($+0.55\,\text{N} \to 4.40\,\text{N}$) while verifying headroom against the structural fragility limit ($38\,\text{N}$).
7. **Grasp Re-Stabilized**: Elevated clamp force suppresses slip displacement, re-stabilizing in `STABLE_GRIP`.
8. **Release & Trial Logging**: Commands opening back to `OPEN`, logs full trial telemetry to `experiment_log.json` and updates profile adaptation.

### How to Run the Demonstration Scenario

#### Via Interactive UI:
1. Launch dashboard: `streamlit run app.py`
2. In sidebar, ensure **🦾 Digital Twin Simulation** is selected.
3. At the top of the controls deck, click **▶ Run End-to-End AI Demo**.
4. Inspect the real-time SVG hand articulation, Plotly force trajectory, and open the **Detailed Execution Log** expander to inspect the 8-stage step table.

#### Programmatically via Python:
```bash
python -c "from demo_scenario import execute_end_to_end_demonstration; summary = execute_end_to_end_demonstration(); print(f'Outcome: {summary.outcome}, Logged: {summary.trial_id_logged}, Steps: {len(summary.steps_log)}')"
```

---

## 🧪 Offline Research Inference Studio

To preserve strict scientific domain validity, machine learning models trained on physical sensor datasets are **never fed arbitrary simulated scalar FSR readings**.

Instead, the **Offline Research Inference Studio** loads verified held-out test recordings:

1. **Model B1: IntelliMan PVDF Dynamic Slip Classifier**
   - **Modality**: 8-Channel PVDF Piezoelectric Film ($1\,\text{kHz}$ dynamic vibration).
   - **Features**: 40 multi-channel statistical features (Vibration Energy, PTP Amplitude, RMS, Std Dev).
   - **Test Set**: 1,287 held-out window frames from 60 unseen physical trials (ICECS 2025).
   - **Output**: Predicted class (`Stable Contact` vs `Dynamic Slip`), slip probability gauge ($0-100\%$), and baseline comparison.
2. **Model B2: Bristol TacTip Optical Slip Classifier**
   - **Modality**: 127 tracked internal biomimetic papillae displacement trajectories.
   - **Features**: Pin velocity magnitude, standard deviation, max speed, and spatial directional coherence ($R$).
   - **Test Set**: 2,640 held-out frames across 20 independent test runs.
   - **Output**: Papillae slip classification with directional coherence diagnostics.
3. **Model C: 3-Phase Contact & Grasp State Classifier**
   - Classifies physical PVDF sensor windows into `Phase 0: Rest / Detached`, `Phase 1: Static Contact`, and `Phase 2: Dynamic Slippage`.
4. **Model D: NinaPro sEMG Movement Intention Classifier**
   - Classifies 8-channel forearm sEMG envelopes into 5 functional grasp gestures using subject-independent holdout (Subject 5).

---

## ⚖️ Model vs. Baseline Benchmark Comparisons

### Comparison 1: Force Model A vs. Physics Analytical Baseline
- **Physics Baseline Formula**:
  $$F_{\text{baseline}} = \frac{m \cdot g}{2 \cdot \mu_s} \times (1 + 0.15 \cdot \text{compliance}) \times \gamma_{\text{margin}}$$
- **Model A (Random Forest Regressor)**:
  Trained on $N=1,500$ multi-parameter interactions ($R^2 = 0.9628$, $\text{MAE} = 0.657\,\text{N}$).
- **Key Finding**: While the analytical baseline assumes rigid linear elasticity, Model A automatically moderates clamp forces for fragile compressible objects (e.g. Cardboard Box, Sponge) to prevent crushing while providing adequate safety margins for low-friction objects (e.g. Plastic Bottle). Model A achieves a **48.8% error reduction** over the static physics formula.

### Comparison 2: Dynamic Slip Classifiers vs. Static Threshold Detector
- **Static Threshold Controller**: Flags slip only when normal force drops below $F_{\text{min}}$.
  - *Vulnerability*: Blind to tangential shear slip where normal force remains temporarily steady while the object slides. False Negative Rate is $\sim 85\%$ during micro-shear displacement.
- **Model B1 (PVDF Dynamic Vibration)**: Senses acoustic shear oscillations within $20\,\text{ms}$ of slip onset. Held-out accuracy: **85.08%**, F1: **0.781**, ROC-AUC: **0.912**.
- **Model B2 (TacTip Optical Kinematics)**: Senses coordinated spatial marker divergence. Held-out accuracy: **99.24%**, F1: **0.889**, ROC-AUC: **1.000**.

---

## 🏗️ Architecture & Module Organization

```text
GripWise/
│
├── app.py                      # Main Streamlit orchestration & 5-deck view router
├── hardware_view.py            # Hardware Integration & Test Bench view (Mock, Replay, E-Stop)
├── demo_scenario.py            # Reproducible 8-stage End-to-End AI Demonstration Engine
├── offline_inference_view.py   # Offline Research Inference Studio (Models B1, B2, C, D)
├── model_comparison_view.py    # Model vs. Baseline Benchmark Comparison Deck
├── data_lab_view.py            # Data & AI Research Lab UI (6 interactive tabs)
├── data_manager.py             # Singleton data, model, and metric coordinator
│
├── dataset_registry.py         # Central dataset catalog, metadata & licensing
├── dataset_download.py         # Streaming downloader with retry & auto-unzip
├── dataset_preprocessing.py    # Leak-free feature extraction adapters (PVDF, TacTip, EMG)
│
├── slip_model.py               # Models B1 (PVDF Slip) and B2 (TacTip Slip)
├── grasp_model.py              # Model C (3-Phase Contact State Classifier)
├── emg_model.py                # Model D (sEMG Movement Intent Classifier)
├── prediction.py               # Model A (Minimum-Safe-Force Regressor)
├── model_evaluation.py         # Leakage-resistant evaluator & confusion matrices
│
├── controller.py               # Closed-loop adaptive controller, legal transitions, watchdog
├── simulation.py               # Hand kinematics, object specifications & physics
├── virtual_hand.py             # Dynamic SVG prosthetic hand visualizer
├── personalization.py          # User profiles, preference adaptation, filtering, backups
├── hardware.py                 # Cyber-physical layer: PacketParser, mock stream, replay, PySerial
│
├── data/                       # Data storage
│   ├── raw/                    # Original zip archives and extracted source files
│   ├── processed/              # Extracted leak-free train/test feature CSVs
│   ├── telemetry_recordings/   # Pre-recorded telemetry CSV traces for replay testing
│   └── backups/                # Timestamped JSON experiment & profile snapshot backups
│
├── models/saved/               # Serialized scikit-learn model weights (.joblib)
├── evaluation/metrics/         # Evaluation metric JSONs and confusion matrix arrays
│
├── tests/                      # Automated unit and integration test suite
│   └── test_pipeline_research.py # 26 comprehensive tests (100% pass)
│
├── SOFTWARE_STATUS.md          # 5-tier software readiness & physical separation audit
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation & research report
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Run the Research Test Suite
Execute all 26 automated unit and integration tests:
```bash
python tests/test_pipeline_research.py
# or
python -m unittest discover tests
```

### 4. Launch the Dashboard
Run the dashboard with a single command:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Use the sidebar radio buttons to navigate between:
- **🦾 Digital Twin Simulation**: Test kinematics, run grasp cycles, inject micro-slips, adjust personalized safety margins, or click **▶ Run End-to-End AI Demo**.
- **🔌 Hardware Integration & Test Bench**: Manage actuator safety arming, test cyber-physical fault injection via Mock Serial, replay telemetry from CSV, or monitor raw ESP32 packets.
- **🧪 Offline Research Inference Studio**: Select real physical recordings from held-out public datasets and run live inference with Model B1, B2, C, or D.
- **⚖️ Model vs Baseline Comparison**: Inspect empirical benchmarks comparing ML models against analytical and threshold baselines.
- **🔬 Data & AI Research Lab**: Download datasets, execute leak-free preprocessing, retrain models, and inspect metrics.

---

## 🔌 Hardware-Ready Layer & Microcontroller Protocol

The codebase uses a cyber-physical hardware abstraction layer ([`hardware.py`](hardware.py)). When physical hardware is procured, an ESP32 or Arduino microcontroller can replace the simulated stream with zero changes to controller or UI code.

### Standard Wire Protocol Specification (115200 Baud, UART/USB)
Incoming telemetry is parsed by [`PacketParser`](hardware.py). Every 20 milliseconds, the microcontroller outputs a newline-terminated comma-delimited ASCII packet:
```text
<total_force>,<slip_flag>,<thumb_force>,<index_force>,<middle_force>,<ring_force>,<pinky_force>,<current_ma>\n
```
*Example:* `3.85,0,1.54,0.85,0.77,0.42,0.27,245.0\n`

### Microcontroller Command Ingestion
The dashboard sends commands to the microcontroller:
| Command | Effect |
| :--- | :--- |
| `ARM\n` | Arms motor command lines; enables positive force targets. |
| `DISARM\n` | Disarms motor driver; sets commanded target to 0.0 N. |
| `TARGET:<f>\n` | Commands clamp force (e.g., `TARGET:3.85\n`). Clamped by software limit. |
| `EMERGENCY_STOP\n` | Immediate actuator shutdown; drives PWM to 0% and latches disarmed state. |
| `INJECT_SLIP\n` | (Bench/Test mode) Signals test bench to induce shear displacement. |

### Hardware Readiness Audit
For an in-depth breakdown of software implementation, synthetic tests, public dataset dependencies, and items requiring physical hardware bench testing, refer to **[`SOFTWARE_STATUS.md`](SOFTWARE_STATUS.md)**.

---

## ⚠️ Scientific Integrity & Disclaimers

> [!WARNING]
> **RESEARCH & SIMULATION PROTOTYPE ONLY**
> 1. **Synthetic vs. Real Data**: Model A (force target regression) is trained on synthetic demonstration distributions derived from mechanical friction heuristics. Models B1, B2, C, and D are trained on real, openly licensed academic research datasets.
> 2. **No Clinical Validation**: Results, force targets, slip probabilities, and intention classifications are **NOT clinically certified** for patient fitting or functional prosthesis use.
> 3. **Hardware Deployment**: Physical operation requires calibrated load cells, analog conditioning circuits, real-world bench testing, and electrical emergency limit switches.
> 4. **No Waterproof / Harsh Environment Rating**: Not rated for moisture ingress or high-impact industrial cycles.

---

## 📄 Licensing & Citations
- **GripWise Platform Code**: MIT License.
- **Bristol Tactile Slip Dataset**: Open Data under University of Bristol Non-Commercial Research Licence (Ward-Cherrier et al., 2018).
- **IntelliMan Contact & Slippage / Texture Datasets**: Creative Commons Attribution 4.0 International (CC-BY 4.0; D'Abbraccio et al., 2025; Zenodo).
- **NinaPro EMG Dataset**: NinaPro Non-Commercial Research License (Atzori et al., 2014; HEVS).
