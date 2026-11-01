"""
offline_inference_view.py - Interactive Offline Research Inference Studio
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Executes offline inference on correctly preprocessed held-out public research
samples for Models B1, B2, C, and D.

Strictly preserves scientific integrity:
- Models are evaluated ONLY on their true matching sensor feature representations.
- Arbitrary simulated FSR scalar readings are NEVER fed into PVDF or TacTip models.
- Features, labels, citations, and confidence scores are drawn from verified held-out data.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_manager import DataManager
from slip_model import IntelliManPVDFSlipClassifier, BristolTacTipSlipClassifier
from grasp_model import ContactGraspStateClassifier
from emg_model import EMGIntentionClassifier


def render_offline_inference_studio(dm: DataManager):
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <div>
            <h2 style="margin:0; font-size:1.55rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">
                🧪 Offline Research Inference Studio
            </h2>
            <div style="font-size:0.80rem; color:#94A3B8;">
                Interactive Machine Learning Inference on Verified Held-Out Public Research Datasets
            </div>
        </div>
        <div>
            <span style="background:rgba(6, 182, 212, 0.15); color:#22D3EE; border:1px solid #06B6D4; padding:4px 10px; border-radius:9999px; font-size:0.75rem; font-weight:700;">
                OFFLINE BENCHMARK SAMPLES
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(15, 23, 42, 0.85); border:1px solid #334155; border-left:4px solid #06B6D4; border-radius:6px; padding:10px 14px; margin-bottom:16px; font-size:0.80rem; color:#CBD5E1; line-height:1.5;">
        <strong>🔬 SCIENTIFIC DOMAIN INTEGRITY NOTICE:</strong><br/>
        This studio executes trained models on <strong>genuine physical test recordings</strong> held out from public datasets.
        In accordance with sensor domain segregation, the <strong>PVDF Dynamic Slip Classifier</strong> requires high-frequency piezoelectric vibration features,
        the <strong>Bristol TacTip Classifier</strong> requires 127 tracked optical marker kinematics, and the <strong>sEMG Classifier</strong> requires 8-channel muscle activation envelopes.
        <em>Arbitrary scalar FSR values from the Digital Twin are never coerced into these models.</em>
    </div>
    """, unsafe_allow_html=True)

    # 1. Model Selector Tabs
    model_tabs = st.tabs([
        "Model B1: IntelliMan PVDF Dynamic Slip",
        "Model B2: Bristol TacTip Optical Slip",
        "Model C: 3-Phase Contact State",
        "Model D: sEMG Movement Intention"
    ])

    # =========================================================================
    # TAB 1: MODEL B1 (INTELLIMAN PVDF SLIP)
    # =========================================================================
    with model_tabs[0]:
        _render_b1_pvdf_inference(dm)

    # =========================================================================
    # TAB 2: MODEL B2 (BRISTOL TACTIP SLIP)
    # =========================================================================
    with model_tabs[1]:
        _render_b2_tactip_inference(dm)

    # =========================================================================
    # TAB 3: MODEL C (CONTACT & GRASP-STATE 3-PHASE)
    # =========================================================================
    with model_tabs[2]:
        _render_c_contact_state_inference(dm)

    # =========================================================================
    # TAB 4: MODEL D (sEMG INTENTION)
    # =========================================================================
    with model_tabs[3]:
        _render_d_emg_inference(dm)


# -----------------------------------------------------------------------------
# HELPER: MODEL B1 INFERENCE
# -----------------------------------------------------------------------------
def _render_b1_pvdf_inference(dm: DataManager):
    test_csv_path = os.path.abspath("data/processed/intelliman_slip/intelliman_slip_test.csv")
    if not os.path.exists(test_csv_path):
        st.warning("⚠️ IntelliMan processed test dataset not found. Run preprocessing in the Data & AI Lab first.")
        return

    df_test = pd.read_csv(test_csv_path)
    clf = IntelliManPVDFSlipClassifier()
    if clf.model is None:
        clf.train_and_evaluate()

    st.markdown("##### 1. Select Held-Out Test Recording Window")
    c_sel1, c_sel2 = st.columns([3, 1])
    
    unique_trials = df_test["trial_file"].unique()
    with c_sel1:
        chosen_trial = st.selectbox(
            "Select Unseen Physical Trial File (Held-Out Test Set)",
            unique_trials,
            index=0,
            key="pvdf_trial_select"
        )
    
    df_trial_windows = df_test[df_test["trial_file"] == chosen_trial].reset_index(drop=True)
    
    with c_sel2:
        chosen_win_idx = st.selectbox(
            f"Window Frame (Total: {len(df_trial_windows)})",
            range(len(df_trial_windows)),
            index=0,
            key="pvdf_win_select"
        )

    sample_row = df_trial_windows.iloc[chosen_win_idx]
    actual_label = int(sample_row["is_slip"])
    actual_label_str = "⚠️ Dynamic Slip (Phase 2)" if actual_label == 1 else "✅ Static Contact / Stable"

    # Feature extraction matching model schema
    feat_cols = [c for c in clf.FEATURE_COLS if c in sample_row.index]
    sample_vec = sample_row[feat_cols].values
    pred_class, slip_prob = clf.predict(sample_vec)
    pred_str = "⚠️ SLIP DETECTED" if pred_class == 1 else "✅ STABLE CONTACT"
    is_correct = (pred_class == actual_label)

    # Display Metrics & Prediction Results
    col_res1, col_res2, col_res3 = st.columns([1.2, 1.2, 1.6])

    with col_res1:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px;">
            <div style="font-size:0.75rem; color:#64748B; font-weight:700;">GROUND TRUTH (PHYSICAL SENSOR)</div>
            <div style="font-size:1.1rem; font-weight:800; color:{'#EF4444' if actual_label == 1 else '#10B981'}; margin-top:4px;">
                {actual_label_str}
            </div>
            <div style="font-size:0.72rem; color:#94A3B8; margin-top:4px;">
                Texture: <code>{sample_row.get('texture', 'N/A')}</code><br/>
                Trial: <code>{chosen_trial}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        pred_color = "#EF4444" if pred_class == 1 else "#10B981"
        badge = "🎯 MATCH" if is_correct else "❌ MISCLASSIFICATION"
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px;">
            <div style="font-size:0.75rem; color:#64748B; font-weight:700;">MODEL PREDICTION</div>
            <div style="font-size:1.1rem; font-weight:800; color:{pred_color}; margin-top:4px;">
                {pred_str}
            </div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                Slip Probability: <strong style="color:#06B6D4;">{slip_prob*100:.1f}%</strong> ({badge})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res3:
        # Mini probability gauge
        prob_fig = go.Figure(go.Bar(
            x=[(1.0 - slip_prob) * 100, slip_prob * 100],
            y=["Stable Contact", "Dynamic Slip"],
            orientation='h',
            marker=dict(color=['#10B981', '#EF4444'])
        ))
        prob_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0B132B",
            margin=dict(l=10, r=10, t=10, b=20),
            height=100,
            xaxis=dict(title="Probability (%)", range=[0, 100]),
            yaxis=dict(title="")
        )
        st.plotly_chart(prob_fig, use_container_width=True)

    # Detailed Model Metadata & Schema
    col_meta, col_feat = st.columns([1, 1.2])

    with col_meta:
        st.markdown("##### 📋 Model Architecture & Provenance")
        st.markdown(f"""
        <table style="width:100%; font-size:0.80rem; border-collapse:collapse; color:#CBD5E1;">
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Model Name</td><td style="padding:6px; font-weight:700;">RandomForestClassifier (PVDF Dynamic Vibration)</td></tr>
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Source Dataset</td><td style="padding:6px;">IntelliMan ICECS 2025 (Zenodo 18759556)</td></tr>
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">License</td><td style="padding:6px;">CC-BY-4.0</td></tr>
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Sensor Modality</td><td style="padding:6px;">8-Channel PVDF Piezoelectric Film (1 kHz)</td></tr>
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Input Feature Dimension</td><td style="padding:6px;">{len(feat_cols)} Statistical Features</td></tr>
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Test Set Accuracy</td><td style="padding:6px; font-weight:700; color:#10B981;">{clf.metrics.accuracy*100:.2f}% (ROC-AUC: {clf.metrics.roc_auc:.3f})</td></tr>
            <tr><td style="padding:6px; color:#64748B;">Baseline Comparison</td><td style="padding:6px;">Dummy F1: {clf.baseline_f1:.3f} vs Model F1: {clf.metrics.f1_score:.3f}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.caption("⚠️ **Limitation**: Requires AC dynamic piezoelectric vibration sensing. Cannot be triggered by static DC resistance load cells alone.")

    with col_feat:
        st.markdown("##### 🔍 Sensor Feature Vector (Physical Signal)")
        features_preview = {
            "Global Mean Std (V)": round(float(sample_row.get("global_vibration_mean_std", 0.0)), 6),
            "Global Max Std (V)": round(float(sample_row.get("global_vibration_max_std", 0.0)), 6),
            "Vibration Energy (V²)": round(float(sample_row.get("global_vibration_energy", 0.0)), 6),
            "Signal PTP Amplitude (V)": round(float(sample_row.get("global_signal_ptp", 0.0)), 6),
            "Ch 1 RMS (V)": round(float(sample_row.get("ch1_rms", 0.0)), 6),
            "Ch 2 RMS (V)": round(float(sample_row.get("ch2_rms", 0.0)), 6),
            "Ch 3 RMS (V)": round(float(sample_row.get("ch3_rms", 0.0)), 6),
            "Ch 4 RMS (V)": round(float(sample_row.get("ch4_rms", 0.0)), 6),
        }
        df_feat = pd.DataFrame(list(features_preview.items()), columns=["Sensor Feature", "Physical Value"])
        st.dataframe(df_feat, use_container_width=True, height=210)


# -----------------------------------------------------------------------------
# HELPER: MODEL B2 INFERENCE
# -----------------------------------------------------------------------------
def _render_b2_tactip_inference(dm: DataManager):
    test_csv_path = os.path.abspath("data/processed/bristol_slip/bristol_slip_test.csv")
    if not os.path.exists(test_csv_path):
        st.warning("⚠️ Bristol processed test dataset not found. Run preprocessing in the Data & AI Lab first.")
        return

    df_test = pd.read_csv(test_csv_path)
    clf = BristolTacTipSlipClassifier()
    if clf.model is None:
        clf.train_and_evaluate()

    st.markdown("##### 1. Select Held-Out Bristol TacTip Test Run")
    c_sel1, c_sel2 = st.columns([3, 1])

    unique_runs = df_test["recording"].unique()
    with c_sel1:
        chosen_run = st.selectbox(
            "Select Independent Experimental Test Run (20 Held-Out Runs)",
            unique_runs,
            index=0,
            key="bristol_run_select"
        )

    df_run_frames = df_test[df_test["recording"] == chosen_run].reset_index(drop=True)
    with c_sel2:
        chosen_frame_idx = st.selectbox(
            f"Frame Window (Total: {len(df_run_frames)})",
            range(len(df_run_frames)),
            index=min(10, len(df_run_frames)-1),
            key="bristol_frame_select"
        )

    sample_row = df_run_frames.iloc[chosen_frame_idx]
    actual_label = int(sample_row["is_slip"])
    actual_label_str = "⚠️ Dynamic Slip" if actual_label == 1 else "✅ Static Contact / Stable"

    feat_cols = clf.FEATURE_COLS
    sample_vec = sample_row[feat_cols].values.reshape(1, -1)
    pred_class = int(clf.model.predict(sample_vec)[0])
    slip_prob = float(clf.model.predict_proba(sample_vec)[0, 1])
    pred_str = "⚠️ SLIP DETECTED" if pred_class == 1 else "✅ STABLE CONTACT"
    is_correct = (pred_class == actual_label)

    # Display Metrics & Prediction Results
    col_res1, col_res2, col_res3 = st.columns([1.2, 1.2, 1.6])

    with col_res1:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px;">
            <div style="font-size:0.75rem; color:#64748B; font-weight:700;">GROUND TRUTH (OPTICAL TACTIP)</div>
            <div style="font-size:1.1rem; font-weight:800; color:{'#EF4444' if actual_label == 1 else '#10B981'}; margin-top:4px;">
                {actual_label_str}
            </div>
            <div style="font-size:0.72rem; color:#94A3B8; margin-top:4px;">
                Run ID: <code>{chosen_run}</code><br/>
                Split: <code>Independent Test Run</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        pred_color = "#EF4444" if pred_class == 1 else "#10B981"
        badge = "🎯 MATCH" if is_correct else "❌ MISCLASSIFICATION"
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px;">
            <div style="font-size:0.75rem; color:#64748B; font-weight:700;">TACTIP MODEL PREDICTION</div>
            <div style="font-size:1.1rem; font-weight:800; color:{pred_color}; margin-top:4px;">
                {pred_str}
            </div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                Slip Probability: <strong style="color:#06B6D4;">{slip_prob*100:.1f}%</strong> ({badge})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res3:
        prob_fig = go.Figure(go.Bar(
            x=[(1.0 - slip_prob) * 100, slip_prob * 100],
            y=["Stable Contact", "Dynamic Slip"],
            orientation='h',
            marker=dict(color=['#10B981', '#EF4444'])
        ))
        prob_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0B132B",
            margin=dict(l=10, r=10, t=10, b=20),
            height=100,
            xaxis=dict(title="Probability (%)", range=[0, 100]),
            yaxis=dict(title="")
        )
        st.plotly_chart(prob_fig, use_container_width=True)

    col_meta, col_feat = st.columns([1, 1.2])

    with col_meta:
        st.markdown("##### 📋 Model Architecture & Provenance")
        st.markdown(f"""
        <table style="width:100%; font-size:0.80rem; border-collapse:collapse; color:#CBD5E1;">
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Model Name</td><td style="padding:6px; font-weight:700;">RandomForestClassifier (TacTip Optical Kinematics)</td></tr>
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Source Dataset</td><td style="padding:6px;">Bristol Biomimetic Tactile Slip Dataset (Ward-Cherrier et al.)</td></tr>
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Sensor Modality</td><td style="padding:6px;">127 Internal Elastomer Papillae Coordinates</td></tr>
            <tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px; color:#64748B;">Test Accuracy</td><td style="padding:6px; font-weight:700; color:#10B981;">{clf.metrics.accuracy*100:.2f}% (ROC-AUC: {clf.metrics.roc_auc:.3f})</td></tr>
            <tr><td style="padding:6px; color:#64748B;">Baseline Comparison</td><td style="padding:6px;">Dummy F1: {clf.baseline_f1:.3f} vs Model F1: {clf.metrics.f1_score:.3f}</td></tr>
        </table>
        """, unsafe_allow_html=True)
        st.caption("⚠️ **Limitation**: Requires an internal camera and elastomer dome tracking 127 marker pins. Cannot be evaluated from simple single-point force pads.")

    with col_feat:
        st.markdown("##### 🔍 Kinematic Feature Values (Physical Markers)")
        tactip_feat = {
            "Mean Pin Velocity (px/frame)": round(float(sample_row.get("mean_pin_speed", 0.0)), 4),
            "Max Pin Velocity (px/frame)": round(float(sample_row.get("max_pin_speed", 0.0)), 4),
            "Velocity Std Dev (px/frame)": round(float(sample_row.get("std_pin_speed", 0.0)), 4),
            "Spatial Directional Coherence R": round(float(sample_row.get("spatial_coherence", 0.0)), 4),
            "Coherence Std Dev": round(float(sample_row.get("coherence_std", 0.0)), 4),
        }
        df_tf = pd.DataFrame(list(tactip_feat.items()), columns=["Kinematic Feature", "Measured Value"])
        st.dataframe(df_tf, use_container_width=True, height=180)


# -----------------------------------------------------------------------------
# HELPER: MODEL C INFERENCE
# -----------------------------------------------------------------------------
def _render_c_contact_state_inference(dm: DataManager):
    test_csv_path = os.path.abspath("data/processed/intelliman_slip/intelliman_slip_test.csv")
    if not os.path.exists(test_csv_path):
        st.warning("⚠️ IntelliMan test dataset not found.")
        return

    df_test = pd.read_csv(test_csv_path)
    clf = ContactGraspStateClassifier()
    if clf.model is None:
        clf.train_and_evaluate()

    sample_row = df_test.sample(n=1, random_state=int(np.random.randint(0, 1000))).iloc[0]
    actual_state = int(sample_row.get("contact_state", 1))
    
    avail_cols = [c for c in clf.FEATURE_COLS if c in sample_row.index]
    x_sc = clf.scaler.transform(sample_row[avail_cols].values.reshape(1, -1))
    pred_state = int(clf.model.predict(x_sc)[0])
    probs = clf.model.predict_proba(x_sc)[0]

    st.markdown(f"""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between;">
            <div>
                <strong>Model C: 3-Phase Multi-Class Contact & Slip Classifier</strong><br/>
                <span style="font-size:0.80rem; color:#94A3B8;">Ground Truth: <strong>{clf.CLASS_NAMES[actual_state]}</strong></span>
            </div>
            <div>
                Predicted: <strong style="color:#06B6D4;">{clf.CLASS_NAMES[pred_state]}</strong>
                ({'🎯 Match' if pred_state == actual_state else '❌ Mismatch'})
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    prob_fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],
        y=clf.CLASS_NAMES,
        orientation='h',
        marker=dict(color=['#64748B', '#10B981', '#EF4444'])
    ))
    prob_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B132B",
        margin=dict(l=10, r=10, t=10, b=20),
        height=140,
        xaxis=dict(title="Probability (%)", range=[0, 100]),
        yaxis=dict(title="")
    )
    st.plotly_chart(prob_fig, use_container_width=True)


# -----------------------------------------------------------------------------
# HELPER: MODEL D INFERENCE
# -----------------------------------------------------------------------------
def _render_d_emg_inference(dm: DataManager):
    test_csv_path = os.path.abspath("data/processed/ninapro_emg/ninapro_emg_test.csv")
    if not os.path.exists(test_csv_path):
        st.warning("⚠️ NinaPro test dataset not found.")
        return

    df_test = pd.read_csv(test_csv_path)
    clf = EMGIntentionClassifier()
    if clf.model is None:
        clf.train_and_evaluate()

    sample_row = df_test.sample(n=1, random_state=int(np.random.randint(0, 1000))).iloc[0]
    actual_g = int(sample_row.get("gesture_id", 0))

    feat_cols = [c for c in df_test.columns if c.startswith("emg_")]
    x_sc = clf.scaler.transform(sample_row[feat_cols].values.reshape(1, -1))
    pred_g = int(clf.model.predict(x_sc)[0])
    probs = clf.model.predict_proba(x_sc)[0]

    st.markdown(f"""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between;">
            <div>
                <strong>Model D: NinaPro sEMG Movement Intention Classifier</strong><br/>
                <span style="font-size:0.80rem; color:#94A3B8;">Held-Out Subject: <strong>Subject 5 (Subject-Independent)</strong></span>
            </div>
            <div>
                Target: <strong>{clf.GESTURE_NAMES[actual_g]}</strong> | 
                Predicted: <strong style="color:#06B6D4;">{clf.GESTURE_NAMES[pred_g]}</strong>
                ({'🎯 Match' if pred_g == actual_g else '❌ Mismatch'})
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    prob_fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],
        y=clf.GESTURE_NAMES,
        orientation='h',
        marker=dict(color='#06B6D4')
    ))
    prob_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B132B",
        margin=dict(l=10, r=10, t=10, b=20),
        height=160,
        xaxis=dict(title="Probability (%)", range=[0, 100]),
        yaxis=dict(title="")
    )
    st.plotly_chart(prob_fig, use_container_width=True)

