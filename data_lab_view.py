"""
data_lab_view.py - Data & AI Research Laboratory UI Component
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Provides interactive Streamlit tabs:
1. Dataset Registry (sources, citations, licenses, real vs synthetic tags, downloads)
2. Model Laboratory (architectures, feature selections, retrain triggers)
3. Evaluation Dashboard (confusion matrices, ROC/PR curves, baselines vs ML)
4. Dataset Explorer (waveforms, distributions, sample inspections, CSV export)
5. Tactile ML vs Simulation Studio (real sensor waveforms vs digital twin simulation)
6. Hardware Readiness & ESP32 Integration (pinouts, packet specs, calibration)
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_manager import DataManager
from dataset_registry import DataNature, MLTask


def render_data_lab(dm: DataManager):
    """Renders the comprehensive Data & AI Research Lab."""

    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h2 style="margin:0; font-size:1.7rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">
            🔬 Data & AI Research Laboratory
        </h2>
        <div style="font-size:0.85rem; color:#94A3B8;">
            Multi-Dataset Research Pipeline • Real Physical Tactile Sensors • Leakage-Resistant ML Benchmarks
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Dataset Registry",
        "🧪 Model Laboratory",
        "📊 Evaluation Dashboard",
        "🔍 Dataset Explorer",
        "⚡ Tactile ML Studio",
        "🔌 Hardware & Protocol"
    ])

    # =========================================================================
    # TAB 1: DATASET REGISTRY
    # =========================================================================
    with tab1:
        st.markdown("#### 📚 Central Research Dataset Registry")
        st.caption("Investigated public and benchmark datasets for prosthetic tactile slip, force, and grasp modeling.")

        summary_rows = dm.registry.get_summary_table()
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.markdown("##### 📦 Detailed Dataset Catalog & Actions")

        datasets = dm.get_all_datasets()
        for d in datasets:
            with st.expander(f"{'🟢' if d.is_downloaded() else '⏳'} {d.name} ({d.primary_ml_task.value})", expanded=False):
                col_info1, col_info2 = st.columns([2, 1])
                with col_info1:
                    nature_color = "#38BDF8" if d.data_nature == DataNature.REAL_RESEARCH_DATA else "#F59E0B"
                    st.markdown(f"**Data Nature:** <span style='color:{nature_color}; font-weight:700;'>{d.data_nature.value}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Primary Task:** `{d.primary_ml_task.value}`")
                    st.markdown(f"**Sensor Modality:** {d.sensor_modality}")
                    st.markdown(f"**Source URL:** [{d.source_url}]({d.source_url})")
                    st.markdown(f"**Citation:** *{d.citation}*")
                    st.markdown(f"**License:** `{d.license_type}` ([Details]({d.license_url}))")
                    st.markdown(f"**File Formats:** {', '.join(d.file_formats)}")
                    st.markdown(f"**Expected Labels:** {', '.join(d.expected_labels)}")
                    st.markdown(f"**Access Notes:** {d.access_requirements}")

                with col_info2:
                    st.markdown(f"""
                    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; font-size:0.85rem;">
                        <div><strong>Download Status:</strong> {'✅ Present' if d.is_downloaded() else '⏳ Not Downloaded'}</div>
                        <div style="margin-top:4px;"><strong>Preprocessing:</strong> {'✅ Processed' if d.is_processed() else '⏳ Pending'}</div>
                        <div style="margin-top:4px;"><strong>Approx Size:</strong> {d.download_size_approx}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if d.direct_download_url:
                        if st.button(f"📥 Download / Verify {d.key}", key=f"dl_{d.key}", use_container_width=True):
                            with st.spinner(f"Processing download for {d.name}..."):
                                success, msg = dm.download(d.key)
                                if success:
                                    st.success(msg)
                                else:
                                    st.error(msg)
                                st.rerun()

                    if st.button(f"⚙️ Run Preprocessing Adapter", key=f"prep_{d.key}", use_container_width=True):
                        with st.spinner(f"Running preprocessing for {d.name}..."):
                            success, msg = dm.preprocess(d.key)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
                            st.rerun()

    # =========================================================================
    # TAB 2: MODEL LABORATORY
    # =========================================================================
    with tab2:
        st.markdown("#### 🧪 Task-Appropriate Model Architectures")
        st.caption("Dedicated task models strictly separated by physical sensor modality to prevent incompatible feature conflation.")

        c_m1, c_m2 = st.columns(2)

        with c_m1:
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:14px; margin-bottom:12px;">
                <div style="color:#06B6D4; font-weight:700; font-size:0.8rem; text-transform:uppercase;">Model A • Force Regression</div>
                <div style="font-size:1.1rem; font-weight:800; color:#F8FAFC;">Minimum-Safe-Force Regressor</div>
                <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">
                    <strong>Algorithm:</strong> Random Forest Regressor (50 trees, max depth 7)<br/>
                    <strong>Dataset:</strong> Synthetic Force Dataset (1,500 samples)<br/>
                    <strong>Inputs:</strong> Mass, Friction (μ), Compliance (mm/N), Fragility (N), Margin<br/>
                    <strong>Target:</strong> Safe holding clamp force (Newtons)<br/>
                    <span style="color:#F59E0B;">⚠️ Synthetic demonstration model only.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Retrain Model A (Force)", use_container_width=True):
                with st.spinner("Retraining Synthetic Force Regressor..."):
                    succ, msg = dm.train_model("gripwise_force")
                    st.success(msg)
                    st.rerun()

            st.markdown("""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:14px; margin-bottom:12px; margin-top:16px;">
                <div style="color:#10B981; font-weight:700; font-size:0.8rem; text-transform:uppercase;">Model B1 • Tactile Slip (PVDF)</div>
                <div style="font-size:1.1rem; font-weight:800; color:#F8FAFC;">IntelliMan PVDF Slip Classifier</div>
                <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">
                    <strong>Algorithm:</strong> Random Forest Classifier (65 trees, balanced)<br/>
                    <strong>Dataset:</strong> IntelliMan ICECS 2025 (Physical PVDF sensor recordings)<br/>
                    <strong>Inputs:</strong> 8-channel dynamic vibration std, peak-to-peak, energy<br/>
                    <strong>Evaluation:</strong> Strict recording-level trial partition (zero leakage)
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Retrain Model B1 (IntelliMan PVDF)", use_container_width=True):
                with st.spinner("Retraining IntelliMan PVDF Slip Classifier..."):
                    succ, msg = dm.train_model("intelliman_slip")
                    st.success(msg)
                    st.rerun()

        with c_m2:
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:14px; margin-bottom:12px;">
                <div style="color:#38BDF8; font-weight:700; font-size:0.8rem; text-transform:uppercase;">Model B2 • Tactile Slip (TacTip)</div>
                <div style="font-size:1.1rem; font-weight:800; color:#F8FAFC;">Bristol TacTip Slip Classifier</div>
                <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">
                    <strong>Algorithm:</strong> Random Forest Classifier (50 trees, max depth 7)<br/>
                    <strong>Dataset:</strong> Bristol Biomimetic Tactile Slip Dataset (University of Bristol)<br/>
                    <strong>Inputs:</strong> 127 papillae marker speed, spatial coherence, directional jerk<br/>
                    <strong>Evaluation:</strong> 20 train runs vs 20 held-out test runs
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Retrain Model B2 (Bristol TacTip)", use_container_width=True):
                with st.spinner("Retraining Bristol TacTip Slip Classifier..."):
                    succ, msg = dm.train_model("bristol_slip")
                    st.success(msg)
                    st.rerun()

            st.markdown("""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:14px; margin-bottom:12px; margin-top:16px;">
                <div style="color:#EC4899; font-weight:700; font-size:0.8rem; text-transform:uppercase;">Model C • Contact & Grasp State</div>
                <div style="font-size:1.1rem; font-weight:800; color:#F8FAFC;">Three-Phase Grasp State Classifier</div>
                <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">
                    <strong>Algorithm:</strong> Multi-class Random Forest (55 trees)<br/>
                    <strong>Dataset:</strong> IntelliMan ICECS 2025 ground-truth phase annotations<br/>
                    <strong>Classes:</strong> 0 = Rest / Separation, 1 = Static Contact, 2 = Dynamic Slippage
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Retrain Model C (Contact & Grasp Phase)", use_container_width=True):
                with st.spinner("Retraining Contact & Grasp-State Classifier..."):
                    succ, msg = dm.train_model("contact_state")
                    st.success(msg)
                    st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:14px;">
            <div style="color:#A78BFA; font-weight:700; font-size:0.8rem; text-transform:uppercase;">Model D • Optional sEMG Movement Intent</div>
            <div style="font-size:1.1rem; font-weight:800; color:#F8FAFC;">NinaPro Forearm sEMG Grasp Classifier</div>
            <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">
                <strong>Algorithm:</strong> Random Forest Classifier (45 trees) on 8-channel sEMG MAV/RMS/WL features<br/>
                <strong>Classes:</strong> Rest, Power Grasp, Precision Pinch, Hand Open, Lateral Key<br/>
                <strong>Evaluation:</strong> Subject-independent held-out test split (tests inter-subject generalization)<br/>
                <span style="color:#94A3B8;">Notice: Forearm sEMG classifies intended movement gesture, NOT minimum safe holding force.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("🔄 Train / Evaluate Model D (sEMG Movement Intent)", use_container_width=True):
            with st.spinner("Training sEMG Classifier..."):
                succ, msg = dm.train_model("ninapro_emg")
                st.success(msg)
                st.rerun()

    # =========================================================================
    # TAB 3: EVALUATION DASHBOARD
    # =========================================================================
    with tab3:
        st.markdown("#### 📊 Leakage-Resistant Model Benchmarks & Metrics")
        st.caption("All evaluations executed on held-out physical trial recordings or unseen subjects. Zero window leakage.")

        summaries = dm.get_evaluation_summaries()
        if summaries:
            st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)
        else:
            st.info("No saved evaluation metrics found yet. Click 'Run Full Pipeline' below.")

        col_ev_btn, _ = st.columns([1.5, 3])
        with col_ev_btn:
            if st.button("🚀 Re-Run Full Multi-Model Benchmark Pipeline", use_container_width=True):
                with st.spinner("Evaluating all models with strict leakage protection..."):
                    dm.evaluator.run_full_evaluation_pipeline()
                    st.success("All models evaluated! Metrics refreshed.")
                    st.rerun()

        st.markdown("---")
        st.markdown("##### 🔍 Deep-Dive: Confusion Matrices & Class Distributions")

        col_c1, col_c2 = st.columns(2)

        # 1. IntelliMan Confusion Matrix
        with col_c1:
            st.markdown("###### 1. IntelliMan PVDF Slip Classification (Binary)")
            im_path = os.path.join(dm.evaluator.metrics_dir, "intelliman_pvdf_slip_metrics.json")
            if os.path.exists(im_path):
                with open(im_path, "r", encoding="utf-8") as f:
                    im_data = json.load(f)
                
                cm = np.array(im_data["confusion_matrix"])
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted Class", y="True Label", color="Samples"),
                    x=["Non-Slip (Static/Rest)", "Dynamic Slip"],
                    y=["Non-Slip (Static/Rest)", "Dynamic Slip"],
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#0B132B",
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption(f"**Metrics:** Accuracy: {im_data['accuracy']*100:.1f}% | Precision: {im_data['precision']:.3f} | Recall: {im_data['recall']:.3f} | F1: {im_data['f1_score']:.3f} | ROC-AUC: {im_data['roc_auc']:.3f}")
            else:
                st.warning("IntelliMan metrics not yet generated.")

        # 2. Contact State Confusion Matrix (3x3)
        with col_c2:
            st.markdown("###### 2. Contact & Grasp Phase Classification (Multi-Class)")
            cs_path = os.path.join(dm.evaluator.metrics_dir, "intelliman_contact_state_metrics.json")
            if os.path.exists(cs_path):
                with open(cs_path, "r", encoding="utf-8") as f:
                    cs_data = json.load(f)
                
                cm_cs = np.array(cs_data["confusion_matrix"])
                classes_short = ["0: Rest", "1: Contact", "2: Slip"]
                fig_cs = px.imshow(
                    cm_cs,
                    text_auto=True,
                    labels=dict(x="Predicted Phase", y="Ground Truth Phase", color="Windows"),
                    x=classes_short,
                    y=classes_short,
                    color_continuous_scale="Purples"
                )
                fig_cs.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#0B132B",
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_cs, use_container_width=True)
                st.caption(f"**Metrics:** Overall Accuracy: {cs_data['accuracy']*100:.1f}% | Macro F1: {cs_data['macro_f1']:.3f} | Weighted F1: {cs_data['weighted_f1']:.3f}")
            else:
                st.warning("Contact state metrics not yet generated.")

        # 3. Bristol TacTip feature importances
        st.markdown("---")
        st.markdown("###### 3. Bristol TacTip Biomimetic Kinematic Feature Importances")
        br_path = os.path.join(dm.evaluator.metrics_dir, "bristol_tactip_slip_metrics.json")
        if os.path.exists(br_path):
            with open(br_path, "r", encoding="utf-8") as f:
                br_data = json.load(f)
            
            imp_df = pd.DataFrame([
                {"Kinematic Feature": k, "Importance": v}
                for k, v in br_data["feature_importances"].items()
            ]).sort_values("Importance", ascending=True)

            fig_br = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Kinematic Feature"],
                orientation='h',
                marker=dict(color='#38BDF8')
            ))
            fig_br.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0B132B",
                height=220,
                margin=dict(l=10, r=10, t=10, b=20),
                xaxis=dict(title="Gini Feature Importance"),
                yaxis=dict(title="")
            )
            st.plotly_chart(fig_br, use_container_width=True)
            st.caption(f"**Bristol TacTip Results (20 Train vs 20 Held-Out Test Runs):** Accuracy: {br_data['accuracy']*100:.2f}% | F1: {br_data['f1_score']:.3f} | ROC-AUC: {br_data['roc_auc']:.3f}")

    # =========================================================================
    # TAB 4: DATASET EXPLORER
    # =========================================================================
    with tab4:
        st.markdown("#### 🔍 Dataset Record & Waveform Explorer")
        st.caption("Inspect raw and processed records, feature summary tables, and export preprocessed feature matrices.")

        sel_dataset = st.selectbox(
            "Select Processed Dataset to Inspect",
            ["intelliman_slip", "bristol_slip", "gripwise_force_synthetic", "ninapro_emg"],
            format_func=lambda k: dm.registry.get_dataset(k).name if dm.registry.get_dataset(k) else k
        )

        sample_df = dm.load_sample_dataset(sel_dataset, max_rows=100)
        if sample_df is not None:
            col_d_stats1, col_d_stats2, col_d_stats3 = st.columns(3)
            with col_d_stats1:
                st.metric("Preview Columns", len(sample_df.columns))
            with col_d_stats2:
                st.metric("Missing Values", sample_df.isna().sum().sum())
            with col_d_stats3:
                st.metric("Data Sample Rows", len(sample_df))

            st.dataframe(sample_df.head(25), use_container_width=True)

            # Export button
            csv_exp = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"📥 Download Sample Data ({sel_dataset}.csv)",
                data=csv_exp,
                file_name=f"{sel_dataset}_sample.csv",
                mime="text/csv"
            )

            # If IntelliMan dataset, plot vibration signals across channels
            if "intelliman" in sel_dataset:
                st.markdown("##### 📈 PVDF Tactile Vibration Signals Across Channels")
                ch_cols = [c for c in sample_df.columns if "_std" in c][:8]
                if ch_cols:
                    fig_waves = go.Figure()
                    for ch in ch_cols:
                        fig_waves.add_trace(go.Scatter(
                            y=sample_df[ch].head(60),
                            mode="lines",
                            name=ch
                        ))
                    fig_waves.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0F172A",
                        plot_bgcolor="#0B132B",
                        height=280,
                        margin=dict(l=30, r=20, t=20, b=30),
                        xaxis=dict(title="Sliding Window Index"),
                        yaxis=dict(title="Vibration Std Dev (V)")
                    )
                    st.plotly_chart(fig_waves, use_container_width=True)
        else:
            st.info(f"Dataset '{sel_dataset}' has not been preprocessed yet. Run preprocessing in the Dataset Registry tab.")

    # =========================================================================
    # TAB 5: TACTILE ML STUDIO
    # =========================================================================
    with tab5:
        st.markdown("#### ⚡ Tactile Sensor Modality Comparative Studio")
        st.caption("Demonstrating the physical distinction between real optical/piezoelectric research signals and simulated piezoresistive digital twin telemetry.")

        st.markdown("""
        <div style="background:#0F172A; border-left:4px solid #06B6D4; padding:12px 14px; border-radius:4px; font-size:0.85rem; color:#E2E8F0; margin-bottom:14px;">
            <strong>SCIENTIFIC MODALITY ISOLATION PRINCIPLE:</strong><br/>
            Real prosthetic research utilizes varied tactile transducers:
            <ul>
                <li><strong>TacTip (Optical Biomimetic):</strong> Measures pin shear displacement fields (pixels/frame).</li>
                <li><strong>PVDF (Piezoelectric):</strong> Measures dynamic transient AC voltage vibrations (mV).</li>
                <li><strong>FSR / Piezoresistive:</strong> Measures static compressive DC resistance / normal force (Newtons).</li>
            </ul>
            GripWise does NOT force these disparate physical signals into a single artificial model. Models B1 and B2 operate strictly on their native sensor domains.
        </div>
        """, unsafe_allow_html=True)

        col_st1, col_st2 = st.columns(2)

        with col_st1:
            st.markdown("##### 🔬 Real Sensor Benchmark: IntelliMan PVDF Inference")
            from slip_model import IntelliManPVDFSlipClassifier
            clf_im = IntelliManPVDFSlipClassifier()

            if clf_im.model is not None:
                # Load test sample
                df_te = dm.load_sample_dataset("intelliman_slip", max_rows=50)
                if df_te is not None and not df_te.empty:
                    sample_row_idx = st.slider("Select Physical Test Window", 0, len(df_te)-1, 12)
                    sample_vec = df_te[clf_im.FEATURE_COLS].iloc[sample_row_idx].values
                    ground_truth = int(df_te["is_slip"].iloc[sample_row_idx])
                    pred_cls, pred_prob = clf_im.predict(sample_vec)

                    st.markdown(f"""
                    <div style="background:#0B132B; border:1px solid #1E293B; border-radius:8px; padding:12px;">
                        <div>Ground Truth Label: <strong>{'⚠️ SLIP (Phase 2)' if ground_truth == 1 else '🟢 NON-SLIP (Phase 1/3)'}</strong></div>
                        <div style="margin-top:4px;">ML Classifier Output: <strong style="color:{'#EF4444' if pred_cls == 1 else '#10B981'};">{'⚠️ SLIP DETECTED' if pred_cls == 1 else '🟢 STABLE CONTACT'}</strong></div>
                        <div style="margin-top:4px;">Slip Probability: <strong>{pred_prob*100:.1f}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Train Model B1 to test live inference on physical PVDF recordings.")

        with col_st2:
            st.markdown("##### 🦾 Digital Twin Simulation Sensor: Normal Force")
            st.markdown("""
            <div style="background:#0B132B; border:1px solid #1E293B; border-radius:8px; padding:12px; font-size:0.85rem;">
                <div>Sensor Physics: <strong>Piezoresistive 5-Fingertip Normal Force</strong></div>
                <div style="margin-top:4px;">Signal Nature: <strong>DC Newton Clamp Force (0.0N – 40.0N)</strong></div>
                <div style="margin-top:4px;">Slip Mechanism: <strong>Tangential Friction Drop & Adaptive Boost</strong></div>
                <div style="margin-top:4px; color:#94A3B8;">The Digital Twin uses its verified adaptive closed-loop controller in the main dashboard tab.</div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 6: HARDWARE & PROTOCOL
    # =========================================================================
    with tab6:
        st.markdown("#### 🔌 Hardware Readiness & ESP32 Microcontroller Integration")
        st.caption("Standardized hardware abstraction protocol enabling physical microcontroller deployment.")

        st.markdown("""
        ### Microcontroller Serial Bridge Specification (115200 Baud UART)

        The `hardware.py` module exposes an abstract interface (`BaseSensorInterface`).
        When physical hardware is connected, switch the sidebar mode to **Hardware Serial (ESP32/Arduino)**.

        #### 1. Inbound Packet Format (ESP32 $\\rightarrow$ Dashboard)
        The microcontroller must transmit a comma-separated ASCII string every **20 milliseconds (50 Hz)**:
        ```text
        <total_force>,<slip_flag>,<thumb_force>,<index_force>,<middle_force>,<ring_force>,<pinky_force>,<motor_current_ma>\\n
        ```
        *Example:* `4.15,0,1.66,0.91,0.83,0.46,0.29,268.4\\n`

        #### 2. Outbound Target Command (Dashboard $\\rightarrow$ ESP32)
        The adaptive controller commands finger tendon actuators:
        ```text
        TARGET:<target_force_in_newtons>\\n
        ```
        *Example:* `TARGET:5.80\\n`

        #### 3. Recommended Pinout Wiring (ESP32 DevKit V1)
        - **GPIO 32 (ADC1_CH4):** Thumb FSR Piezoresistive Sensor (10kΩ voltage divider)
        - **GPIO 33 (ADC1_CH5):** Index FSR Sensor
        - **GPIO 34 (ADC1_CH6):** Middle FSR Sensor
        - **GPIO 35 (ADC1_CH7):** Ring FSR Sensor
        - **GPIO 36 (ADC1_CH0):** Pinky FSR Sensor
        - **GPIO 4 (Digital Interrupt):** Piezoelectric vibration disc / PVDF slip shear sensor
        - **GPIO 18, 19, 21, 22, 23 (PWM):** Micro servo tendon spooling actuators
        """)

