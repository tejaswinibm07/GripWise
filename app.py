"""
app.py - GripWise Digital Twin Engineering Dashboard
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand for Personalized
Grip Strategies and Minimum-Safe-Force Prediction.

Runs in 100% simulation mode by default; ready for future ESP32 serial communication.
Launch command: streamlit run app.py
"""

import time
import copy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from hardware import (
    BaseSensorInterface, SimulatedSensorInterface, SerialSensorInterface,
    MockSerialSensorInterface, RecordedReplaySensorInterface, SensorPacket
)
from simulation import STANDARD_OBJECTS, ObjectSpec, HandState
from prediction import ForcePredictor, PredictionResult
from controller import AdaptiveGripController, ControllerTelemetryPoint
from personalization import PersonalizationManager, UserProfile, TrialRecord, DEFAULT_PRESETS
from virtual_hand import generate_hand_svg
from data_manager import DataManager
from data_lab_view import render_data_lab
from demo_scenario import execute_end_to_end_demonstration, DemonstrationExecutionSummary
from offline_inference_view import render_offline_inference_studio
from model_comparison_view import render_model_comparison_deck
from hardware_view import render_hardware_view

# ==============================================================================
# Page Configuration & Custom Dark Navy Styling
# ==============================================================================
st.set_page_config(
    page_title="GripWise | Digital Twin Dashboard",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark Navy Core Backgrounds */
    .stApp {
        background-color: #070B14;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0B132B;
        border-right: 1px solid #1E293B;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #0F172A 0%, #131E36 100%);
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        margin-bottom: 8px;
    }
    .metric-card:hover {
        border-color: #0284C7;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* State Pills */
    .pill-open { background: rgba(56, 189, 248, 0.12); color: #38BDF8; border: 1px solid #0284C7; }
    .pill-closing { background: rgba(14, 165, 233, 0.15); color: #38BDF8; border: 1px solid #0EA5E9; }
    .pill-grasping { background: rgba(6, 182, 212, 0.20); color: #22D3EE; border: 1px solid #06B6D4; }
    .pill-stable { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid #10B981; }
    .pill-slipping { background: rgba(239, 68, 68, 0.25); color: #FCA5A5; border: 1px solid #EF4444; }
    .pill-releasing { background: rgba(129, 140, 248, 0.18); color: #A5B4FC; border: 1px solid #818CF8; }

    .state-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.80rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* Action Banner */
    .action-banner {
        background-color: #0F172A;
        border-left: 4px solid #06B6D4;
        padding: 10px 14px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #38BDF8;
        margin-bottom: 12px;
    }

    /* Scientific Disclaimer Banner */
    .disclaimer-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.45;
    }

    /* Sleek buttons */
    div.stButton > button:first-child {
        background-color: #0F172A;
        color: #E2E8F0;
        border: 1px solid #334155;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1E293B;
        border-color: #06B6D4;
        color: #38BDF8;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# Initialization of Session State
# ==============================================================================
if "personalization_mgr" not in st.session_state:
    st.session_state.personalization_mgr = PersonalizationManager()

if "predictor" not in st.session_state:
    st.session_state.predictor = ForcePredictor()

if "sensor_interface" not in st.session_state:
    st.session_state.sensor_interface = SimulatedSensorInterface()

if "active_object_key" not in st.session_state:
    st.session_state.active_object_key = "Cup"

if "custom_objects" not in st.session_state:
    st.session_state.custom_objects = {k: copy.deepcopy(v) for k, v in STANDARD_OBJECTS.items()}

if "controller" not in st.session_state:
    active_obj = st.session_state.custom_objects[st.session_state.active_object_key]
    st.session_state.controller = AdaptiveGripController(active_obj, base_target=3.8)

if "active_profile_name" not in st.session_state:
    st.session_state.active_profile_name = "Alex (Standard)"

if "trial_counter" not in st.session_state:
    st.session_state.trial_counter = 4

if "active_feedback_msg" not in st.session_state:
    st.session_state.active_feedback_msg = ""

if "last_demo_summary" not in st.session_state:
    st.session_state.last_demo_summary = None


pm: PersonalizationManager = st.session_state.personalization_mgr
predictor: ForcePredictor = st.session_state.predictor
controller: AdaptiveGripController = st.session_state.controller
sensor: BaseSensorInterface = st.session_state.sensor_interface

if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()
dm: DataManager = st.session_state.data_manager


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom: 6px;">
        <span style="font-size:2rem;">🦾</span>
        <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.01em;">GripWise</div>
            <div style="font-size:0.70rem; color:#06B6D4; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">
                Digital Twin • Bio-Mechatronics
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 8px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)

    # 0. NAVIGATION DECK SELECTOR
    st.markdown("##### 🧭 Navigation Deck")
    view_selection = st.radio(
        "Navigation",
        [
            "🦾 Digital Twin Simulation",
            "🔌 Hardware Integration & Test Bench",
            "🧪 Offline Research Inference Studio",
            "⚖️ Model vs Baseline Comparison",
            "🔬 Data & AI Research Lab"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='margin: 8px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)

    if view_selection == "🔬 Data & AI Research Lab":
        st.markdown("##### 🔬 Research Status")
        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:10px; font-size:0.78rem;">
            <div>Registered Datasets: <strong>6</strong></div>
            <div style="margin-top:2px;">Trained ML Models: <strong>5</strong></div>
            <div style="margin-top:4px; color:#10B981; font-weight:700;">● Leakage-Resistant Benchmarks</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("🚀 Re-Run Multi-Model Pipeline", use_container_width=True):
            with st.spinner("Evaluating all models..."):
                dm.evaluator.run_full_evaluation_pipeline()
                st.success("Benchmarks refreshed!")
                st.rerun()

    st.markdown("<hr style='margin: 8px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)

    # 1. HARDWARE / SIMULATION MODE
    st.markdown("##### 🔌 Hardware & Sensor Mode")
    mode_selection = st.radio(
        "Sensor Source",
        ["Simulation (Default)", "Hardware Serial (ESP32/Arduino)"],
        label_visibility="collapsed"
    )

    if mode_selection == "Simulation (Default)":
        if not isinstance(sensor, SimulatedSensorInterface):
            st.session_state.sensor_interface = SimulatedSensorInterface()
            sensor = st.session_state.sensor_interface
        st.caption("🟢 **Simulation Engine Active** (Piezoresistive tactile model + Gaussian noise)")
    else:
        if not isinstance(sensor, SerialSensorInterface):
            st.session_state.sensor_interface = SerialSensorInterface()
            sensor = st.session_state.sensor_interface
        c1, c2 = st.columns([2, 1])
        with c1:
            serial_port = st.text_input("Port", value="COM3", help="ESP32 COM port on Windows")
        with c2:
            baud_rate = st.selectbox("Baud", [115200, 9600, 57600], index=0)
        
        if st.button("🔗 Connect Serial"):
            success = sensor.connect(serial_port, baud_rate)
            if success:
                st.success(f"Connected to {serial_port}")
            else:
                st.warning(f"Could not open {serial_port}. Fallback simulated packet active.")

    st.markdown("<hr style='margin: 10px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)

    # 2. OBJECT SELECTION & PARAMETERS
    st.markdown("##### 📦 Demonstration Test Object")
    obj_keys = list(STANDARD_OBJECTS.keys())
    if "prev_selected_obj" not in st.session_state:
        st.session_state.prev_selected_obj = st.session_state.active_object_key

    selected_key = st.selectbox(
        "Select Object",
        obj_keys,
        index=obj_keys.index(st.session_state.active_object_key),
        format_func=lambda k: f"{STANDARD_OBJECTS[k].icon} {STANDARD_OBJECTS[k].name}",
        label_visibility="collapsed"
    )
    if selected_key != st.session_state.prev_selected_obj:
        st.session_state.active_object_key = selected_key
        st.session_state.prev_selected_obj = selected_key
        controller.reset(sensor)
    else:
        st.session_state.active_object_key = selected_key
    active_obj = st.session_state.custom_objects[selected_key]

    with st.expander("⚙️ Fine-Tune Object Properties", expanded=False):
        st.caption("Demonstration modeling parameters (theoretical mechanics):")
        new_mass = st.slider("Mass (g)", 30.0, 900.0, float(active_obj.mass_g), step=10.0)
        new_mu = st.slider("Friction Coeff (μ)", 0.15, 0.95, float(active_obj.friction_coeff), step=0.01)
        new_comp = st.slider("Compliance (mm/N)", 0.02, 3.50, float(active_obj.compliance_mm_per_n), step=0.05)
        new_frag = st.slider("Fragility Limit (N)", 8.0, 60.0, float(active_obj.fragility_limit_n), step=1.0)
        
        active_obj.mass_g = new_mass
        active_obj.friction_coeff = new_mu
        active_obj.compliance_mm_per_n = new_comp
        active_obj.fragility_limit_n = new_frag

    st.markdown("<hr style='margin: 10px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)

    # 3. USER PROFILE & PERSONALIZATION
    st.markdown("##### 👤 User Profile & Preferences")
    profile_names = list(pm.profiles.keys())
    active_prof_name = st.selectbox(
        "Active Profile",
        profile_names,
        index=profile_names.index(st.session_state.active_profile_name) if st.session_state.active_profile_name in profile_names else 0,
        label_visibility="collapsed"
    )
    st.session_state.active_profile_name = active_prof_name
    active_prof: UserProfile = pm.profiles[active_prof_name]

    col_pr1, col_pr2 = st.columns(2)
    with col_pr1:
        new_preset = st.selectbox("Preset", ["Standard", "Delicate", "Firm", "Custom"], index=["Standard", "Delicate", "Firm", "Custom"].index(active_prof.preset_mode))
        if new_preset != active_prof.preset_mode:
            active_prof.preset_mode = new_preset
            if new_preset in DEFAULT_PRESETS:
                active_prof.base_safety_margin = DEFAULT_PRESETS[new_preset]["safety_margin"]
            pm.save_profiles()
    with col_pr2:
        st.markdown(f"""
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:5px;">
            Learned Bias:<br/><strong style="color:#06B6D4;">{active_prof.learned_adaptation_bias:+.2f}x</strong>
        </div>
        """, unsafe_allow_html=True)

    custom_margin = st.slider(
        "Base Safety Margin Factor",
        1.05, 2.20,
        float(active_prof.base_safety_margin),
        step=0.05,
        help="Multiplier applied above minimum theoretical slip force"
    )
    active_prof.base_safety_margin = custom_margin
    pm.save_profiles()

    st.markdown(f"""
    <div style="background:#0F172A; padding:8px 10px; border-radius:6px; border:1px solid #1E293B; font-size:0.75rem;">
        Effective Safety Margin: <strong style="color:#38BDF8;">{active_prof.effective_safety_margin:.2f}x</strong><br/>
        <span style="color:#64748B;">Trials Completed: {active_prof.total_trials} ({active_prof.successful_trials} stable)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 10px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)

    # 4. PREDICTION ALGORITHM
    st.markdown("##### 🧠 ML & Prediction Engine")
    algorithm_choice = st.radio(
        "Algorithm",
        ["Random Forest (ML)", "Physics Baseline Heuristic"],
        label_visibility="collapsed"
    )

    if algorithm_choice == "Random Forest (ML)":
        st.caption(f"🌲 **Random Forest Regressor** (Trained on synthetic demonstration data. $R^2={predictor.r2:.3f}$, MAE={predictor.mae:.2f}N)")
    else:
        st.caption("📐 **Physics Analytical Model** (Dual-shear opposing static friction)")


# ==============================================================================
# MAIN DIGITAL TWIN VIEW FUNCTION
# ==============================================================================
def render_digital_twin_view():
    # FORCE PREDICTION STEP
    prediction_res: PredictionResult = predictor.predict(
        obj=active_obj,
        user_safety_margin=active_prof.effective_safety_margin,
        algorithm=algorithm_choice
    )
    controller.configure(active_obj, base_target=prediction_res.recommended_target_n)

    # MAIN DASHBOARD HEADER & TELEMETRY ROW
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <div>
            <h2 style="margin:0; font-size:1.6rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">
                Prosthetic Hand Digital Twin
            </h2>
            <div style="font-size:0.80rem; color:#94A3B8;">
                Personalized Closed-Loop Tactile Grasp Strategy & Dynamic Slip Recovery
            </div>
        </div>
        <div style="text-align:right;">
            <span style="font-size:0.72rem; color:#64748B; text-transform:uppercase; letter-spacing:0.05em;">SYSTEM CLOCK</span><br/>
            <span style="font-family:monospace; color:#38BDF8; font-weight:700;">LIVE SIMULATION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Telemetry Cards
    latest_packet: SensorPacket = sensor.read_packet() if not controller.history else sensor.read_packet()
    current_measured_f = latest_packet.total_force if not controller.history else controller.history[-1].measured_force
    current_target_f = controller.current_target

    state_classes = {
        HandState.OPEN: ("pill-open", "OPEN (IDLE)"),
        HandState.CLOSING: ("pill-closing", "CLOSING"),
        HandState.GRASPING: ("pill-grasping", "GRASPING"),
        HandState.STABLE_GRIP: ("pill-stable", "STABLE GRIP"),
        HandState.SLIPPING: ("pill-slipping", "⚠️ SLIPPING"),
        HandState.RELEASING: ("pill-releasing", "RELEASING"),
    }
    pill_class, pill_text = state_classes.get(controller.state, ("pill-open", controller.state.value))

    min_slip_threshold = (active_obj.mass_g / 1000.0 * 9.80665) / (2.0 * max(0.1, active_obj.friction_coeff))
    safety_headroom = current_measured_f - min_slip_threshold

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prosthetic State</div>
            <div style="margin-top:6px;"><span class="state-pill {pill_class}">{pill_text}</span></div>
            <div class="metric-sub" style="margin-top:6px;">Flexion: {int(controller.flexion_ratio * 100)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Min Safe Force</div>
            <div class="metric-value" style="color:#06B6D4;">{prediction_res.recommended_target_n:.2f} <span style="font-size:0.85rem;">N</span></div>
            <div class="metric-sub">Base Heuristic: {prediction_res.baseline_heuristic_n:.2f}N</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Commanded Target</div>
            <div class="metric-value" style="color:#38BDF8;">{current_target_f:.2f} <span style="font-size:0.85rem;">N</span></div>
            <div class="metric-sub">Adaptation Boost: +{controller.slip_adaptation_total:.2f}N</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        color_meas = "#EF4444" if controller.state == HandState.SLIPPING else "#10B981"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Measured Grip Force</div>
            <div class="metric-value" style="color:{color_meas};">{current_measured_f:.2f} <span style="font-size:0.85rem;">N</span></div>
            <div class="metric-sub">Slip Threshold: {min_slip_threshold:.2f}N</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Structural Fragility Cap</div>
            <div class="metric-value" style="color:#F59E0B;">{active_obj.fragility_limit_n:.1f} <span style="font-size:0.85rem;">N</span></div>
            <div class="metric-sub">Cap: {controller.max_allowable_cap:.1f}N (92% safe)</div>
        </div>
        """, unsafe_allow_html=True)

    # ACTION LOG BANNER
    st.markdown(f"""
    <div class="action-banner">
        <strong>CONTROLLER TELEMETRY:</strong> {controller.last_action_note}
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # REPRODUCIBLE END-TO-END AI DEMONSTRATION SCENARIO CARD
    # --------------------------------------------------------------------------
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #172554 100%); border: 1px solid #1D4ED8; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:0.72rem; color:#60A5FA; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">
                        AUTOMATED RESEARCH BENCHMARK
                    </span>
                    <h4 style="margin:2px 0 0 0; color:#F8FAFC; font-size:1.05rem;">
                        🚀 End-to-End AI Demonstration Scenario
                    </h4>
                    <div style="font-size:0.78rem; color:#94A3B8; margin-top:2px;">
                        Executes full autonomous pipeline: Object Resolution → Model A Force Target → Kinematic Grasp → Slip Perturbation → Adaptive Controller Boost → Safe Restabilization → Trial Logging.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_sc1, col_sc2 = st.columns([3, 1.2])
        with col_sc1:
            st.caption(f"Active Scenario: **{active_obj.name}** | Target: **{prediction_res.recommended_target_n:.2f}N** | User Margin: **{active_prof.effective_safety_margin:.2f}x** | Fragility Ceiling: **{active_obj.fragility_limit_n:.1f}N**")
        with col_sc2:
            run_e2e_btn = st.button("▶ Run End-to-End AI Demo", type="primary", use_container_width=True)

        if run_e2e_btn:
            with st.spinner("Executing End-to-End Demonstration Sequence..."):
                demo_summary: DemonstrationExecutionSummary = execute_end_to_end_demonstration(
                    object_key=selected_key,
                    profile_name=active_prof.name,
                    custom_obj=active_obj,
                    predictor=predictor,
                    personalization_mgr=pm
                )
                st.session_state.last_demo_summary = demo_summary
                controller.history.clear()
                controller.reset(sensor)
                controller.configure(active_obj, base_target=demo_summary.initial_predicted_force_n)
                controller.start_grasp_sequence(sensor)
                for _ in range(10):
                    controller.step(sensor)
                controller.trigger_slip_event(sensor)
                for _ in range(8):
                    controller.step(sensor)
                st.session_state.active_feedback_msg = f"✅ End-to-End AI Demonstration Completed! Logged Trial: {demo_summary.trial_id_logged} ({demo_summary.outcome})"
                st.rerun()

        if st.session_state.last_demo_summary is not None:
            summary: DemonstrationExecutionSummary = st.session_state.last_demo_summary
            with st.expander(f"📜 Detailed Execution Log: {summary.scenario_name} (Trial {summary.trial_id_logged})", expanded=False):
                st.markdown(f"""
                <div style="background:#0B132B; border:1px solid #1E293B; border-radius:6px; padding:10px; font-size:0.78rem; margin-bottom:8px;">
                    <strong>Pipeline Outcome:</strong> <span style="color:#10B981; font-weight:700;">{summary.outcome}</span> | 
                    <strong>Model A Target:</strong> {summary.initial_predicted_force_n:.2f}N | 
                    <strong>Adapted Target:</strong> {summary.adapted_target_force_n:.2f}N (+{summary.force_boost_applied_n:.2f}N boost) | 
                    <strong>Fragility Headroom:</strong> {summary.fragility_headroom_n:.2f}N safe
                </div>
                """, unsafe_allow_html=True)
                step_rows = []
                for s in summary.steps_log:
                    step_rows.append({
                        "Stage": f"#{s.stage_number} {s.stage_name}",
                        "State": s.hand_state,
                        "Flexion": f"{s.flexion_percent}%",
                        "Target (N)": s.commanded_target_n,
                        "Measured (N)": s.simulated_measured_n,
                        "Slip Alert": "⚠️ YES" if s.slip_detected else "No",
                        "Telemetry Details": s.action_description
                    })
                st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)

    # MANUAL CONTROL DECK (INTERACTIVE SIMULATION TRIGGER)
    st.markdown("<div style='font-size:0.75rem; color:#64748B; font-weight:700; text-transform:uppercase; margin-top:6px; margin-bottom:4px;'>Manual Step-by-Step Simulation Controls:</div>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns([1.3, 1.3, 1.1, 1.1, 1.4])

    with col_btn1:
        if st.button("▶ Run Full Grasp Sequence", use_container_width=True):
            controller.start_grasp_sequence(sensor)
            for _ in range(10):
                controller.step(sensor)
            st.rerun()

    with col_btn2:
        if st.button("⚡ Inject Micro-Slip Event", use_container_width=True):
            if controller.state in [HandState.STABLE_GRIP, HandState.GRASPING]:
                controller.trigger_slip_event(sensor)
                for _ in range(8):
                    controller.step(sensor)
                st.rerun()
            else:
                st.warning("Grasp must be active/holding before triggering a slip event. Click 'Run Full Grasp Sequence' first.")

    with col_btn3:
        if st.button("⏭ Step Cycle", use_container_width=True):
            controller.step(sensor)
            st.rerun()

    with col_btn4:
        if st.button("↺ Open / Release", use_container_width=True):
            controller.release_grasp(sensor)
            for _ in range(7):
                controller.step(sensor)
            st.rerun()

    with col_btn5:
        if st.button("💾 Log Trial Record", use_container_width=True):
            st.session_state.trial_counter += 1
            trial_id = f"TR-{st.session_state.trial_counter:03d}"
            
            peak_f = max([p.measured_force for p in controller.history], default=current_measured_f)
            
            if controller.slip_counter > 0 and controller.state == HandState.STABLE_GRIP:
                outcome = "ADAPTED_SUCCESS"
                note = f"Micro-slip recovered automatically via +{controller.slip_adaptation_total:.2f}N adaptive boost."
            elif controller.state == HandState.STABLE_GRIP:
                outcome = "STABLE_SUCCESS"
                note = "Direct stable grip achieved without slip."
            elif current_measured_f >= active_obj.fragility_limit_n:
                outcome = "CRUSH_RISK"
                note = "Exceeded safe object fragility threshold."
            else:
                outcome = "MANUAL_CHECK"
                note = f"Logged at state {controller.state.value}."

            record = TrialRecord(
                trial_id=trial_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                object_name=active_obj.name,
                user_profile=active_prof.name,
                preset_mode=active_prof.preset_mode,
                initial_target_f=round(prediction_res.recommended_target_n, 2),
                final_target_f=round(controller.current_target, 2),
                peak_measured_f=round(peak_f, 2),
                slip_events=controller.slip_counter,
                outcome=outcome,
                feedback_rating="Ideal" if "SUCCESS" in outcome else "Pending",
                notes=note
            )
            pm.log_trial(record)
            st.session_state.active_feedback_msg = f"Logged trial {trial_id} ({outcome}) to profile {active_prof.name}!"
            st.rerun()

    if st.session_state.active_feedback_msg:
        st.success(st.session_state.active_feedback_msg)
        st.session_state.active_feedback_msg = ""

    # MAIN VISUALIZATION: VIRTUAL HAND SVG & PLOTLY TIME-SERIES
    col_viz_left, col_viz_right = st.columns([1, 1.25])

    with col_viz_left:
        st.markdown("##### 🦾 Digital Twin 5-Finger Articulation")
        st.markdown("<div style='font-size:0.72rem; color:#64748B; margin-bottom:6px;'>🟢 SIMULATED PIEZORESISTIVE FSR SIGNALS (±0.04N GAUSSIAN NOISE) • 100% SIMULATION MODE</div>", unsafe_allow_html=True)
        svg_markup = generate_hand_svg(
            state=controller.state,
            flexion_ratio=controller.flexion_ratio,
            current_force=current_measured_f,
            target_force=current_target_f,
            obj=active_obj,
            packet=latest_packet
        )
        components.html(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8">
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background-color: transparent;
                    overflow: hidden;
                    width: 100%;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
            </style>
            </head>
            <body>
                {svg_markup}
            </body>
            </html>
            """,
            height=510,
            scrolling=False
        )

        with st.expander("🖐️ Individual Fingertip Tactile Distribution", expanded=False):
            c_th, c_ix, c_mid, c_rg, c_pk = st.columns(5)
            with c_th:
                st.metric("Thumb", f"{latest_packet.thumb_force:.2f}N")
            with c_ix:
                st.metric("Index", f"{latest_packet.index_force:.2f}N")
            with c_mid:
                st.metric("Middle", f"{latest_packet.middle_force:.2f}N")
            with c_rg:
                st.metric("Ring", f"{latest_packet.ring_force:.2f}N")
            with c_pk:
                st.metric("Pinky", f"{latest_packet.pinky_force:.2f}N")

    with col_viz_right:
        st.markdown("##### 📈 Real-Time Closed-Loop Force Telemetry")
        if controller.history:
            df_hist = pd.DataFrame([
                {
                    "Step": p.step,
                    "Time (s)": p.time_offset_s,
                    "Target Force (N)": p.target_force,
                    "Measured Force (N)": p.measured_force,
                    "Slip": p.slip_detected,
                    "Fragility Limit (N)": p.fragility_limit,
                    "Slip Threshold (N)": p.min_slip_threshold
                }
                for p in controller.history
            ])
        else:
            df_hist = pd.DataFrame([{
                "Step": 0,
                "Time (s)": 0.0,
                "Target Force (N)": 0.0,
                "Measured Force (N)": 0.0,
                "Slip": False,
                "Fragility Limit (N)": active_obj.fragility_limit_n,
                "Slip Threshold (N)": min_slip_threshold
            }])

        fig = go.Figure()
        min_t = float(df_hist["Time (s)"].min())
        max_t = float(df_hist["Time (s)"].max())
        if max_t <= min_t:
            max_t = min_t + 1.0

        peak_force = max(
            float(df_hist["Target Force (N)"].max()),
            float(df_hist["Measured Force (N)"].max()),
            min_slip_threshold,
            1.0
        )
        if active_obj.fragility_limit_n <= peak_force * 1.8:
            y_ceiling = max(active_obj.fragility_limit_n + 1.5, peak_force + 2.0)
        else:
            y_ceiling = max(peak_force * 1.5 + 1.5, 6.0)

        # 1. Shaded Safe Grasp Corridor
        fig.add_trace(go.Scatter(
            x=[min_t, max_t, max_t, min_t],
            y=[min_slip_threshold, min_slip_threshold, min(y_ceiling, active_obj.fragility_limit_n), min(y_ceiling, active_obj.fragility_limit_n)],
            fill='toself',
            fillcolor='rgba(6, 182, 212, 0.07)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='Safe Operation Band'
        ))

        # 2. Fragility Limit
        if active_obj.fragility_limit_n <= y_ceiling:
            fig.add_trace(go.Scatter(
                x=[min_t, max_t],
                y=[active_obj.fragility_limit_n, active_obj.fragility_limit_n],
                mode='lines',
                name=f'Fragility Limit ({active_obj.fragility_limit_n:.0f}N)',
                line=dict(color='#EF4444', width=2.0, dash='dot')
            ))

        # 3. Minimum Slip Threshold
        fig.add_trace(go.Scatter(
            x=[min_t, max_t],
            y=[min_slip_threshold, min_slip_threshold],
            mode='lines',
            name=f'Slip Threshold ({min_slip_threshold:.2f}N)',
            line=dict(color='#F59E0B', width=1.8, dash='dash')
        ))

        # 4. Commanded Target Force
        fig.add_trace(go.Scatter(
            x=df_hist["Time (s)"],
            y=df_hist["Target Force (N)"],
            mode='lines',
            name='Commanded Target (N)',
            line=dict(color='#06B6D4', width=2.5)
        ))

        # 5. Simulated Measured Force
        fig.add_trace(go.Scatter(
            x=df_hist["Time (s)"],
            y=df_hist["Measured Force (N)"],
            mode='lines+markers',
            name='Measured Tactile Force (N)',
            line=dict(color='#38BDF8', width=2.2),
            marker=dict(size=4)
        ))

        # 6. Mark Slip Events on Graph
        slip_points = df_hist[df_hist["Slip"] == True]
        if not slip_points.empty:
            fig.add_trace(go.Scatter(
                x=slip_points["Time (s)"],
                y=slip_points["Measured Force (N)"],
                mode='markers',
                name='Micro-Slip Event',
                marker=dict(color='#EF4444', size=12, symbol='x-thin', line=dict(width=3, color='#EF4444'))
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0B132B",
            margin=dict(l=40, r=20, t=20, b=40),
            height=380,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(size=10)
            ),
            xaxis=dict(
                title="Simulation Time (seconds)",
                gridcolor="#1E293B",
                zerolinecolor="#1E293B"
            ),
            yaxis=dict(
                title="Force (Newtons)",
                gridcolor="#1E293B",
                zerolinecolor="#1E293B",
                range=[0, y_ceiling]
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model A Provenance & Input Features Card
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:10px 14px; margin-top:4px; font-size:0.77rem; color:#94A3B8;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong style="color:#F8FAFC;">🌲 Model A: Minimum-Safe-Force Regressor (Active)</strong>
                <span style="color:#06B6D4; font-family:monospace; font-weight:700;">RandomForestRegressor (N=50, d=7)</span>
            </div>
            <div style="margin-top:4px;">
                <strong>Source:</strong> GripWise Synthetic Demonstration Mechanics (N=1,500) • 
                <strong>R²:</strong> {predictor.r2:.4f} • <strong>MAE:</strong> {predictor.mae:.3f} N
            </div>
            <div style="margin-top:2px;">
                <strong>Expected Features:</strong> <code>[mass={active_obj.mass_g:.0f}g, μ={active_obj.friction_coeff:.2f}, comp={active_obj.compliance_mm_per_n:.2f}, frag={active_obj.fragility_limit_n:.0f}N, margin={active_prof.effective_safety_margin:.2f}x]</code>
            </div>
            <div style="margin-top:2px; color:#64748B;">
                <strong>Limitation:</strong> Synthetic mechanics estimation. Not clinically certified for patient biological load bearing.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # PERSONALIZATION & SELF-LEARNING INTERACTIVE CARD
    st.markdown("<hr style='margin: 16px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)
    col_learn1, col_learn2 = st.columns([1.2, 1])

    with col_learn1:
        st.markdown("##### 🧬 Self-Learning Grip Personalization")
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="color:#F8FAFC;">{active_prof.name}</strong> 
                    <span style="color:#64748B;">({active_prof.preset_mode} Mode)</span>
                </div>
                <div style="color:#06B6D4; font-weight:700;">
                    Adaptation Bias: {active_prof.learned_adaptation_bias:+.2f}x
                </div>
            </div>
            <p style="font-size:0.80rem; color:#94A3B8; margin: 8px 0;">
                Provide post-trial feedback to recalibrate safety margin factor. The controller
                progressively tunes clamp force recommendations for your preferred grip style.
            </p>
        </div>
        """, unsafe_allow_html=True)

        fb_c1, fb_c2, fb_c3 = st.columns(3)
        with fb_c1:
            if st.button("👍 Ideal Hold", use_container_width=True):
                delta = pm.apply_feedback_learning(active_prof.name, "Ideal")
                st.toast("Profile calibrated for steady hold!")
                st.rerun()
        with fb_c2:
            if st.button("⚠️ Too Loose (Slipped)", use_container_width=True):
                delta = pm.apply_feedback_learning(active_prof.name, "Too Loose (Slipped)")
                st.toast(f"Adapted: Increased safety margin bias by {delta:+.2f}x")
                st.rerun()
        with fb_c3:
            if st.button("⚡ Too Tight (Overclamp)", use_container_width=True):
                delta = pm.apply_feedback_learning(active_prof.name, "Too Tight")
                st.toast(f"Adapted: Decreased safety margin bias by {delta:+.2f}x")
                st.rerun()

    with col_learn2:
        st.markdown("##### 📊 ML Feature Importance (Model A)")
        importances_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in prediction_res.feature_importances.items()
        ]).sort_values("Importance", ascending=True)

        bar_fig = go.Figure(go.Bar(
            x=importances_df["Importance"],
            y=importances_df["Feature"],
            orientation='h',
            marker=dict(color='#06B6D4')
        ))
        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0B132B",
            margin=dict(l=10, r=10, t=10, b=20),
            height=140,
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(title="", showgrid=False)
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # EXPERIMENT LOG & CSV/JSON EXPORT, IMPORT, BACKUP
    st.markdown("<hr style='margin: 16px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)
    st.markdown("##### 📜 Experiment Trial History & Record Management")

    # Filter Controls
    c_flt1, c_flt2, c_flt3, c_flt4 = st.columns([1.5, 1.5, 1.5, 2])
    with c_flt1:
        flt_obj = st.selectbox("Filter Object", ["All Objects"] + list(STANDARD_OBJECTS.keys()), index=0)
    with c_flt2:
        flt_prof = st.selectbox("Filter Profile", ["All Profiles"] + list(pm.profiles.keys()), index=0)
    with c_flt3:
        flt_out = st.selectbox("Filter Outcome", ["All Outcomes", "SUCCESS", "ADAPTED", "CRUSH_RISK"], index=0)
    with c_flt4:
        flt_query = st.text_input("Search Trials (ID, Notes)", placeholder="e.g. TR-001, slip, soft")

    filtered_trials = pm.filter_trials(
        object_name=flt_obj,
        profile_name=flt_prof,
        outcome_filter=flt_out,
        search_query=flt_query
    )
    trials_df = pm.get_trials_dataframe(filtered_trials)

    # Action buttons: Export CSV, Export JSON, Backup
    c_act1, c_act2, c_act3 = st.columns([1, 1, 1.5])
    with c_act1:
        csv_bytes = trials_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Trials (CSV)",
            data=csv_bytes,
            file_name=f"gripwise_trials_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_act2:
        json_str = pm.export_trials_json()
        st.download_button(
            label="📥 Export Trials (JSON)",
            data=json_str.encode('utf-8'),
            file_name=f"gripwise_trials_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    with c_act3:
        if st.button("💾 Create System Snapshot Backup", use_container_width=True):
            b_name = pm.create_backup()
            st.toast(f"Created backup: {b_name} in data/backups/")
            st.rerun()

    # Dataframe display
    st.dataframe(
        trials_df,
        use_container_width=True,
        height=210,
        column_config={
            "Init Target (N)": st.column_config.NumberColumn(format="%.2f N"),
            "Final Target (N)": st.column_config.NumberColumn(format="%.2f N"),
            "Peak Measured (N)": st.column_config.NumberColumn(format="%.2f N"),
            "Slips": st.column_config.NumberColumn(format="%d"),
        }
    )

    # Expander: Import Trials & Restore from Backup
    with st.expander("🛠️ Advanced Experiment Tools (Import JSON / Restore Backup)", expanded=False):
        c_imp, c_res = st.columns(2)
        with c_imp:
            st.markdown("###### 📤 Import External Trial Records (JSON)")
            uploaded_file = st.file_uploader("Upload trial records JSON", type=["json"], label_visibility="collapsed")
            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8")
                if st.button("Merge Imported Records"):
                    count, errs = pm.import_trials_json(content, overwrite=False)
                    if count > 0:
                        st.success(f"Successfully imported {count} trial records!")
                        st.rerun()
                    else:
                        st.error(f"Import failed: {errs}")

        with c_res:
            st.markdown("###### 🔄 Restore System Snapshot")
            backups = pm.list_backups()
            if backups:
                b_options = [b["filename"] for b in backups]
                chosen_b = st.selectbox("Select Snapshot", b_options, label_visibility="collapsed")
                if st.button("Restore from Selected Snapshot"):
                    ok, msg = pm.restore_backup(chosen_b)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.caption("No backup snapshots found in data/backups/.")

    # TRANSPARENT SCIENTIFIC DISCLAIMER & ARCHITECTURE FOOTER
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ SCIENTIFIC DEMONSTRATION & SAFETY DISCLAIMER:</strong><br/>
        This software is a simulation and algorithmic proof-of-concept prototype for the GripWise research project.
        All object mechanics, friction coefficients, tactile readings, and minimum-safe-force values are 
        <strong>demonstration estimates derived from synthetic modeling</strong>.
        They are <strong>NOT clinically validated</strong>, nor guaranteed safe bounds for real functional prosthetic hands.
        Physical hardware deployment requires calibrated piezoresistive strain gauges, functional force sensors,
        and mechanical bench validation.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🛠️ Architecture & Future Hardware Integration Guide (ESP32/Arduino)", expanded=False):
        st.markdown("""
        ### Hardware Integration Bridge
        The `hardware.py` module uses an abstract interface (`BaseSensorInterface`).
        To connect a physical ESP32 or Arduino microcontroller:
        
        1. **Firmware Packet Specification (UART 115200 Baud):**
           Transmit a newline-terminated comma-separated string every 20ms:
           ```
           <total_force_n>,<slip_flag_0_or_1>,<thumb_n>,<index_n>,<middle_n>,<ring_n>,<pinky_n>,<motor_current_ma>\\n
           ```
           *Example:* `3.85,0,1.54,0.85,0.77,0.42,0.27,245.0`
        
        2. **Actuator Command Ingestion:**
           The dashboard transmits target clamp commands over serial:
           ```
           TARGET:<force_in_newtons>\\n
           ```
           Your ESP32 servo/tendon controller translates this to PWM pulses for finger tendon spooling.
        
        3. **Sensor Layout:**
           - **Thumb, Index, Middle, Ring, Pinky Tips:** 5x Piezoresistive force sensors (e.g. FSR 402 or Velostat matrix).
           - **Palm / Shear:** Piezoelectric vibration disc or optical flow sensor for micro-slip shear detection.
        """)


# ==============================================================================
# MAIN ROUTING EXECUTION
# ==============================================================================
if view_selection == "🔬 Data & AI Research Lab":
    render_data_lab(dm)
elif view_selection == "🔌 Hardware Integration & Test Bench":
    render_hardware_view(controller, sensor)
elif view_selection == "🧪 Offline Research Inference Studio":
    render_offline_inference_studio(dm)
elif view_selection == "⚖️ Model vs Baseline Comparison":
    render_model_comparison_deck(dm)
else:
    render_digital_twin_view()


