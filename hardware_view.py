"""
hardware_view.py - Hardware Integration & Hardware-Free Test Bench View
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Provides a complete, hardware-ready test bench and UART integration workbench:
1. Mode Switcher: Simulation, Mock Serial, Recorded Replay, Physical ESP32.
2. Actuator Safety & Software Emergency Stop (Disarmed by default).
3. Packet Diagnostics & Raw Telemetry Inspector.
4. Hardware-Free Fault Injection Bench (malformed packets, stalls, drift).
5. Recorded Telemetry Player with scrub and loop controls.
6. ESP32 Firmware Protocol Specification Reference.
"""

import time
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hardware import (
    BaseSensorInterface,
    SimulatedSensorInterface,
    MockSerialSensorInterface,
    RecordedReplaySensorInterface,
    SerialSensorInterface,
    SensorPacket,
    PacketParser,
    REPLAY_DIR
)
from controller import AdaptiveGripController


def render_hardware_view(
    controller: AdaptiveGripController,
    active_sensor: BaseSensorInterface
):
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <div>
            <h2 style="margin:0; font-size:1.55rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">
                🔌 Hardware Integration & Software Test Bench
            </h2>
            <div style="font-size:0.80rem; color:#94A3B8;">
                Pre-Hardware Verification Suite • Mock Serial Engine • Recorded Replay • UART Protocol
            </div>
        </div>
        <div>
            <span style="background:rgba(239, 68, 68, 0.15); color:#FCA5A5; border:1px solid #EF4444; padding:4px 10px; border-radius:9999px; font-size:0.75rem; font-weight:700;">
                NO PHYSICAL HW REQUIRED
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 1. ACTUATOR ARMING & SOFTWARE EMERGENCY STOP (CRITICAL SAFETY)
    # --------------------------------------------------------------------------
    st.markdown("""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong style="color:#F8FAFC; font-size:0.95rem;">Actuator Power & Safety Watchdog</strong>
                <div style="font-size:0.78rem; color:#94A3B8; margin-top:2px;">
                    Actuator commands are <strong>DISARMED BY DEFAULT</strong>. Commands to servos/tendons are suppressed until explicitly armed.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_arm1, c_arm2, c_arm3, c_arm4 = st.columns([1.2, 1.2, 1.6, 2])

    with c_arm1:
        if active_sensor.emergency_stopped:
            st.error("🚨 E-STOP ACTIVE")
            if st.button("🔄 Reset E-Stop Latch", use_container_width=True):
                active_sensor.reset_emergency_stop()
                st.rerun()
        elif active_sensor.is_armed():
            st.success("⚡ ACTUATORS ARMED")
            if st.button("🛡️ Disarm Actuators", use_container_width=True):
                active_sensor.disarm_actuators()
                st.rerun()
        else:
            st.warning("🔒 DISARMED (SAFE)")
            if st.button("⚡ Arm Actuators", type="secondary", use_container_width=True):
                active_sensor.arm_actuators()
                st.rerun()

    with c_arm2:
        if st.button("🚨 EMERGENCY STOP", type="primary", use_container_width=True):
            active_sensor.emergency_stop()
            controller.emergency_stop(active_sensor)
            st.toast("EMERGENCY STOP TRIGGERED: Force dumped to 0.0N, actuators disarmed.")
            st.rerun()

    with c_arm3:
        new_lim = st.slider(
            "Software Force Limit (N)",
            10.0, 45.0,
            float(active_sensor.max_software_limit_n),
            step=1.0,
            help="Software ceiling capping all commanded forces"
        )
        active_sensor.max_software_limit_n = new_lim

    with c_arm4:
        st.markdown(f"""
        <div style="font-size:0.75rem; color:#64748B; background:#0B132B; padding:8px 10px; border-radius:6px; border:1px solid #1E293B;">
            <strong>Hardware Safety Notice:</strong> Software limits and e-stops operate on application logic.
            Physical prosthetics require hardware current limits, physical limit switches, and thermal fuses.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 12px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. OPERATING MODE SELECTOR
    # --------------------------------------------------------------------------
    st.markdown("##### ⚙️ Active Sensor & Telemetry Source")
    mode_tabs = st.tabs([
        "1. Mock Serial Stream (Wire-Protocol Test)",
        "2. Recorded Telemetry Replay",
        "3. Simulation Engine (Digital Twin)",
        "4. Physical ESP32 Bridge (PySerial UART)"
    ])

    # TAB 1: MOCK SERIAL
    with mode_tabs[0]:
        _render_mock_serial_tab(controller, active_sensor)

    # TAB 2: RECORDED REPLAY
    with mode_tabs[1]:
        _render_replay_tab(controller, active_sensor)

    # TAB 3: SIMULATION ENGINE
    with mode_tabs[2]:
        _render_simulation_tab(controller, active_sensor)

    # TAB 4: PHYSICAL ESP32 UART
    with mode_tabs[3]:
        _render_serial_tab(controller, active_sensor)

    # --------------------------------------------------------------------------
    # 3. LIVE PACKET DIAGNOSTICS & RAW PROTOCOL INSPECTOR
    # --------------------------------------------------------------------------
    st.markdown("<hr style='margin: 16px 0; border-color:#1E293B;'/>", unsafe_allow_html=True)
    st.markdown("##### 📊 Live Telemetry Diagnostics & Wire-Protocol Terminal")

    pkt = active_sensor.read_packet()

    c_diag1, c_diag2, c_diag3, c_diag4 = st.columns(4)
    with c_diag1:
        st.metric("Source Tag", pkt.source)
    with c_diag2:
        status_text = "VALID" if pkt.is_valid else "ERROR"
        st.metric("Packet Health", status_text, delta="Stale" if pkt.is_stale else "Fresh")
    with c_diag3:
        st.metric("Total Force (N)", f"{pkt.total_force:.2f} N")
    with c_diag4:
        st.metric("Motor Current (mA)", f"{pkt.motor_current_ma:.1f} mA")

    col_raw, col_bars = st.columns([1.2, 1])

    with col_raw:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; font-family:monospace; font-size:0.80rem;">
            <div style="color:#64748B; font-weight:700; margin-bottom:4px;">RAW UART ASCII BYTESTREAM:</div>
            <div style="color:#38BDF8; background:#0B132B; padding:8px; border-radius:4px; overflow-x:auto;">
                {pkt.raw_packet or '[NO BYTES RECEIVED]'}
            </div>
            <div style="margin-top:8px; color:#94A3B8; font-size:0.75rem;">
                <strong>Packet Fields:</strong> <code>&lt;total&gt;,&lt;slip&gt;,&lt;thumb&gt;,&lt;index&gt;,&lt;middle&gt;,&lt;ring&gt;,&lt;pinky&gt;,&lt;motor_ma&gt;</code><br/>
                <strong>Parse Status:</strong> {'✅ Successfully parsed' if pkt.is_valid else f'❌ Error: {pkt.parse_error}'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_bars:
        ff = pkt.finger_forces
        fig_bar = go.Figure(go.Bar(
            x=list(ff.keys()),
            y=list(ff.values()),
            marker_color=["#0EA5E9", "#06B6D4", "#10B981", "#F59E0B", "#EC4899"]
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0B132B",
            height=160,
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis=dict(title="Force (N)", range=[0, max(5.0, max(ff.values(), default=1.0) + 1.0)])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --------------------------------------------------------------------------
    # 4. ESP32 FIRMWARE PROTOCOL SPECIFICATION REFERENCE
    # --------------------------------------------------------------------------
    with st.expander("📖 ESP32 / Arduino Firmware Wire Protocol Specification", expanded=False):
        st.markdown("""
        ### Telemetry Wire Protocol (Microcontroller to PC)
        Every 20 milliseconds (50 Hz), the ESP32 outputs a newline-terminated comma-separated ASCII packet:
        ```text
        <total_force>,<slip_flag>,<thumb_force>,<index_force>,<middle_force>,<ring_force>,<pinky_force>,<current_ma>\\n
        ```
        - `total_force`: Floating-point scalar sum in Newtons (e.g. `3.85`).
        - `slip_flag`: Integer `0` (stable) or `1` (slip shear detected).
        - `thumb..pinky`: 5x individual fingertip force sensors in Newtons.
        - `current_ma`: Actuator motor current consumption in milliamperes (e.g. `245.0`).
        
        *Example Line:* `3.85,0,1.54,0.85,0.77,0.42,0.27,245.0\\n`

        ---

        ### Actuator Command Protocol (PC to Microcontroller)
        Commands transmitted from GripWise to the ESP32:
        - `ARM\\n`: Enables motor PWM drivers. Response: `ACK:ARM\\n`
        - `DISARM\\n`: Disables motor PWM (safe freewheel). Response: `ACK:DISARM\\n`
        - `TARGET:<force_n>\\n`: Sets commanded closed-loop grip force target (e.g. `TARGET:4.20\\n`).
        - `EMERGENCY_STOP\\n`: Immediate zero-force command and driver cutoff. Response: `ACK:ESTOP\\n`
        - `PING\\n`: Watchdog heartbeat. Response: `PONG\\n`
        """)


# -----------------------------------------------------------------------------
# TAB HELPERS
# -----------------------------------------------------------------------------
def _render_mock_serial_tab(controller: AdaptiveGripController, active_sensor: BaseSensorInterface):
    st.markdown("""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:10px; font-size:0.80rem; margin-bottom:10px;">
        <strong>Mock Serial Stream:</strong> Emulates physical ESP32 UART transmission without hardware.
        Feeds generated ASCII bytestrings into the exact same <code>PacketParser</code> used for physical microcontrollers.
    </div>
    """, unsafe_allow_html=True)

    if not isinstance(active_sensor, MockSerialSensorInterface):
        if st.button("🔌 Switch to Mock Serial Stream", use_container_width=True):
            st.session_state.sensor_interface = MockSerialSensorInterface()
            st.rerun()
        st.info("Click the button above to activate the Mock Serial Stream.")
        return

    # Fault Injection Controls
    st.markdown("##### 🧪 Fault & Edge-Case Injection Bench")
    c_f1, c_f2, c_f3, c_f4 = st.columns(4)

    with c_f1:
        if st.button("⚠️ Inject Malformed Packet", use_container_width=True):
            active_sensor.inject_corrupt_packet()
            st.toast("Injected malformed ASCII packet into parser.")
            st.rerun()

    with c_f2:
        is_stalled = active_sensor.simulated_stall
        stall_label = "▶ Resume Stream" if is_stalled else "❄️ Simulate Stream Stall"
        if st.button(stall_label, use_container_width=True):
            active_sensor.set_stall(not is_stalled)
            st.rerun()

    with c_f3:
        if st.button("⚡ Inject Shear Slip", use_container_width=True):
            active_sensor.trigger_slip_event()
            st.toast("Triggered micro-slip flag in mock packet.")
            st.rerun()

    with c_f4:
        if st.button("🔄 Step Mock Cycle", use_container_width=True):
            controller.step(active_sensor)
            st.rerun()

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        active_sensor.noise_std = st.slider("Sensor Noise (Std Dev)", 0.0, 0.20, float(active_sensor.noise_std), step=0.01)
    with c_p2:
        active_sensor.drift = st.slider("Sensor DC Bias Drift (N)", -1.0, 1.0, float(active_sensor.drift), step=0.05)


def _render_replay_tab(controller: AdaptiveGripController, active_sensor: BaseSensorInterface):
    st.markdown("""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:10px; font-size:0.80rem; margin-bottom:10px;">
        <strong>Recorded Telemetry Replay:</strong> Replays pre-recorded experimental trials or stress traces.
        Ideal for deterministic benchmarking and repeatable regression tests.
    </div>
    """, unsafe_allow_html=True)

    if not isinstance(active_sensor, RecordedReplaySensorInterface):
        if st.button("📼 Switch to Recorded Replay Engine", use_container_width=True):
            st.session_state.sensor_interface = RecordedReplaySensorInterface()
            st.rerun()
        st.info("Click the button above to activate the Telemetry Replay Engine.")
        return

    diag = active_sensor.get_diagnostics()
    total_frames = len(active_sensor.df) if not active_sensor.df.empty else 1

    c_r1, c_r2, c_r3, c_r4 = st.columns([1, 1, 1, 2])
    with c_r1:
        if st.button("⏮ Reset to Frame 0", use_container_width=True):
            active_sensor.reset_replay()
            st.rerun()
    with c_r2:
        play_label = "⏸ Pause" if active_sensor.is_playing else "▶ Play"
        if st.button(play_label, use_container_width=True):
            active_sensor.is_playing = not active_sensor.is_playing
            st.rerun()
    with c_r3:
        if st.button("⏭ Step Single Frame", use_container_width=True):
            active_sensor.step_replay()
            controller.step(active_sensor)
            st.rerun()
    with c_r4:
        active_sensor.loop = st.checkbox("Loop Recording", value=active_sensor.loop)

    new_frame = st.slider(
        "Replay Scrubber (Frame Index)",
        0, max(0, total_frames - 1),
        int(active_sensor.current_idx)
    )
    if new_frame != active_sensor.current_idx:
        active_sensor.current_idx = new_frame
        controller.step(active_sensor)
        st.rerun()


def _render_simulation_tab(controller: AdaptiveGripController, active_sensor: BaseSensorInterface):
    st.markdown("""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:10px; font-size:0.80rem; margin-bottom:10px;">
        <strong>Physics Simulation Engine:</strong> First-order dynamic response modeling compliance delay,
        Gaussian noise (±0.04N), and micro-slip friction decay.
    </div>
    """, unsafe_allow_html=True)

    if not isinstance(active_sensor, SimulatedSensorInterface):
        if st.button("🟢 Switch to Simulation Engine", use_container_width=True):
            st.session_state.sensor_interface = SimulatedSensorInterface()
            st.rerun()
    else:
        st.success("🟢 Simulation Engine is currently ACTIVE.")


def _render_serial_tab(controller: AdaptiveGripController, active_sensor: BaseSensorInterface):
    st.markdown("""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:10px; font-size:0.80rem; margin-bottom:10px;">
        <strong>Physical ESP32 / Arduino Bridge:</strong> Connects to a physical microcontroller via USB-UART serial.
        <em>Note: If no physical microcontroller is plugged in, use the Mock Serial or Simulation modes.</em>
    </div>
    """, unsafe_allow_html=True)

    if not isinstance(active_sensor, SerialSensorInterface):
        if st.button("🔌 Switch to Serial Bridge", use_container_width=True):
            st.session_state.sensor_interface = SerialSensorInterface()
            st.rerun()
        return

    c_s1, c_s2, c_s3 = st.columns([2, 1, 1.5])
    with c_s1:
        port = st.text_input("COM Port", value=active_sensor.port)
    with c_s2:
        baud = st.selectbox("Baud Rate", [115200, 9600, 57600], index=0)
    with c_s3:
        if active_sensor.is_connected():
            if st.button("Disconnect Serial", use_container_width=True):
                active_sensor.disconnect()
                st.rerun()
        else:
            if st.button("Connect Serial", use_container_width=True):
                active_sensor.connect(port, baud)
                st.rerun()

    if active_sensor.is_connected():
        st.success(f"Connected to {active_sensor.port} at {active_sensor.baud} baud.")
    else:
        st.info("No physical hardware detected. Running in disconnected fallback state.")

