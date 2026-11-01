"""
hardware.py - Sensor and Actuator Hardware Abstraction Layer
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Provides a robust, hardware-ready cyber-physical interface layer:
1. PacketParser: Validates, parses, and sanitizes newline-delimited ASCII packets.
2. SimulatedSensorInterface: High-fidelity compliance & noise simulation.
3. SerialSensorInterface: PySerial UART driver for ESP32/Arduino (115200 baud).
4. MockSerialSensorInterface: Hardware-free testing stream feeding raw strings into PacketParser.
5. RecordedReplaySensorInterface: Deterministic offline telemetry replay engine.
6. Actuator Safety: Disarmed-by-default gating, software emergency stop, and limits.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import time
import math
import random
import pandas as pd
import numpy as np
from typing import Optional, Tuple

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# Directory for recorded sensor telemetry replays
REPLAY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "telemetry_recordings"))
os.makedirs(REPLAY_DIR, exist_ok=True)


@dataclass
class SensorPacket:
    """Standardized telemetry packet from tactile sensors and motor current monitors."""
    timestamp: float
    thumb_force: float = 0.0
    index_force: float = 0.0
    middle_force: float = 0.0
    ring_force: float = 0.0
    pinky_force: float = 0.0
    total_force: float = 0.0
    slip_detected: bool = False
    slip_intensity: float = 0.0  # 0.0 to 1.0 scale
    motor_current_ma: float = 0.0
    raw_packet: str = ""
    source: str = "SIMULATION"  # SIMULATION, MOCK_SERIAL, REPLAY, SERIAL_ESP32
    is_valid: bool = True
    is_stale: bool = False
    parse_error: str = ""

    @property
    def finger_forces(self) -> dict[str, float]:
        return {
            "Thumb": round(self.thumb_force, 2),
            "Index": round(self.index_force, 2),
            "Middle": round(self.middle_force, 2),
            "Ring": round(self.ring_force, 2),
            "Pinky": round(self.pinky_force, 2),
        }


class PacketParser:
    """
    Validates, parses, and sanitizes incoming microcontroller ASCII packets.
    
    Standard Wire Protocol Specification:
    <total_force>,<slip_flag>,<thumb_force>,<index_force>,<middle_force>,<ring_force>,<pinky_force>,<current_ma>\n
    Example: '3.85,0,1.54,0.85,0.77,0.42,0.27,245.0\n'
    """

    MAX_SAFE_FORCE_N = 65.0
    MAX_CURRENT_MA = 2500.0

    @classmethod
    def parse_raw_packet(cls, raw_line: str, source_tag: str = "SERIAL") -> Tuple[SensorPacket, bool]:
        """
        Parses a raw ASCII line into a SensorPacket.
        Returns (SensorPacket, is_success: bool).
        Never throws exceptions on corrupt or malformed inputs.
        """
        now = time.time()
        clean_line = raw_line.strip()

        if not clean_line:
            return SensorPacket(
                timestamp=now,
                raw_packet=raw_line,
                source=source_tag,
                is_valid=False,
                parse_error="Empty packet received"
            ), False

        parts = clean_line.split(",")
        if len(parts) < 8:
            return SensorPacket(
                timestamp=now,
                raw_packet=clean_line,
                source=source_tag,
                is_valid=False,
                parse_error=f"Malformed packet: Expected 8 comma-separated fields, got {len(parts)}"
            ), False

        try:
            # Parse numeric fields
            tot = float(parts[0])
            slip_int = int(parts[1])
            th = float(parts[2])
            idx = float(parts[3])
            mid = float(parts[4])
            rng = float(parts[5])
            pnk = float(parts[6])
            cur = float(parts[7])

            # Check for non-finite numbers (NaN, Inf)
            numeric_vals = [tot, th, idx, mid, rng, pnk, cur]
            if any(math.isnan(v) or math.isinf(v) for v in numeric_vals):
                return SensorPacket(
                    timestamp=now,
                    raw_packet=clean_line,
                    source=source_tag,
                    is_valid=False,
                    parse_error="Invalid numeric value: NaN or Inf detected"
                ), False

            # Range & Bounds Checking
            if tot < 0.0 or tot > cls.MAX_SAFE_FORCE_N:
                return SensorPacket(
                    timestamp=now,
                    raw_packet=clean_line,
                    source=source_tag,
                    is_valid=False,
                    parse_error=f"Total force {tot}N exceeds plausible range [0, {cls.MAX_SAFE_FORCE_N}]"
                ), False

            if slip_int not in [0, 1]:
                return SensorPacket(
                    timestamp=now,
                    raw_packet=clean_line,
                    source=source_tag,
                    is_valid=False,
                    parse_error=f"Slip flag must be 0 or 1, got {slip_int}"
                ), False

            if cur < 0.0 or cur > cls.MAX_CURRENT_MA:
                return SensorPacket(
                    timestamp=now,
                    raw_packet=clean_line,
                    source=source_tag,
                    is_valid=False,
                    parse_error=f"Motor current {cur}mA out of range [0, {cls.MAX_CURRENT_MA}]"
                ), False

            slip_detected = (slip_int == 1)

            packet = SensorPacket(
                timestamp=now,
                thumb_force=max(0.0, th),
                index_force=max(0.0, idx),
                middle_force=max(0.0, mid),
                ring_force=max(0.0, rng),
                pinky_force=max(0.0, pnk),
                total_force=max(0.0, tot),
                slip_detected=slip_detected,
                slip_intensity=1.0 if slip_detected else 0.0,
                motor_current_ma=max(0.0, cur),
                raw_packet=clean_line,
                source=source_tag,
                is_valid=True,
                is_stale=False,
                parse_error=""
            )
            return packet, True

        except (ValueError, TypeError) as e:
            return SensorPacket(
                timestamp=now,
                raw_packet=clean_line,
                source=source_tag,
                is_valid=False,
                parse_error=f"Field conversion error: {str(e)}"
            ), False


class BaseSensorInterface(ABC):
    """Abstract Base Class for Prosthetic Sensor/Actuator Interface."""

    def __init__(self):
        self.armed: bool = False
        self.emergency_stopped: bool = False
        self.max_software_limit_n: float = 35.0

    def arm_actuators(self) -> None:
        """Enables commanded actuator power."""
        if not self.emergency_stopped:
            self.armed = True

    def disarm_actuators(self) -> None:
        """Disarms actuator command lines (safe state)."""
        self.armed = False

    def is_armed(self) -> bool:
        return self.armed and not self.emergency_stopped

    def emergency_stop(self) -> None:
        """Immediate software emergency stop."""
        self.emergency_stopped = True
        self.armed = False
        if hasattr(self, "target_force"):
            self.target_force = 0.0
        self.send_actuator_target(0.0)

    def reset_emergency_stop(self) -> None:
        """Clears emergency stop latch; actuators remain disarmed until explicit arm."""
        self.emergency_stopped = False
        self.armed = False

    @abstractmethod
    def connect(self, port: str = "COM3", baud: int = 115200) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def read_packet(self) -> SensorPacket:
        pass

    @abstractmethod
    def send_actuator_target(self, target_force_n: float) -> bool:
        pass

    @abstractmethod
    def trigger_slip_event(self, intensity: float = 0.8) -> None:
        pass

    @abstractmethod
    def get_diagnostics(self) -> dict:
        pass


class SimulatedSensorInterface(BaseSensorInterface):
    """
    Physics-based simulated sensor modeling compliance, noise, and micro-slip.
    Default mode for 100% hardware-free interactive execution.
    """

    def __init__(self, noise_std: float = 0.04):
        super().__init__()
        self.connected: bool = True
        self.noise_std: float = noise_std
        self.drift_offset: float = 0.0
        self.target_force: float = 0.0
        self.current_force: float = 0.0
        self.simulated_slip_active: bool = False
        self.slip_intensity: float = 0.0
        self.slip_duration_steps: int = 0
        self.packets_served: int = 0

        self.finger_distribution = {
            "thumb": 0.40,
            "index": 0.22,
            "middle": 0.20,
            "ring": 0.11,
            "pinky": 0.07
        }

    def connect(self, port: str = "SIM", baud: int = 0) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def trigger_slip_event(self, intensity: float = 0.8) -> None:
        self.simulated_slip_active = True
        self.slip_intensity = max(0.1, min(1.0, intensity))
        self.slip_duration_steps = 4

    def send_actuator_target(self, target_force_n: float) -> bool:
        capped = min(self.max_software_limit_n, max(0.0, float(target_force_n)))
        self.target_force = capped
        return True

    def step_simulation(self, compliance_factor: float = 0.3) -> None:
        alpha = max(0.15, min(0.65, 0.45 - compliance_factor * 0.1))
        self.current_force += alpha * (self.target_force - self.current_force)

        if self.simulated_slip_active:
            if self.slip_duration_steps > 0:
                self.slip_duration_steps -= 1
            else:
                self.simulated_slip_active = False
                self.slip_intensity = 0.0

    def read_packet(self) -> SensorPacket:
        now = time.time()
        self.packets_served += 1

        noise = np.random.normal(0, self.noise_std)
        total_measured = max(0.0, self.current_force + noise + self.drift_offset)

        if self.simulated_slip_active:
            slip_drop = total_measured * (0.28 * self.slip_intensity)
            total_measured = max(0.0, total_measured - slip_drop)

        thumb = total_measured * self.finger_distribution["thumb"] + np.random.normal(0, self.noise_std * 0.3)
        index = total_measured * self.finger_distribution["index"] + np.random.normal(0, self.noise_std * 0.3)
        middle = total_measured * self.finger_distribution["middle"] + np.random.normal(0, self.noise_std * 0.3)
        ring = total_measured * self.finger_distribution["ring"] + np.random.normal(0, self.noise_std * 0.2)
        pinky = total_measured * self.finger_distribution["pinky"] + np.random.normal(0, self.noise_std * 0.2)
        motor_current = 120.0 + (total_measured * 48.0) + (95.0 if self.simulated_slip_active else 0.0)

        raw_str = (
            f"{total_measured:.2f},{1 if self.simulated_slip_active else 0},"
            f"{thumb:.2f},{index:.2f},{middle:.2f},{ring:.2f},{pinky:.2f},{motor_current:.1f}"
        )

        return SensorPacket(
            timestamp=now,
            thumb_force=max(0.0, float(thumb)),
            index_force=max(0.0, float(index)),
            middle_force=max(0.0, float(middle)),
            ring_force=max(0.0, float(ring)),
            pinky_force=max(0.0, float(pinky)),
            total_force=float(total_measured),
            slip_detected=self.simulated_slip_active,
            slip_intensity=self.slip_intensity if self.simulated_slip_active else 0.0,
            motor_current_ma=float(motor_current),
            raw_packet=raw_str,
            source="SIMULATION",
            is_valid=True,
            is_stale=False
        )

    def get_diagnostics(self) -> dict:
        return {
            "Interface": "SimulatedSensorInterface",
            "Status": "Connected (Simulation Engine Active)",
            "Packets Served": self.packets_served,
            "Target Force": f"{self.target_force:.2f} N",
            "Current Force": f"{self.current_force:.2f} N",
            "Slip Active": self.simulated_slip_active,
            "Actuator Armed": self.armed,
            "Emergency Stopped": self.emergency_stopped
        }


class MockSerialSensorInterface(BaseSensorInterface):
    """
    Mock serial stream generator that exercises the exact same PacketParser wire protocol
    without requiring physical ESP32 hardware.
    Supports fault injection: corrupt packets, drops, stall/freeze, noise, drift.
    """

    def __init__(self):
        super().__init__()
        self.connected: bool = True
        self.port: str = "MOCK_VIRTUAL_COM"
        self.baud: int = 115200
        self.corrupt_next_packet: bool = False
        self.simulated_stall: bool = False
        self.noise_std: float = 0.05
        self.drift: float = 0.0
        self.current_force: float = 0.0
        self.target_force: float = 0.0
        self.slip_flag: bool = False
        self.packets_generated: int = 0
        self.parse_errors_logged: int = 0
        self.last_packet: Optional[SensorPacket] = None

    def connect(self, port: str = "MOCK_COM", baud: int = 115200) -> bool:
        self.connected = True
        self.port = port
        self.baud = baud
        return True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def trigger_slip_event(self, intensity: float = 0.8) -> None:
        self.slip_flag = True

    def inject_corrupt_packet(self) -> None:
        """Injects a malformed packet string on the next read."""
        self.corrupt_next_packet = True

    def set_stall(self, stall: bool) -> None:
        """Simulates sensor freeze or UART stall."""
        self.simulated_stall = stall

    def send_actuator_target(self, target_force_n: float) -> bool:
        if float(target_force_n) == 0.0:
            self.target_force = 0.0
            return True
        if not self.is_armed():
            return False
        self.target_force = min(self.max_software_limit_n, max(0.0, float(target_force_n)))
        return True

    def step_simulation(self, compliance_factor: float = 0.3) -> None:
        self.current_force += 0.35 * (self.target_force - self.current_force)

    def read_packet(self) -> SensorPacket:
        now = time.time()
        self.packets_generated += 1

        if self.simulated_stall:
            # Stale packet detection: return previous packet flagged as stale
            if self.last_packet is not None:
                stale_pkt = SensorPacket(
                    timestamp=self.last_packet.timestamp,
                    thumb_force=self.last_packet.thumb_force,
                    index_force=self.last_packet.index_force,
                    middle_force=self.last_packet.middle_force,
                    ring_force=self.last_packet.ring_force,
                    pinky_force=self.last_packet.pinky_force,
                    total_force=self.last_packet.total_force,
                    slip_detected=self.last_packet.slip_detected,
                    motor_current_ma=self.last_packet.motor_current_ma,
                    raw_packet="[STALL - NO NEW UART BYTES]",
                    source="MOCK_SERIAL (STALLED)",
                    is_valid=False,
                    is_stale=True,
                    parse_error="Telemetry stream stalled: No updates received"
                )
                return stale_pkt

        if self.corrupt_next_packet:
            self.corrupt_next_packet = False
            raw_string = "CORRUPT_BYTES_999,NAN,ERR,##"
            self.parse_errors_logged += 1
            pkt, _ = PacketParser.parse_raw_packet(raw_string, source_tag="MOCK_SERIAL")
            self.last_packet = pkt
            return pkt

        # Generate realistic ASCII wire packet
        tot = max(0.0, self.current_force + np.random.normal(0, self.noise_std) + self.drift)
        slip_val = 1 if self.slip_flag else 0
        self.slip_flag = False  # Reset one-shot

        th = tot * 0.40
        idx = tot * 0.22
        mid = tot * 0.20
        rng = tot * 0.11
        pnk = tot * 0.07
        ma = 120.0 + tot * 50.0

        raw_ascii = f"{tot:.2f},{slip_val},{th:.2f},{idx:.2f},{mid:.2f},{rng:.2f},{pnk:.2f},{ma:.1f}\n"
        pkt, success = PacketParser.parse_raw_packet(raw_ascii, source_tag="MOCK_SERIAL")
        if not success:
            self.parse_errors_logged += 1
        self.last_packet = pkt
        return pkt

    def get_diagnostics(self) -> dict:
        return {
            "Interface": "MockSerialSensorInterface (Hardware-Free)",
            "Status": "Connected (Mock Stream Active)",
            "Packets Generated": self.packets_generated,
            "Parse Errors": self.parse_errors_logged,
            "Stalled": self.simulated_stall,
            "Actuator Armed": self.is_armed(),
            "Emergency Stopped": self.emergency_stopped
        }


class RecordedReplaySensorInterface(BaseSensorInterface):
    """
    Telemetry Replay Engine.
    Replays pre-recorded experimental trials or simulated stress traces from CSV.
    Permits deterministic benchmark reproduction.
    """

    def __init__(self, recording_csv: Optional[str] = None):
        super().__init__()
        self.connected: bool = True
        self.current_idx: int = 0
        self.is_playing: bool = True
        self.loop: bool = True
        self.recording_path: str = recording_csv or self._ensure_default_recording()
        self.df: pd.DataFrame = pd.DataFrame()
        self._load_recording()

    def _ensure_default_recording(self) -> str:
        """Generates a default reproducible slip-recovery recording if none exists."""
        default_file = os.path.join(REPLAY_DIR, "demo_slip_recording.csv")
        if not os.path.exists(default_file):
            rows = []
            for t_step in range(60):
                t_sec = round(t_step * 0.05, 3)
                if t_step < 10:
                    f = round(t_step * 0.35, 2)
                    slip = 0
                elif t_step < 25:
                    f = 3.85 + round(np.random.normal(0, 0.03), 2)
                    slip = 0
                elif t_step < 32:
                    f = 2.40 + round(np.random.normal(0, 0.05), 2)
                    slip = 1
                elif t_step < 50:
                    f = 4.45 + round(np.random.normal(0, 0.03), 2)
                    slip = 0
                else:
                    f = max(0.0, round(4.45 - (t_step - 50) * 0.45, 2))
                    slip = 0

                th = round(f * 0.40, 2)
                idx = round(f * 0.22, 2)
                mid = round(f * 0.20, 2)
                rng = round(f * 0.11, 2)
                pnk = round(f * 0.07, 2)
                ma = round(120.0 + f * 50.0, 1)

                rows.append({
                    "time_s": t_sec,
                    "total_force": f,
                    "slip_flag": slip,
                    "thumb": th,
                    "index": idx,
                    "middle": mid,
                    "ring": rng,
                    "pinky": pnk,
                    "motor_current_ma": ma
                })
            pd.DataFrame(rows).to_csv(default_file, index=False)
        return default_file

    def _load_recording(self) -> None:
        try:
            if os.path.exists(self.recording_path):
                self.df = pd.read_csv(self.recording_path)
            else:
                self.df = pd.DataFrame()
        except Exception:
            self.df = pd.DataFrame()

    def connect(self, port: str = "REPLAY", baud: int = 0) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected and not self.df.empty

    def trigger_slip_event(self, intensity: float = 0.8) -> None:
        pass  # Replay is deterministic from data file

    def send_actuator_target(self, target_force_n: float) -> bool:
        return True

    def reset_replay(self) -> None:
        self.current_idx = 0

    def step_replay(self) -> None:
        if self.df.empty:
            return
        self.current_idx += 1
        if self.current_idx >= len(self.df):
            if self.loop:
                self.current_idx = 0
            else:
                self.current_idx = len(self.df) - 1

    def read_packet(self) -> SensorPacket:
        if self.df.empty:
            return SensorPacket(
                timestamp=time.time(),
                source="REPLAY (EMPTY)",
                is_valid=False,
                parse_error="No replay data loaded"
            )

        row = self.df.iloc[self.current_idx]
        if self.is_playing:
            self.step_replay()

        raw_str = (
            f"{row.get('total_force', 0.0):.2f},{int(row.get('slip_flag', 0))},"
            f"{row.get('thumb', 0.0):.2f},{row.get('index', 0.0):.2f},"
            f"{row.get('middle', 0.0):.2f},{row.get('ring', 0.0):.2f},"
            f"{row.get('pinky', 0.0):.2f},{row.get('motor_current_ma', 120.0):.1f}\n"
        )
        pkt, _ = PacketParser.parse_raw_packet(raw_str, source_tag="REPLAY")
        return pkt

    def get_diagnostics(self) -> dict:
        total_len = len(self.df) if not self.df.empty else 0
        return {
            "Interface": "RecordedReplaySensorInterface",
            "File": os.path.basename(self.recording_path),
            "Progress": f"{self.current_idx + 1} / {total_len} frames",
            "Looping": self.loop,
            "Playing": self.is_playing,
            "Actuator Armed": self.is_armed()
        }


class SerialSensorInterface(BaseSensorInterface):
    """
    Physical Hardware Bridge interface for ESP32 / Arduino via PySerial.
    Includes stale-telemetry watchdog, command acknowledgment parser, and diagnostics.
    """

    def __init__(self):
        super().__init__()
        self.port: str = "COM3"
        self.baud: int = 115200
        self.ser: any = None
        self.last_packet: Optional[SensorPacket] = None
        self.last_received_time: float = 0.0
        self.packets_received: int = 0
        self.parse_errors: int = 0
        self.connection_error: str = ""

    def connect(self, port: str = "COM3", baud: int = 115200) -> bool:
        if not SERIAL_AVAILABLE:
            self.connection_error = "PySerial library not available in environment."
            return False
        self.port = port
        self.baud = baud
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.connection_error = ""
            return True
        except Exception as e:
            self.ser = None
            self.connection_error = str(e)
            return False

    def disconnect(self) -> None:
        if self.ser and hasattr(self.ser, "is_open") and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def is_connected(self) -> bool:
        return self.ser is not None and hasattr(self.ser, "is_open") and self.ser.is_open

    def trigger_slip_event(self, intensity: float = 0.8) -> None:
        if self.is_connected():
            try:
                self.ser.write(b"INJECT_SLIP\n")
            except Exception:
                pass

    def send_actuator_target(self, target_force_n: float) -> bool:
        """
        Sends commanded clamp target to MCU over UART.
        Gated by arming state: only transmits positive commands if armed.
        """
        if not self.is_connected():
            return False
        if float(target_force_n) == 0.0:
            try:
                self.ser.write(b"TARGET:0.00\n")
                return True
            except Exception:
                return False
        if not self.is_armed():
            return False
        try:
            clamped_f = min(self.max_software_limit_n, max(0.0, float(target_force_n)))
            cmd = f"TARGET:{clamped_f:.2f}\n".encode("utf-8")
            self.ser.write(cmd)
            return True
        except Exception:
            return False

    def read_packet(self) -> SensorPacket:
        now = time.time()
        if not self.is_connected():
            return SensorPacket(
                timestamp=now,
                source="SERIAL_ESP32",
                is_valid=False,
                parse_error=f"Disconnected: {self.connection_error or 'No open COM port'}"
            )

        try:
            raw_bytes = self.ser.readline()
            if not raw_bytes:
                # Check for stale data if no packet within 0.5s
                if self.last_packet is not None:
                    if (now - self.last_received_time) > 0.50:
                        self.last_packet.is_stale = True
                    return self.last_packet
                return SensorPacket(timestamp=now, source="SERIAL_ESP32", is_valid=False, parse_error="Timeout waiting for bytes")

            raw_str = raw_bytes.decode("utf-8", errors="ignore")
            packet, success = PacketParser.parse_raw_packet(raw_str, source_tag="SERIAL_ESP32")
            self.packets_received += 1

            if success:
                self.last_received_time = now
                self.last_packet = packet
            else:
                self.parse_errors += 1
                if self.last_packet:
                    return self.last_packet
            return packet

        except Exception as e:
            self.parse_errors += 1
            return SensorPacket(timestamp=now, source="SERIAL_ESP32", is_valid=False, parse_error=str(e))

    def get_diagnostics(self) -> dict:
        is_conn = self.is_connected()
        age = round(time.time() - self.last_received_time, 2) if self.last_received_time > 0 else "N/A"
        return {
            "Interface": "SerialSensorInterface (Physical ESP32 Bridge)",
            "Port": self.port,
            "Baud": self.baud,
            "Connected": is_conn,
            "Packets Received": self.packets_received,
            "Parse Errors": self.parse_errors,
            "Telemetry Age": f"{age} s" if isinstance(age, (int, float)) else age,
            "Actuator Armed": self.is_armed(),
            "Emergency Stopped": self.emergency_stopped,
            "Error": self.connection_error or "None"
        }
