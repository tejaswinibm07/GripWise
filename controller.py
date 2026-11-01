"""
controller.py - Closed-Loop Adaptive Grip Force Controller with Slip Recovery
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Implements closed-loop adaptive force feedback:
1. Contact detection and gentle ramp-up
2. Active slip detection and incremental force adaptation
3. Fragility-limit cap and Grasp Infeasibility Detection
4. Stale-telemetry watchdog and sensor fault handling
5. Software Emergency Stop and legal state machine transitions
6. Telemetry logging for real-time visualization
"""

import time
from typing import Optional
from dataclasses import dataclass, field
from simulation import HandState, ObjectSpec, STANDARD_OBJECTS
from hardware import BaseSensorInterface, SensorPacket


@dataclass
class ControllerTelemetryPoint:
    """Snapshot of closed-loop telemetry at time step t."""
    step: int
    time_offset_s: float
    target_force: float
    measured_force: float
    state: str
    slip_detected: bool
    fragility_limit: float
    min_slip_threshold: float
    motor_current_ma: float
    action_log: str
    is_infeasible: bool = False
    sensor_status: str = "HEALTHY"


class AdaptiveGripController:
    """
    Adaptive Closed-Loop Controller for the GripWise prosthetic hand.
    Monitors tactile sensor feedback, commands actuator targets,
    and rapidly stabilizes grasp upon micro-slip events.
    """

    # Enforce legal physical state transitions
    LEGAL_TRANSITIONS: dict[HandState, list[HandState]] = {
        HandState.OPEN: [HandState.CLOSING, HandState.OPEN],
        HandState.CLOSING: [HandState.GRASPING, HandState.OPEN, HandState.RELEASING],
        HandState.GRASPING: [HandState.STABLE_GRIP, HandState.SLIPPING, HandState.RELEASING, HandState.OPEN],
        HandState.STABLE_GRIP: [HandState.SLIPPING, HandState.RELEASING, HandState.OPEN],
        HandState.SLIPPING: [HandState.STABLE_GRIP, HandState.RELEASING, HandState.OPEN],
        HandState.RELEASING: [HandState.OPEN, HandState.RELEASING]
    }

    def __init__(self, obj: Optional[ObjectSpec] = None, base_target: float = 3.5):
        if obj is None:
            obj = STANDARD_OBJECTS.get("Cup") or ObjectSpec("Default Cup", 250.0, 0.40, 0.05, 35.0)
        self.obj: ObjectSpec = obj
        self.base_target: float = base_target
        self.current_target: float = 0.0
        self.state: HandState = HandState.OPEN
        self.flexion_ratio: float = 0.0  # 0.0 = wide open, 1.0 = fully closed
        self.step_count: int = 0
        self.start_time: float = time.time()
        self.slip_counter: int = 0
        self.slip_adaptation_total: float = 0.0
        self.history: list[ControllerTelemetryPoint] = []
        self.last_action_note: str = "System idle. Ready to grasp."
        self.max_allowable_cap: float = obj.fragility_limit_n * 0.92

        # Infeasibility & Fault Detection
        self.is_infeasible: bool = False
        self.infeasibility_message: str = ""
        self.stale_packet_count: int = 0
        self.sensor_fault_active: bool = False
        self._check_grasp_feasibility()

    def _check_grasp_feasibility(self) -> None:
        """
        Calculates theoretical static friction minimum dual-shear force:
        F_slip_min = (m * g) / (2 * mu)
        If F_slip_min > fragility_limit_n, grasp is physically infeasible.
        """
        mass_kg = self.obj.mass_g / 1000.0
        g = 9.80665
        mu = max(0.08, self.obj.friction_coeff)
        f_min_slip = (mass_kg * g) / (2.0 * mu)

        if f_min_slip > self.obj.fragility_limit_n:
            self.is_infeasible = True
            self.infeasibility_message = (
                f"CRITICAL WARNING: Grasp is physically infeasible! "
                f"Theoretical minimum slip force ({f_min_slip:.2f}N) exceeds object fragility limit ({self.obj.fragility_limit_n:.1f}N). "
                f"Object will either crush or slip."
            )
        else:
            self.is_infeasible = False
            self.infeasibility_message = ""

    def configure(self, obj: ObjectSpec, base_target: float) -> None:
        """Update active object and predicted target force."""
        self.obj = obj
        self.base_target = base_target
        self.max_allowable_cap = obj.fragility_limit_n * 0.92
        self._check_grasp_feasibility()

    def _transition_to(self, new_state: HandState) -> bool:
        """Enforces physical transition safety."""
        allowed = self.LEGAL_TRANSITIONS.get(self.state, [])
        if new_state in allowed:
            self.state = new_state
            return True
        # If emergency release or abort, permit transition to OPEN
        if new_state == HandState.OPEN or new_state == HandState.RELEASING:
            self.state = new_state
            return True
        return False

    def start_grasp_sequence(self, sensor: BaseSensorInterface) -> None:
        """Initiate sequence from Open -> Closing -> Grasping."""
        self.history.clear()
        self.step_count = 0
        self.slip_counter = 0
        self.slip_adaptation_total = 0.0
        self.flexion_ratio = 0.0
        self._transition_to(HandState.CLOSING)
        self.current_target = 0.8
        self.last_action_note = f"Approaching {self.obj.name}... Articulating fingers."
        if hasattr(sensor, "current_force"):
            sensor.current_force = 0.0
        sensor.send_actuator_target(self.current_target)

    def release_grasp(self, sensor: BaseSensorInterface) -> None:
        """Open hand and release object."""
        self._transition_to(HandState.RELEASING)
        self.current_target = 0.0
        self.last_action_note = f"Opening hand. Releasing {self.obj.name}."
        sensor.send_actuator_target(0.0)

    def trigger_slip_event(self, sensor: BaseSensorInterface) -> None:
        """User-triggered simulated shear slip event."""
        sensor.trigger_slip_event(intensity=0.85)
        self.last_action_note = "ALERT: External shear slip event injected!"

    def emergency_stop(self, sensor: BaseSensorInterface) -> None:
        """Immediate software emergency stop: dumps force and opens hand."""
        self.state = HandState.OPEN
        self.flexion_ratio = 0.0
        self.current_target = 0.0
        self.slip_adaptation_total = 0.0
        self.last_action_note = "EMERGENCY STOP TRIGGERED! Commanded force set to 0.0N. Hand disarmed."
        sensor.emergency_stop()

    def step(self, sensor: BaseSensorInterface) -> ControllerTelemetryPoint:
        """
        Executes one controller cycle:
        1. Read sensor packet & check for stale telemetry
        2. Evaluate state transitions & slip detection
        3. Compute adaptive target adjustments
        4. Transmit actuator command
        5. Log telemetry snapshot
        """
        self.step_count += 1
        t_now = round(self.step_count * 0.1, 2)
        if hasattr(sensor, "step_simulation"):
            sensor.step_simulation(compliance_factor=self.obj.compliance_mm_per_n)

        packet: SensorPacket = sensor.read_packet()
        measured_f = packet.total_force

        # Telemetry health watchdog
        sensor_status = "HEALTHY"
        if packet.is_stale:
            self.stale_packet_count += 1
            sensor_status = "STALE"
        elif not packet.is_valid:
            self.stale_packet_count += 1
            sensor_status = "INVALID"
        else:
            self.stale_packet_count = 0

        action_log = ""
        min_slip_f = (self.obj.mass_g / 1000.0 * 9.80665) / (2.0 * max(0.1, self.obj.friction_coeff))

        # Check for grasp infeasibility
        if self.is_infeasible:
            action_log += f"[INFEASIBILITY ALERT: {self.infeasibility_message}] "

        # Check for persistent stale telemetry fault (watchdog)
        if self.stale_packet_count >= 5:
            self.sensor_fault_active = True
            action_log += " [WATCHDOG TIMEOUT: TELEMETRY STALE. TRIGGERING SAFE OPEN] "
            self._transition_to(HandState.OPEN)
            self.current_target = 0.0

        # ---------------- State Machine Execution ----------------
        if self.state == HandState.OPEN:
            self.flexion_ratio = 0.0
            self.current_target = 0.0
            action_log += "Hand fully open. Actuator relaxed."

        elif self.state == HandState.CLOSING:
            self.flexion_ratio = min(0.65, self.flexion_ratio + 0.12)
            if measured_f >= 0.40 or self.flexion_ratio >= 0.60:
                self._transition_to(HandState.GRASPING)
                self.current_target = self.base_target
                action_log += f"Contact detected ({measured_f:.2f}N). Ramping to predicted target {self.base_target:.2f}N."
            else:
                action_log += f"Fingers closing... Flexion {int(self.flexion_ratio*100)}%."

        elif self.state == HandState.GRASPING:
            self.flexion_ratio = min(0.85, self.flexion_ratio + 0.08)
            if measured_f >= (self.current_target * 0.88):
                self._transition_to(HandState.STABLE_GRIP)
                action_log += f"Target force reached ({measured_f:.2f}N). Grasp stabilized."
            else:
                action_log += f"Clamping to target {self.current_target:.2f}N... Current: {measured_f:.2f}N."

        elif self.state == HandState.STABLE_GRIP:
            self.flexion_ratio = 0.85
            # Slip Detection Condition
            if packet.slip_detected or (measured_f < min_slip_f and measured_f > 0.2):
                self._transition_to(HandState.SLIPPING)
                self.slip_counter += 1

                headroom = self.max_allowable_cap - self.current_target
                boost_step = min(1.60, max(0.40, headroom * 0.45))

                if headroom > 0.2:
                    self.current_target += boost_step
                    self.slip_adaptation_total += boost_step
                    action_log += (
                        f"MICRO-SLIP DETECTED! Adapted target +{boost_step:.2f}N "
                        f"-> New Target: {self.current_target:.2f}N."
                    )
                else:
                    action_log += (
                        f"MICRO-SLIP DETECTED! At maximum safe clamp limit ({self.max_allowable_cap:.2f}N). "
                        "Cannot boost further without crushing object."
                    )
            else:
                action_log += f"Stable grip maintained at {measured_f:.2f}N. No slip detected."

        elif self.state == HandState.SLIPPING:
            self.flexion_ratio = 0.88
            if not packet.slip_detected and measured_f >= (self.current_target * 0.85):
                self._transition_to(HandState.STABLE_GRIP)
                action_log += f"SLIP RECOVERED! Grip re-stabilized at {measured_f:.2f}N."
            else:
                if packet.slip_detected and (self.current_target + 0.35 <= self.max_allowable_cap):
                    self.current_target += 0.35
                    self.slip_adaptation_total += 0.35
                    action_log += f"Slip persisting: Nudging target to {self.current_target:.2f}N."
                else:
                    action_log += f"Recovering from slip... Measured: {measured_f:.2f}N."

        elif self.state == HandState.RELEASING:
            self.flexion_ratio = max(0.0, self.flexion_ratio - 0.20)
            self.current_target = 0.0
            if self.flexion_ratio <= 0.05:
                self._transition_to(HandState.OPEN)
                self.flexion_ratio = 0.0
                action_log += "Hand fully opened. Ready for next trial."
            else:
                action_log += f"Releasing grasp... Opening fingers ({int(self.flexion_ratio*100)}%)."

        # Structural Fragility Safeguard
        if self.current_target > self.max_allowable_cap:
            self.current_target = self.max_allowable_cap
            action_log += " [SAFETY CAP ACTIVE]"

        # Send target command to sensor/actuator interface
        sensor.send_actuator_target(self.current_target)
        self.last_action_note = action_log

        point = ControllerTelemetryPoint(
            step=self.step_count,
            time_offset_s=round(t_now, 2),
            target_force=round(self.current_target, 2),
            measured_force=round(measured_f, 2),
            state=self.state.value,
            slip_detected=packet.slip_detected,
            fragility_limit=round(self.obj.fragility_limit_n, 2),
            min_slip_threshold=round(min_slip_f, 2),
            motor_current_ma=round(packet.motor_current_ma, 1),
            action_log=action_log,
            is_infeasible=self.is_infeasible,
            sensor_status=sensor_status
        )
        self.history.append(point)
        if len(self.history) > 200:
            self.history.pop(0)

        return point

    def reset(self, sensor: BaseSensorInterface) -> None:
        """Reset controller to baseline state."""
        self.state = HandState.OPEN
        self.flexion_ratio = 0.0
        self.current_target = 0.0
        self.slip_counter = 0
        self.slip_adaptation_total = 0.0
        self.history.clear()
        self.step_count = 0
        self.start_time = time.time()
        self.stale_packet_count = 0
        self.sensor_fault_active = False
        self.last_action_note = "Controller reset. Hand opened."
        self._check_grasp_feasibility()
        sensor.send_actuator_target(0.0)
