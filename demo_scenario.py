"""
demo_scenario.py - End-to-End AI Demonstration Scenario Engine
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Executes a complete, reproducible 8-stage sequence:
1. Object Selection & Preference Resolution
2. Initial Minimum-Safe-Force Prediction (Model A - Random Forest)
3. Finger Articulation & Grasp Approach (OPEN -> CLOSING)
4. Contact Detection & Target Force Stabilization (GRASPING -> STABLE_GRIP)
5. Micro-Slip Shear Injection (SLIPPING)
6. Autonomous Adaptive Force Boost within Fragility Limits
7. Slip Recovery & Grasp Restabilization (STABLE_GRIP)
8. Release & Automatic Trial History Logging (OPEN)
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable

from simulation import STANDARD_OBJECTS, ObjectSpec, HandState
from hardware import SimulatedSensorInterface
from prediction import ForcePredictor, PredictionResult
from controller import AdaptiveGripController, ControllerTelemetryPoint
from personalization import PersonalizationManager, TrialRecord, UserProfile


@dataclass
class ScenarioStepResult:
    """Telemetry and narrative snapshot for one stage of the demo scenario."""
    stage_number: int
    stage_name: str
    action_description: str
    hand_state: str
    flexion_percent: int
    commanded_target_n: float
    simulated_measured_n: float
    slip_detected: bool
    fragility_cap_n: float
    explanation_note: str


@dataclass
class DemonstrationExecutionSummary:
    """Complete summary of an end-to-end demonstration execution."""
    scenario_name: str
    object_name: str
    user_profile_name: str
    initial_predicted_force_n: float
    model_source: str
    stabilized_force_n: float
    slip_injected: bool
    adapted_target_force_n: float
    force_boost_applied_n: float
    fragility_headroom_n: float
    trial_id_logged: str
    outcome: str
    total_steps: int
    steps_log: list[ScenarioStepResult]


def execute_end_to_end_demonstration(
    object_key: str = "Cup",
    profile_name: str = "Alex (Standard)",
    custom_obj: Optional[ObjectSpec] = None,
    predictor: Optional[ForcePredictor] = None,
    personalization_mgr: Optional[PersonalizationManager] = None,
    step_callback: Optional[Callable[[ScenarioStepResult], None]] = None
) -> DemonstrationExecutionSummary:
    """
    Executes a reproducible, multi-stage demonstration scenario using
    the trained Random Forest regressor, physical simulation, and adaptive controller.
    """
    if predictor is None:
        predictor = ForcePredictor()
    if personalization_mgr is None:
        personalization_mgr = PersonalizationManager()

    # 1. Resolve Object & User Preferences
    if custom_obj is not None:
        obj = custom_obj
    else:
        obj = STANDARD_OBJECTS.get(object_key, STANDARD_OBJECTS["Cup"])

    profile: UserProfile = personalization_mgr.profiles.get(
        profile_name,
        list(personalization_mgr.profiles.values())[0]
    )
    user_margin = profile.effective_safety_margin

    # 2. Model A: Predict Initial Minimum Safe Force
    pred_res: PredictionResult = predictor.predict(
        obj=obj,
        user_safety_margin=user_margin,
        algorithm="Random Forest (ML)"
    )
    predicted_target_f = float(pred_res.recommended_target_n)

    # 3. Initialize Controller & Sensor Interface
    sensor = SimulatedSensorInterface()
    controller = AdaptiveGripController(obj=obj, base_target=predicted_target_f)
    controller.reset(sensor)
    controller.configure(obj, base_target=predicted_target_f)

    steps_log: list[ScenarioStepResult] = []

    def record_stage(
        stage_num: int,
        stage_name: str,
        action_desc: str,
        note: str
    ) -> ScenarioStepResult:
        packet = sensor.read_packet()
        meas_f = packet.total_force if not controller.history else controller.history[-1].measured_force
        step_res = ScenarioStepResult(
            stage_number=stage_num,
            stage_name=stage_name,
            action_description=action_desc,
            hand_state=controller.state.value,
            flexion_percent=int(controller.flexion_ratio * 100),
            commanded_target_n=round(controller.current_target, 2),
            simulated_measured_n=round(meas_f, 2),
            slip_detected=packet.slip_detected or (controller.state == HandState.SLIPPING),
            fragility_cap_n=round(controller.max_allowable_cap, 2),
            explanation_note=note
        )
        steps_log.append(step_res)
        if step_callback:
            step_callback(step_res)
        return step_res

    # STAGE 1: Object & Preferences Resolution
    record_stage(
        1,
        "Object & User Preference Selection",
        f"Selected {obj.name} (Mass: {obj.mass_g:.0f}g, Friction μ={obj.friction_coeff:.2f}) with user margin {user_margin:.2f}x.",
        "Object mechanical properties and user adaptation profile are loaded into the pipeline."
    )

    # STAGE 2: Minimum-Safe-Force Prediction (Model A)
    record_stage(
        2,
        "Force Prediction (Model A)",
        f"Model A (Random Forest) computed candidate clamp force target: {predicted_target_f:.2f}N.",
        f"Analytical static baseline estimated {pred_res.baseline_heuristic_n:.2f}N. Random Forest adjusted for compliance ({obj.compliance_mm_per_n:.2f} mm/N)."
    )

    # STAGE 3: Initiate Grasp Sequence (Approach)
    controller.start_grasp_sequence(sensor)
    for _ in range(4):
        controller.step(sensor)
    record_stage(
        3,
        "Fingers Articulating (Approach)",
        f"Hand closing... Flexion at {int(controller.flexion_ratio * 100)}%. Approaching surface contact.",
        "Actuator drives 5-finger kinematics in anatomical power cascade (Thumb leads, fingers follow)."
    )

    # STAGE 4: Contact & Target Clamping
    for _ in range(7):
        controller.step(sensor)
    record_stage(
        4,
        "Contact & Grasp Stabilization",
        f"Stable contact achieved. Measured force stabilized at {controller.history[-1].measured_force:.2f}N (Target: {controller.current_target:.2f}N).",
        "Closed-loop controller regulates actuator target to match Model A's predicted safe force."
    )
    stabilized_f = controller.history[-1].measured_force

    # STAGE 5: Micro-Slip Shear Injection
    controller.trigger_slip_event(sensor)
    controller.step(sensor)
    record_stage(
        5,
        "Micro-Slip Shear Injected",
        f"External shear perturbation injected! Measured force dropped to {controller.history[-1].measured_force:.2f}N. Hand in SLIPPING state.",
        "Simulated tangential slip alert triggered. Controller detects shear displacement before gross drop."
    )

    # STAGE 6: Autonomous Adaptive Force Boost
    for _ in range(5):
        controller.step(sensor)
    adapted_target = controller.current_target
    boost_applied = controller.slip_adaptation_total
    record_stage(
        6,
        "Adaptive Closed-Loop Recovery",
        f"Controller adapted target force by +{boost_applied:.2f}N -> New Target: {adapted_target:.2f}N (Cap: {controller.max_allowable_cap:.1f}N).",
        "Adaptive controller steps up commanded normal force while verifying headroom against the structural fragility limit."
    )

    # STAGE 7: Slip Suppressed & Re-Stabilized
    for _ in range(3):
        controller.step(sensor)
    record_stage(
        7,
        "Grasp Re-Stabilized",
        f"Slip suppressed! Restabilized in STABLE_GRIP at {controller.history[-1].measured_force:.2f}N.",
        "Grip force now safely exceeds updated dynamic friction threshold. Object is secure."
    )

    # STAGE 8: Releasing Grasp
    controller.release_grasp(sensor)
    for _ in range(7):
        controller.step(sensor)
    record_stage(
        8,
        "Release & Rest",
        f"Hand relaxed to {controller.state.value}. Flexion returned to {int(controller.flexion_ratio * 100)}%.",
        "Actuator de-energized. Hand returns to wide open state."
    )

    # STAGE 9: Automatic Experiment Logging
    trial_count = len(personalization_mgr.trials) + 1
    trial_id = f"DEMO-{trial_count:03d}"
    peak_f = max([p.measured_force for p in controller.history], default=stabilized_f)
    headroom = obj.fragility_limit_n - peak_f

    trial_record = TrialRecord(
        trial_id=trial_id,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        object_name=obj.name,
        user_profile=profile.name,
        preset_mode=profile.preset_mode,
        initial_target_f=round(predicted_target_f, 2),
        final_target_f=round(adapted_target, 2),
        peak_measured_f=round(peak_f, 2),
        slip_events=controller.slip_counter,
        outcome="ADAPTED_SUCCESS",
        feedback_rating="Ideal",
        notes=f"End-to-End AI Demo: Model A predicted {predicted_target_f:.2f}N. Slip recovered via +{boost_applied:.2f}N boost."
    )
    personalization_mgr.log_trial(trial_record)

    return DemonstrationExecutionSummary(
        scenario_name="Fragile Object Grasp & Micro-Slip Recovery",
        object_name=obj.name,
        user_profile_name=profile.name,
        initial_predicted_force_n=round(predicted_target_f, 2),
        model_source=pred_res.model_type,
        stabilized_force_n=round(stabilized_f, 2),
        slip_injected=True,
        adapted_target_force_n=round(adapted_target, 2),
        force_boost_applied_n=round(boost_applied, 2),
        fragility_headroom_n=round(headroom, 2),
        trial_id_logged=trial_id,
        outcome="ADAPTED_SUCCESS",
        total_steps=len(controller.history),
        steps_log=steps_log
    )

