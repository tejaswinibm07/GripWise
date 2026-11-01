"""
simulation.py - Physics, Hand Kinematics, and Object Simulation Engine
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Simulates physical interactions between a 5-finger articulated prosthetic hand
and everyday test objects. All physical properties are illustrative demonstration
parameters designed to mimic typical real-world physical behavior.
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class HandState(str, Enum):
    """Discrete state machine stages for prosthetic grasp cycle."""
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    GRASPING = "GRASPING"
    STABLE_GRIP = "STABLE_GRIP"
    SLIPPING = "SLIPPING"
    RELEASING = "RELEASING"


@dataclass
class ObjectSpec:
    """
    Demonstration object characteristics.
    Note: These parameters are theoretical engineering models for simulation
    and do not represent measured clinical calibration.
    """
    name: str
    icon: str = "📦"
    mass_g: float = 250.0                 # Mass in grams
    friction_coeff: float = 0.40         # Estimated static friction coefficient (mu)
    compliance_mm_per_n: float = 0.10    # Mechanical compliance: mm deformation per Newton
    fragility_limit_n: float = 30.0      # Maximum safe clamp force before crushing/damage (Newtons)
    recommended_grasp: str = "Standard Grasp"        # e.g., "Cylindrical", "Spherical", "Delicate Pinch"
    description: str = "Simulation test object."
    color_hex: str = "#0EA5E9"
    is_synthetic: bool = True     # Explicit flag distinguishing demo assumptions


# Default demonstration objects specified in requirements
STANDARD_OBJECTS: dict[str, ObjectSpec] = {
    "Cup": ObjectSpec(
        name="Ceramic Cup",
        icon="☕",
        mass_g=240.0,
        friction_coeff=0.42,
        compliance_mm_per_n=0.04,   # Very rigid
        fragility_limit_n=38.0,     # Rigid ceramic, cracks at excessive localized pressure
        recommended_grasp="Cylindrical Power",
        description="Rigid ceramic body with moderate surface glaze friction. Low compliance, requires firm normal force.",
        color_hex="#0EA5E9"
    ),
    "Bottle": ObjectSpec(
        name="Plastic Bottle",
        icon="🍾",
        mass_g=360.0,               # Filled/semi-filled beverage
        friction_coeff=0.32,        # Smooth PET plastic (slippery)
        compliance_mm_per_n=0.45,   # Moderate compliance / thin wall deflection
        fragility_limit_n=24.0,     # Buckles if squeezed beyond thin-wall yield point
        recommended_grasp="Cylindrical Grasp",
        description="Smooth PET plastic with lower friction and moderate wall deformation. High slip risk if under-clamped.",
        color_hex="#38BDF8"
    ),
    "Ball": ObjectSpec(
        name="Rubber Tennis Ball",
        icon="🎾",
        mass_g=125.0,
        friction_coeff=0.72,        # High surface friction
        compliance_mm_per_n=0.90,   # Yielding rubber core
        fragility_limit_n=42.0,     # Resilient rubber elasticity
        recommended_grasp="Spherical Grasp",
        description="High friction elastomeric surface with significant contact compliance. High grip stability.",
        color_hex="#A3E635"
    ),
    "Box": ObjectSpec(
        name="Cardboard Box",
        icon="📦",
        mass_g=190.0,
        friction_coeff=0.48,        # Dry corrugated paper
        compliance_mm_per_n=0.65,   # Corrugation compression
        fragility_limit_n=16.0,     # Readily crushed if squeezed hard
        recommended_grasp="Parallel Jaw / Power",
        description="Compressible corrugated cellulose. Requires sensitive force moderation to prevent box crushing.",
        color_hex="#F59E0B"
    ),
    "Sponge": ObjectSpec(
        name="Polyurethane Sponge",
        icon="🧽",
        mass_g=45.0,
        friction_coeff=0.65,        # High surface contact area under low loads
        compliance_mm_per_n=2.80,   # Highly compliant open-cell foam
        fragility_limit_n=10.0,     # Total deformation occurs quickly
        recommended_grasp="Delicate Pinch",
        description="Extreme compliance with low mass. Force easily exceeds functional deformation limits if uncurbed.",
        color_hex="#EC4899"
    )
}


@dataclass
class FingerFlexion:
    """Flexion degree of fingers from 0.0 (wide open) to 1.0 (fully flexed)."""
    thumb: float = 0.0
    index: float = 0.0
    middle: float = 0.0
    ring: float = 0.0
    pinky: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Thumb": round(self.thumb, 3),
            "Index": round(self.index, 3),
            "Middle": round(self.middle, 3),
            "Ring": round(self.ring, 3),
            "Pinky": round(self.pinky, 3),
        }


class ProstheticHandPhysics:
    """
    Kinematic and dynamic simulation model of the 5-finger prosthetic hand.
    Computes finger flexion, contact deformation, tangential shear forces,
    and slip risk indices.
    """

    def __init__(self, obj: ObjectSpec):
        self.object: ObjectSpec = obj
        self.state: HandState = HandState.OPEN
        self.flexion: FingerFlexion = FingerFlexion()
        self.contact_detected: bool = False
        self.current_normal_force: float = 0.0
        self.crushed_flag: bool = False

    def update_object(self, obj: ObjectSpec) -> None:
        self.object = obj
        self.crushed_flag = False

    def get_kinematic_flexion(self, progress: float) -> FingerFlexion:
        """
        Interpolates finger flexion based on grip state progress (0.0 to 1.0).
        Different grasp types articulate fingers with anatomical phase delays.
        """
        p = max(0.0, min(1.0, progress))
        # Thumb opposes earlier; fingers follow in power cascade
        thumb_f = p ** 1.1
        index_f = p ** 0.95
        middle_f = p ** 0.90
        ring_f = p ** 0.85
        pinky_f = p ** 0.80
        return FingerFlexion(thumb_f, index_f, middle_f, ring_f, pinky_f)

    def calculate_theoretical_min_slip_force(self) -> float:
        """
        Calculates theoretical static friction minimum normal force:
        F_friction = mu * F_normal >= m * g
        F_normal_min = (m * g) / mu
        For dual-sided opposing grasp (Thumb opposing fingers, 2 effective shear faces):
        F_normal_min = (m * g) / (2 * mu)
        """
        mass_kg = self.object.mass_g / 1000.0
        g = 9.80665  # m/s^2
        mu = max(0.1, self.object.friction_coeff)
        
        # Two opposing contact surfaces (e.g., thumb on one side, fingers on opposite)
        f_min = (mass_kg * g) / (2.0 * mu)
        return float(round(f_min, 3))

    def evaluate_slip_risk(self, current_force: float) -> tuple[float, bool]:
        """
        Returns:
            (slip_risk_ratio: float from 0.0 [rock solid] to 1.0+ [imminent slip],
             slip_flag: bool)
        """
        f_min = self.calculate_theoretical_min_slip_force()
        if current_force <= 0.05:
            # No force applied yet
            return 1.0, False

        ratio = f_min / current_force
        is_slipping = (current_force < f_min) and (current_force > 0.3)
        return float(round(min(2.0, ratio), 3)), is_slipping

    def check_fragility_violation(self, current_force: float) -> bool:
        """Check if grip force exceeds structural fragility limit."""
        if current_force >= self.object.fragility_limit_n:
            self.crushed_flag = True
            return True
        return False

