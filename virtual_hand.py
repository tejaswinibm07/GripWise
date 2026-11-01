"""
virtual_hand.py - Dynamic Articulated 5-Finger Prosthetic Hand SVG Generator
Part of GripWise: A Low-Cost Self-Learning Prosthetic Hand

Renders a recognizable, articulated five-finger prosthetic hand with tactile sensor pads,
chassis plate, joint pivots, and grasped object visualization.
Dynamically responds to hand states:
- OPEN
- CLOSING
- GRASPING
- STABLE_GRIP
- SLIPPING
- RELEASING
"""

import math
from simulation import HandState, ObjectSpec
from hardware import SensorPacket


def generate_hand_svg(
    state: HandState | str,
    flexion_ratio: float,
    current_force: float,
    target_force: float,
    obj: ObjectSpec,
    packet: SensorPacket | None = None
) -> str:
    """
    Generates a standalone, responsive SVG markup representing the 5-finger
    prosthetic hand and active object grasp state.
    """
    if isinstance(state, str):
        try:
            state = HandState(state)
        except ValueError:
            state = HandState.OPEN

    # Clamp flexion between 0.0 (wide open) and 1.0 (fully flexed)
    p = max(0.0, min(1.0, flexion_ratio))

    # Color palette based on hand state
    state_styles = {
        HandState.OPEN: {
            "primary": "#38BDF8",       # Sky cyan
            "accent": "#0EA5E9",
            "glow": "#0284C7",
            "state_bg": "rgba(14, 165, 233, 0.15)",
            "state_border": "#0284C7",
            "state_text": "#38BDF8",
            "sensor_color": "#06B6D4",
            "status_label": "HAND OPEN (IDLE)",
            "pulse_class": ""
        },
        HandState.CLOSING: {
            "primary": "#38BDF8",
            "accent": "#06B6D4",
            "glow": "#0EA5E9",
            "state_bg": "rgba(6, 182, 212, 0.18)",
            "state_border": "#06B6D4",
            "state_text": "#67E8F9",
            "status_label": "CLOSING (APPROACH)",
            "sensor_color": "#38BDF8",
            "pulse_class": "pulse-cyan"
        },
        HandState.GRASPING: {
            "primary": "#06B6D4",
            "accent": "#22D3EE",
            "glow": "#06B6D4",
            "state_bg": "rgba(6, 182, 212, 0.22)",
            "state_border": "#22D3EE",
            "state_text": "#A5F3FC",
            "status_label": "GRASPING (RAMP-UP)",
            "sensor_color": "#22D3EE",
            "pulse_class": "pulse-cyan"
        },
        HandState.STABLE_GRIP: {
            "primary": "#10B981",       # Emerald Green
            "accent": "#34D399",
            "glow": "#059669",
            "state_bg": "rgba(16, 185, 129, 0.20)",
            "state_border": "#10B981",
            "state_text": "#6EE7B7",
            "status_label": "STABLE GRIP (LOCKED)",
            "sensor_color": "#10B981",
            "pulse_class": "pulse-emerald"
        },
        HandState.SLIPPING: {
            "primary": "#EF4444",       # Warning Crimson
            "accent": "#F87171",
            "glow": "#DC2626",
            "state_bg": "rgba(239, 68, 68, 0.25)",
            "state_border": "#EF4444",
            "state_text": "#FCA5A5",
            "status_label": "⚠️ SLIP DETECTED (ADAPTING)",
            "sensor_color": "#EF4444",
            "pulse_class": "pulse-crimson"
        },
        HandState.RELEASING: {
            "primary": "#818CF8",       # Indigo / Slate
            "accent": "#A5B4FC",
            "glow": "#6366F1",
            "state_bg": "rgba(129, 140, 248, 0.18)",
            "state_border": "#818CF8",
            "state_text": "#C7D2FE",
            "status_label": "RELEASING GRASP",
            "sensor_color": "#818CF8",
            "pulse_class": ""
        }
    }

    style = state_styles.get(state, state_styles[HandState.OPEN])

    # ---------------- Kinematic Joint Computation ----------------
    # Base palm knuckle positions (MCP)
    mcp_thumb = (175, 345)
    mcp_index = (220, 255)
    mcp_middle = (270, 240)
    mcp_ring = (320, 250)
    mcp_pinky = (365, 275)

    # Parametric finger articulators
    # As flexion p increases from 0 -> 1:
    # 1. Thumb rotates inwards (+x, -y) towards center
    thumb_pip = (
        mcp_thumb[0] - 25 + (p * 45),
        mcp_thumb[1] - 40 - (p * 15)
    )
    thumb_tip = (
        thumb_pip[0] + 10 + (p * 40),
        thumb_pip[1] - 30 + (p * 42)
    )

    # 2. Index finger articulates downwards towards object contact
    index_pip = (
        mcp_index[0] - 5 + (p * 12),
        mcp_index[1] - 65 + (p * 40)
    )
    index_tip = (
        index_pip[0] + 2 + (p * 14),
        index_pip[1] - 55 + (p * 68)
    )

    # 3. Middle finger
    middle_pip = (
        mcp_middle[0],
        mcp_middle[1] - 75 + (p * 46)
    )
    middle_tip = (
        middle_pip[0],
        middle_pip[1] - 60 + (p * 74)
    )

    # 4. Ring finger
    ring_pip = (
        mcp_ring[0] + 4 - (p * 10),
        mcp_ring[1] - 68 + (p * 42)
    )
    ring_tip = (
        ring_pip[0] - 2 - (p * 14),
        ring_pip[1] - 55 + (p * 70)
    )

    # 5. Pinky finger
    pinky_pip = (
        mcp_pinky[0] + 18 - (p * 20),
        mcp_pinky[1] - 50 + (p * 34)
    )
    pinky_tip = (
        pinky_pip[0] - 5 - (p * 22),
        pinky_pip[1] - 42 + (p * 55)
    )

    # Tactile sensor pressure radii and opacity
    f_max_scale = max(1.0, obj.fragility_limit_n)
    force_ratio = min(1.0, current_force / f_max_scale)
    halo_radius = 8 + (force_ratio * 12)

    # Object representation in palm grasp
    obj_center = (270, 315)
    obj_markup = _generate_object_svg(obj, obj_center, state, p)

    # Slip vibration wave rings if slipping
    slip_waves = ""
    if state == HandState.SLIPPING:
        slip_waves = f"""
        <g class="slip-vibration" stroke="#EF4444" stroke-width="2.5" fill="none" opacity="0.85">
            <circle cx="{obj_center[0]}" cy="{obj_center[1]}" r="68" stroke-dasharray="6,4">
                <animate attributeName="r" values="60;85" dur="0.8s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.9;0.0" dur="0.8s" repeatCount="indefinite"/>
            </circle>
            <circle cx="{obj_center[0]}" cy="{obj_center[1]}" r="75" stroke-dasharray="8,6">
                <animate attributeName="r" values="65;95" dur="0.8s" begin="0.3s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.8;0.0" dur="0.8s" begin="0.3s" repeatCount="indefinite"/>
            </circle>
            <text x="{obj_center[0]}" y="{obj_center[1] + 95}" fill="#F87171" font-size="12" font-weight="700" text-anchor="middle" letter-spacing="1">
                SHEAR DISPLACEMENT DETECTED
            </text>
        </g>
        """

    svg = f"""
    <div style="display: flex; justify-content: center; width: 100%;">
    <svg viewBox="0 0 540 500" width="100%" height="auto" style="max-width: 520px; background: radial-gradient(circle at 50% 45%, #0F172A 0%, #070B14 100%); border-radius: 14px; border: 1px solid #1E293B; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.6);" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <!-- Glow Filters -->
            <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <filter id="glow-crimson" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="6" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <!-- Carbon Fiber Pattern for Palm Chassis -->
            <pattern id="carbon-plate" width="10" height="10" patternUnits="userSpaceOnUse">
                <rect width="10" height="10" fill="#0B132B"/>
                <line x1="0" y1="0" x2="10" y2="10" stroke="#1E293B" stroke-width="1"/>
                <circle cx="5" cy="5" r="1.5" fill="#172554"/>
            </pattern>
        </defs>

        <style>
            .pulse-cyan {{ animation: pulseC 1.8s ease-in-out infinite alternate; }}
            .pulse-emerald {{ animation: pulseE 2.0s ease-in-out infinite alternate; }}
            .pulse-crimson {{ animation: pulseR 0.6s ease-in-out infinite alternate; }}
            @keyframes pulseC {{ from {{ opacity: 0.75; }} to {{ opacity: 1.0; filter: drop-shadow(0 0 8px #06B6D4); }} }}
            @keyframes pulseE {{ from {{ opacity: 0.85; }} to {{ opacity: 1.0; filter: drop-shadow(0 0 8px #10B981); }} }}
            @keyframes pulseR {{ from {{ opacity: 0.65; }} to {{ opacity: 1.0; filter: drop-shadow(0 0 12px #EF4444); }} }}
            .cyber-line {{ stroke: #1E293B; stroke-dasharray: 4,4; stroke-width: 1; }}
            .metric-txt {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        </style>

        <!-- Background Coordinate Grid -->
        <g opacity="0.25">
            <line x1="50" y1="250" x2="490" y2="250" class="cyber-line"/>
            <line x1="270" y1="40" x2="270" y2="460" class="cyber-line"/>
            <circle cx="270" cy="315" r="120" stroke="#1E293B" fill="none" stroke-width="1"/>
            <circle cx="270" cy="315" r="170" stroke="#1E293B" fill="none" stroke-width="1" stroke-dasharray="3,6"/>
        </g>

        <!-- Top Telemetry Status Pill -->
        <g transform="translate(20, 20)">
            <rect x="0" y="0" width="220" height="34" rx="8" fill="{style['state_bg']}" stroke="{style['state_border']}" stroke-width="1.5"/>
            <circle cx="16" cy="17" r="6" fill="{style['primary']}" class="{style['pulse_class']}"/>
            <text x="32" y="22" fill="{style['state_text']}" font-size="12" font-weight="700" letter-spacing="0.5" class="metric-txt">
                {style['status_label']}
            </text>
        </g>

        <!-- Top Right Mini Telemetry Info -->
        <g transform="translate(370, 20)">
            <rect x="0" y="0" width="150" height="34" rx="8" fill="#0B132B" stroke="#1E293B" stroke-width="1"/>
            <text x="14" y="15" fill="#64748B" font-size="9" font-weight="600" class="metric-txt">CLAMP / TARGET</text>
            <text x="14" y="28" fill="{style['primary']}" font-size="13" font-weight="700" class="metric-txt">
                {current_force:.2f}N <tspan fill="#64748B" font-size="11">/ {target_force:.2f}N</tspan>
            </text>
        </g>

        <!-- 0. Grasped Object in Center of Palm Grasp -->
        {obj_markup}

        <!-- Slip Wave Indicators -->
        {slip_waves}

        <!-- 1. Wrist & Metacarpal Palm Chassis -->
        <g id="palm-base">
            <!-- Forearm / Wrist Socket -->
            <path d="M 235 480 L 235 440 L 305 440 L 305 480 Z" fill="#0F172A" stroke="#334155" stroke-width="2"/>
            <rect x="245" y="445" width="50" height="8" rx="2" fill="#1E293B"/>

            <!-- Wrist Pivot Joint -->
            <circle cx="270" cy="425" r="22" fill="#1E293B" stroke="{style['primary']}" stroke-width="1.8"/>
            <circle cx="270" cy="425" r="10" fill="#0F172A" stroke="#475569" stroke-width="1"/>

            <!-- Metacarpal Palm Plate -->
            <path d="M 195 365 
                     C 180 325, 200 270, 220 255 
                     L 270 240 
                     L 320 250 
                     L 365 275 
                     C 385 320, 360 385, 335 415 
                     L 270 425 
                     L 205 415 Z" 
                  fill="url(#carbon-plate)" stroke="#334155" stroke-width="2.5"/>

            <!-- Inner Structural Reinforcement Ribs -->
            <path d="M 220 275 L 270 340 L 320 275" fill="none" stroke="#1E293B" stroke-width="3"/>
            <circle cx="270" cy="340" r="12" fill="#0F172A" stroke="{style['accent']}" stroke-width="1.5"/>
            <text x="270" y="344" fill="{style['primary']}" font-size="9" font-weight="700" text-anchor="middle" class="metric-txt">GW-5</text>
        </g>

        <!-- 2. Articulated Robotic Fingers -->
        <!-- Finger Helper Function Call Renders below -->
        {_render_finger("thumb", mcp_thumb, thumb_pip, thumb_tip, style, packet.thumb_force if packet else 0.0, halo_radius, is_thumb=True)}
        {_render_finger("index", mcp_index, index_pip, index_tip, style, packet.index_force if packet else 0.0, halo_radius)}
        {_render_finger("middle", mcp_middle, middle_pip, middle_tip, style, packet.middle_force if packet else 0.0, halo_radius)}
        {_render_finger("ring", mcp_ring, ring_pip, ring_tip, style, packet.ring_force if packet else 0.0, halo_radius)}
        {_render_finger("pinky", mcp_pinky, pinky_pip, pinky_tip, style, packet.pinky_force if packet else 0.0, halo_radius)}

        <!-- Bottom Telemetry Bar -->
        <g transform="translate(30, 455)">
            <text x="0" y="14" fill="#64748B" font-size="10" font-weight="600" class="metric-txt">FLEXION:</text>
            <rect x="60" y="5" width="120" height="10" rx="5" fill="#1E293B"/>
            <rect x="60" y="5" width="{int(120 * p)}" height="10" rx="5" fill="{style['primary']}"/>
            <text x="188" y="14" fill="#94A3B8" font-size="10" font-weight="700" class="metric-txt">{int(p * 100)}%</text>

            <text x="250" y="14" fill="#64748B" font-size="10" font-weight="600" class="metric-txt">OBJECT:</text>
            <text x="305" y="14" fill="#E2E8F0" font-size="11" font-weight="700" class="metric-txt">{obj.icon} {obj.name}</text>
            
            <text x="440" y="14" fill="#64748B" font-size="10" font-weight="600" class="metric-txt">LIMIT:</text>
            <text x="480" y="14" fill="#F59E0B" font-size="11" font-weight="700" class="metric-txt">{obj.fragility_limit_n:.0f}N</text>
        </g>
    </svg>
    </div>
    """
    return svg


def _render_finger(
    name: str,
    mcp: tuple[float, float],
    pip: tuple[float, float],
    tip: tuple[float, float],
    style: dict,
    finger_f: float,
    halo_radius: float,
    is_thumb: bool = False
) -> str:
    """Renders single articulated finger segments, joints, and tactile pad."""
    joint_radius = 8 if not is_thumb else 9
    tip_radius = 6.5
    pad_glow = style["sensor_color"]
    f_val = f"{finger_f:.1f}N" if finger_f > 0.05 else "0.0N"

    return f"""
    <!-- Finger: {name.upper()} -->
    <g id="finger-{name}">
        <!-- Phalanx Segment 1: Proximal (MCP to PIP) -->
        <line x1="{mcp[0]}" y1="{mcp[1]}" x2="{pip[0]}" y2="{pip[1]}" 
              stroke="#334155" stroke-width="12" stroke-linecap="round"/>
        <line x1="{mcp[0]}" y1="{mcp[1]}" x2="{pip[0]}" y2="{pip[1]}" 
              stroke="{style['accent']}" stroke-width="2.5" stroke-linecap="round"/>

        <!-- Phalanx Segment 2: Distal (PIP to Tip) -->
        <line x1="{pip[0]}" y1="{pip[1]}" x2="{tip[0]}" y2="{tip[1]}" 
              stroke="#1E293B" stroke-width="9.5" stroke-linecap="round"/>
        <line x1="{pip[0]}" y1="{pip[1]}" x2="{tip[0]}" y2="{tip[1]}" 
              stroke="{style['primary']}" stroke-width="2" stroke-linecap="round"/>

        <!-- MCP Base Knuckle Pivot -->
        <circle cx="{mcp[0]}" cy="{mcp[1]}" r="{joint_radius}" fill="#0F172A" stroke="#475569" stroke-width="2"/>
        <circle cx="{mcp[0]}" cy="{mcp[1]}" r="3.5" fill="{style['accent']}"/>

        <!-- PIP Intermediate Joint Pivot -->
        <circle cx="{pip[0]}" cy="{pip[1]}" r="{joint_radius - 2}" fill="#0F172A" stroke="{style['accent']}" stroke-width="1.8"/>

        <!-- Distal Fingertip Tactile Sensor Pad -->
        <!-- Sensor Force Glow Halo -->
        <circle cx="{tip[0]}" cy="{tip[1]}" r="{halo_radius}" fill="{pad_glow}" opacity="0.35"/>
        <circle cx="{tip[0]}" cy="{tip[1]}" r="{tip_radius}" fill="#0F172A" stroke="{pad_glow}" stroke-width="2.5"/>
        <circle cx="{tip[0]}" cy="{tip[1]}" r="2.5" fill="{pad_glow}"/>

        <!-- Tactile Force Label near fingertip -->
        <text x="{tip[0]}" y="{tip[1] - 12}" fill="#94A3B8" font-size="8.5" font-weight="700" text-anchor="middle" font-family="sans-serif">
            {f_val}
        </text>
    </g>
    """


def _generate_object_svg(obj: ObjectSpec, center: tuple[int, int], state: HandState, p: float) -> str:
    """Generates visual silhouette for the active test object in the palm grasp."""
    cx, cy = center
    color = obj.color_hex

    # Object scale slightly responds to compression if compliant
    compress_scale = 1.0 - (p * 0.08 * (obj.compliance_mm_per_n / 2.5))

    if "Cup" in obj.name:
        # Ceramic cup geometry with handle
        return f"""
        <g id="grasped-cup" transform="scale({compress_scale}) transform-origin: {cx}px {cy}px;">
            <!-- Cup Body -->
            <path d="M {cx - 36} {cy - 35} L {cx + 36} {cy - 35} L {cx + 28} {cy + 40} L {cx - 28} {cy + 40} Z" 
                  fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="2.5"/>
            <!-- Cup Rim ellipse -->
            <ellipse cx="{cx}" cy="{cy - 35}" rx="36" ry="10" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="2"/>
            <!-- Handle -->
            <path d="M {cx + 34} {cy - 20} C {cx + 56} {cy - 15}, {cx + 56} {cy + 20}, {cx + 26} {cy + 25}" 
                  fill="none" stroke="{color}" stroke-width="3"/>
            <text x="{cx}" y="{cy + 10}" fill="#F8FAFC" font-size="12" font-weight="700" text-anchor="middle" font-family="sans-serif">
                {obj.name}
            </text>
            <text x="{cx}" y="{cy + 24}" fill="#94A3B8" font-size="9" text-anchor="middle" font-family="sans-serif">
                {obj.mass_g:.0f}g | μ={obj.friction_coeff:.2f}
            </text>
        </g>
        """
    elif "Bottle" in obj.name:
        # Plastic bottle geometry
        return f"""
        <g id="grasped-bottle">
            <!-- Bottle Cap -->
            <rect x="{cx - 10}" y="{cy - 65}" width="20" height="12" rx="2" fill="#38BDF8" stroke="#0284C7" stroke-width="1.5"/>
            <!-- Neck -->
            <path d="M {cx - 12} {cy - 53} L {cx + 12} {cy - 53} L {cx + 28} {cy - 28} L {cx - 28} {cy - 28} Z" 
                  fill="{color}" fill-opacity="0.2" stroke="{color}" stroke-width="2"/>
            <!-- Cylinder Body -->
            <rect x="{cx - 28}" y="{cy - 28}" width="56" height="75" rx="10" 
                  fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="2.5"/>
            <!-- Ribbed Grip Rings -->
            <line x1="{cx - 24}" y1="{cy}" x2="{cx + 24}" y2="{cy}" stroke="{color}" stroke-width="1.5" stroke-dasharray="4,3"/>
            <line x1="{cx - 24}" y1="{cy + 20}" x2="{cx + 24}" y2="{cy + 20}" stroke="{color}" stroke-width="1.5" stroke-dasharray="4,3"/>
            <text x="{cx}" y="{cy - 8}" fill="#F8FAFC" font-size="11" font-weight="700" text-anchor="middle" font-family="sans-serif">
                {obj.name}
            </text>
            <text x="{cx}" y="{cy + 36}" fill="#94A3B8" font-size="9" text-anchor="middle" font-family="sans-serif">
                {obj.mass_g:.0f}g | μ={obj.friction_coeff:.2f}
            </text>
        </g>
        """
    elif "Ball" in obj.name:
        # Rubber tennis ball geometry
        return f"""
        <g id="grasped-ball">
            <circle cx="{cx}" cy="{cy}" r="45" fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="2.8"/>
            <!-- Tennis Seam curves -->
            <path d="M {cx - 32} {cy - 30} C {cx - 5} {cy - 10}, {cx - 5} {cy + 10}, {cx - 32} {cy + 30}" 
                  fill="none" stroke="#FFFFFF" stroke-opacity="0.6" stroke-width="2"/>
            <path d="M {cx + 32} {cy - 30} C {cx + 5} {cy - 10}, {cx + 5} {cy + 10}, {cx + 32} {cy + 30}" 
                  fill="none" stroke="#FFFFFF" stroke-opacity="0.6" stroke-width="2"/>
            <text x="{cx}" y="{cy - 4}" fill="#F8FAFC" font-size="11" font-weight="700" text-anchor="middle" font-family="sans-serif">
                {obj.name}
            </text>
            <text x="{cx}" y="{cy + 12}" fill="#94A3B8" font-size="9" text-anchor="middle" font-family="sans-serif">
                {obj.mass_g:.0f}g | μ={obj.friction_coeff:.2f}
            </text>
        </g>
        """
    elif "Box" in obj.name:
        # Cardboard box isometric/ortho view
        return f"""
        <g id="grasped-box">
            <rect x="{cx - 42}" y="{cy - 38}" width="84" height="76" rx="5" 
                  fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="2.5"/>
            <!-- Box Seam Tape -->
            <line x1="{cx}" y1="{cy - 38}" x2="{cx}" y2="{cy + 38}" stroke="#D97706" stroke-width="6" stroke-opacity="0.45"/>
            <text x="{cx}" y="{cy - 8}" fill="#F8FAFC" font-size="11" font-weight="700" text-anchor="middle" font-family="sans-serif">
                {obj.name}
            </text>
            <text x="{cx}" y="{cy + 14}" fill="#94A3B8" font-size="9" text-anchor="middle" font-family="sans-serif">
                {obj.mass_g:.0f}g | μ={obj.friction_coeff:.2f}
            </text>
        </g>
        """
    else:  # Sponge
        # Highly compliant foam block with pores
        squeeze_x = 45 + (p * 8)
        squeeze_y = 35 - (p * 10)
        return f"""
        <g id="grasped-sponge">
            <rect x="{cx - squeeze_x}" y="{cy - squeeze_y}" width="{squeeze_x * 2}" height="{squeeze_y * 2}" rx="8" 
                  fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="2.5"/>
            <!-- Foam Pores -->
            <circle cx="{cx - 20}" cy="{cy - 10}" r="4" fill="{color}" fill-opacity="0.6"/>
            <circle cx="{cx + 15}" cy="{cy - 8}" r="5" fill="{color}" fill-opacity="0.6"/>
            <circle cx="{cx - 5}" cy="{cy + 12}" r="3.5" fill="{color}" fill-opacity="0.6"/>
            <circle cx="{cx + 25}" cy="{cy + 10}" r="4" fill="{color}" fill-opacity="0.6"/>
            <text x="{cx}" y="{cy - 4}" fill="#F8FAFC" font-size="11" font-weight="700" text-anchor="middle" font-family="sans-serif">
                {obj.name}
            </text>
            <text x="{cx}" y="{cy + 12}" fill="#94A3B8" font-size="9" text-anchor="middle" font-family="sans-serif">
                Compliance: {obj.compliance_mm_per_n:.1f} mm/N
            </text>
        </g>
        """

