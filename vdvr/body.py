"""Synthetic full-body pose generation.

VRChat's OSC trackers are absolute world-space poses (Unity coordinates, metres,
euler degrees applied in Z-X-Y order).  We only have a head and two hands, so we
invent a plausible spine + leg pose from:

* head position / yaw / pitch (driven by the mouse),
* hand positions (drive the elbows),
* locomotion input (drives a walk cycle),
* crouch / jump state.

The result is what VRChat's calibrated full-body IK consumes, so the avatar
looks like it is wearing real trackers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .math3d import (
    Vec3,
    clamp,
    forward_from_yaw,
    right_from_yaw,
    v,
    vadd,
    vlerp,
    vmul,
    vsub,
)

TrackerPose = tuple[Vec3, tuple[float, float, float]]

# VRChat's fixed OSC tracker slot assignment.
SLOT_HIP = 1
SLOT_CHEST = 2
SLOT_LEFT_FOOT = 3
SLOT_RIGHT_FOOT = 4
SLOT_LEFT_KNEE = 5
SLOT_RIGHT_KNEE = 6
SLOT_LEFT_ELBOW = 7
SLOT_RIGHT_ELBOW = 8

PROFILES: dict[str, tuple[int, ...]] = {
    # Fewer trackers = VRChat's IK compensates more = more forgiving.
    "minimal": (SLOT_HIP, SLOT_LEFT_FOOT, SLOT_RIGHT_FOOT),
    "standard": (SLOT_HIP, SLOT_CHEST, SLOT_LEFT_FOOT, SLOT_RIGHT_FOOT),
    "full": tuple(range(1, 9)),
}


@dataclass
class BodyInput:
    head_pos: Vec3 = (0.0, 1.6, 0.0)
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    left_wrist: Vec3 = (-0.2, 1.2, 0.25)
    right_wrist: Vec3 = (0.2, 1.2, 0.25)
    move_forward: float = 0.0
    move_strafe: float = 0.0
    running: bool = False
    crouch: float = 0.0  # 0..1, already applied to head_pos by the caller
    jumping: bool = False


@dataclass
class BodyModel:
    """Anthropometric model, all proportions relative to the user's height."""

    height: float = 1.70
    profile: str = "standard"

    # internal animation state
    time: float = 0.0
    phase: float = 0.0
    move_amp: float = 0.0
    _breath: float = 0.0
    _idle: float = 0.0
    _last: BodyInput = field(default_factory=BodyInput)

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown tracker profile: {self.profile!r}")
        self.height = clamp(float(self.height), 1.0, 2.3)

    @property
    def eye_height(self) -> float:
        return self.height - 0.13

    def tracker_indices(self) -> tuple[int, ...]:
        return PROFILES[self.profile]

    # ------------------------------------------------------------------ update
    def update(self, dt: float, state: BodyInput) -> dict[int, TrackerPose]:
        dt = clamp(dt, 0.0, 0.1)
        self.time += dt
        self._last = state
        s = self.height / 1.70  # uniform scale factor

        hx, hy, hz = state.head_pos
        yaw = state.head_yaw
        pitch = state.head_pitch
        fwd = forward_from_yaw(yaw)
        right = right_from_yaw(yaw)

        # Locomotion amplitude with a little attack/release smoothing.
        planar = math.hypot(state.move_forward, state.move_strafe)
        target_amp = clamp(planar, 0.0, 1.0)
        self.move_amp += (target_amp - self.move_amp) * clamp(dt * 8.0, 0.0, 1.0)

        cadence = 2.35 if state.running else 1.75
        if self.move_amp > 0.01:
            self.phase += dt * 2.0 * math.pi * cadence * max(self.move_amp, 0.25)
        self.phase %= 2.0 * math.pi

        # Idle breathing / weight shift keeps the pose alive when standing.
        self._breath = 0.006 * s * math.sin(self.time * 1.7)
        self._idle = 0.010 * s * math.sin(self.time * 0.55)

        swing = math.sin(self.phase)
        lift_scale = 0.085 * s * self.move_amp
        stride = 0.16 * s * self.move_amp
        hip_bob = 0.018 * s * self.move_amp * math.sin(self.phase * 2.0)

        hip_y = hy - 0.42 * self.height
        neck_y = hy - 0.07 * s
        chest_y = hip_y + 0.58 * (neck_y - hip_y)
        shoulder_y = neck_y - 0.005 * s
        shoulder_half = 0.100 * self.height

        hip_pos = (
            hx + self._idle * (1.0 - self.move_amp),
            hip_y + hip_bob + self._breath,
            hz,
        )
        chest_pos = (
            hx + self._idle * 0.5 * (1.0 - self.move_amp) + fwd[0] * 0.012 * s,
            chest_y + hip_bob * 0.7 + self._breath * 0.8,
            hz + fwd[2] * 0.012 * s,
        )

        # ---- feet ---------------------------------------------------------
        foot_x = 0.088 * s
        foot_y = 0.038 * s
        left_lift = max(0.0, swing) * lift_scale
        right_lift = max(0.0, -swing) * lift_scale
        jump_lift = 0.10 * s if state.jumping else 0.0
        crouch_forward = 0.05 * s * state.crouch

        left_foot = (
            hx - right[0] * foot_x + fwd[0] * (stride * swing + crouch_forward),
            foot_y + left_lift + jump_lift,
            hz - right[2] * foot_x + fwd[2] * (stride * swing + crouch_forward),
        )
        right_foot = (
            hx + right[0] * foot_x + fwd[0] * (-stride * swing + crouch_forward),
            foot_y + right_lift + jump_lift,
            hz + right[2] * foot_x + fwd[2] * (-stride * swing + crouch_forward),
        )

        left_toe = -18.0 * (left_lift / lift_scale) if lift_scale > 1e-6 else 0.0
        right_toe = -18.0 * (right_lift / lift_scale) if lift_scale > 1e-6 else 0.0
        left_foot_rot = (clamp(left_toe, -18.0, 0.0), yaw, 0.0)
        right_foot_rot = (clamp(right_toe, -18.0, 0.0), yaw, 0.0)

        # ---- knees --------------------------------------------------------
        knee_forward = 0.045 * s + 0.075 * s * state.crouch
        left_knee = vlerp(hip_pos, left_foot, 0.5)
        right_knee = vlerp(hip_pos, right_foot, 0.5)
        left_knee = vadd(left_knee, vadd(vmul(fwd, knee_forward), vmul(right, -0.012 * s)))
        right_knee = vadd(right_knee, vadd(vmul(fwd, knee_forward), vmul(right, 0.012 * s)))
        knee_pitch = 8.0 * state.crouch + 6.0 * self.move_amp
        left_knee_rot = (knee_pitch, yaw, 0.0)
        right_knee_rot = (knee_pitch, yaw, 0.0)

        # ---- elbows (derived from the virtual hands) ----------------------
        left_shoulder = (
            hx - right[0] * shoulder_half,
            shoulder_y,
            hz - right[2] * shoulder_half,
        )
        right_shoulder = (
            hx + right[0] * shoulder_half,
            shoulder_y,
            hz + right[2] * shoulder_half,
        )

        left_elbow = vlerp(left_shoulder, state.left_wrist, 0.5)
        right_elbow = vlerp(right_shoulder, state.right_wrist, 0.5)
        left_elbow = vadd(left_elbow, vmul(right, -0.025 * s))
        right_elbow = vadd(right_elbow, vmul(right, 0.025 * s))

        left_arm_yaw = math.degrees(
            math.atan2(state.left_wrist[0] - left_shoulder[0], state.left_wrist[2] - left_shoulder[2])
        )
        right_arm_yaw = math.degrees(
            math.atan2(state.right_wrist[0] - right_shoulder[0], state.right_wrist[2] - right_shoulder[2])
        )

        # ---- assemble -----------------------------------------------------
        poses: dict[int, TrackerPose] = {
            SLOT_HIP: (hip_pos, (0.0, yaw, 4.0 * self.move_amp * swing)),
            SLOT_CHEST: (chest_pos, (pitch * 0.35, yaw, 0.0)),
            SLOT_LEFT_FOOT: (left_foot, left_foot_rot),
            SLOT_RIGHT_FOOT: (right_foot, right_foot_rot),
            SLOT_LEFT_KNEE: (left_knee, left_knee_rot),
            SLOT_RIGHT_KNEE: (right_knee, right_knee_rot),
            SLOT_LEFT_ELBOW: (left_elbow, (0.0, left_arm_yaw, 0.0)),
            SLOT_RIGHT_ELBOW: (right_elbow, (0.0, right_arm_yaw, 0.0)),
        }

        wanted = self.tracker_indices()
        return {slot: poses[slot] for slot in wanted}

    # --------------------------------------------------------------- helpers
    def head_euler(self, state: BodyInput) -> tuple[float, float, float]:
        return (state.head_pitch, state.head_yaw, 0.0)

    def neutral_state(self) -> BodyInput:
        """A relaxed standing pose, used for dry-run previews."""
        return BodyInput(
            head_pos=v(0.0, self.eye_height, 0.0),
            left_wrist=v(-0.18, self.eye_height - 0.40, 0.28),
            right_wrist=v(0.18, self.eye_height - 0.40, 0.28),
        )


def sanitize_tracker(pose: TrackerPose) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Clamp and round values to keep float32 OSC payloads sane."""
    (px, py, pz), (rx, ry, rz) = pose
    return (
        (round(float(px), 4), round(float(py), 4), round(float(pz), 4)),
        (round(float(rx), 3), round(float(ry), 3), round(float(rz), 3)),
    )


def distance(a: Vec3, b: Vec3) -> float:
    d = vsub(a, b)
    return math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
