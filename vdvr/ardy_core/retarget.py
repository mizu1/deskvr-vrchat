"""Retarget SOMA77 motion onto the VRChat OSC tracker rig.

Pipeline for every frame
------------------------
1. **Root stabilisation** - optionally strip the horizontal root trajectory so the
   avatar performs the motion in place (sway is kept, locomotion is removed).
2. **Coordinate conversion** - mirror X (right-handed SOMA -> left-handed Unity),
   scale by ``target_height / source_height`` and apply a floor offset.
3. **Forward-kinematics retarget** - walk the skeleton parent -> child, keep each
   source bone *direction* and use the target rig's bone *length*. This transfers
   the pose to a differently proportioned avatar while keeping every limb rigid.
4. **Two-bone IK** - solve legs and arms so the end effectors hit their targets
   with the target bone lengths. For legs this also plants a contacting foot on
   the floor and clamps unreachable targets instead of stretching the limb.
5. **Tracker synthesis** - hip / chest / knees / feet / elbows (+ head), with
   rotations carried over from the source bone and corrected for the IK delta.

The result is a sequence of :class:`TrackerFrame` values sampled at the source
frame rate; :mod:`ardy_vrc.player` interpolates it to the OSC send rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .mathutil import (
    lerp,
    matrix_from_quat,
    mirror_x_rotation,
    normalize,
    quat_from_matrix,
    quat_from_two_vectors,
    quat_mul,
    quat_normalize,
    quat_slerp,
)
from .skeleton import JOINT, SOMA77_PARENTS, SomaMotion, bone_lengths, chain_length

# VRChat OSC tracker slots.  These are the eight numbered trackers from the
# official tracker contract; one body point per slot, every frame.
# Slot order used by VRChat's OSC tracker consumers (matches the working
# deskvr-vrchat/vdvr implementation): hip, chest, feet, knees, elbows.
SLOT_INDEX = {
    "hip": 1,
    "chest": 2,
    "lfoot": 3,
    "rfoot": 4,
    "lknee": 5,
    "rknee": 6,
    "lelbow": 7,
    "relbow": 8,
}
ALL_SLOTS = tuple(SLOT_INDEX)
HEAD_SLOT = "head"

SLOT_PRESETS = {
    "all": ALL_SLOTS,
    "lite": ("hip", "lfoot", "rfoot"),
    "legs": ("hip", "lknee", "rknee", "lfoot", "rfoot"),
    "upper": ("chest", "hip", "lelbow", "relbow"),
}

# Left/right landmark names used by the solver.
SIDES = {
    "L": {
        "hip": "LeftLeg",
        "knee": "LeftShin",
        "foot": "LeftFoot",
        "toe": "LeftToeBase",
        "thigh": "LeftLeg",
        "shoulder": "LeftArm",
        "elbow": "LeftForeArm",
        "wrist": "LeftHand",
        "upper_arm": "LeftArm",
        "prefix": "l",
    },
    "R": {
        "hip": "RightLeg",
        "knee": "RightShin",
        "foot": "RightFoot",
        "toe": "RightToeBase",
        "thigh": "RightLeg",
        "shoulder": "RightArm",
        "elbow": "RightForeArm",
        "wrist": "RightHand",
        "upper_arm": "RightArm",
        "prefix": "r",
    },
}


# --------------------------------------------------------------------------- #
# Rig description
# --------------------------------------------------------------------------- #
@dataclass
class Rig:
    """Target body proportions (metres) and per-bone lengths."""

    height: float
    length: np.ndarray            # (77,) bone length, child index -> parent
    scale: float
    hip_width: float
    shoulder_width: float
    ankle_height: float
    hip_drop: float               # Hips joint -> hip joint vertical distance
    shoulder_drop: float          # Chest joint -> shoulder joint vertical distance

    def bone(self, name: str) -> float:
        return float(self.length[JOINT[name]])


def _source_height(motion: SomaMotion) -> float:
    """Height of the source character in its first (neutral) frame."""
    first = float(motion.joints[0, JOINT["HeadEnd"], 1])
    if 0.5 < first < 2.5:
        return first
    # Lying/rolling clips can start low; fall back to the tallest frame.
    tallest = float(np.max(motion.joints[:, JOINT["HeadEnd"], 1]))
    if 0.5 < tallest < 2.5:
        return tallest
    return 1.75


def build_rig(
    motion: SomaMotion,
    height: float = 1.70,
    source_height: float | None = None,
    scale: float | None = None,
    profile: Mapping | None = None,
) -> Rig:
    """Build target proportions from the clip's own skeleton, with overrides.

    ``profile`` (JSON-compatible mapping) may contain::

        {"height": 1.72,
         "bone_lengths": {"LeftShin": 0.44, "RightShin": 0.44, ...},
         "hip_width": 0.22, "shoulder_width": 0.38, "ankle_height": 0.07}
    """
    src_h = float(source_height) if source_height else _source_height(motion)
    if src_h <= 0:
        raise ValueError("source_height must be positive")

    if scale:
        k = float(scale)
        target_height = src_h * k
    else:
        target_height = float(profile.get("height", height)) if profile else float(height)
        k = target_height / src_h
    if k <= 0:
        raise ValueError("scale must be positive")

    frame0 = motion.joints[0]
    lengths = bone_lengths(motion.joints) * k

    hip_width = abs(float(frame0[JOINT["LeftLeg"], 0] - frame0[JOINT["RightLeg"], 0])) * k
    shoulder_width = abs(float(frame0[JOINT["LeftArm"], 0] - frame0[JOINT["RightArm"], 0])) * k
    ankle_height = 0.5 * float(frame0[JOINT["LeftFoot"], 1] + frame0[JOINT["RightFoot"], 1]) * k
    hip_drop = (float(frame0[JOINT["Hips"], 1]) - 0.5 * float(frame0[JOINT["LeftLeg"], 1] + frame0[JOINT["RightLeg"], 1])) * k
    shoulder_drop = (0.5 * float(frame0[JOINT["LeftArm"], 1] + frame0[JOINT["RightArm"], 1]) - float(frame0[JOINT["Chest"], 1])) * k

    if profile:
        hip_width = float(profile.get("hip_width", hip_width))
        shoulder_width = float(profile.get("shoulder_width", shoulder_width))
        ankle_height = float(profile.get("ankle_height", ankle_height))
        for name, value in (profile.get("bone_lengths") or {}).items():
            if name not in JOINT:
                raise ValueError(f"unknown bone in rig profile: {name}")
            lengths[JOINT[name]] = float(value)

    # Keep the rig physically plausible.
    lengths = np.maximum(lengths, 1e-4)
    ankle_height = float(np.clip(ankle_height, 0.005, 0.25))

    return Rig(
        height=target_height,
        length=lengths,
        scale=k,
        hip_width=max(hip_width, 0.05),
        shoulder_width=max(shoulder_width, 0.05),
        ankle_height=ankle_height,
        hip_drop=float(hip_drop),
        shoulder_drop=float(shoulder_drop),
    )


# --------------------------------------------------------------------------- #
# Poses
# --------------------------------------------------------------------------- #
@dataclass
class TrackerFrame:
    positions: dict[str, np.ndarray]
    quats: dict[str, np.ndarray]

    def slots(self) -> Iterable[str]:
        return self.positions.keys()


@dataclass
class PoseSequence:
    """Interpolatable tracker track: positions (T,S,3) and quats (T,S,4)."""

    slots: tuple[str, ...]
    positions: np.ndarray
    quats: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.quats = np.asarray(self.quats, dtype=np.float64)

    @property
    def frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def duration(self) -> float:
        return (self.frames - 1) / self.fps if self.frames > 1 else 0.0

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate at time ``t`` seconds (clamped to the clip)."""
        if self.frames == 1:
            return self.positions[0].copy(), self.quats[0].copy()
        x = float(np.clip(t, 0.0, self.duration)) * self.fps
        i0 = min(int(np.floor(x)), self.frames - 1)
        i1 = min(i0 + 1, self.frames - 1)
        f = x - i0
        pos = self.positions[i0] * (1.0 - f) + self.positions[i1] * f
        quats = np.stack([quat_slerp(self.quats[i0, s], self.quats[i1, s], f) for s in range(len(self.slots))])
        return pos, quats

    @classmethod
    def single(cls, frame: TrackerFrame, slots: Iterable[str], fps: float = 30.0) -> "PoseSequence":
        slots = tuple(slots)
        pos = np.stack([frame.positions[s] for s in slots])[None]
        quat = np.stack([frame.quats[s] for s in slots])[None]
        return cls(slots=slots, positions=pos, quats=quat, fps=fps)


# --------------------------------------------------------------------------- #
# IK
# --------------------------------------------------------------------------- #
def two_bone_ik(
    root: np.ndarray,
    target: np.ndarray,
    l1: float,
    l2: float,
    pole_point: np.ndarray,
    fallback_pole: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic two-bone IK.

    Returns ``(mid, end)``.  ``pole_point`` (a hint joint such as the source knee
    or elbow) chooses the bend plane; ``fallback_pole`` is used only when the hint
    is collinear with the root->target axis.
    """
    root = np.asarray(root, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    v = target - root
    d = float(np.linalg.norm(v))
    if d < 1e-8:
        v = np.array([0.0, -1.0, 0.0])
        d = 1e-8
    axis = v / d

    d = float(np.clip(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))

    pole = np.asarray(pole_point, dtype=np.float64) - root
    pole = pole - axis * float(np.dot(pole, axis))
    if np.linalg.norm(pole) < 1e-6:
        base = np.array([0.0, 0.0, 1.0]) if fallback_pole is None else np.asarray(fallback_pole, dtype=np.float64)
        pole = base - axis * float(np.dot(base, axis))
        if np.linalg.norm(pole) < 1e-6:
            base = np.array([1.0, 0.0, 0.0])
            pole = base - axis * float(np.dot(base, axis))
    pole = normalize(pole)

    a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    h = float(np.sqrt(max(l1 * l1 - a * a, 0.0)))
    mid = root + axis * a + pole * h
    end = root + axis * d
    return mid, end


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
@dataclass
class RetargetOptions:
    height: float = 1.70
    source_height: float | None = None
    scale: float | None = None
    in_place: bool = True
    ground_lock: bool = True
    flat_hip: bool = False
    floor_offset: float = 0.0
    send_head: bool = True
    slots: tuple[str, ...] = ALL_SLOTS
    chest_offset: float = 0.35       # 0 = Chest joint, 1 = Neck1
    hip_offset: float = 0.0          # metres, along the hip bone
    hip_forward: float = 0.0
    chest_forward: float = 0.0
    knee_ratio: float = 0.85         # 0 = hip joint, 1 = knee joint
    elbow_ratio: float = 0.5         # 0 = shoulder, 1 = elbow
    rig_profile: Mapping | None = None

    def selected_slots(self) -> tuple[str, ...]:
        slots = tuple(dict.fromkeys(self.slots))
        unknown = [s for s in slots if s not in SLOT_INDEX]
        if unknown:
            raise ValueError(f"unknown tracker slot(s): {unknown}")
        return slots


# --------------------------------------------------------------------------- #
# Retargeter
# --------------------------------------------------------------------------- #
class Retargeter:
    def __init__(self, motion: SomaMotion, options: RetargetOptions | None = None):
        self.motion = motion
        self.options = options or RetargetOptions()
        self.rig = build_rig(
            motion,
            height=self.options.height,
            source_height=self.options.source_height,
            scale=self.options.scale,
            profile=self.options.rig_profile,
        )
        self.scale = self.rig.scale

        if self.options.in_place:
            delta = motion.smooth_root - motion.smooth_root[0]
            delta = delta.copy()
            delta[:, 1] = 0.0
            self.root_delta = delta
        else:
            self.root_delta = np.zeros_like(motion.smooth_root)

        self.rest = self._build_rest_frame()
        self.sequence = self._build_sequence()

    # -- public ------------------------------------------------------------- #
    def solve_frame(self, frame: int) -> TrackerFrame:
        return self._solve_frame_with_joints(frame)[0]

    def _solve_frame_with_joints(self, frame: int) -> tuple[TrackerFrame, np.ndarray]:
        frame = int(np.clip(frame, 0, self.motion.frames - 1))
        p = self._source_positions(frame)
        tp = self._fk_retarget(p)
        tp = self._solve_legs(tp, frame)
        tp = self._solve_arms(tp)
        positions, quats = self._trackers(p, tp, frame)
        return TrackerFrame(positions=positions, quats=quats), tp

    def sequence_slots(self) -> tuple[str, ...]:
        slots = self.options.selected_slots()
        if self.options.send_head:
            slots = slots + (HEAD_SLOT,)
        return slots

    # -- stages ------------------------------------------------------------- #
    def _source_positions(self, frame: int) -> np.ndarray:
        joints = self.motion.joints[frame]
        joints = joints - self.root_delta[frame]
        p = joints * np.array([-self.scale, self.scale, self.scale])
        p[:, 1] += self.options.floor_offset
        return p

    def _fk_retarget(self, p: np.ndarray) -> np.ndarray:
        tp = np.empty_like(p)
        tp[0] = p[0]
        for joint in range(1, len(SOMA77_PARENTS)):
            parent = SOMA77_PARENTS[joint]
            d = p[joint] - p[parent]
            n = float(np.linalg.norm(d))
            if n < 1e-7:
                d = np.array([0.0, 1.0, 0.0])
                n = 1.0
            tp[joint] = tp[parent] + (d / n) * self.rig.length[joint]
        return tp

    def _solve_legs(self, tp: np.ndarray, frame: int) -> np.ndarray:
        ground = self.options.ground_lock and self.motion.foot_contacts is not None
        for side, names in SIDES.items():
            hip_i = JOINT[names["hip"]]
            knee_i = JOINT[names["knee"]]
            foot_i = JOINT[names["foot"]]
            target = tp[foot_i].copy()
            if ground and self.motion.contact(frame, "Left" if side == "L" else "Right"):
                target[1] = self.rig.ankle_height + self.options.floor_offset
            knee, foot = two_bone_ik(
                tp[hip_i],
                target,
                self.rig.length[knee_i],
                self.rig.length[foot_i],
                tp[knee_i],
                fallback_pole=np.array([0.0, 0.0, 1.0]),
            )
            tp[knee_i] = knee
            tp[foot_i] = foot
        return tp

    def _solve_arms(self, tp: np.ndarray) -> np.ndarray:
        for side, names in SIDES.items():
            shoulder_i = JOINT[names["shoulder"]]
            elbow_i = JOINT[names["elbow"]]
            wrist_i = JOINT[names["wrist"]]
            elbow, wrist = two_bone_ik(
                tp[shoulder_i],
                tp[wrist_i],
                self.rig.length[elbow_i],
                self.rig.length[wrist_i],
                tp[elbow_i],
                fallback_pole=np.array([0.0, 0.0, -1.0]),
            )
            tp[elbow_i] = elbow
            tp[wrist_i] = wrist
        return tp

    def _trackers(
        self, p: np.ndarray, tp: np.ndarray, frame: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        opts = self.options
        q_src = {
            name: quat_from_matrix(mirror_x_rotation(self.motion.rotations[frame, JOINT[name]]))
            for name in ("Hips", "Chest", "Head")
        }

        positions: dict[str, np.ndarray] = {}
        quats: dict[str, np.ndarray] = {}

        hip_q = q_src["Hips"]
        chest_q = q_src["Chest"]
        up = np.array([0.0, 1.0, 0.0])
        hip_fwd = matrix_from_quat(hip_q) @ np.array([0.0, 0.0, 1.0])
        chest_fwd = matrix_from_quat(chest_q) @ np.array([0.0, 0.0, 1.0])

        hip_pos = tp[JOINT["Hips"]] + up * opts.hip_offset + hip_fwd * opts.hip_forward
        if opts.flat_hip:
            hip_pos = hip_pos.copy()
            hip_pos[1] = self.rest.positions["hip"][1]
        positions["hip"] = hip_pos
        quats["hip"] = hip_q

        chest_point = tp[JOINT["Chest"]] + (tp[JOINT["Neck1"]] - tp[JOINT["Chest"]]) * opts.chest_offset
        positions["chest"] = chest_point + chest_fwd * opts.chest_forward
        quats["chest"] = chest_q

        for side, names in SIDES.items():
            prefix = names["prefix"]
            hip_i, knee_i, foot_i = JOINT[names["hip"]], JOINT[names["knee"]], JOINT[names["foot"]]
            shoulder_i, elbow_i = JOINT[names["shoulder"]], JOINT[names["elbow"]]

            # Knee tracker sits on the thigh (0 = hip joint, 1 = knee joint).
            positions[f"{prefix}knee"] = lerp(tp[hip_i], tp[knee_i], opts.knee_ratio)
            # Elbow tracker sits above the elbow, on the upper arm.
            positions[f"{prefix}elbow"] = lerp(tp[shoulder_i], tp[elbow_i], opts.elbow_ratio)
            positions[f"{prefix}foot"] = tp[foot_i].copy()

            # The knee tracker sits on the *thigh* and the elbow tracker on the
            # *upper arm*, so their orientation must be the parent bone's
            # orientation (SOMA stores the bone starting at each joint), adjusted
            # for the IK delta of that segment.  The foot uses the foot bone.
            thigh_i = JOINT[names["thigh"]]
            upper_arm_i = JOINT[names["upper_arm"]]
            knee_q = self._delta_rotation(
                p, tp, thigh_i, knee_i, mirror_x_rotation(self.motion.rotations[frame, thigh_i])
            )
            foot_q = quat_from_matrix(mirror_x_rotation(self.motion.rotations[frame, foot_i]))
            elbow_q = self._delta_rotation(
                p, tp, upper_arm_i, elbow_i, mirror_x_rotation(self.motion.rotations[frame, upper_arm_i])
            )

            quats[f"{prefix}knee"] = knee_q
            quats[f"{prefix}foot"] = foot_q
            quats[f"{prefix}elbow"] = elbow_q

        if opts.send_head:
            positions[HEAD_SLOT] = tp[JOINT["Head"]].copy()
            quats[HEAD_SLOT] = q_src["Head"]

        return positions, quats

    @staticmethod
    def _delta_rotation(p: np.ndarray, tp: np.ndarray, joint_i: int, child_i: int, R_unity: np.ndarray) -> np.ndarray:
        src_dir = p[child_i] - p[joint_i]
        dst_dir = tp[child_i] - tp[joint_i]
        if np.linalg.norm(src_dir) < 1e-8 or np.linalg.norm(dst_dir) < 1e-8:
            return quat_from_matrix(R_unity)
        delta = quat_from_two_vectors(src_dir, dst_dir)
        return quat_normalize(quat_mul(delta, quat_from_matrix(R_unity)))

    # -- sequences ---------------------------------------------------------- #
    def _build_sequence(self) -> PoseSequence:
        slots = self.sequence_slots()
        frames: list[TrackerFrame] = []
        joints: list[np.ndarray] = []
        for t in range(self.motion.frames):
            frame, tp = self._solve_frame_with_joints(t)
            frames.append(frame)
            joints.append(tp)
        # Retargeted joint positions (after IK) kept for export/preview tooling.
        self.joint_positions = np.stack(joints).astype(np.float32)
        positions = np.stack([np.stack([f.positions[s] for s in slots]) for f in frames])
        quats = np.stack([np.stack([f.quats[s] for s in slots]) for f in frames])
        return PoseSequence(slots=slots, positions=positions, quats=quats, fps=self.motion.fps)

    # -- rest pose ---------------------------------------------------------- #
    def _build_rest_frame(self) -> TrackerFrame:
        """Relaxed A-pose built from the target rig, used for fade-in/return."""
        opts = self.options
        rig = self.rig
        L = rig.length

        thigh = float(L[JOINT["LeftShin"]])
        shin = float(L[JOINT["LeftFoot"]])
        hip_joint_y = rig.ankle_height + shin + thigh
        hips_y = hip_joint_y + rig.hip_drop
        hips = np.array([0.0, hips_y, 0.0])

        spine = chain_length(L, ["Spine1", "Spine2", "Chest"])
        chest = hips + np.array([0.0, spine, 0.0])
        neck = chain_length(L, ["Neck1", "Neck2", "Head"])
        head = chest + np.array([0.0, neck, 0.0])

        half_hip = rig.hip_width * 0.5
        half_shoulder = rig.shoulder_width * 0.5
        shoulder_y = chest[1] + rig.shoulder_drop

        upper_arm = float(L[JOINT["LeftForeArm"]])
        forearm = float(L[JOINT["LeftHand"]])
        arm_angle = np.radians(20.0)
        arm_dir = np.array([np.sin(arm_angle), -np.cos(arm_angle), 0.0])

        # Orientations: the canonical rest pose faces +Z with flat feet, so the
        # body trackers use Unity identity.  Knee/elbow orientations are derived
        # from frame 0's bone orientation with the rest-pose direction delta; they
        # are only transmitted in --all-rotations mode.
        p0 = self._source_positions(0)

        def source_quat(name: str) -> np.ndarray:
            return quat_from_matrix(mirror_x_rotation(self.motion.rotations[0, JOINT[name]]))

        identity = np.array([1.0, 0.0, 0.0, 0.0])
        positions: dict[str, np.ndarray] = {}
        quats: dict[str, np.ndarray] = {}

        positions["hip"] = hips
        quats["hip"] = identity
        positions["chest"] = chest + np.array([0.0, opts.chest_offset * float(L[JOINT["Neck1"]]), 0.0])
        quats["chest"] = identity

        for side, names in SIDES.items():
            # Unity/VRChat space: character left is -X, right is +X.
            sign = -1.0 if side == "L" else 1.0
            hip_joint = np.array([sign * half_hip, hip_joint_y, 0.0])
            knee = hip_joint + np.array([0.0, -thigh, 0.005])
            ankle = knee + np.array([0.0, -shin, 0.0])
            shoulder = np.array([sign * half_shoulder, shoulder_y, 0.0])
            direction = np.array([sign * np.sin(arm_angle), -np.cos(arm_angle), 0.0])
            elbow = shoulder + direction * upper_arm
            wrist = elbow + direction * forearm

            prefix = names["prefix"]
            positions[f"{prefix}knee"] = lerp(hip_joint, knee, opts.knee_ratio)
            positions[f"{prefix}foot"] = ankle
            positions[f"{prefix}elbow"] = lerp(shoulder, elbow, opts.elbow_ratio)

            thigh_i = JOINT[names["thigh"]]
            knee_i = JOINT[names["knee"]]
            upper_arm_i = JOINT[names["upper_arm"]]
            elbow_i = JOINT[names["elbow"]]
            src_thigh = normalize(p0[knee_i] - p0[thigh_i])
            dst_thigh = normalize(knee - hip_joint)
            src_upper = normalize(p0[elbow_i] - p0[upper_arm_i])
            dst_upper = normalize(elbow - shoulder)
            quats[f"{prefix}knee"] = quat_normalize(
                quat_mul(quat_from_two_vectors(src_thigh, dst_thigh), source_quat(names["thigh"]))
            )
            quats[f"{prefix}foot"] = identity
            quats[f"{prefix}elbow"] = quat_normalize(
                quat_mul(quat_from_two_vectors(src_upper, dst_upper), source_quat(names["upper_arm"]))
            )

        if opts.send_head:
            positions[HEAD_SLOT] = head.copy()
            quats[HEAD_SLOT] = identity

        return TrackerFrame(positions=positions, quats=quats)

    def rest_sequence(self) -> PoseSequence:
        return PoseSequence.single(self.rest, self.sequence_slots(), fps=self.motion.fps)

    # -- calibration pose --------------------------------------------------- #
    def tpose_frame(self) -> TrackerFrame:
        """Static T-pose used while calibrating VRChat's FBT.

        VRChat calibrates the virtual trackers against the avatar's T-pose, so
        the sender must present a *static* T-pose (arms horizontal) during
        calibration.  Sending a moving clip while calibrating bakes a wrong
        offset into VRChat and the resulting body looks broken.
        """
        opts = self.options
        rig = self.rig
        L = rig.length

        thigh = float(L[JOINT["LeftShin"]])
        shin = float(L[JOINT["LeftFoot"]])
        hip_joint_y = rig.ankle_height + shin + thigh
        hips_y = hip_joint_y + rig.hip_drop
        hips = np.array([0.0, hips_y, 0.0])

        spine = chain_length(L, ["Spine1", "Spine2", "Chest"])
        chest = hips + np.array([0.0, spine, 0.0])
        neck = chain_length(L, ["Neck1", "Neck2", "Head"])
        head = chest + np.array([0.0, neck, 0.0])

        half_hip = rig.hip_width * 0.5
        half_shoulder = rig.shoulder_width * 0.5
        shoulder_y = chest[1] + rig.shoulder_drop
        upper_arm = float(L[JOINT["LeftForeArm"]])

        identity = np.array([1.0, 0.0, 0.0, 0.0])
        positions: dict[str, np.ndarray] = {}
        quats: dict[str, np.ndarray] = {}

        positions["hip"] = hips
        quats["hip"] = identity
        positions["chest"] = chest + np.array([0.0, opts.chest_offset * float(L[JOINT["Neck1"]]), 0.0])
        quats["chest"] = identity

        for side, names in SIDES.items():
            # Unity/VRChat space: character left is -X, right is +X.
            sign = -1.0 if side == "L" else 1.0
            hip_joint = np.array([sign * half_hip, hip_joint_y, 0.0])
            knee = hip_joint + np.array([0.0, -thigh, 0.005])
            ankle = knee + np.array([0.0, -shin, 0.0])
            shoulder = np.array([sign * half_shoulder, shoulder_y, 0.0])
            elbow = shoulder + np.array([sign * upper_arm, 0.0, 0.0])

            prefix = names["prefix"]
            positions[f"{prefix}knee"] = lerp(hip_joint, knee, opts.knee_ratio)
            positions[f"{prefix}foot"] = ankle
            positions[f"{prefix}elbow"] = lerp(shoulder, elbow, opts.elbow_ratio)
            quats[f"{prefix}knee"] = identity
            quats[f"{prefix}foot"] = identity
            quats[f"{prefix}elbow"] = identity

        if opts.send_head:
            positions[HEAD_SLOT] = head.copy()
            quats[HEAD_SLOT] = identity

        return TrackerFrame(positions=positions, quats=quats)

    def tpose_sequence(self) -> PoseSequence:
        return PoseSequence.single(self.tpose_frame(), self.sequence_slots(), fps=self.motion.fps)
