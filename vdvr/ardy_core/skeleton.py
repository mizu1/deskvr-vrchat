"""Loading ARDY SOMA77 motion clips and forward kinematics.

An ARDY clip (``motion.npz``) contains, per frame:

=============== ====================== =========================================
key             shape                  meaning
=============== ====================== =========================================
posed_joints    (1, T, 77, 3)          **world-space** joint positions (metres)
global_rot_mats (1, T, 77, 3, 3)       world-space bone rotation matrices
local_rot_mats  (1, T, 77, 3, 3)       rotation relative to the parent bone
root_positions  (1, T, 3)              root (Hips) translation
smooth_root_pos (1, T, 3)              smoothed root trajectory (used for
                                       in-place playback)
foot_contacts   (1, T, 6)              [L0,L1,L2,R0,R1,R2] contact flags
fps             scalar                 frames per second (30 for ARDY)
text            scalar                 generation prompt
=============== ====================== =========================================

The SOMA77 rig is rigid, so ``R_parent^T @ (p_child - p_parent)`` is constant for
every frame.  We call that constant the *rest offset* and use it to run forward
kinematics, which lets us validate / repair the clip and rebuild joints when a
file only stores rotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton_data import JOINT, SOMA77_NAMES, SOMA77_PARENTS

N_JOINTS = 77


def _squeeze_leading(array, dims: int) -> np.ndarray:
    """Drop ARDY's leading batch dimension (``(1, T, ...)`` -> ``(T, ...)``)."""
    array = np.asarray(array)
    while array.ndim > dims and array.shape[0] == 1:
        array = array[0]
    return array


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or len(x) < 2:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(x.shape[1])], axis=1)
    return smoothed[: len(x)]


@dataclass
class SomaMotion:
    """A decoded, single-clip SOMA77 animation."""

    joints: np.ndarray                 # (T, 77, 3) world positions, metres
    rotations: np.ndarray              # (T, 77, 3, 3) world rotations
    root: np.ndarray                   # (T, 3)
    smooth_root: np.ndarray            # (T, 3)
    foot_contacts: np.ndarray | None   # (T, 6) bool
    fps: float = 30.0
    text: str = ""
    source: str = ""
    local_rotations: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return int(self.joints.shape[0])

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    def joint(self, name: str) -> np.ndarray:
        return self.joints[:, JOINT[name]]

    def frame_joint(self, frame: int, name: str) -> np.ndarray:
        return self.joints[int(frame), JOINT[name]]

    # -- contact helpers ---------------------------------------------------- #
    def contact(self, frame: int, side: str) -> bool:
        """True when one of the three contact flags for ``side`` is set."""
        if self.foot_contacts is None:
            return False
        flags = self.foot_contacts[int(frame)]
        return bool(np.any(flags[:3] if side == "Left" else flags[3:]))

    def contacts(self, side: str) -> np.ndarray:
        if self.foot_contacts is None:
            return np.zeros(self.frames, dtype=bool)
        flags = self.foot_contacts
        return np.any(flags[:, :3] if side == "Left" else flags[:, 3:], axis=1)


def load_motion(path: str | Path) -> SomaMotion:
    """Load an ARDY ``motion.npz`` (or any npz with the same fields)."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in keys else 30.0
    text = str(np.asarray(data["text"]).reshape(-1)[0]) if "text" in keys else ""

    local = None
    if "local_rot_mats" in keys:
        local = _squeeze_leading(data["local_rot_mats"], 4).astype(np.float64)
        if local.shape[1:] != (N_JOINTS, 3, 3):
            raise ValueError(f"unexpected local_rot_mats shape {local.shape}")

    if "global_rot_mats" in keys:
        rotations = _squeeze_leading(data["global_rot_mats"], 4).astype(np.float64)
    elif local is not None:
        rotations = compose_global_rotations(local)
    else:
        raise ValueError("motion.npz needs global_rot_mats or local_rot_mats")

    if "posed_joints" in keys:
        joints = _squeeze_leading(data["posed_joints"], 3).astype(np.float64)
    else:
        # Rotations are enough to rebuild the pose once rest offsets are known;
        # without a bind pose we fall back to zero offsets, which only works for
        # clips that also ship joint positions.
        raise ValueError("motion.npz needs posed_joints for retargeting")

    if "root_positions" in keys:
        root = _squeeze_leading(data["root_positions"], 2).astype(np.float64)
    else:
        root = joints[:, 0].copy()

    if "smooth_root_pos" in keys:
        smooth_root = _squeeze_leading(data["smooth_root_pos"], 2).astype(np.float64)
    else:
        smooth_root = _moving_average(root, max(1, int(round(fps))))

    foot_contacts = None
    if "foot_contacts" in keys:
        foot_contacts = _squeeze_leading(data["foot_contacts"], 2).astype(bool)

    motion = SomaMotion(
        joints=joints,
        rotations=rotations,
        root=root,
        smooth_root=smooth_root,
        foot_contacts=foot_contacts,
        fps=fps if fps > 0 else 30.0,
        text=text,
        source=str(path),
        local_rotations=local,
    )
    _validate(motion)
    return motion


def _validate(motion: SomaMotion) -> None:
    frames = motion.joints.shape[0]
    if motion.joints.ndim != 3 or motion.joints.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"unexpected posed_joints shape {motion.joints.shape}")
    if motion.rotations.shape != (frames, N_JOINTS, 3, 3):
        raise ValueError(f"unexpected global_rot_mats shape {motion.rotations.shape}")
    if not np.isfinite(motion.joints).all():
        raise ValueError("posed_joints contains NaN/Inf")
    if motion.frames < 1:
        raise ValueError("motion has no frames")


def compose_global_rotations(local: np.ndarray) -> np.ndarray:
    """Compose per-bone local rotations into world rotations (parent order)."""
    frames = local.shape[0]
    out = np.empty_like(local)
    out[:, 0] = local[:, 0]
    for joint in range(1, N_JOINTS):
        parent = SOMA77_PARENTS[joint]
        out[:, joint] = out[:, parent] @ local[:, joint]
    return out


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def calibrate_rest_offsets(motion: SomaMotion) -> np.ndarray:
    """Recover constant rest offsets ``R_parent^T (p_child - p_parent)``.

    For a rigid skeleton the value is identical every frame; the median makes the
    estimate robust to tiny post-processing corrections.  Index 0 is the root and
    is left at zero (the root position is used directly by FK).
    """
    offsets = np.zeros((N_JOINTS, 3), dtype=np.float64)
    for joint in range(1, N_JOINTS):
        parent = SOMA77_PARENTS[joint]
        rel = np.einsum("tji,tj->ti", motion.rotations[:, parent], motion.joints[:, joint] - motion.joints[:, parent])
        offsets[joint] = np.median(rel, axis=0)
    return offsets


def forward_kinematics(root: np.ndarray, rotations: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Rebuild world joint positions from root translation, rotations and offsets.

    ``root``       : (T, 3)
    ``rotations``  : (T, 77, 3, 3) world bone rotations
    ``offsets``    : (77, 3) rest offset of each bone in its parent's frame
    """
    root = np.asarray(root, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    frames = root.shape[0]
    joints = np.zeros((frames, N_JOINTS, 3), dtype=np.float64)
    joints[:, 0] = root
    for joint in range(1, N_JOINTS):
        parent = SOMA77_PARENTS[joint]
        joints[:, joint] = joints[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], offsets[joint])
    return joints


def fk_error(motion: SomaMotion, offsets: np.ndarray | None = None) -> float:
    """Max reconstruction error (metres) of FK against ``posed_joints``."""
    offsets = calibrate_rest_offsets(motion) if offsets is None else offsets
    rebuilt = forward_kinematics(motion.root, motion.rotations, offsets)
    return float(np.abs(rebuilt - motion.joints).max())


def bone_lengths(joints: np.ndarray) -> np.ndarray:
    """Median length of each bone (child index -> parent distance)."""
    joints = np.asarray(joints, dtype=np.float64)
    lengths = np.zeros(N_JOINTS, dtype=np.float64)
    for joint in range(1, N_JOINTS):
        parent = SOMA77_PARENTS[joint]
        lengths[joint] = float(np.median(np.linalg.norm(joints[:, joint] - joints[:, parent], axis=1)))
    return lengths


def bone_directions(joints: np.ndarray) -> np.ndarray:
    """Unit direction of every bone (parent -> child)."""
    directions = np.zeros_like(joints, dtype=np.float64)
    for joint in range(1, N_JOINTS):
        parent = SOMA77_PARENTS[joint]
        v = joints[:, joint] - joints[:, parent]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        directions[:, joint] = v / np.maximum(n, 1e-9)
    return directions


def chain_length(lengths: np.ndarray, names: list[str]) -> float:
    return float(sum(lengths[JOINT[name]] for name in names))


SPINE_CHAIN = ["Spine1", "Spine2", "Chest"]
NECK_CHAIN = ["Neck1", "Neck2", "Head"]
