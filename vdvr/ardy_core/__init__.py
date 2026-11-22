"""Vendored ARDY SOMA77 motion core (FK, calibration and FK/IK retargeting).

Copied from the ardy-action-to-vrchat `ardy_vrc` package so `vdvr` is
self-contained: loading an ARDY motion.npz, forward kinematics checks,
proportional FK retargeting and the two-bone IK foot/arm pass.
"""

from .mathutil import euler_zxy_from_matrix, matrix_from_quat, mirror_x_rotation, quat_from_matrix
from .retarget import PoseSequence, RetargetOptions, Retargeter, TrackerFrame
from .skeleton import (
    SomaMotion,
    calibrate_rest_offsets,
    fk_error,
    forward_kinematics,
    load_motion,
)

__all__ = [
    "PoseSequence",
    "RetargetOptions",
    "Retargeter",
    "SomaMotion",
    "TrackerFrame",
    "calibrate_rest_offsets",
    "euler_zxy_from_matrix",
    "fk_error",
    "forward_kinematics",
    "load_motion",
    "matrix_from_quat",
    "mirror_x_rotation",
    "quat_from_matrix",
]
