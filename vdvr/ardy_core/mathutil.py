"""Small dependency-free vector / rotation helpers.

Conventions used across the package
-----------------------------------
* Column vectors: ``p_world = R @ p_local``.
* Rotation matrices are 3x3 ``float64`` arrays with orthonormal columns (det +1).
* Quaternions are stored as ``(w, x, y, z)``.
* ARDY/SOMA space is right-handed, ``+Y`` up, ``+Z`` forward, and ``+X`` points to
  the character's **left** (the glTF convention: the asset faces ``+Z``).
* VRChat/Unity space is left-handed, ``+Y`` up, ``+Z`` forward, ``+X`` right.

The two spaces are related by mirroring the X axis::

    p_unity = M @ p_soma,          M = diag(-1, 1, 1)
    R_unity = M @ R_soma @ M

Mirroring a proper rotation this way is still a proper rotation, so it can be
converted to a Unity quaternion / Euler triplet without special cases.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-9
MIRROR_X = np.array([-1.0, 1.0, 1.0])


# --------------------------------------------------------------------------- #
# Vectors
# --------------------------------------------------------------------------- #
def normalize(v: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:
    """Return ``v`` scaled to unit length (safe for zero vectors)."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def lerp(a, b, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t


def smoothstep(t: float) -> float:
    """Classic 3t^2 - 2t^3 ease-in/out on [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# --------------------------------------------------------------------------- #
# Handedness / coordinate conversion
# --------------------------------------------------------------------------- #
def mirror_x_position(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=np.float64) * MIRROR_X


def mirror_x_direction(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) * MIRROR_X


def mirror_x_rotation(R: np.ndarray) -> np.ndarray:
    """Convert a proper SOMA rotation to the equivalent Unity rotation."""
    R = np.asarray(R, dtype=np.float64)
    return R * np.outer(MIRROR_X, MIRROR_X)


# --------------------------------------------------------------------------- #
# Quaternions (w, x, y, z)
# --------------------------------------------------------------------------- #
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Robust matrix -> quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s])
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])
    return quat_normalize(q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return matrix_from_quat(q) @ np.asarray(v, dtype=np.float64)


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc spherical interpolation (falls back to lerp for tiny angles)."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(a + (b - a) * float(np.clip(t, 0.0, 1.0)))
    theta0 = math.acos(min(dot, 1.0))
    theta = theta0 * float(np.clip(t, 0.0, 1.0))
    sin0 = math.sin(theta0)
    return quat_normalize(a * (math.sin(theta0 - theta) / sin0) + b * (math.sin(theta) / sin0))


def quat_from_two_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Minimal rotation taking direction ``src`` to ``dst`` (both 3-vectors)."""
    a = normalize(src)
    b = normalize(dst)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-12:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(a, b)
    s = math.sqrt((1.0 + dot) * 2.0)
    return quat_normalize(np.array([s * 0.5, axis[0] / s, axis[1] / s, axis[2] / s]))


def quat_to_euler_zxy_degrees(q: np.ndarray) -> np.ndarray:
    return euler_zxy_from_matrix(matrix_from_quat(q), degrees=True)


# --------------------------------------------------------------------------- #
# Unity-style Euler angles (ZXY order)
# --------------------------------------------------------------------------- #
def euler_zxy_from_matrix(R: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Decompose ``R = Ry(y) @ Rx(x) @ Rz(z)``.

    VRChat applies Euler angles in Z, X, Y order, which for column vectors means
    ``R = Ry * Rx * Rz``.  Returned as ``(x, y, z)`` in degrees by default.
    """
    R = np.asarray(R, dtype=np.float64)
    sx = float(np.clip(-R[1, 2], -1.0, 1.0))
    x = math.asin(sx)
    cx = math.cos(x)
    if abs(cx) > 1e-6:
        y = math.atan2(R[0, 2], R[2, 2])
        z = math.atan2(R[1, 0], R[1, 1])
    else:  # gimbal lock: fold the free angle into yaw
        z = 0.0
        if sx > 0.0:  # x = +90
            y = math.atan2(R[0, 1], R[0, 0])
        else:  # x = -90
            y = math.atan2(-R[0, 1], R[0, 0])
    out = np.array([x, y, z])
    return np.degrees(out) if degrees else out


def matrix_from_euler_zxy(euler: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Inverse of :func:`euler_zxy_from_matrix` (used by tests and trace replay)."""
    x, y, z = np.asarray(euler, dtype=np.float64)
    if degrees:
        x, y, z = np.radians([x, y, z])
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def is_rotation(R: np.ndarray, atol: float = 1e-5) -> bool:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    return bool(np.allclose(R.T @ R, np.eye(3), atol=atol) and abs(np.linalg.det(R) - 1.0) < 1e-4)
