"""Small vector/quaternion helpers matching Unity's conventions.

Coordinate system (Unity): left handed, +Y up, +Z forward, +X right.
Angles are degrees. ``euler_deg(x, y, z)`` is interpreted as Unity's
``Quaternion.Euler(x, y, z)`` which applies rotations in Z, X, Y order, i.e.
``q = Ry * Rx * Rz``.
"""

from __future__ import annotations

import math
from typing import Iterable

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z


def v(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    return (float(x), float(y), float(z))


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def vlerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def vlen(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vnorm(a: Vec3) -> Vec3:
    length = vlen(a)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def wrap180(angle: float) -> float:
    angle = (angle + 180.0) % 360.0 - 180.0
    return angle


def _axis_quat(axis: str, degrees: float) -> Quat:
    half = math.radians(degrees) * 0.5
    s = math.sin(half)
    c = math.cos(half)
    if axis == "x":
        return (c, s, 0.0, 0.0)
    if axis == "y":
        return (c, 0.0, s, 0.0)
    if axis == "z":
        return (c, 0.0, 0.0, s)
    raise ValueError(axis)


def qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qnorm(q: Quat) -> Quat:
    length = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    if length < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / length, q[1] / length, q[2] / length, q[3] / length)


def q_from_euler_deg(pitch: float, yaw: float, roll: float) -> Quat:
    """Unity ``Quaternion.Euler(pitch, yaw, roll)`` -> (w, x, y, z)."""
    return qnorm(qmul(qmul(_axis_quat("y", yaw), _axis_quat("x", pitch)), _axis_quat("z", roll)))


def q_from_yaw(yaw: float) -> Quat:
    return _axis_quat("y", yaw)


def qrotate(q: Quat, vec: Vec3) -> Vec3:
    """Rotate ``vec`` by quaternion ``q``."""
    qw, qx, qy, qz = q
    x, y, z = vec
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def q_slerp(a: Quat, b: Quat, t: float) -> Quat:
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if dot < 0.0:
        b = tuple(-c for c in b)  # type: ignore[assignment]
        dot = -dot
    if dot > 0.9995:
        return qnorm(
            (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                a[3] + (b[3] - a[3]) * t,
            )
        )
    theta0 = math.acos(clamp(dot, -1.0, 1.0))
    theta = theta0 * t
    sin_theta = math.sin(theta)
    sin_theta0 = math.sin(theta0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return qnorm(
        (
            a[0] * s0 + b[0] * s1,
            a[1] * s0 + b[1] * s1,
            a[2] * s0 + b[2] * s1,
            a[3] * s0 + b[3] * s1,
        )
    )


def forward_from_yaw(yaw: float) -> Vec3:
    """Unity forward (+Z) rotated by yaw."""
    rad = math.radians(yaw)
    return (math.sin(rad), 0.0, math.cos(rad))


def right_from_yaw(yaw: float) -> Vec3:
    """Unity right (+X) rotated by yaw."""
    rad = math.radians(yaw)
    return (math.cos(rad), 0.0, -math.sin(rad))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
