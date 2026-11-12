from __future__ import annotations

import math
import unittest

from vdvr.math3d import (
    forward_from_yaw,
    q_from_euler_deg,
    q_from_yaw,
    qmul,
    qrotate,
    right_from_yaw,
    wrap180,
)


class MathTests(unittest.TestCase):
    def test_identity(self) -> None:
        q = q_from_euler_deg(0.0, 0.0, 0.0)
        self.assertAlmostEqual(q[0], 1.0)
        rotated = qrotate(q, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(rotated[2], 1.0, places=6)

    def test_yaw_90_maps_forward_to_right(self) -> None:
        q = q_from_euler_deg(0.0, 90.0, 0.0)
        rotated = qrotate(q, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(rotated[0], 1.0, places=6)
        self.assertAlmostEqual(rotated[2], 0.0, places=6)

    def test_pitch_90_maps_forward_to_down(self) -> None:
        # Unity: positive X rotation pitches forward downward.
        q = q_from_euler_deg(90.0, 0.0, 0.0)
        rotated = qrotate(q, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(rotated[1], -1.0, places=6)

    def test_yaw_helpers_match_quaternion(self) -> None:
        for yaw in (-135.0, -30.0, 0.0, 45.0, 179.0):
            expected = qrotate(q_from_yaw(yaw), (0.0, 0.0, 1.0))
            actual = forward_from_yaw(yaw)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b, places=6)
            right = qrotate(q_from_yaw(yaw), (1.0, 0.0, 0.0))
            for a, b in zip(right_from_yaw(yaw), right):
                self.assertAlmostEqual(a, b, places=6)

    def test_qmul_identity(self) -> None:
        q = q_from_euler_deg(10.0, 20.0, 30.0)
        self.assertEqual(qmul((1.0, 0.0, 0.0, 0.0), q), q)

    def test_wrap(self) -> None:
        self.assertAlmostEqual(wrap180(190.0), -170.0)
        self.assertAlmostEqual(wrap180(-190.0), 170.0)
        self.assertAlmostEqual(wrap180(0.0), 0.0)
        self.assertTrue(math.isfinite(wrap180(1e9)))


if __name__ == "__main__":
    unittest.main()
