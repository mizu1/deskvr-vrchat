import math
import unittest

import numpy as np

from vdvr.ardy_core.mathutil import (
    euler_zxy_from_matrix,
    is_rotation,
    matrix_from_euler_zxy,
    matrix_from_quat,
    mirror_x_rotation,
    quat_from_matrix,
    quat_slerp,
)


class TestEulerZXY(unittest.TestCase):
    def test_roundtrip_matrix(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            euler = np.array([
                rng.uniform(-80.0, 80.0),
                rng.uniform(-179.0, 179.0),
                rng.uniform(-179.0, 179.0),
            ])
            R = matrix_from_euler_zxy(euler)
            recovered = euler_zxy_from_matrix(R)
            R2 = matrix_from_euler_zxy(recovered)
            self.assertTrue(np.allclose(R, R2, atol=1e-9), f"failed for {euler} -> {recovered}")
            self.assertTrue(np.allclose(matrix_from_euler_zxy(euler), matrix_from_euler_zxy(euler), atol=0))

    def test_gimbal_lock_roundtrip(self):
        for x in (90.0, -90.0):
            euler = np.array([x, 37.0, 12.0])
            R = matrix_from_euler_zxy(euler)
            recovered = euler_zxy_from_matrix(R)
            self.assertAlmostEqual(recovered[0], x, places=6)
            self.assertAlmostEqual(recovered[2], 0.0, places=6)
            self.assertTrue(np.allclose(matrix_from_euler_zxy(recovered), R, atol=1e-9))

    def test_identity(self):
        np.testing.assert_allclose(euler_zxy_from_matrix(np.eye(3)), np.zeros(3), atol=1e-12)


class TestHandedness(unittest.TestCase):
    def test_mirror_keeps_proper_rotation(self):
        R = matrix_from_euler_zxy([10.0, 30.0, -20.0])
        mirrored = mirror_x_rotation(R)
        self.assertTrue(is_rotation(mirrored))

    def test_mirror_flips_yaw(self):
        R = matrix_from_euler_zxy([0.0, 30.0, 0.0])
        yaw = euler_zxy_from_matrix(mirror_x_rotation(R))
        self.assertAlmostEqual(yaw[1], -30.0, places=9)
        self.assertAlmostEqual(yaw[0], 0.0, places=9)
        self.assertAlmostEqual(yaw[2], 0.0, places=9)

    def test_mirror_flips_roll(self):
        R = matrix_from_euler_zxy([0.0, 0.0, 25.0])
        euler = euler_zxy_from_matrix(mirror_x_rotation(R))
        self.assertAlmostEqual(euler[2], -25.0, places=6)


class TestQuaternions(unittest.TestCase):
    def test_matrix_roundtrip(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            euler = np.array([rng.uniform(-80, 80), rng.uniform(-179, 179), rng.uniform(-179, 179)])
            R = matrix_from_euler_zxy(euler)
            R2 = matrix_from_quat(quat_from_matrix(R))
            self.assertTrue(np.allclose(R, R2, atol=1e-9))

    def test_slerp_endpoints(self):
        a = quat_from_matrix(matrix_from_euler_zxy([10, 20, 30]))
        b = quat_from_matrix(matrix_from_euler_zxy([-40, 80, 15]))
        np.testing.assert_allclose(quat_slerp(a, b, 0.0), a, atol=1e-12)
        np.testing.assert_allclose(np.abs(quat_slerp(a, b, 1.0)), np.abs(b), atol=1e-12)
        mid = quat_slerp(a, b, 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(mid)), 1.0, places=9)

    def test_slerp_shortest_arc(self):
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = -a
        mid = quat_slerp(a, b, 0.5)
        self.assertAlmostEqual(abs(float(np.dot(a, mid))), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
