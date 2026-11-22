import unittest

import numpy as np

from vdvr.ardy_core.mathutil import matrix_from_euler_zxy
from vdvr.ardy_core.skeleton import (
    JOINT,
    SOMA77_PARENTS,
    SomaMotion,
    calibrate_rest_offsets,
    fk_error,
    forward_kinematics,
)


def make_fk_motion(frames: int = 24, fps: float = 30.0, seed: int = 11) -> tuple[SomaMotion, np.ndarray]:
    """Build a synthetic but kinematically consistent clip for FK tests."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 0.06, size=(77, 3))
    offsets[0] = 0.0
    # Make the chain point mostly up so the skeleton is not degenerate.
    for joint in range(1, 77):
        offsets[joint, 1] = abs(offsets[joint, 1]) * 0.5 + 0.02

    rotations = np.empty((frames, 77, 3, 3))
    for frame in range(frames):
        phase = frame / max(frames - 1, 1)
        for joint in range(77):
            euler = np.array([
                5.0 * np.sin(phase * 2.0 * np.pi + joint * 0.01),
                10.0 * np.sin(phase * 2.0 * np.pi + joint * 0.02),
                0.0,
            ])
            rotations[frame, joint] = matrix_from_euler_zxy(euler)

    root = np.stack([
        np.linspace(0.0, 0.4, frames),
        np.full(frames, 1.0),
        np.linspace(0.0, -0.3, frames),
    ], axis=1)

    joints = forward_kinematics(root, rotations, offsets)
    motion = SomaMotion(
        joints=joints,
        rotations=rotations,
        root=root,
        smooth_root=root.copy(),
        foot_contacts=np.ones((frames, 6), dtype=bool),
        fps=fps,
        text="synthetic",
    )
    return motion, offsets


class TestForwardKinematics(unittest.TestCase):
    def test_recover_rest_offsets(self):
        motion, offsets = make_fk_motion()
        recovered = calibrate_rest_offsets(motion)
        np.testing.assert_allclose(recovered[1:], offsets[1:], atol=1e-9)

    def test_reconstruction_error_is_zero(self):
        motion, _ = make_fk_motion()
        self.assertLess(fk_error(motion), 1e-9)

    def test_root_matches_hips(self):
        motion, _ = make_fk_motion()
        np.testing.assert_allclose(motion.joints[:, 0], motion.root, atol=1e-12)

    def test_parent_index_ordering(self):
        # FK walks the array in order, so parents must precede their children.
        for child, parent in enumerate(SOMA77_PARENTS):
            if parent >= 0:
                self.assertLess(parent, child, f"{child} has parent {parent}")

    def test_degenerate_direction_does_not_create_nan(self):
        motion, offsets = make_fk_motion(frames=2)
        motion.joints[:, JOINT["LeftHand"]] = motion.joints[:, JOINT["LeftForeArm"]]
        rebuilt = forward_kinematics(motion.root, motion.rotations, offsets)
        self.assertTrue(np.isfinite(rebuilt).all())


if __name__ == "__main__":
    unittest.main()
