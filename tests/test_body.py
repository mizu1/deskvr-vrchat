from __future__ import annotations

import math
import unittest

from vdvr.body import (
    PROFILES,
    SLOT_CHEST,
    SLOT_HIP,
    SLOT_LEFT_FOOT,
    SLOT_LEFT_KNEE,
    SLOT_RIGHT_FOOT,
    BodyInput,
    BodyModel,
    distance,
)
from vdvr.math3d import v


class BodyModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = BodyModel(height=1.70, profile="full")
        self.state = BodyInput(
            head_pos=v(0.0, 1.57, 0.0),
            head_pitch=0.0,
            head_yaw=0.0,
            left_wrist=v(-0.2, 1.2, 0.3),
            right_wrist=v(0.2, 1.2, 0.3),
        )

    def test_idle_pose_is_sane(self) -> None:
        poses = self.model.update(1 / 60, self.state)
        self.assertEqual(set(poses), set(range(1, 9)))

        hip_pos, _ = poses[SLOT_HIP]
        self.assertLess(hip_pos[1], 1.57)
        self.assertAlmostEqual(hip_pos[1], 1.57 - 0.42 * 1.70, places=3)

        left_foot, _ = poses[SLOT_LEFT_FOOT]
        right_foot, _ = poses[SLOT_RIGHT_FOOT]
        self.assertAlmostEqual(left_foot[1], 0.038 * (1.70 / 1.70), places=3)
        self.assertLess(left_foot[0], 0.0)
        self.assertGreater(right_foot[0], 0.0)
        self.assertAlmostEqual(distance(left_foot, right_foot), 0.176, places=2)

    def test_walk_cycle_alternates_feet(self) -> None:
        walking = BodyInput(
            head_pos=v(0.0, 1.57, 0.0),
            left_wrist=v(-0.2, 1.2, 0.3),
            right_wrist=v(0.2, 1.2, 0.3),
            move_forward=1.0,
        )
        z_samples = []
        lift_samples = []
        for _ in range(120):
            poses = self.model.update(1 / 60, walking)
            z_samples.append(poses[SLOT_LEFT_FOOT][0][2] - poses[SLOT_RIGHT_FOOT][0][2])
            lift_samples.append(max(poses[SLOT_LEFT_FOOT][0][1], poses[SLOT_RIGHT_FOOT][0][1]))

        self.assertGreater(max(z_samples) - min(z_samples), 0.1)
        self.assertGreater(max(lift_samples), 0.06)

    def test_profile_filters_slots(self) -> None:
        for profile, expected in PROFILES.items():
            model = BodyModel(height=1.75, profile=profile)
            poses = model.update(1 / 60, self.state)
            self.assertEqual(tuple(sorted(poses)), tuple(sorted(expected)))
        self.assertIn(SLOT_CHEST, PROFILES["standard"])
        self.assertIn(SLOT_LEFT_KNEE, PROFILES["full"])

    def test_crouch_bends_knees_forward(self) -> None:
        standing = BodyModel(height=1.70, profile="full")
        crouched = BodyModel(height=1.70, profile="full")
        state_stand = BodyInput(head_pos=v(0.0, 1.57, 0.0), left_wrist=v(-0.2, 1.2, 0.3), right_wrist=v(0.2, 1.2, 0.3))
        state_crouch = BodyInput(
            head_pos=v(0.0, 1.57 - 0.44, 0.0),
            left_wrist=v(-0.2, 1.2 - 0.44, 0.3),
            right_wrist=v(0.2, 1.2 - 0.44, 0.3),
            crouch=1.0,
        )
        knee_stand = standing.update(1 / 60, state_stand)[SLOT_LEFT_KNEE][0]
        knee_crouch = crouched.update(1 / 60, state_crouch)[SLOT_LEFT_KNEE][0]
        self.assertLess(knee_crouch[1], knee_stand[1])
        self.assertGreater(knee_crouch[2], knee_stand[2])

    def test_no_nan_under_random_input(self) -> None:
        import random

        random.seed(0)
        model = BodyModel(height=1.80, profile="full")
        for _ in range(200):
            state = BodyInput(
                head_pos=v(0.0, random.uniform(0.8, 1.8), 0.0),
                head_pitch=random.uniform(-80, 80),
                head_yaw=random.uniform(-180, 180),
                left_wrist=v(random.uniform(-1, 0), random.uniform(0.5, 1.8), random.uniform(-1, 1)),
                right_wrist=v(random.uniform(0, 1), random.uniform(0.5, 1.8), random.uniform(-1, 1)),
                move_forward=random.uniform(-1, 1),
                move_strafe=random.uniform(-1, 1),
                crouch=random.uniform(0, 1),
                jumping=random.random() > 0.8,
            )
            poses = model.update(random.uniform(0.0, 0.05), state)
            for position, euler in poses.values():
                for value in (*position, *euler):
                    self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
