import os
import unittest
from pathlib import Path

import numpy as np

from vdvr.ardy_core.mathutil import (
    is_rotation,
    matrix_from_euler_zxy,
    matrix_from_quat,
    mirror_x_rotation,
    quat_from_matrix,
)
from vdvr.ardy_core.retarget import (
    ALL_SLOTS,
    HEAD_SLOT,
    RetargetOptions,
    Retargeter,
    build_rig,
    two_bone_ik,
)
from vdvr.ardy_core.skeleton import JOINT, load_motion
from tests.test_skeleton import make_fk_motion


def real_motion_path() -> Path | None:
    candidate = Path(
        os.environ.get(
            "ARDY_MOTION",
            r"E:\360Downloads\SeedData\repos\ardy-0.5b-infer\outputs\pipeline\motion.npz",
        )
    )
    return candidate if candidate.exists() else None


class TestTwoBoneIK(unittest.TestCase):
    def test_reaches_target(self):
        root = np.array([0.0, 1.0, 0.0])
        target = np.array([0.1, 0.5, 0.1])
        knee, end = two_bone_ik(root, target, 0.4, 0.4, np.array([0.0, 0.7, 0.1]))
        self.assertAlmostEqual(float(np.linalg.norm(knee - root)), 0.4, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(end - knee)), 0.4, places=6)
        self.assertLess(float(np.linalg.norm(end - target)), 1e-6)

    def test_clamps_unreachable_target(self):
        root = np.array([0.0, 1.0, 0.0])
        target = np.array([5.0, 1.0, 0.0])
        knee, end = two_bone_ik(root, target, 0.4, 0.3, np.array([0.0, 0.5, 0.0]))
        self.assertAlmostEqual(float(np.linalg.norm(end - root)), 0.7, places=3)
        self.assertAlmostEqual(float(np.linalg.norm(knee - root)), 0.4, places=6)

    def test_collinear_pole_uses_fallback(self):
        root = np.array([0.0, 1.0, 0.0])
        target = np.array([0.0, 0.2, 0.0])
        knee, _ = two_bone_ik(root, target, 0.4, 0.4, np.array([0.0, 0.6, 0.0]))
        self.assertTrue(np.isfinite(knee).all())
        # Knee must bend forward (+Z fallback pole).
        self.assertGreater(knee[2], 0.0)


class TestRig(unittest.TestCase):
    def test_height_scale(self):
        motion, _ = make_fk_motion()
        rig = build_rig(motion, height=1.8)
        self.assertGreater(rig.scale, 0.0)
        self.assertGreater(rig.hip_width, 0.05)
        self.assertGreater(rig.shoulder_width, 0.05)

    def test_profile_override(self):
        motion, _ = make_fk_motion()
        rig = build_rig(motion, height=1.7, profile={"bone_lengths": {"LeftShin": 0.5}})
        self.assertAlmostEqual(rig.bone("LeftShin"), 0.5, places=9)

    def test_profile_height_sets_scale(self):
        motion, _ = make_fk_motion()
        source_height = float(motion.joints[0, JOINT["HeadEnd"], 1])
        rig = build_rig(motion, height=1.7, profile={"height": 1.9})
        self.assertAlmostEqual(rig.height, 1.9, places=9)
        self.assertAlmostEqual(rig.scale, 1.9 / source_height, places=9)


class TestRetargeter(unittest.TestCase):
    def setUp(self):
        self.motion, _ = make_fk_motion(frames=8)

    def test_shapes_and_finiteness(self):
        rt = Retargeter(self.motion, RetargetOptions(height=1.7, ground_lock=False))
        seq = rt.sequence
        self.assertEqual(seq.frames, 8)
        self.assertEqual(seq.slots, ALL_SLOTS + (HEAD_SLOT,))
        self.assertEqual(seq.positions.shape, (8, 9, 3))
        self.assertTrue(np.isfinite(seq.positions).all())
        self.assertTrue(np.isfinite(seq.quats).all())

    def test_tracker_rotations_are_proper(self):
        rt = Retargeter(self.motion)
        for slot in rt.sequence.slots:
            index = rt.sequence.slots.index(slot)
            for frame in range(rt.sequence.frames):
                q = rt.sequence.quats[frame, index]
                self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=9)

    def test_ground_lock_plants_feet_with_real_leg_lengths(self):
        path = real_motion_path()
        if path is None:
            self.skipTest("ARDY sample motion.npz not available")
        assert path is not None
        motion = load_motion(path)
        rt = Retargeter(motion, RetargetOptions(height=1.72, ground_lock=True))
        lfoot = rt.sequence.slots.index("lfoot")
        rfoot = rt.sequence.slots.index("rfoot")
        for frame in np.where(motion.contacts("Left"))[0][::15]:
            self.assertAlmostEqual(float(rt.sequence.positions[frame, lfoot, 1]), rt.rig.ankle_height, places=5)
        for frame in np.where(motion.contacts("Right"))[0][::15]:
            self.assertAlmostEqual(float(rt.sequence.positions[frame, rfoot, 1]), rt.rig.ankle_height, places=5)

    def test_in_place_removes_root_travel(self):
        rt = Retargeter(self.motion, RetargetOptions(in_place=True, ground_lock=False))
        hip = rt.sequence.slots.index("hip")
        xs = rt.sequence.positions[:, hip, 0]
        zs = rt.sequence.positions[:, hip, 2]
        # Synthetic root travels 0.4m in X / 0.3m in Z; smoothing keeps a small
        # residual at the very first frame but the total drift must collapse.
        self.assertLess(max(xs) - min(xs), 0.25)
        self.assertLess(max(zs) - min(zs), 0.25)

    def test_world_motion_keeps_root_travel(self):
        rt = Retargeter(self.motion, RetargetOptions(in_place=False, ground_lock=False))
        hip = rt.sequence.slots.index("hip")
        xs = rt.sequence.positions[:, hip, 0]
        self.assertGreater(max(xs) - min(xs), 0.2)

    def test_rest_pose_is_grounded_and_upright(self):
        rt = Retargeter(self.motion, RetargetOptions(height=1.7, ground_lock=False))
        rest = rt.rest
        self.assertGreater(rest.positions["hip"][1], rest.positions["lknee"][1])
        self.assertGreater(rest.positions["lknee"][1], rest.positions["lfoot"][1])
        self.assertGreater(rest.positions["chest"][1], rest.positions["hip"][1])
        self.assertGreater(rest.positions[HEAD_SLOT][1], rest.positions["chest"][1])
        self.assertAlmostEqual(rest.positions["lfoot"][1], rt.rig.ankle_height, places=6)
        self.assertAlmostEqual(rest.positions["rfoot"][1], rt.rig.ankle_height, places=6)
        for q in rest.quats.values():
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=9)

    def test_head_can_be_disabled(self):
        rt = Retargeter(self.motion, RetargetOptions(send_head=False))
        self.assertNotIn(HEAD_SLOT, rt.sequence.slots)
        self.assertNotIn(HEAD_SLOT, rt.rest.positions)

    def test_tpose_has_horizontal_arms_and_grounded_feet(self):
        rt = Retargeter(self.motion, RetargetOptions(ground_lock=False))
        tp = rt.tpose_frame()
        left = tp.positions["lelbow"]
        right = tp.positions["relbow"]
        self.assertLess(left[0], -0.1)
        self.assertGreater(right[0], 0.1)
        self.assertAlmostEqual(float(left[1]), float(right[1]), places=6)
        self.assertAlmostEqual(float(tp.positions["lfoot"][1]), rt.rig.ankle_height, places=6)
        self.assertAlmostEqual(float(tp.positions["rfoot"][1]), rt.rig.ankle_height, places=6)
        # T-pose arms must be wider than the relaxed A-pose rest.
        self.assertGreater(abs(float(left[0])), abs(float(rt.rest.positions["lelbow"][0])))
        for q in tp.quats.values():
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=9)

    def test_rest_pose_body_rotations_are_identity(self):
        rt = Retargeter(self.motion)
        for slot in ("hip", "chest", "lfoot", "rfoot", HEAD_SLOT):
            if slot in rt.rest.quats:
                self.assertTrue(np.allclose(np.abs(rt.rest.quats[slot]), (1.0, 0.0, 0.0, 0.0), atol=1e-9), slot)

    def test_lite_preset_slots(self):
        rt = Retargeter(self.motion, RetargetOptions(slots=("hip", "lfoot", "rfoot")))
        self.assertEqual(rt.sequence.slots, ("hip", "lfoot", "rfoot", HEAD_SLOT))

    def test_segment_trackers_use_parent_bone_rotation(self):
        # Knee trackers sit on the thigh, elbow trackers on the upper arm, so
        # their orientation must follow LeftLeg/RightLeg and LeftArm/RightArm.
        rt = Retargeter(self.motion, RetargetOptions(ground_lock=False))
        frame = 0
        for slot, bone in (
            ("lknee", "LeftLeg"),
            ("rknee", "RightLeg"),
            ("lelbow", "LeftArm"),
            ("relbow", "RightArm"),
        ):
            index = rt.sequence.slots.index(slot)
            got = rt.sequence.quats[frame, index]
            expected = quat_from_matrix(mirror_x_rotation(self.motion.rotations[frame, JOINT[bone]]))
            dot = abs(float(np.dot(got, expected)))
            self.assertGreater(dot, 1.0 - 1e-6, f"{slot} does not follow {bone}")


class TestRealArdyClip(unittest.TestCase):
    def setUp(self):
        path = real_motion_path()
        if path is None:
            self.skipTest("ARDY sample motion.npz not available")
        self.motion = load_motion(path)

    def test_retarget_real_clip(self):
        rt = Retargeter(self.motion, RetargetOptions(height=1.72))
        seq = rt.sequence
        self.assertEqual(seq.frames, self.motion.frames)
        self.assertTrue(np.isfinite(seq.positions).all())

        hip = seq.slots.index("hip")
        head = seq.slots.index(HEAD_SLOT)
        lfoot = seq.slots.index("lfoot")
        # Plausible human height range after scaling.
        self.assertTrue(0.85 < float(seq.positions[:, hip, 1].mean()) < 1.15)
        self.assertTrue(1.35 < float(seq.positions[:, head, 1].max()) < 2.0)
        self.assertTrue(np.all(seq.positions[:, lfoot, 1] >= -0.02))

    def test_real_fk_error_is_tiny(self):
        from vdvr.ardy_core.skeleton import fk_error

        self.assertLess(fk_error(self.motion), 1e-4)

    def test_uniform_rig_reproduces_scaled_source(self):
        rt = Retargeter(self.motion, RetargetOptions(height=1.72, ground_lock=False))
        k = rt.rig.scale
        for frame in (0, self.motion.frames // 2, self.motion.frames - 1):
            source = (self.motion.joints[frame] - rt.root_delta[frame]) * np.array([-k, k, k])
            deviation = np.linalg.norm(source[1:] - rt.joint_positions[frame][1:], axis=1).max()
            self.assertLess(float(deviation), 5e-3)

    def test_hip_and_chest_rotations_follow_source(self):
        rt = Retargeter(self.motion, RetargetOptions(height=1.72))
        frame = 0
        for slot, bone in (("hip", "Hips"), ("chest", "Chest")):
            index = rt.sequence.slots.index(slot)
            got = matrix_from_quat(rt.sequence.quats[frame, index])
            expected = mirror_x_rotation(self.motion.rotations[frame, JOINT[bone]])
            self.assertTrue(np.allclose(got, expected, atol=1e-6), slot)


if __name__ == "__main__":
    unittest.main()
