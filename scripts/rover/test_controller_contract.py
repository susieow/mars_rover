"""Regression checks for dataset acceptance and closed-loop control."""
import unittest
from unittest.mock import Mock
import math
import mujoco
import numpy as np
from climb import ROBOT_XML
from closed_loop_controller import ClosedLoopController

class ControllerContract(unittest.TestCase):
    def setUp(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.d = mujoco.MjData(self.m)
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(self.m.njnt)]
        kw = {}
        for key, suffix in [("wheels", "_wheel_joint"), ("steer", "_steer_joint"), ("rockers", "_rocker_joint")]:
            ids = [j for j, n in enumerate(names) if n.endswith(suffix)]
            kw[key] = ids
            actkey = {"wheels": "wheel_act", "steer": "steer_act", "rockers": "rocker_act"}[key]
            kw[actkey] = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, names[j] + "_act") for j in ids]
        self.c = ClosedLoopController(self.m, self.d, **kw, settle_time=1.0)
        self.ground = {"all_ground": True, "ground_wheels": list(range(6)), "platform_top": False,
                       "down": False, "vertical": {"front":set(),"middle":set(),"rear":set()},
                       "top": {"front":set(),"middle":set(),"rear":set()}, "events":[]}

    def prepare_stop(self):
        self.c.platform_seen = self.c.down_seen = True
        self.c.finishing = self.c.braking = True
        self.c._contacts = lambda: self.ground
        self.c._command_wheels = Mock()
        self.c._command_rockers = Mock()

    def test_settle_requires_continuous_physical_second(self):
        self.prepare_stop()
        self.c.update(0)
        self.d.qvel[0] = 0.1
        self.c.update(0.5)
        self.d.qvel[0] = 0
        self.c.update(0.6)
        self.c.update(1.5)
        self.assertFalse(self.c.settle_ok)
        self.c.update(1.62)
        self.assertTrue(self.c.settle_ok)

    def test_contact_loss_resets_stability(self):
        self.prepare_stop()
        self.c.update(0)
        self.ground["all_ground"] = False
        self.c.update(0.9)
        self.ground["all_ground"] = True
        self.c.update(1.1)
        self.assertFalse(self.c.settle_ok)
        self.assertEqual(self.c.settle_since, 1.1)

    def test_joint_posture_must_return_to_neutral(self):
        self.prepare_stop()
        j = self.c.rockers[0]
        self.d.qpos[self.m.jnt_qposadr[j]] = math.radians(15)
        self.c.update(0)
        self.c.update(2)
        self.assertFalse(self.c.settle_ok)

    def test_yaw_rate_is_damped(self):
        self.c._command_wheels(1.0, 0.0, 0.2)
        left = [self.d.ctrl[self.c.wheel_act[k]] for k in self.c.left]
        right = [self.d.ctrl[self.c.wheel_act[k]] for k in range(6) if k not in self.c.left]
        self.assertLess(np.mean(right), np.mean(left))

    def test_braking_does_not_turn_to_correct_lateral_position(self):
        self.c.braking = True
        self.d.qpos[1] = 0.2
        self.c._command_wheels(1.0, 0.1, 0.2)
        np.testing.assert_array_equal(self.d.ctrl[self.c.wheel_act], np.zeros(6))

    def test_middle_lift_has_a_landing_transition(self):
        self.c.contact_counts[("middle", "vertical", "up0")] = 3
        self.c._update_target(0, self.ground)
        self.assertEqual(self.c.phase, "MIDDLE_LIFT")
        self.c.contact_counts.clear()
        self.c._update_target(0.2, self.ground)
        self.c._update_target(0.32, self.ground)
        self.assertEqual(self.c.phase, "MIDDLE_LAND")
        self.c._update_target(0.34, self.ground)
        self.assertEqual(self.c.target_deg, 0)

    def test_settle_phase_is_not_overwritten_by_descent(self):
        self.prepare_stop()
        self.c.update(0)
        self.c.update(0.5)
        self.assertEqual(self.c.phase, "SETTLE")

    def assert_latest_riser_selected(self, names, expected):
        for role, phase in [("front", "FRONT_LIFT"), ("middle", "MIDDLE_LIFT")]:
            with self.subTest(role=role, contacts=names):
                self.c.phase = "APPROACH"
                self.c.contact_counts.clear()
                for name in names:
                    self.c.contact_counts[(role, "vertical", name)] = 3
                self.c._update_target(0.0, self.ground)
                self.assertEqual(self.c.phase, phase)
                self.assertEqual(self.c.active_riser, expected)

    def test_up10_follows_up9_for_simultaneous_contacts(self):
        self.assert_latest_riser_selected(["up9", "up10"], "up10")

    def test_up14_follows_earlier_risers_for_simultaneous_contacts(self):
        self.assert_latest_riser_selected(["up9", "up10", "up14"], "up14")

    def test_platform_follows_numbered_risers_for_simultaneous_contacts(self):
        self.assert_latest_riser_selected(["up9", "up10", "up14", "platform"], "platform")

    def test_next_riser_does_not_block_landing_from_previous_riser(self):
        self.c.phase = "MIDDLE_LIFT"
        self.c.active_riser = "up0"
        self.c.contact_counts[("middle", "vertical", "up1")] = 3
        self.c._update_target(0.2, self.ground)
        self.c._update_target(0.32, self.ground)
        self.assertEqual(self.c.phase, "MIDDLE_LAND")

    def test_lost_parking_support_triggers_bounded_reseating(self):
        self.prepare_stop()
        self.ground["all_ground"] = False
        self.c.update(0.0)
        self.c.update(0.26)
        self.assertFalse(self.c.braking)
        self.assertFalse(self.c.settle_ok)
        self.assertEqual(self.c.settle_retries, 1)
        self.ground["all_ground"] = True
        self.c.update(0.50)
        self.assertFalse(self.c.braking)
        self.c.update(1.28)
        self.assertTrue(self.c.braking)
        self.assertFalse(self.c.settle_ok)


    def test_descent_brakes_overspeed_wheels_more_strongly(self):
        for joint in self.c.wheels:
            self.d.qvel[self.m.jnt_dofadr[joint]] = 2.0
        self.c._command_wheels(1.0, 0.0, 0.0)
        cruise = self.d.ctrl[self.c.wheel_act].copy()
        self.c.phase = "DESCENT"
        self.c._command_wheels(1.0, 0.0, 0.0)
        downhill = self.d.ctrl[self.c.wheel_act].copy()
        self.assertTrue(np.all(downhill < cruise))
        self.assertTrue(np.all(np.abs(downhill) <= 50.0))

    def test_displaced_steering_blocks_drive_but_preserves_braking(self):
        joint = self.c.steer[0]
        wheel_name = self.c.joint_names[joint].replace("_steer_joint", "_wheel_joint")
        wheel = self.c.joint_names.index(wheel_name)
        actuator = self.c.wheel_act[self.c.wheels.index(wheel)]
        self.d.qpos[self.m.jnt_qposadr[joint]] = math.radians(30.0)
        self.c._command_wheels(1.0, 0.0, 0.0)
        self.assertEqual(self.d.ctrl[actuator], 0.0)
        self.d.qvel[self.m.jnt_dofadr[wheel]] = 2.0
        self.c._command_wheels(1.0, 0.0, 0.0)
        self.assertLess(self.d.ctrl[actuator], 0.0)

if __name__ == "__main__":
    unittest.main()
