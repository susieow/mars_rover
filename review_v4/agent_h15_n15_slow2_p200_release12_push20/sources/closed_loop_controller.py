"""Contact-driven traction controller for the six-wheel Mars rover."""

from __future__ import annotations

import math
from collections import defaultdict

import mujoco
import numpy as np


class ClosedLoopController:
    def __init__(self, model, data, *, wheels, wheel_act, steer, steer_act,
                 rockers, rocker_act, wheel_radius=0.163, speed=6.0,
                 tau_max=50.0, yaw0=0.0, front_lift_deg=0.0,
                 middle_lift_deg=0.0, rocker_gain=600.0, rocker_damp=150.0,
                 rocker_slew_deg_s=30.0, dt=0.001, settle_time=1.2, wheel_gain=10.0,
                 descent_speed=1.0, descent_wheel_gain=40.0, steer_damp=50.0):
        self.m = model
        self.d = data
        self.wheels = wheels
        self.wheel_act = wheel_act
        self.steer = steer
        self.steer_act = steer_act
        self.rockers = rockers
        self.rocker_act = rocker_act
        self.radius = float(wheel_radius)
        self.base_speed = float(speed)
        self.wheel_gain = float(wheel_gain)
        self.descent_wheel_gain = float(descent_wheel_gain)
        self.steer_damp = float(steer_damp)
        self.descent_speed = float(descent_speed)
        self.phase_since = 0.0
        self.clear_since = None
        self.active_riser = None
        self.speed_command = 0.0
        self.tau_max = min(float(tau_max), 50.0)
        self.yaw0 = float(yaw0)
        self.front_lift_deg = float(front_lift_deg)
        self.middle_lift_deg = float(middle_lift_deg)
        self.rocker_gain = float(rocker_gain)
        self.rocker_damp = float(rocker_damp)
        self.rocker_step = math.radians(float(rocker_slew_deg_s)) * float(dt)
        self.dt = float(dt)
        self.joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)]
        self.geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
        self.front = [k for k, j in enumerate(wheels) if "front" in self.joint_names[j]]
        self.middle = [k for k, j in enumerate(wheels) if "middle" in self.joint_names[j]]
        self.rear = [k for k, j in enumerate(wheels) if "rear" in self.joint_names[j]]
        self.left = [k for k, j in enumerate(wheels) if "left" in self.joint_names[j]]
        self.target_deg = 0.0
        self.rocker_command = 0.0
        self.phase = "APPROACH"
        self.platform_seen = False
        self.down_seen = False
        self.settle_started = False
        self.settle_since = None
        self.settle_ok = False
        self.events_seen = set()
        self.contact_counts = defaultdict(int)
        self.force = np.zeros(6)
        self.event_log = []
        self.max_slip = 0.0
        self.max_abs_y = 0.0
        self.max_yaw_error = 0.0
        self.abort_reason = None
        self.settle_time = float(settle_time)
        self.last_contacts = {}
        self.yaw_previous = None
        self.yaw_rate = 0.0
        self.finishing = False
        self.braking = False
        self.brake_retry_after = 0.0
        self.support_lost_since = None
        self.settle_retries = 0
        self.final_pose_ok = False
        self.wheel_geom = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                             self.joint_names[j].replace("_joint", "") + "_col_1"): k
            for k, j in enumerate(wheels)
        }

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _rpy(self):
        w, x, y, z = [float(v) for v in self.d.qpos[3:7]]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1, 1)))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    def _contacts(self):
        roles = {"front": self.front, "middle": self.middle, "rear": self.rear}
        role_by_geom = {}
        for role, indices in roles.items():
            for k in indices:
                stem = self.joint_names[self.wheels[k]].replace("_joint", "")
                role_by_geom[stem + "_col_1"] = role
        forces = {r: {"ground": 0.0, "up": 0.0, "down": 0.0, "platform": 0.0} for r in roles}
        vertical = {r: set() for r in roles}
        top = {r: set() for r in roles}
        active = set()
        ground_wheels = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            n1 = self.geom_names[int(c.geom1)]
            n2 = self.geom_names[int(c.geom2)]
            if n1.startswith(("ground", "up", "down", "platform")) and "_wheel" in n2:
                terrain, wheel = n1, n2
            elif n2.startswith(("ground", "up", "down", "platform")) and "_wheel" in n1:
                terrain, wheel = n2, n1
            else:
                continue
            role = role_by_geom.get(wheel)
            if role is None:
                role = "front" if "front" in wheel else "middle" if "middle" in wheel else "rear"
            mujoco.mj_contactForce(self.m, self.d, i, self.force)
            fn = abs(float(self.force[0]))
            if fn < 5.0:
                continue
            kind = "ground" if terrain == "ground" else "platform" if terrain == "platform" else "up" if terrain.startswith("up") else "down"
            forces[role][kind] += fn
            normal = np.asarray(c.frame[:3], dtype=float)
            is_vertical = abs(float(normal[0])) > 0.70 and abs(float(normal[2])) < 0.70
            is_top = abs(float(normal[2])) > 0.70
            if kind == "ground" and abs(float(normal[2])) > 0.70:
                gid = int(c.geom1) if int(c.geom1) in self.wheel_geom else int(c.geom2)
                if gid in self.wheel_geom:
                    ground_wheels.add(self.wheel_geom[gid])
            if is_vertical:
                vertical[role].add(terrain)
                active.add((role, "vertical", terrain))
            elif is_top:
                top[role].add(terrain)
                active.add((role, "top", terrain))
        for key in list(self.contact_counts):
            if key not in active:
                self.contact_counts[key] = max(0, self.contact_counts[key] - 1)
        for key in active:
            self.contact_counts[key] = min(6, self.contact_counts[key] + 1)
        events = []
        for key, count in self.contact_counts.items():
            if count >= 3 and key not in self.events_seen:
                self.events_seen.add(key)
                events.append(key)
        return {
            "vertical": vertical,
            "top": top,
            "events": events,
            "all_ground": len(ground_wheels) == len(self.wheels),
            "ground_wheels": sorted(ground_wheels),
            "platform_top": any("platform" in top[r] for r in roles),
            "down": any(forces[r]["down"] > 10.0 for r in roles),
        }

    def _change_phase(self, phase, t, riser=None):
        if self.phase != phase:
            self.event_log.append({"t": round(t, 4), "phase": phase, "riser": riser})
            self.phase = phase
            self.phase_since = t
            self.clear_since = None
        if riser is not None:
            self.active_riser = riser

    def _update_target(self, t, c):
        if self.finishing:
            self.target_deg = 0.0
            return
        if c["platform_top"]:
            self.platform_seen = True
        if c["down"]:
            self.down_seen = True
        if self.platform_seen and self.down_seen:
            self._change_phase("DESCENT", t)
            self.target_deg = 0.0
            return

        stable = {key for key, count in self.contact_counts.items() if count >= 3}
        def terrain(role, kind):
            return {name for r, k, name in stable
                    if r == role and k == kind and (name.startswith("up") or name == "platform")}
        fv, ft = terrain("front", "vertical"), terrain("front", "top")
        mv, mt = terrain("middle", "vertical"), terrain("middle", "top")
        elapsed = t - self.phase_since
        if self.phase in {"APPROACH", "CRUISE", "STEP_SETTLE", "MIDDLE_APPROACH"}:
            self.target_deg = 0.0
            if mv:
                self._change_phase("MIDDLE_LIFT", t, sorted(mv)[-1])
            elif fv:
                self._change_phase("FRONT_LIFT", t, sorted(fv)[-1])
        elif self.phase == "FRONT_LIFT":
            self.target_deg = -self.front_lift_deg
            if elapsed >= 0.12 and self.active_riser in ft:
                self._change_phase("MIDDLE_APPROACH", t)
            elif self.active_riser not in fv and elapsed >= 0.12:
                if self.clear_since is None:
                    self.clear_since = t
                if t - self.clear_since >= 0.10:
                    self._change_phase("FRONT_LAND", t)
            else:
                self.clear_since = None
        elif self.phase == "FRONT_LAND":
            self.target_deg = 0.0
            if elapsed >= 0.12 and (ft or fv):
                self._change_phase("MIDDLE_APPROACH", t)
        elif self.phase == "MIDDLE_LIFT":
            self.target_deg = self.middle_lift_deg
            if elapsed >= 0.12 and self.active_riser in mt:
                self._change_phase("STEP_SETTLE", t)
            elif self.active_riser not in mv and elapsed >= 0.12:
                if self.clear_since is None:
                    self.clear_since = t
                if t - self.clear_since >= 0.10:
                    self._change_phase("MIDDLE_LAND", t)
            else:
                self.clear_since = None
        elif self.phase == "MIDDLE_LAND":
            self.target_deg = 0.0
            if elapsed >= 0.20 and (mt or mv):
                self._change_phase("STEP_SETTLE", t)

    def _slip_scale(self, wheel_index, body_speed, target_speed):
        joint = self.wheels[wheel_index]
        wheel_speed = float(self.d.qvel[self.m.jnt_dofadr[joint]])
        slip = (self.radius * wheel_speed - body_speed) / max(abs(body_speed), 0.25)
        self.max_slip = max(self.max_slip, abs(slip))
        if abs(body_speed) < 0.25 or slip <= 0.12:
            return 1.0
        if abs(wheel_speed) > 1.5 * max(abs(target_speed), 0.5):
            return float(np.clip(1.0 - (slip - 0.12) / 0.45, 0.45, 1.0))
        return 1.0

    def _command_wheels(self, speed_target, yaw, yaw_rate):
        prebraking = False
        if self.platform_seen and not self.finishing:
            platform = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "platform")
            front_x = max(float(self.d.geom_xpos[g, 0])
                          for g, k in self.wheel_geom.items() if k in self.front)
            remaining = (float(self.m.geom_pos[platform, 0] + self.m.geom_size[platform, 0])
                         - front_x - self.radius - 0.10)
            # Waiting for downhill contact leaves too little distance to brake.
            approach = math.sqrt((self.descent_speed * self.radius) ** 2
                                 + 2.0 * 0.35 * max(remaining, 0.0)) / self.radius
            prebraking = approach < speed_target
            speed_target = min(speed_target, approach)
        wheel_gain = self.descent_wheel_gain if prebraking or self.phase == "DESCENT" else self.wheel_gain
        y = float(self.d.qpos[1])
        vy = float(self.d.qvel[1])
        path_heading = self.yaw0 + float(np.clip(-math.atan(1.5 * y + 0.3 * vy), -0.35, 0.35))
        yaw_error = self._wrap(path_heading - yaw)
        self.max_yaw_error = max(self.max_yaw_error, abs(self._wrap(self.yaw0 - yaw)))
        self.max_abs_y = max(self.max_abs_y, abs(y))
        correction = float(np.clip(4.0 * yaw_error - 1.2 * yaw_rate, -1.5, 1.5))
        steer_target = float(np.clip(0.45 * yaw_error - 0.10 * yaw_rate, -0.15, 0.15))
        if self.braking or self.abort_reason:
            correction = 0.0
            steer_target = 0.0
            speed_target = 0.0
        body_speed = math.cos(yaw) * float(self.d.qvel[0]) + math.sin(yaw) * float(self.d.qvel[1])
        for k, (joint, actuator) in enumerate(zip(self.wheels, self.wheel_act)):
            target = speed_target - correction if k in self.left else speed_target + correction
            vel = float(self.d.qvel[self.m.jnt_dofadr[joint]])
            torque = wheel_gain * (target - vel) * self._slip_scale(k, body_speed, target)
            self.d.ctrl[actuator] = float(np.clip(torque, -self.tau_max, self.tau_max))
        for joint, actuator in zip(self.steer, self.steer_act):
            target = steer_target if "front" in self.joint_names[joint] else -steer_target
            q = float(self.d.qpos[self.m.jnt_qposadr[joint]])
            v = float(self.d.qvel[self.m.jnt_dofadr[joint]])
            error = target - q
            wheel_name = self.joint_names[joint].replace("_steer_joint", "_wheel_joint")
            wheel_index = self.wheels.index(self.joint_names.index(wheel_name))
            wheel_actuator = self.wheel_act[wheel_index]
            # Tire scrub must not gain more drive torque while steering is displaced.
            traction_scale = float(np.clip((25.0 - abs(math.degrees(error))) / 15.0, 0.0, 1.0))
            if self.d.ctrl[wheel_actuator] > 0:
                self.d.ctrl[wheel_actuator] *= traction_scale
            desired_velocity = float(np.clip(8.0 * error, -1.2, 1.2))
            self.d.ctrl[actuator] = float(np.clip(self.steer_damp * (desired_velocity - v), -40.0, 40.0))

    def _command_rockers(self):
        target = math.radians(self.target_deg)
        self.rocker_command += float(np.clip(target - self.rocker_command, -self.rocker_step, self.rocker_step))
        for joint, actuator in zip(self.rockers, self.rocker_act):
            q = float(self.d.qpos[self.m.jnt_qposadr[joint]])
            qvel = float(self.d.qvel[self.m.jnt_dofadr[joint]])
            desired_velocity = float(np.clip(
                self.rocker_gain / self.rocker_damp * (self.rocker_command - q), -0.7, 0.7))
            self.d.ctrl[actuator] = float(np.clip(
                self.rocker_damp * (desired_velocity - qvel), -100.0, 100.0))

    def update(self, t):
        c = self._contacts()
        self.last_contacts = c
        roll, pitch, yaw = self._rpy()
        if self.yaw_previous is not None:
            measured_rate = self._wrap(yaw - self.yaw_previous) / self.dt
            self.yaw_rate = 0.65 * self.yaw_rate + 0.35 * measured_rate
        self.yaw_previous = yaw
        if abs(roll) > math.radians(40):
            self.abort_reason = "roll_over_40deg"
        if abs(pitch) > math.radians(65):
            self.abort_reason = "pitch_over_65deg"
        if abs(float(self.d.qpos[1])) > 0.8:
            self.abort_reason = "lateral_drift_over_0.8m"
        self._update_target(t, c)
        final_ground = c["all_ground"] and not c["platform_top"] and not c["down"]
        if self.platform_seen and self.down_seen and final_ground:
            self.finishing = True
        if self.finishing:
            self.target_deg = 0.0
            if t >= self.brake_retry_after and abs(float(self.d.qpos[1])) < 0.18 and abs(self._wrap(self.yaw0 - yaw)) < math.radians(3):
                self.braking = True
        if self.braking and not final_ground:
            if self.support_lost_since is None:
                self.support_lost_since = t
            # A short crawl releases tire scrub before trying to park again.
            if t - self.support_lost_since >= 0.25 and self.settle_retries < 3:
                self.braking = False
                self.brake_retry_after = t + 1.0
                self.settle_retries += 1
                self.support_lost_since = None
        else:
            self.support_lost_since = None
        self.phase = ("SETTLE" if self.braking else "ALIGN") if self.finishing else self.phase
        speed = min(self.base_speed, 1.5) if self.finishing else self.base_speed
        if self.phase == "DESCENT":
            speed = min(speed, self.descent_speed)
        self.speed_command += float(np.clip(speed - self.speed_command, -4.0 * self.dt, 4.0 * self.dt))
        speed = self.speed_command
        self._command_wheels(speed, yaw, self.yaw_rate)
        self._command_rockers()
        wheel_slow = max(abs(float(self.d.qvel[self.m.jnt_dofadr[j]])) for j in self.wheels) < 0.20
        pose_ok = abs(roll) < math.radians(3) and abs(pitch) < math.radians(3)
        heading_ok = abs(self._wrap(self.yaw0 - yaw)) < math.radians(5) and abs(float(self.d.qpos[1])) < 0.25
        motion_ok = np.linalg.norm(self.d.qvel[:3]) < 0.03 and np.linalg.norm(self.d.qvel[3:6]) < 0.08
        posture_ok = all(abs(float(self.d.qpos[self.m.jnt_qposadr[j]])) < math.radians(3)
                         and abs(float(self.d.qvel[self.m.jnt_dofadr[j]])) < 0.10
                         for j in [*self.rockers, *self.steer])
        stable = bool(self.braking and wheel_slow and pose_ok and heading_ok and motion_ok
                      and posture_ok and final_ground and self.abort_reason is None)
        self.final_pose_ok = stable
        if stable:
            if self.settle_since is None:
                self.settle_since = t
        else:
            self.settle_since = None
        self.settle_ok = bool(stable and t - self.settle_since >= self.settle_time)
        return {"phase": self.phase, "target_deg": self.target_deg,
                "rocker_deg": math.degrees(self.rocker_command)}
