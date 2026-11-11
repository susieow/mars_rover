"""Contact-driven traction controller for the six-wheel Mars rover."""

from __future__ import annotations

import math
from collections import defaultdict

import mujoco
import numpy as np


class ClosedLoopController:
    def __init__(self, model, data, *, wheels, wheel_act, steer, steer_act,
                 rockers, rocker_act, wheel_radius=0.163, speed=6.0,
                 tau_max=50.0, yaw0=0.0, front_lift_deg=20.0,
                 middle_lift_deg=20.0, rocker_gain=80.0, rocker_damp=3.0,
                 rocker_slew_deg_s=70.0, dt=0.02):
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
        all_ground = {r: False for r in roles}
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
            if kind == "ground":
                # Ground contact is enough; contact-frame orientation varies across MuJoCo geom ordering.
                all_ground[role] = True
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
            "all_ground": all(all_ground.values()),
            "platform_top": any("platform" in top[r] for r in roles),
            "down": any(forces[r]["down"] > 10.0 for r in roles),
        }

    def _update_target(self, t, c):
        if c["platform_top"]:
            self.platform_seen = True
        if c["down"] or any(k[2].startswith("down") for k in c["events"]):
            self.down_seen = True
        events = []
        event_keys = set(c["events"])
        for role, kind, terrain in c["events"]:
            name = "front" if role == "front" else "middle" if role == "middle" else "rear"
            events.append(f"{name}_{'riser' if kind == 'vertical' else 'top'}")

        front_v = bool(c["vertical"]["front"])
        middle_v = bool(c["vertical"]["middle"])
        front_top_up = any(name.startswith("up") for name in c["top"]["front"])
        middle_top_up = any(name.startswith("up") for name in c["top"]["middle"])
        front_v_event = any(role == "front" and kind == "vertical" and terrain.startswith("up")
                            for role, kind, terrain in event_keys)
        middle_v_event = any(role == "middle" and kind == "vertical" and terrain.startswith("up")
                             for role, kind, terrain in event_keys)
        front_top_event = any(role == "front" and kind == "top" and terrain.startswith("up")
                              for role, kind, terrain in event_keys)
        middle_top_event = any(role == "middle" and kind == "top" and terrain.startswith("up")
                               for role, kind, terrain in event_keys)

        if self.platform_seen and self.down_seen:
            self.target_deg = 0.0
            self.phase = "DESCENT"
        elif self.phase in {"APPROACH", "CRUISE", "STEP_SETTLE"} and (front_v or front_v_event):
            self.target_deg = -self.front_lift_deg
            self.phase = "FRONT_LIFT"
        elif self.phase == "FRONT_LIFT" and (front_top_up or front_top_event):
            self.target_deg = self.middle_lift_deg
            self.phase = "MIDDLE_APPROACH"
        elif self.phase == "MIDDLE_APPROACH" and (middle_v or middle_v_event):
            self.target_deg = self.middle_lift_deg
            self.phase = "MIDDLE_LIFT"
        elif self.phase == "MIDDLE_LIFT" and (middle_top_up or middle_top_event):
            self.target_deg = 0.0
            self.phase = "STEP_SETTLE"
        elif self.phase == "STEP_SETTLE" and not front_v and not front_v_event:
            self.target_deg = 0.0
            self.phase = "CRUISE"
        elif self.phase == "APPROACH" and not front_v:
            self.target_deg = 0.0
        if events:
            self.event_log.append({"t": round(t, 3), "events": events, "phase": self.phase})

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
        y = float(self.d.qpos[1])
        vy = float(self.d.qvel[1])
        yaw_error = self._wrap(self.yaw0 - yaw)
        self.max_yaw_error = max(self.max_yaw_error, abs(yaw_error))
        self.max_abs_y = max(self.max_abs_y, abs(y))
        correction = 2.8 * yaw_error + 0.9 * yaw_rate - 1.20 * y - 0.40 * vy
        correction = float(np.clip(correction, -0.55, 0.55))
        if abs(yaw_error) > math.radians(20) or abs(y) > 0.25:
            speed_target = min(speed_target, 1.2)
            correction = float(np.clip(4.5 * yaw_error - 1.1 * y - 0.35 * vy, -0.75, 0.75))
        if abs(yaw_error) > math.radians(50) or abs(y) > 0.65:
            self.abort_reason = "heading_or_lateral_drift"
            speed_target = 0.0
        body_speed = math.cos(yaw) * float(self.d.qvel[0]) + math.sin(yaw) * float(self.d.qvel[1])
        for k, (joint, actuator) in enumerate(zip(self.wheels, self.wheel_act)):
            target = speed_target - correction if k in self.left else speed_target + correction
            vel = float(self.d.qvel[self.m.jnt_dofadr[joint]])
            torque = 24.0 * (target - vel) * self._slip_scale(k, body_speed, target)
            self.d.ctrl[actuator] = float(np.clip(torque, -self.tau_max, self.tau_max))

    def _command_rockers(self):
        target = math.radians(self.target_deg)
        self.rocker_command += float(np.clip(target - self.rocker_command, -self.rocker_step, self.rocker_step))
        for joint, actuator in zip(self.rockers, self.rocker_act):
            q = float(self.d.qpos[self.m.jnt_qposadr[joint]])
            qvel = float(self.d.qvel[self.m.jnt_dofadr[joint]])
            self.d.ctrl[actuator] = float(np.clip(self.rocker_gain * (self.rocker_command - q) - self.rocker_damp * qvel, -100, 100))

    def update(self, t):
        c = self._contacts()
        self._update_target(t, c)
        roll, pitch, yaw = self._rpy()
        final_ground = c["all_ground"] and not c["platform_top"] and not c["down"]
        if self.platform_seen and self.down_seen and final_ground:
            if not self.settle_started:
                self.settle_started = True
                self.settle_since = t
                self.phase = "SETTLE"
            self.target_deg = 0.0
        self._command_wheels(0.0 if self.settle_started else self.base_speed, yaw, float(self.d.qvel[5]))
        self._command_rockers()
        wheel_slow = max((abs(float(self.d.qvel[self.m.jnt_dofadr[j]])) for j in self.wheels), default=0.0) < 0.20
        pose_ok = abs(roll) < math.radians(5) and abs(pitch) < math.radians(5)
        heading_ok = abs(self._wrap(self.yaw0 - yaw)) < math.radians(10) and abs(float(self.d.qpos[1])) < 0.25
        motion_ok = np.linalg.norm(self.d.qvel[:3]) < 0.05 and np.linalg.norm(self.d.qvel[3:6]) < 0.10
        if self.settle_started and self.settle_since is not None and t - self.settle_since >= 0.80:
            self.settle_ok = bool(wheel_slow and pose_ok and heading_ok and motion_ok and final_ground)
        return {"phase": self.phase, "target_deg": round(self.target_deg, 2), "rocker_deg": round(math.degrees(self.rocker_command), 2)}
