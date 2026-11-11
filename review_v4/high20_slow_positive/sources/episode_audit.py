"""Independent physical-quality checks for a completed rover episode."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


class EpisodeAudit:
    MAX_PENETRATION_M = 0.02
    MAX_NON_WHEEL_CONTACT_S = 0.02
    MAX_LIMIT_ERROR_RAD = math.radians(1.0)
    FORCE_TOLERANCE_NM = 0.001
    MIN_CONTACT_FORCE_N = 5.0
    MIN_FINAL_STABILITY_S = 1.0
    VELOCITY_TOLERANCE_RAD_S = 0.01

    def __init__(self, model, urdf_path=None):
        self.urdf_path = Path(urdf_path) if urdf_path is not None else (
            Path(__file__).resolve().parents[2] / "assets/mars_rover/urdf/mars_rover.urdf")
        urdf_bytes = self.urdf_path.read_bytes()
        self.urdf_sha256 = hashlib.sha256(urdf_bytes).hexdigest()
        self.velocity_limits = {}
        for joint in ET.fromstring(urdf_bytes).findall("joint"):
            if joint.get("type") == "fixed":
                continue
            name, limit = joint.get("name"), joint.find("limit")
            if name is None or limit is None or limit.get("velocity") is None:
                raise ValueError("every movable URDF joint must declare a velocity limit")
            velocity = float(limit.get("velocity"))
            if not math.isfinite(velocity) or velocity <= 0 or name in self.velocity_limits:
                raise ValueError(f"invalid or duplicate URDF velocity limit: {name}")
            self.velocity_limits[name] = velocity
        self.dt = float(model.opt.timestep)
        self.geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                           for i in range(model.ngeom)]
        self.joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
                            for i in range(model.njnt)]
        self.joints = []
        wheel_bodies = {}
        self.limited_joints = []
        for j, name in enumerate(self.joint_names):
            kind = next((kind for kind in ("wheel", "steer", "rocker")
                         if name.endswith(f"_{kind}_joint")), None)
            if kind:
                row = (name, kind, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
                self.joints.append(row)
                if kind == "wheel":
                    wheel_bodies[int(model.jnt_bodyid[j])] = name.removesuffix("_joint")
            if bool(model.jnt_limited[j]) and int(model.jnt_type[j]) in (2, 3):
                low, high = map(float, model.jnt_range[j])
                if math.isfinite(low) and math.isfinite(high):
                    self.limited_joints.append((name, int(model.jnt_qposadr[j]), low, high))
        self.wheel_names = sorted(wheel_bodies.values())
        model_movable = {name for j, name in enumerate(self.joint_names)
                         if int(model.jnt_type[j]) in (2, 3)}
        if model_movable != set(self.velocity_limits):
            raise ValueError("model movable joints differ from the URDF velocity-limit joints")
        self.velocity_dofs = {name: int(model.jnt_dofadr[j]) for j, name in enumerate(self.joint_names)
                              if name in self.velocity_limits}
        self.velocity_peak_by_joint_rad_s = {name: 0.0 for name in self.velocity_limits}
        self.velocity_excess_duration_by_joint_s = {name: 0.0 for name in self.velocity_limits}
        self.terrain_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == 0}
        self.ground_geoms = {g for g in self.terrain_geoms if self.geom_names[g] == "ground"}
        self.wheel_by_geom = {}
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            while body:
                if body in wheel_bodies:
                    self.wheel_by_geom[g] = wheel_bodies[body]
                    break
                body = int(model.body_parentid[body])
        self.warning_names = {int(value): name for name, value in mujoco.mjtWarning.__members__.items()
                              if name != "mjNWARNING"}
        self.warning_previous = {}
        self.warning_counts = {}
        self.force = np.zeros(6)
        self.step_count = 0
        self.last_time = None
        self.nonfinite_seen = False
        self.time_nonmonotonic_seen = False
        self.max_contact_penetration_m = 0.0
        self.worst_contact = None
        self.non_wheel_contact_duration_s = 0.0
        self.non_wheel_contact_duration_by_geom_s = {}
        self.max_joint_limit_error_rad = 0.0
        self.joint_limit_error_by_joint_rad = {name: 0.0 for name, *_ in self.limited_joints}
        self.actuator_force_peak_by_joint_nm = {name: 0.0 for name, *_ in self.joints}
        self.final_ground_normal_force_n = {name: 0.0 for name in self.wheel_names}
        self.final_joint_state = {}
        self.final_stability_criteria = {}
        self.continuous_final_stability_s = 0.0
        self._stable_since = None

    @staticmethod
    def _number(value):
        value = float(value)
        return value if math.isfinite(value) else None

    def step(self, model, data, t):
        self.step_count += 1
        now = float(t)
        dt = self.dt if self.last_time is None else now - self.last_time
        if self.last_time is not None and (not math.isfinite(dt) or dt <= 0.0):
            self.time_nonmonotonic_seen = True
            dt = self.dt
        self.last_time = now
        self.final_ground_normal_force_n = {name: 0.0 for name in self.wheel_names}
        for i, warning in enumerate(data.warning):
            count = int(warning.number)
            previous = self.warning_previous.get(i, 0)
            delta = count - previous if count >= previous else count
            if delta:
                name = self.warning_names.get(i, f"warning_{i}")
                self.warning_counts[name] = self.warning_counts.get(name, 0) + delta
            self.warning_previous[i] = count
        arrays = (data.qpos, data.qvel, data.qacc, data.ctrl, data.qfrc_actuator,
                  data.actuator_force)
        finite = math.isfinite(now) and all(bool(np.isfinite(a).all()) for a in arrays)
        if not finite:
            self.nonfinite_seen = True
        self.final_joint_state = {
            name: {"position_deg": self._number(np.degrees(data.qpos[q])),
                   "velocity_deg_s": self._number(np.degrees(data.qvel[v]))}
            for name, kind, q, v in self.joints if kind in ("rocker", "steer")
        }
        if not finite:
            self.final_stability_criteria = {"finite": False}
            self._stable_since = None
            self.continuous_final_stability_s = 0.0
            return
        for name, dof in self.velocity_dofs.items():
            speed = abs(float(data.qvel[dof]))
            self.velocity_peak_by_joint_rad_s[name] = max(self.velocity_peak_by_joint_rad_s[name], speed)
            if speed > self.velocity_limits[name] + self.VELOCITY_TOLERANCE_RAD_S:
                self.velocity_excess_duration_by_joint_s[name] += dt
        for name, _, _, dof in self.joints:
            self.actuator_force_peak_by_joint_nm[name] = max(
                self.actuator_force_peak_by_joint_nm[name], abs(float(data.qfrc_actuator[dof])))
        for name, q, low, high in self.limited_joints:
            error = max(low - float(data.qpos[q]), float(data.qpos[q]) - high, 0.0)
            self.joint_limit_error_by_joint_rad[name] = max(self.joint_limit_error_by_joint_rad[name], error)
            self.max_joint_limit_error_rad = max(self.max_joint_limit_error_rad, error)
        non_wheel_geoms = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = int(contact.geom1), int(contact.geom2)
            a_terrain, b_terrain = a in self.terrain_geoms, b in self.terrain_geoms
            if a_terrain == b_terrain or int(contact.efc_address) < 0:
                continue
            terrain, robot = (a, b) if a_terrain else (b, a)
            mujoco.mj_contactForce(model, data, i, self.force)
            if not (np.isfinite(self.force).all() and math.isfinite(float(contact.dist))
                    and np.isfinite(contact.frame[:3]).all()):
                self.nonfinite_seen = True
                continue
            penetration = max(0.0, -float(contact.dist))
            if penetration > self.max_contact_penetration_m:
                self.max_contact_penetration_m = penetration
                self.worst_contact = {"time_s": now, "robot_geom": self.geom_names[robot],
                                      "terrain_geom": self.geom_names[terrain],
                                      "penetration_m": penetration}
            wheel = self.wheel_by_geom.get(robot)
            if wheel is None:
                if float(np.linalg.norm(self.force[:3])) > self.MIN_CONTACT_FORCE_N:
                    non_wheel_geoms.add(robot)
            elif terrain in self.ground_geoms and abs(float(contact.frame[2])) > 0.70:
                self.final_ground_normal_force_n[wheel] += max(0.0, float(self.force[0]))
        if non_wheel_geoms:
            self.non_wheel_contact_duration_s += dt
            for geom in non_wheel_geoms:
                name = self.geom_names[geom]
                self.non_wheel_contact_duration_by_geom_s[name] = (
                    self.non_wheel_contact_duration_by_geom_s.get(name, 0.0) + dt)

        w, x, y, z = map(float, data.qpos[3:7])
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        wheels = [(q, v) for _, kind, q, v in self.joints if kind == "wheel"]
        posture = [(q, v) for _, kind, q, v in self.joints if kind in ("rocker", "steer")]
        self.final_stability_criteria = {
            "finite": finite,
            "six_wheels_on_ground": len(self.wheel_names) == 6 and all(
                force > self.MIN_CONTACT_FORCE_N for force in self.final_ground_normal_force_n.values()),
            "roll_pitch": abs(roll) < math.radians(3) and abs(pitch) < math.radians(3),
            "heading": abs(yaw) < math.radians(5),
            "lateral_position": abs(float(data.qpos[1])) < 0.25,
            "body_translation": float(np.linalg.norm(data.qvel[:3])) < 0.03,
            "body_rotation": float(np.linalg.norm(data.qvel[3:6])) < 0.08,
            "wheel_speed": len(wheels) == 6 and all(abs(float(data.qvel[v])) < 0.2 for _, v in wheels),
            "joint_position": len(posture) == 6 and all(abs(float(data.qpos[q])) < math.radians(3) for q, _ in posture),
            "joint_speed": len(posture) == 6 and all(abs(float(data.qvel[v])) < 0.1 for _, v in posture),
        }
        if all(self.final_stability_criteria.values()):
            if self._stable_since is None:
                self._stable_since = now
            self.continuous_final_stability_s = max(0.0, now - self._stable_since)
        else:
            self._stable_since = None
            self.continuous_final_stability_s = 0.0

    def result(self):
        limits = {"wheel": 50.0, "steer": 40.0, "rocker": 100.0}
        reasons = []
        if self.step_count == 0:
            reasons.append("no_audit_steps")
        if len(self.wheel_names) != 6:
            reasons.append("wheel_count_not_six")
        if self.nonfinite_seen:
            reasons.append("nonfinite_state_or_contact")
        if self.time_nonmonotonic_seen:
            reasons.append("nonmonotonic_simulation_time")
        if any(self.warning_counts.values()):
            reasons.append("mujoco_warning")
        if self.max_contact_penetration_m > self.MAX_PENETRATION_M + 1e-12:
            reasons.append("contact_penetration_over_20mm")
        if self.non_wheel_contact_duration_s > self.MAX_NON_WHEEL_CONTACT_S + 1e-12:
            reasons.append("non_wheel_contact_over_20ms")
        if self.max_joint_limit_error_rad > self.MAX_LIMIT_ERROR_RAD + 1e-12:
            reasons.append("joint_limit_error_over_1deg")
        for name, duration in self.velocity_excess_duration_by_joint_s.items():
            if duration > 0:
                reasons.append(f"joint_velocity_limit:{name}")
        category_peaks = {kind: 0.0 for kind in limits}
        for name, kind, _, _ in self.joints:
            peak = self.actuator_force_peak_by_joint_nm[name]
            category_peaks[kind] = max(category_peaks[kind], peak)
            if peak > limits[kind] + self.FORCE_TOLERANCE_NM:
                reasons.append(f"actuator_force_limit:{name}")
        final_ground_contact = {name: force >= self.MIN_CONTACT_FORCE_N
                                for name, force in self.final_ground_normal_force_n.items()}
        all_ground = len(final_ground_contact) == 6 and all(final_ground_contact.values())
        if not all_ground:
            reasons.append("final_six_wheels_not_supported_on_ground")
        if self.continuous_final_stability_s + 1e-9 < self.MIN_FINAL_STABILITY_S:
            reasons.append("final_stability_under_1s")
        return {
            "audit_steps": self.step_count,
            "audit_time_s": self._number(self.last_time) if self.last_time is not None else None,
            "nonfinite_seen": self.nonfinite_seen,
            "time_nonmonotonic_seen": self.time_nonmonotonic_seen,
            "mujoco_warning_counts": dict(self.warning_counts),
            "max_contact_penetration_m": self.max_contact_penetration_m,
            "worst_contact": self.worst_contact,
            "non_wheel_contact_duration_s": self.non_wheel_contact_duration_s,
            "non_wheel_contact_duration_by_geom_s": dict(self.non_wheel_contact_duration_by_geom_s),
            "max_joint_limit_error_deg": math.degrees(self.max_joint_limit_error_rad),
            "joint_limit_error_by_joint_deg": {name: math.degrees(value)
                                               for name, value in self.joint_limit_error_by_joint_rad.items()},
            "actuator_force_peak_by_joint_nm": dict(self.actuator_force_peak_by_joint_nm),
            "actuator_force_peak_by_kind_nm": category_peaks,
            "velocity_limit_source": {"path": str(self.urdf_path), "sha256": self.urdf_sha256},
            "joint_velocity_limit_rad_s": dict(self.velocity_limits),
            "joint_velocity_threshold_rad_s": {name: limit + self.VELOCITY_TOLERANCE_RAD_S
                                               for name, limit in self.velocity_limits.items()},
            "joint_velocity_peak_rad_s": dict(self.velocity_peak_by_joint_rad_s),
            "joint_velocity_excess_duration_s": dict(self.velocity_excess_duration_by_joint_s),
            "final_ground_normal_force_n": dict(self.final_ground_normal_force_n),
            "final_ground_contact_by_wheel": final_ground_contact,
            "final_all_six_ground": all_ground,
            "final_joint_state": self.final_joint_state,
            "continuous_final_stability_s": self.continuous_final_stability_s,
            "final_stability_criteria": dict(self.final_stability_criteria),
            "quality_rejection_reasons": reasons,
            "audit_quality_ok": not reasons,
            "thresholds": {"minimum_final_stability_s": self.MIN_FINAL_STABILITY_S,
                           "final_roll_pitch_deg": 3.0, "final_yaw_deg": 5.0,
                           "final_lateral_position_m": 0.25, "final_linear_speed_m_s": 0.03,
                           "final_angular_speed_rad_s": 0.08, "final_wheel_speed_rad_s": 0.2,
                           "final_rocker_steer_position_deg": 3.0, "final_rocker_steer_speed_rad_s": 0.1,
                           "max_penetration_m": self.MAX_PENETRATION_M,
                           "max_non_wheel_contact_s": self.MAX_NON_WHEEL_CONTACT_S,
                           "max_joint_limit_error_deg": math.degrees(self.MAX_LIMIT_ERROR_RAD),
                           "minimum_ground_normal_force_n": self.MIN_CONTACT_FORCE_N,
                           "actuator_force_limits_nm": limits,
                           "joint_velocity_tolerance_rad_s": self.VELOCITY_TOLERANCE_RAD_S,
                           "actuator_force_tolerance_nm": self.FORCE_TOLERANCE_NM},
        }
