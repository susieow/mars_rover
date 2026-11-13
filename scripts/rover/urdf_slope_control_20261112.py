"""Run URDF-constrained contact recovery candidates on the experiment-two staircase."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from recovery_urdf_model import build_scene, model_compliance
import urdf_slope_model as p0
from overspeed20_stairs import summarize_speed


URDF = ROOT / "assets/mars_rover/urdf/mars_rover.urdf"
LEGACY_RUN_ROOT = ROOT / "runs/urdf_slope_control_20261112"
RUN_ROOT = ROOT / "runs/urdf_slope_control_c12222df"
VIDEO_ROOT = Path("/home/agilex/wzy/videos/review_high_stairs/urdf_slope_control_c12222df")
DT = 0.001
HEIGHT = 0.2
STAIRS = 15
DEFAULT_ANGLE_DEG = 15.0
APPROACH_M = 8.0
MAX_SECONDS = 240.0
PLATFORM_CONFIRM_S = 1.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def wrap(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def orientation(data):
    w, x, y, z = map(float, data.qpos[3:7])
    return (
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(float(np.clip(2 * (w * y - z * x), -1, 1))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    )


class Controller:
    def __init__(self, model, data):
        self.m = model
        self.limits = get_joint_limits()
        self.d = data
        self.wheel_commands = np.zeros(3)
        self.rocker_commands = np.zeros(2)
        self.yaw_previous = None
        self.yaw_rate = 0.0
        self.names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or ""
            for index in range(model.njnt)
        ]
        actuators = {
            int(model.actuator_trnid[index, 0]): index for index in range(model.nu)
        }
        self.wheels = []
        self.steer = []
        self.rockers = []
        for joint_id, name in enumerate(self.names):
            if joint_id not in actuators:
                continue
            row = (
                joint_id,
                actuators[joint_id],
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
            if name.endswith("_wheel_joint"):
                role = next(
                    index
                    for index, role_name in enumerate(("front", "middle", "rear"))
                    if role_name in name
                )
                geom = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name.replace("_joint", "") + "_col_1",
                )
                self.wheels.append((*row, role, "left" in name, geom))
            elif name.endswith("_steer_joint"):
                self.steer.append(row)
            elif name.endswith("_rocker_joint"):
                self.rockers.append(row)
        if (len(self.wheels), len(self.steer), len(self.rockers)) != (6, 4, 2):
            raise ValueError("unexpected movable joint mapping")
        self.wheel_geom = {row[-1]: index for index, row in enumerate(self.wheels)}

    def update(self, targets, rocker_targets, heading_target=None):
        model, data = self.m, self.d
        _, _, yaw = orientation(data)
        if self.yaw_previous is not None:
            self.yaw_rate = (
                0.65 * self.yaw_rate
                + 0.35 * wrap(yaw - self.yaw_previous) / DT
            )
        self.yaw_previous = yaw
        tracking_heading = float(
            np.clip(
                -math.atan(1.5 * float(data.qpos[1]) + 0.3 * float(data.qvel[1])),
                -0.35,
                0.35,
            )
        )
        path_heading = (
            tracking_heading
            if heading_target is None
            else float(np.clip(heading_target + 0.25 * tracking_heading, -0.35, 0.35))
        )
        error = wrap(path_heading - yaw)
        correction = float(np.clip(4 * error - 1.2 * self.yaw_rate, -1.5, 1.5))
        steer_target = float(np.clip(0.45 * error - 0.1 * self.yaw_rate, -0.15, 0.15))
        self.wheel_commands += np.clip(
            np.asarray(targets, dtype=float) - self.wheel_commands,
            -30 * DT,
            30 * DT,
        )
        actual_targets = np.zeros(6)
        for index, (joint_id, actuator, _, dof, role, left, _) in enumerate(self.wheels):
            limit = self.limits[self.names[joint_id]]
            goal = float(
                np.clip(
                    self.wheel_commands[role] + (-correction if left else correction),
                    -limit["velocity"],
                    limit["velocity"],
                )
            )
            actual_targets[index] = goal
            data.ctrl[actuator] = np.clip(
                40 * (goal - float(data.qvel[dof])), -limit["effort"], limit["effort"]
            )
        for joint_id, actuator, qpos, dof in self.steer:
            limit = self.limits[self.names[joint_id]]
            goal = steer_target if "front" in self.names[joint_id] else -steer_target
            goal = float(np.clip(goal, limit["lower"], limit["upper"]))
            current_error = goal - float(data.qpos[qpos])
            speed = min(1.2, limit["velocity"])
            desired = float(np.clip(8 * current_error, -speed, speed))
            data.ctrl[actuator] = np.clip(
                50 * (desired - float(data.qvel[dof])), -limit["effort"], limit["effort"]
            )
            wheel_name = self.names[joint_id].replace("_steer_joint", "_wheel_joint")
            wheel_actuator = next(
                act
                for joint, act, *_ in self.wheels
                if self.names[joint] == wheel_name
            )
            if data.ctrl[wheel_actuator] > 0:
                data.ctrl[wheel_actuator] *= np.clip(
                    (25 - abs(math.degrees(current_error))) / 15, 0, 1
                )
        self.rocker_commands += np.clip(
            np.asarray(rocker_targets, dtype=float) - self.rocker_commands,
            -math.radians(20) * DT,
            math.radians(20) * DT,
        )
        for index, (joint_id, actuator, qpos, dof) in enumerate(self.rockers):
            limit = self.limits[self.names[joint_id]]
            self.rocker_commands[index] = np.clip(self.rocker_commands[index], limit["lower"], limit["upper"])
            speed = min(0.7, limit["velocity"])
            desired = float(
                np.clip(
                    4 * (self.rocker_commands[index] - float(data.qpos[qpos])),
                    -speed,
                    speed,
                )
            )
            data.ctrl[actuator] = np.clip(
                150 * (desired - float(data.qvel[dof])), -limit["effort"], limit["effort"]
            )
        return actual_targets


def initialize(scene):
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = [0.0, 0.0, 0.54]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    settle_controller = Controller(model, data)
    for _ in range(2000):
        settle_controller.update(np.zeros(3), np.zeros(2))
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("initial ground settle produced a non-finite state")
    data.qpos[0:2] = [0.0, 0.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    front_geoms = [
        index
        for index in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").endswith(
            "front_wheel_col_1"
        )
    ]
    if len(front_geoms) != 2:
        raise ValueError("expected two front wheel collision geoms")
    front_extent = max(
        float(data.geom_xpos[index, 0] + model.geom_size[index, 0])
        for index in front_geoms
    )
    data.qpos[0] += 1.6 - APPROACH_M - front_extent
    mujoco.mj_forward(model, data)
    return model, data


def contact_state(model, data, controller):
    forces = np.zeros(6)
    support = [set() for _ in controller.wheels]
    front_stair_force = 0.0
    platform = np.zeros(6, dtype=bool)
    self_contacts = 0
    min_distance = 0.0
    contact_force = np.zeros(6)
    terrain_names = ("ground", "up", "down", "platform")
    for index in range(data.ncon):
        contact = data.contact[index]
        if int(contact.efc_address) < 0:
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        wheel_id = controller.wheel_geom.get(geom1)
        terrain = name2
        if wheel_id is None:
            wheel_id = controller.wheel_geom.get(geom2)
            terrain = name1
        if wheel_id is None:
            if model.geom_bodyid[geom1] != 0 and model.geom_bodyid[geom2] != 0:
                self_contacts += 1
            continue
        terrain_is_stair = terrain.startswith(("up", "down")) or terrain in (
            "ground",
            "platform",
        )
        if not terrain_is_stair:
            continue
        mujoco.mj_contactForce(model, data, index, contact_force)
        normal = max(0.0, float(abs(contact_force[0])))
        forces[wheel_id] += normal
        support[wheel_id].add(terrain)
        min_distance = min(min_distance, float(contact.dist))
        if terrain == "platform" and normal > 5.0:
            platform[wheel_id] = True
        if controller.wheels[wheel_id][4] == 0 and terrain.startswith("up") and normal > 0:
            front_stair_force += normal
    return forces, support, front_stair_force, platform, self_contacts, min_distance


def wheel_support_levels(support):
    levels = np.full(6, -1, dtype=np.int8)
    for index, terrains in enumerate(support):
        values = [
            int(name[2:]) + 1
            for name in terrains
            if name.startswith("up") and name[2:].isdigit()
        ]
        if "platform" in terrains:
            values.append(STAIRS)
        if values:
            levels[index] = max(values)
    return levels


def wave_rocker(direction, elapsed):
    phase = elapsed % 8.0
    if elapsed >= 24.0:
        angle = 0.0
    elif phase < 1.0:
        angle = 20.0 * phase
    elif phase < 2.0:
        angle = 20.0
    elif phase < 4.0:
        angle = 20.0 - 20.0 * (phase - 2.0)
    elif phase < 5.0:
        angle = -20.0
    elif phase < 6.0:
        angle = -20.0 + 20.0 * (phase - 5.0)
    else:
        angle = 0.0
    sign = -1.0 if direction == "SN" else 1.0
    return np.full(2, math.radians(angle * sign))


def get_joint_limits():
    root = ET.parse(URDF).getroot()
    limits = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        joint_type = joint.get("type")
        limit = joint.find("limit")
        if not name or joint_type == "fixed" or limit is None:
            continue
        limits[name] = {
            "type": joint_type,
            "lower": None if joint_type == "continuous" else float(limit.get("lower")),
            "upper": None if joint_type == "continuous" else float(limit.get("upper")),
            "velocity": float(limit.get("velocity")),
            "effort": float(limit.get("effort")),
        }
    return limits


def exact_audit_limits(model, data, limits):
    violations = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if name not in limits:
            continue
        expected = limits[name]
        qpos = float(data.qpos[model.jnt_qposadr[joint_id]])
        velocity = abs(float(data.qvel[model.jnt_dofadr[joint_id]]))
        torque = abs(float(data.qfrc_actuator[model.jnt_dofadr[joint_id]]))
        if expected["lower"] is not None and (
            qpos < expected["lower"] or qpos > expected["upper"]
        ):
            violations.append(f"position_limit:{name}")
        if velocity > expected["velocity"]:
            violations.append(f"velocity_limit:{name}")
        if torque > expected["effort"]:
            violations.append(f"effort_limit:{name}")
    return violations


class PreLiftPattern:
    def __init__(self, transfer=False, transfer_deg=10.0):
        self.phase = "APPROACH"
        self.started = 0.0
        self.trigger_time = None
        self.trigger_progress = None
        self.events = []
        self.transfer = transfer
        self.transfer_deg = transfer_deg

    def switch(self, phase, t, progress):
        self.phase = phase
        self.started = t
        self.events.append(
            {"time_s": round(float(t), 6), "phase": phase, "progress_m": round(float(progress), 6)}
        )

    def command(self, t, progress, contact, front_level, trigger_front_level):
        if self.phase == "APPROACH":
            if contact and self.trigger_time is None:
                self.trigger_time = float(t)
                self.trigger_progress = float(progress)
                self.switch("UNLOAD", t, progress)
            return np.full(3, 3.0), np.zeros(2)
        elapsed = t - self.started
        if self.phase == "UNLOAD":
            if elapsed >= 0.15:
                self.switch("RECOIL", t, progress)
            return np.array([0.0, 0.5, 1.0]), np.zeros(2)
        if self.phase == "RECOIL":
            if elapsed >= 0.30:
                self.switch("PRELIFT", t, progress)
            return np.array([-0.8, -0.2, 0.6]), np.zeros(2)
        if self.phase == "PRELIFT":
            if elapsed >= 0.90:
                self.switch("REATTACH", t, progress)
            return np.array([0.0, 0.5, 1.0]), np.full(2, math.radians(-12.0))
        if self.phase == "REATTACH":
            if (
                front_level > trigger_front_level
                or progress - self.trigger_progress >= 0.08
            ):
                self.switch("CLIMB", t, progress)
            elif elapsed >= 2.0:
                self.switch("RETRY_LIFT", t, progress)
            return np.array([1.5, 1.5, 2.0]), np.full(2, math.radians(-12.0))
        if self.phase == "RETRY_LIFT":
            if elapsed >= 0.60:
                self.switch("REATTACH", t, progress)
            return np.array([0.0, 0.5, 1.0]), np.full(2, math.radians(-18.0))
        if self.phase == "CLIMB":
            if elapsed >= 1.4:
                self.switch("LOWER_FRONT" if self.transfer else "FOLLOW", t, progress)
            blend = min(1.0, elapsed / 1.4)
            return np.array([2.5, 2.5, 3.0]), np.full(
                2, math.radians(-12.0) * (1.0 - blend)
            )
        if self.phase == "LOWER_FRONT":
            if elapsed >= 1.5:
                self.switch("TRANSFER_SETTLE", t, progress)
            return np.full(3, 3.0), np.full(2, math.radians(self.transfer_deg))
        if self.phase == "TRANSFER_SETTLE":
            if elapsed >= 0.9:
                self.switch("FOLLOW", t, progress)
            return np.full(3, 3.0), np.zeros(2)
        if self.phase == "FOLLOW":
            if self.trigger_progress is not None and progress - self.trigger_progress >= 0.42:
                self.trigger_time = None
                self.trigger_progress = None
                self.switch("APPROACH", t, progress)
            return np.full(3, 3.0), np.zeros(2)
        raise ValueError(f"unknown phase: {self.phase}")


class ZigzagPattern:
    def __init__(self, direction, yaw_deg=6.0, rocker_deg=8.0):
        self.phase = "APPROACH"
        self.started = 0.0
        self.trigger_time = None
        self.trigger_progress = None
        self.contact_side = "left" if direction == "SN" else "right"
        self.yaw_rad = math.radians(yaw_deg)
        self.rocker_rad = math.radians(rocker_deg)
        self.events = []

    def switch(self, phase, t, progress):
        self.phase = phase
        self.started = t
        self.events.append(
            {
                "time_s": round(float(t), 6),
                "phase": phase,
                "progress_m": round(float(progress), 6),
                "contact_side": self.contact_side,
            }
        )

    def command(self, t, progress, front_total_force, front_force, front_indices, controller):
        if self.phase == "APPROACH":
            if front_total_force > 0 and self.trigger_time is None:
                left_force = sum(
                    float(front_force[index])
                    for index in front_indices
                    if controller.wheels[index][5]
                )
                right_force = sum(
                    float(front_force[index])
                    for index in front_indices
                    if not controller.wheels[index][5]
                )
                if abs(left_force - right_force) > 1e-6:
                    self.contact_side = "left" if left_force > right_force else "right"
                self.trigger_time = float(t)
                self.trigger_progress = float(progress)
                self.switch("UNLOAD", t, progress)
            return np.full(3, 3.0), np.zeros(2), 0.0
        elapsed = t - self.started
        if self.phase == "UNLOAD":
            if elapsed >= 0.15:
                self.switch("ZIG_A", t, progress)
            return np.array([0.0, 0.5, 1.0]), np.zeros(2), 0.0
        if self.phase == "ZIG_A":
            if elapsed >= 1.1:
                self.switch("ZIG_B", t, progress)
            if self.contact_side == "left":
                rockers = np.array([-self.rocker_rad, self.rocker_rad])
                heading = self.yaw_rad
            else:
                rockers = np.array([self.rocker_rad, -self.rocker_rad])
                heading = -self.yaw_rad
            return np.array([1.5, 1.5, 2.0]), rockers, heading
        if self.phase == "ZIG_B":
            if elapsed >= 1.1:
                self.switch("RECENTER", t, progress)
            if self.contact_side == "left":
                rockers = np.array([self.rocker_rad, -self.rocker_rad])
                heading = -self.yaw_rad
            else:
                rockers = np.array([-self.rocker_rad, self.rocker_rad])
                heading = self.yaw_rad
            return np.array([2.0, 2.0, 2.5]), rockers, heading
        if self.phase == "RECENTER":
            if elapsed >= 0.8:
                self.switch("TRANSFER", t, progress)
            return np.array([2.5, 2.5, 3.0]), np.zeros(2), 0.0
        if self.phase == "TRANSFER":
            if elapsed >= 0.9:
                self.switch("FOLLOW", t, progress)
            return np.full(3, 3.0), np.full(2, math.radians(8.0)), 0.0
        if self.phase == "FOLLOW":
            if self.trigger_progress is not None and progress - self.trigger_progress >= 0.35:
                self.trigger_time = None
                self.trigger_progress = None
                self.switch("APPROACH", t, progress)
            return np.full(3, 3.0), np.zeros(2), 0.0
        raise ValueError(f"unknown phase: {self.phase}")


def run_case(*, friction, strategy, direction, angle_deg, max_seconds=MAX_SECONDS):
    angle_tag = f"{angle_deg:g}".replace(".", "p")
    case_id = f"urdf{angle_tag}_mu{friction}_{strategy}_{direction}_c12222df"
    folder = RUN_ROOT / "cases" / case_id
    scene = build_scene(folder, friction, angle_deg)
    model = mujoco.MjModel.from_xml_path(str(scene))
    compliance = model_compliance(model, scene, friction)
    write_json(folder / "model_compliance.json", compliance)
    write_json(RUN_ROOT / f"model_compliance_mu{friction}.json", compliance)
    if not compliance["passed"]:
        raise RuntimeError(f"model compliance gate failed: {folder / 'model_compliance.json'}")
    model, data = initialize(scene)
    controller = Controller(model, data)
    limits = get_joint_limits()
    initial_state = np.empty(
        mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    )
    mujoco.mj_getState(
        model, data, initial_state, mujoco.mjtState.mjSTATE_INTEGRATION
    )
    parameters = {
        "id": case_id,
        "experiment": "urdf_slope_control_c12222df",
        "strategy": strategy,
        "direction": direction,
        "height_cm": 20,
        "stairs": 15,
        "stair_angle_deg": angle_deg,
        "depth_m": HEIGHT / math.tan(math.radians(angle_deg)),
        "stair_friction_slide": friction,
        "physics_dt_s": DT,
        "control_dt_s": DT,
        "max_seconds": max_seconds,
        "approach_distance_m": APPROACH_M,
        "base_wheel_speed_rad_s": 3.0,
        "control_version": (
            "micro_zigzag_v1" if strategy == "zigzag" else
            "contact_prelift_transfer18_v3" if strategy == "transfer18" else
            "contact_prelift_transfer_v2" if strategy == "transfer" else
            "contact_prelift_v1" if strategy == "prelift" else "baseline_wave_v1"
        ),
        "contact_trigger": "front wheel positive normal stair force",
        "prelift_deg": -12.0,
        "prelift_retry_deg": -18.0,
        "recoil_requested_wheel_rad_s": [-0.8, -0.2, 0.6],
        "platform_confirmation_s": PLATFORM_CONFIRM_S,
    }
    write_json(folder / "started.json", parameters)
    maximum_steps = round(max_seconds / DT)
    arrays = {
        name: np.zeros((maximum_steps + 1, size))
        for name, size in (("time", 1), ("qpos", model.nq), ("qvel", model.nv))
    }
    for name, size in (
        ("ctrl", model.nu),
        ("qfrc_actuator", model.nv),
        ("wheel_targets", 6),
        ("wheel_group_commands", 3),
        ("rocker_targets_rad", 2),
        ("contact_force_n", 6),
        ("wheel_speed_rad_s", 6),
        ("heading_target_rad", 1),
    ):
        arrays[name] = np.zeros((maximum_steps, size))
    arrays["wheel_support_level"] = np.full((maximum_steps, 6), -1, dtype=np.int8)
    arrays["platform_support"] = np.zeros((maximum_steps, 6), dtype=bool)
    arrays["rear_geometry_level"] = np.zeros(maximum_steps, dtype=np.int8)
    arrays["time"][0], arrays["qpos"][0], arrays["qvel"][0] = (
        data.time,
        data.qpos,
        data.qvel,
    )
    transfer_deg = 18.0 if strategy == "transfer18" else 10.0
    pattern = PreLiftPattern(
        transfer=strategy in {"transfer", "transfer18"},
        transfer_deg=transfer_deg,
    )
    zigzag_pattern = ZigzagPattern(direction)
    front_indices = [
        index for index, row in enumerate(controller.wheels) if row[4] == 0
    ]
    baseline_trigger = None
    first_contact = None
    first_failure = None
    platform_since = None
    stable_since = None
    ascent_completed = False
    termination_reason = "simulation_time_limit"
    peak_velocity = {name: 0.0 for name in limits}
    peak_effort = {name: 0.0 for name in limits}
    penetration = 0.0
    violation_names = set()
    max_self_contacts = 0
    started_wall = time.monotonic()
    for step in range(maximum_steps):
        progress = float(data.qpos[0] - arrays["qpos"][0, 0])
        forces, support, front_force, platform, self_contacts, min_distance = (
            contact_state(model, data, controller)
        )
        max_self_contacts = max(max_self_contacts, self_contacts)
        penetration = max(penetration, -min_distance)
        levels_for_control = wheel_support_levels(support)
        front_level = int(levels_for_control[front_indices].max(initial=-1))
        if first_contact is None and front_force > 0:
            first_contact = float(data.time)
        if strategy == "baseline":
            if baseline_trigger is None:
                targets = np.full(3, 3.0)
                rockers = np.zeros(2)
                if front_force > 0:
                    baseline_trigger = float(data.time)
            else:
                targets = np.full(3, 3.0)
                rockers = wave_rocker(direction, float(data.time) - baseline_trigger)
            heading = None
            trigger_level = -1
        elif strategy == "zigzag":
            targets, rockers, heading = zigzag_pattern.command(
                float(data.time),
                progress,
                front_force,
                forces,
                front_indices,
                controller,
            )
            if baseline_trigger is None and zigzag_pattern.trigger_time is not None:
                baseline_trigger = zigzag_pattern.trigger_time
        else:
            trigger_level = (
                int(wheel_support_levels(support)[0]) if pattern.trigger_time else -1
            )
            targets, rockers = pattern.command(
                float(data.time),
                progress,
                bool(front_force > 0 and pattern.trigger_time is None),
                front_level,
                trigger_level,
            )
            heading = None
            if baseline_trigger is None and pattern.trigger_time is not None:
                baseline_trigger = pattern.trigger_time
        if all(platform) and platform_since is not None:
            targets = np.zeros(3)
            rockers = np.zeros(2)
            heading = 0.0
        goal = controller.update(targets, rockers, heading_target=heading)
        arrays["ctrl"][step] = data.ctrl
        arrays["wheel_targets"][step] = goal
        arrays["wheel_group_commands"][step] = controller.wheel_commands
        arrays["rocker_targets_rad"][step] = controller.rocker_commands
        arrays["heading_target_rad"][step] = 0.0 if heading is None else heading
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            termination_reason = "nonfinite_state"
            break
        velocities = exact_audit_limits(model, data, limits)
        if velocities:
            violation_names.update(velocities)
            if first_failure is None:
                first_failure = float(data.time)
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if name in limits:
                dof = int(model.jnt_dofadr[joint_id])
                peak_velocity[name] = max(
                    peak_velocity[name], abs(float(data.qvel[dof]))
                )
                peak_effort[name] = max(
                    peak_effort[name], abs(float(data.qfrc_actuator[dof]))
                )
        forces, support, front_force, platform, self_contacts, min_distance = (
            contact_state(model, data, controller)
        )
        levels = wheel_support_levels(support)
        arrays["time"][step + 1] = data.time
        arrays["qpos"][step + 1] = data.qpos
        arrays["qvel"][step + 1] = data.qvel
        arrays["qfrc_actuator"][step] = data.qfrc_actuator
        arrays["contact_force_n"][step] = forces
        arrays["wheel_support_level"][step] = levels
        arrays["platform_support"][step] = platform
        arrays["rear_geometry_level"][step] = int(
            wheel_support_levels(support)[4:].max(initial=-1)
        )
        arrays["wheel_speed_rad_s"][step] = [
            float(data.qvel[row[3]]) for row in controller.wheels
        ]
        if np.all(platform):
            if platform_since is None:
                platform_since = float(data.time)
        else:
            platform_since = None
            stable_since = None
        roll, pitch, yaw = orientation(data)
        _, _, _ = roll, pitch, yaw
        wheel_slow = max(
            abs(float(data.qvel[row[3]])) for row in controller.wheels
        ) < 0.2
        body_slow = (
            np.linalg.norm(data.qvel[:3]) < 0.03
            and np.linalg.norm(data.qvel[3:6]) < 0.08
        )
        settled_pose = (
            wheel_slow
            and body_slow
            and abs(roll) < math.radians(3)
            and abs(pitch) < math.radians(3)
            and abs(np.degrees(yaw)) < 5
        )
        if platform_since is not None and settled_pose:
            if stable_since is None:
                stable_since = float(data.time)
            if (
                data.time - stable_since >= PLATFORM_CONFIRM_S - 1e-9
                and data.time - platform_since >= PLATFORM_CONFIRM_S - 1e-9
            ):
                ascent_completed = True
                termination_reason = "platform_complete"
                break
        else:
            stable_since = None
        if abs(roll) > math.radians(40):
            termination_reason = "roll_over_40deg"
            break
        if abs(pitch) > math.radians(65):
            termination_reason = "pitch_over_65deg"
            break
        if abs(float(data.qpos[1])) > 0.8:
            termination_reason = "lateral_drift_over_0.8m"
            break
    completed_steps = step + 1
    arrays = {
        name: value[: completed_steps + 1]
        if name in ("time", "qpos", "qvel")
        else value[:completed_steps]
        for name, value in arrays.items()
    }
    trajectory = folder / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        **arrays,
        initial_integration_state=initial_state,
        simulation_dt=DT,
        scene_sha256=digest(scene),
    )
    overspeed = summarize_speed(
        arrays["time"][1:],
        arrays["qvel"][1:],
        {
            name: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in limits
        },
        {name: value["velocity"] for name, value in limits.items()},
        DT,
    )
    roll, pitch, yaw = orientation(data)
    numerical_quality = (
        termination_reason not in {"nonfinite_state", "roll_over_40deg", "pitch_over_65deg"}
        and penetration <= 0.02
    )
    urdf_compliant = not violation_names
    result = {
        "id": case_id,
        "parameters": parameters,
        "duration_s": float(data.time),
        "termination_reason": termination_reason,
        "first_contact_s": first_contact,
        "intervention_s": baseline_trigger,
        "events": (
            zigzag_pattern.events if strategy == "zigzag" else
            pattern.events if strategy in {"prelift", "transfer", "transfer18"} else []
        ),
        "ascent_completed": ascent_completed,
        "urdf_compliant_exact": urdf_compliant,
        "numerical_quality_ok": numerical_quality,
        "dataset_accepted": False,
        "violation_names": sorted(violation_names),
        "first_failure_s": first_failure,
        "max_contact_penetration_m": penetration,
        "max_self_contacts": max_self_contacts,
        "peak_velocity_rad_s": peak_velocity,
        "peak_effort_nm": peak_effort,
        "overspeed": overspeed,
        "final": {
            "base_x": float(data.qpos[0]),
            "base_z": float(data.qpos[2]),
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "all_platform_support": bool(np.all(platform)),
        },
        "scene_path": str(scene),
        "trajectory_path": str(trajectory),
        "trajectory_sha256": digest(trajectory),
        "scene_sha256": digest(scene),
        "source_sha256": digest(__file__),
        "urdf_sha256": digest(URDF),
        "robot_xml_sha256": digest(folder / "model/rover_p0.xml"),
        "model_builder_sha256": digest(p0.__file__),
        "model_gate_passed": compliance["passed"],
        "audit_scope": "post_initialization_joint_limits_and_legacy_endpoint",
        "current_urdf_certified": False,
        "video_path": str(VIDEO_ROOT / f"{case_id}.mp4"),
        "wall_seconds": time.monotonic() - started_wall,
        "limitations": [
            "Compiled model fields are checked against the pinned URDF; runtime audit covers saved post-initialization steps only.",
            "Legacy endpoint and penetration checks are diagnostic; no independent replay or timestep-convergence certification.",
            "Robot self-collision is enabled for the new model.",
            "Simulation contact parameters and friction are not hardware calibrated.",
            "A platform failure is a statement about this control candidate, not a hardware capability limit.",
        ],
    }
    write_json(folder / "result.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "id",
                    "duration_s",
                    "termination_reason",
                    "first_contact_s",
                    "intervention_s",
                    "ascent_completed",
                    "urdf_compliant_exact",
                    "numerical_quality_ok",
                    "max_contact_penetration_m",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--friction", type=int, choices=(4, 6, 8), required=True)
    parser.add_argument(
        "--strategy",
        choices=("baseline", "prelift", "transfer", "transfer18", "zigzag"),
    )
    parser.add_argument("--direction", choices=("SN", "SP"))
    parser.add_argument("--angle-deg", type=float, default=DEFAULT_ANGLE_DEG)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    if not 5.0 <= args.angle_deg <= 45.0:
        parser.error("--angle-deg must be within [5, 45]")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be positive and finite")
    if args.check_model:
        folder = Path(tempfile.mkdtemp(prefix="urdf_slope_compliance_", dir=ROOT / "runs"))
        try:
            scene = build_scene(folder / "case", args.friction, args.angle_deg)
            model = mujoco.MjModel.from_xml_path(str(scene))
            compliance = model_compliance(model, scene, args.friction)
            print(json.dumps(compliance, ensure_ascii=False, indent=2))
            raise SystemExit(0 if compliance["passed"] else 1)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    if not args.strategy or not args.direction:
        parser.error("--strategy and --direction are required unless --check-model is used")
    run_case(
        friction=args.friction,
        strategy=args.strategy,
        direction=args.direction,
        angle_deg=args.angle_deg,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
