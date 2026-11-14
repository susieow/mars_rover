"""Frozen 15-step run loop shared by Phase D and E.

Same scene, controller, audit and success criterion as the Phase B runner
(a887141c/3dade1ce lineage). Per case it now also accepts `steps`, `speed`,
`trigger` (contact or stall), and will later take `dt` and a control period.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rover"))
import speed20_experiment as common  # noqa: E402
from urdf_slope_p2_20261112 import Audit, rpy  # noqa: E402

RUN = ROOT / "runs/urdf_slope_20261112/pb"
ROBOT = ROOT / "runs/urdf_slope_20261112/model/rover_p0.xml"
URDF = ROOT / "assets/mars_rover/urdf/mars_rover.urdf"
MODEL_SPEC = ROOT / "runs/urdf_slope_20261112/model/model_spec.json"
DT = 0.001
STEP_RISE = 0.20
STEP_COUNT = 15
WIDTH_HALF = 2.5
PLATFORM_LEN = 2.0
RISER_X = 1.6
APPROACH_M = 8.0
SPEED = 3.0
PERIOD_S = 8.0
CYCLES = 3
BRAKE_S = 3.0
SUPPORT_N = 5.0
UNLOADED_GAP_M = 0.001
STRUCTURE_N = 5.0
STRUCTURE_MAX_S = 0.020
ROLLOVER_DEG = 60.0
LATERAL_LIMIT_M = 2.5
SAVE_EVERY = 10
BOOST_EFFORT = 160.0
# 25 and 35 exist only to narrow the boundary once 15 passes and 45 fails.
ANGLES = (15.0, 25.0, 35.0, 45.0)

SETTLE = {
    "roll_deg": 3.0, "pitch_deg": 3.0, "heading_deg": 5.0,
    "linear_m_s": 0.03, "angular_rad_s": 0.08,
    "wheel_rad_s": 0.2, "joint_rad_s": 0.1,
}


def budget(angle_deg: float, rise_m: float = STEP_RISE) -> int:
    tread = rise_m / math.tan(math.radians(angle_deg))
    ideal = (APPROACH_M + STEP_COUNT * tread + PLATFORM_LEN) / (SPEED * 0.162697)
    return min(600, max(240, math.ceil(2 * ideal) + 180))


def cases() -> list[dict]:
    out = []
    for angle in ANGLES:
        tread = STEP_RISE / math.tan(math.radians(angle))
        for friction in (4, 6, 8):
            for order in ("SP", "SN"):
                for amplitude in (18.0, 20.0):
                    out.append({
                        "run_id": f"urdf_slope_20261112/B/{angle:g}deg/mu{friction}/{order}/{amplitude:g}deg",
                        "angle_deg": angle, "tread_m": tread,
                        "friction": friction, "order": order,
                        "rocker_amp_deg": amplitude,
                        "timeout_s": budget(angle),
                    })
    return out


def box(name: str, x: float, half_x: float, top: float, friction: float) -> str:
    return (f'    <geom name="{name}" type="box" pos="{x} 0 {top / 2}" '
            f'size="{half_x} {WIDTH_HALF} {top / 2}" contype="2" conaffinity="1" '
            f'friction="{friction} 0.02 0.001" solref="0.004 1" '
            f'rgba=".58 .47 .36 1" />\n')


def scene_text(case: dict) -> str:
    terrain = ('    <light name="sun" pos="0 0 14" dir="0 0 -1" directional="true" '
               'diffuse=".9 .9 .85" />\n'
               '    <geom name="ground" type="plane" size="80 80 0.1" contype="2" '
               'conaffinity="1" friction="4.0 0.02 0.001" solref="0.004 1" '
               'rgba=".72 .72 .72 1" />\n')
    x = RISER_X
    steps = case.get("steps", STEP_COUNT)
    rise = case.get("rise_m", STEP_RISE)
    for i in range(steps):
        terrain += box(f"up{i}", x + case["tread_m"] / 2, case["tread_m"] / 2,
                       rise * (i + 1), case["friction"])
        x += case["tread_m"]
    platform_len = case.get("platform_len", PLATFORM_LEN)
    terrain += box("platform", x + platform_len / 2, platform_len / 2,
                   rise * steps, 4.0)
    head, tail = ROBOT.read_text().split("<worldbody>", 1)
    return head + "<worldbody>\n" + terrain + tail


def rocker_profile(elapsed: float, amplitude_deg: float, order: str) -> np.ndarray:
    sign = 1.0 if order == "SP" else -1.0
    phase = elapsed % PERIOD_S
    if elapsed >= PERIOD_S * CYCLES:
        angle = 0.0
    elif phase < 1.0:
        angle = sign * amplitude_deg * phase
    elif phase < 2.0:
        angle = sign * amplitude_deg
    elif phase < 4.0:
        angle = sign * amplitude_deg * (1.0 - (phase - 2.0))
    elif phase < 5.0:
        angle = -sign * amplitude_deg
    elif phase < 6.0:
        angle = -sign * amplitude_deg * (2.0 - phase)
    else:
        angle = 0.0
    return math.radians(angle) * np.ones(2)


def wheel_forces(m, d, c, geoms: set[int]) -> np.ndarray:
    forces = np.zeros(6)
    raw = np.zeros(6)
    for i in range(d.ncon):
        ct = d.contact[i]
        if int(ct.efc_address) < 0:
            continue
        a, b = int(ct.geom1), int(ct.geom2)
        if a in c.wheel_geom:
            wheel, terrain = a, b
        elif b in c.wheel_geom:
            wheel, terrain = b, a
        else:
            continue
        if terrain not in geoms:
            continue
        mujoco.mj_contactForce(m, d, i, raw)
        forces[c.wheel_geom[wheel]] += max(0.0, float(raw[0]))
    return forces


def scan_contacts(m, d, c, terrain_ids: set[int]):
    """Return penetration and non-wheel structure support on the terrain."""
    raw = np.zeros(6)
    depth, pair = 0.0, ""
    structure = {}
    for i in range(d.ncon):
        ct = d.contact[i]
        a, b = int(ct.geom1), int(ct.geom2)
        a_world = int(m.geom_bodyid[a]) == 0
        if a_world == (int(m.geom_bodyid[b]) == 0):
            continue
        terrain, robot = (a, b) if a_world else (b, a)
        if float(ct.dist) < 0.0 and int(ct.efc_address) >= 0:
            mujoco.mj_contactForce(m, d, i, raw)
            if raw[0] > 0.0 and -float(ct.dist) > depth:
                depth = -float(ct.dist)
                na = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a) or str(a)
                nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b) or str(b)
                pair = f"{na}|{nb}"
        if int(ct.efc_address) < 0 or terrain not in terrain_ids or robot in c.wheel_geom:
            continue
        mujoco.mj_contactForce(m, d, i, raw)
        if raw[0] <= STRUCTURE_N:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, terrain) or str(terrain)
        structure[name] = structure.get(name, 0.0) + float(raw[0])
    return depth, pair, structure


def stem_for(case: dict) -> str:
    rise = case.get("rise_m", STEP_RISE)
    speed = case.get("speed", SPEED)
    trigger_kind = case.get("trigger", "contact")
    dt = case.get("dt", DT)
    stem = f"{case['friction']}_{case['angle_deg']:g}_{case['order']}_{case['rocker_amp_deg']:g}"
    if "speed" in case or "trigger" in case:
        stem += f"_v{speed:g}_{trigger_kind}"
    if "dt" in case:
        stem += f"_dt{dt * 1e6:g}us"
    if "rise_m" in case:
        stem += f"_rise{rise * 100:g}cm"
    if case.get("rocker_mode", "stock") == "boost":
        stem += "_boost"
    return stem


def run(case: dict) -> dict:
    RUN.mkdir(parents=True, exist_ok=True)
    steps = case.get("steps", STEP_COUNT)
    rise = case.get("rise_m", STEP_RISE)
    rocker_mode = case.get("rocker_mode", "stock")
    speed = case.get("speed", SPEED)
    trigger_kind = case.get("trigger", "contact")
    dt = case.get("dt", DT)
    control_every = max(1, int(round(DT / dt)))
    save_every = max(1, int(round(SAVE_EVERY * DT / dt)))
    stem = stem_for(case)
    path = RUN / f"{stem}.xml"
    path.write_text(scene_text(case))
    m = mujoco.MjModel.from_xml_path(str(path))
    m.opt.timestep = dt
    if rocker_mode == "boost":
        # Same 160 N.m patch the game applies: the joint limit must allow the
        # effort before the boosted loop can use it.
        for j in range(m.njnt):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("rocker_joint"):
                m.jnt_actfrcrange[j] = (-BOOST_EFFORT, BOOST_EFFORT)
        for a in range(m.nu):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").endswith("rocker_joint_act"):
                m.actuator_ctrlrange[a] = (-BOOST_EFFORT, BOOST_EFFORT)
    d = common.initialize(m, APPROACH_M)
    c = common.Controller(m, d)
    # Batch runs of the boosted loop skip the audit: they exceed the 100 N.m
    # cap on purpose, so every step would be flagged.
    audit = Audit(m) if case.get("audit", True) else None

    wheel_roles = [row[4] for row in c.wheels]
    front = [i for i in range(6) if wheel_roles[i] == 0]
    rear = [i for i in range(6) if wheel_roles[i] == 2]
    rocker_gain, rocker_cap, rocker_feedforward, rocker_effort = (
        (12.0, 1.57, 78.0, BOOST_EFFORT) if rocker_mode == "boost"
        else (4.0, 0.7, 0.0, 100.0))
    stair_ids = {g for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("up")}
    terrain_ids = stair_ids | {g for g in range(m.ngeom)
                               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == "platform"}
    platform_ids = {g for g in range(m.ngeom)
                    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == "platform"}
    platform_len = case.get("platform_len", PLATFORM_LEN)
    platform_start = RISER_X + steps * case["tread_m"]
    platform_end = platform_start + platform_len
    platform_top = rise * steps
    # Park centered: braking on the first six-wheel sample leaves the rear
    # wheels on the platform edge, where the 2 m platform cannot hold them.
    park_x = platform_start + platform_len / 2

    timeout = case["timeout_s"]
    cap = int(round((timeout + 240.0) / dt))
    keep = cap // save_every + 2
    time_series = np.zeros(keep)
    qpos = np.zeros((keep, m.nq))
    qvel = np.zeros((keep, m.nv))
    saved = 0
    time_series[0] = d.time
    qpos[0] = d.qpos
    qvel[0] = d.qvel

    first_contact = None
    trigger = None
    stall_since = None
    stall_ref = None
    support_start = None
    brake_start = None
    settle_start = None
    peak = np.zeros(6)
    penetration_max = (0.0, "", 0.0)
    friction_seen = {}
    structure_max = 0.0
    structure_pair = ""
    structure_since = {}
    structure_flagged = None
    steps_climbed = 0
    steps_climbed_front = 0
    settled_steps = 0
    support_steps = 0
    hold_until = None
    judged = False
    judge_counts = None
    x_history = [(float(d.time), float(d.qpos[0]))]
    extensions = 0
    reason = "timeout"
    settle_ok = False
    max_y = 0.0

    for step in range(cap):
        if step % control_every == 0:
            elapsed = 0.0 if trigger is None else float(d.time) - trigger
            if brake_start is None:
                rocker = rocker_profile(elapsed, case["rocker_amp_deg"], case["order"])
                if trigger is None:
                    rocker[:] = 0.0
                goal = c.update(np.full(3, speed))
            else:
                rocker = rocker_profile(0.0, 0.0, case["order"])
                goal = c.update(np.zeros(3))
            c.rocker_commands[:] = rocker
            for i, (_, act, pos, vel) in enumerate(c.rockers):
                if brake_start is not None:
                    # Passive suspension once parked: the rigid wheels differ in
                    # radius by up to 0.16 mm, so a pinned rocker leaves one wheel
                    # unloaded on a flat platform.
                    d.ctrl[act] = 0.0
                    continue
                error = rocker[i] - float(d.qpos[pos])
                desired = float(np.clip(rocker_gain * error, -rocker_cap, rocker_cap))
                feedforward = rocker_feedforward * float(np.clip(error / 0.05, 0.0, 1.0))
                d.ctrl[act] = np.clip(150.0 * (desired - float(d.qvel[vel])) + feedforward,
                                      -rocker_effort, rocker_effort)

        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            reason = "nonfinite"
            break
        if audit is not None:
            audit.check(d, float(d.time))
        max_y = max(max_y, abs(float(d.qpos[1])))

        if first_contact is None:
            for i in range(d.ncon):
                ct = d.contact[i]
                if int(ct.efc_address) < 0:
                    continue
                a, b = int(ct.geom1), int(ct.geom2)
                if ((a in c.wheel_geom and wheel_roles[c.wheel_geom[a]] == 0 and b in stair_ids)
                        or (b in c.wheel_geom and wheel_roles[c.wheel_geom[b]] == 0 and a in stair_ids)):
                    first_contact = float(d.time)
                    break

        if first_contact is not None and trigger is None:
            if trigger_kind == "contact":
                trigger = first_contact
            else:
                # stall: front wheels contacted but net forward progress
                # stayed under 20 mm for a full 2 s window
                x_now = float(d.qpos[0])
                if stall_since is None:
                    stall_since, stall_ref = float(d.time), x_now
                elif x_now - stall_ref >= 0.02:
                    stall_since, stall_ref = float(d.time), x_now
                elif float(d.time) - stall_since >= 2.0:
                    trigger = float(d.time)

        forces = wheel_forces(m, d, c, terrain_ids)
        peak = np.maximum(peak, forces)
        depth, pair, structure = scan_contacts(m, d, c, terrain_ids)
        if depth > penetration_max[0]:
            penetration_max = (depth, pair, float(d.time))
        for wi in range(6):
            if forces[wi] <= SUPPORT_N:
                continue
            geom = c.wheels[wi][6]
            height = float(d.geom_xpos[geom, 2]) - float(m.geom_size[geom, 0])
            level = int(round(height / rise))
            # Report the trailing axle: how far the whole rover has climbed.
            # The front axle reaches a step two points of support earlier.
            if wi in rear:
                steps_climbed = max(steps_climbed, level)
            if wi in front:
                steps_climbed_front = max(steps_climbed_front, level)

        if structure:
            for name, _ in structure.items():
                structure_since.setdefault(name, float(d.time))
                span = float(d.time) - structure_since[name]
                if span > structure_max:
                    structure_max, structure_pair = span, name
            for name in list(structure_since):
                if name not in structure:
                    del structure_since[name]
            if structure_max > STRUCTURE_MAX_S and structure_flagged is None:
                structure_flagged = {"pair": structure_pair, "t": float(d.time)}
        else:
            structure_since.clear()

        if step % 100 == 0:
            raw = np.zeros(6)
            for i in range(d.ncon):
                ct = d.contact[i]
                a, b = int(ct.geom1), int(ct.geom2)
                if int(ct.efc_address) < 0:
                    continue
                a_world = int(m.geom_bodyid[a]) == 0
                if a_world == (int(m.geom_bodyid[b]) == 0):
                    continue
                terrain, robot = (a, b) if a_world else (b, a)
                if c.wheel_geom.get(robot) is None or terrain not in terrain_ids:
                    continue
                mujoco.mj_contactForce(m, d, i, raw)
                if raw[0] <= 0.0:
                    continue
                na = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, robot) or str(robot)
                nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, terrain) or str(terrain)
                key = f"{na.split('_col')[0]}|{nb}"
                friction_seen[key] = round(float(ct.friction[0]), 6)

        if step % save_every == 0:
            time_series[saved + 1] = d.time
            qpos[saved + 1] = d.qpos
            qvel[saved + 1] = d.qvel
            saved += 1

        platform_forces = wheel_forces(m, d, c, platform_ids)
        inside, loaded, gap = platform_state(
            m, d, c, platform_forces, platform_start, platform_end, platform_top)
        support_ok = platform_support(inside, loaded, gap)
        if brake_start is None:
            # Same predicate as the success criterion: braking while a front
            # wheel already hangs on the far edge is not braking on the platform.
            if support_ok and d.qpos[0] >= park_x:
                brake_start = float(d.time)
                support_start = brake_start
                settle_start = brake_start
                support_steps = 0
        else:
            support_steps = support_steps + 1 if support_ok else 0
            settled_steps = settled_steps + 1 if _settled(m, d, c) else 0
            if not judged and d.time - settle_start >= BRAKE_S - 1e-9:
                judged = True
                judge_counts = (support_steps, settled_steps, inside, loaded,
                                gap * 1000.0, _settle_margins(m, d, c))
                settle_ok = (support_steps >= int(round(1.0 / dt))
                             and settled_steps >= int(round(1.0 / dt)))
                if not settle_ok:
                    reason = "platform_not_settled"
                    break
                hold_until = float(d.time) + BRAKE_S
            if hold_until is not None and d.time >= hold_until - 1e-9:
                reason = "platform_settled"
                break

        roll, pitch, yaw = rpy(d.qpos[3:7])
        if abs(math.degrees(roll)) > ROLLOVER_DEG or abs(math.degrees(pitch)) > ROLLOVER_DEG:
            reason = "rollover"
            break
        if max_y > LATERAL_LIMIT_M:
            reason = "left_lane"
            break
        if d.qpos[0] > platform_end + 1.0:
            reason = "passed_platform"
            break
        if step % 1000 == 0:
            x_history.append((float(d.time), float(d.qpos[0])))
            x_history = [row for row in x_history if row[0] > d.time - 60.0]
        if d.time >= timeout:
            past = [row for row in x_history if row[0] <= d.time - 30.0]
            moved = float(d.qpos[0]) - past[-1][1] if past else 0.0
            if support_start is None and moved > 0.05 and extensions < 2:
                timeout += 60.0
                extensions += 1
            else:
                reason = "timeout"
                break

    n = saved + 1
    np.savez_compressed(
        RUN / f"{stem}_trajectory.npz",
        time=time_series[:n], qpos=qpos[:n], qvel=qvel[:n],
        simulation_dt=dt, save_every=save_every, control_every=control_every)

    audit_summary = (audit.summary() if audit is not None
                     else {"compliant": None, "first_violations": [], "violation_kinds": {}})
    numerical_ok = (reason != "nonfinite" and structure_flagged is None)
    result = {
        "run_id": case["run_id"],
        "phase": case.get("phase", "B"),
        "control_id": {"speed_rad_s": speed, "trigger": trigger_kind, "order": case["order"],
                       "rocker_amp_deg": case["rocker_amp_deg"], "rocker_period_s": PERIOD_S,
                       "rocker_cycles": CYCLES, "steer_law": "heading",
                       "brake_law": "zero_hold_3s", "timeout_s": case["timeout_s"],
                       "rocker_mode": rocker_mode},
        "geometry": {"height_m": rise, "rise_cm": rise * 100, "steps": steps,
                     "angle_deg": case["angle_deg"], "tread_m": case["tread_m"],
                     "platform_len_m": platform_len, "width_m": 2 * WIDTH_HALF},
        "model": {
            "urdf_sha256": common.digest(URDF),
            "mjcf_sha256": common.digest(ROBOT),
            "model_spec_sha256": common.digest(MODEL_SPEC),
            "scene_sha256": common.digest(path),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "friction": {"ground": 4.0, "stair": case["friction"], "platform": 4.0,
                     "wheel_field": 0.6, "runtime_pairs": friction_seen},
        "result": {"ascent_completed": bool(settle_ok),
                   "urdf_compliant": audit_summary["compliant"],
                   "numerical_quality_ok": numerical_ok,
                   "steps_climbed": steps_climbed,
                   "steps_climbed_front": steps_climbed_front,
                   "end_reason": reason},
        "violations": audit_summary["first_violations"],
        "violation_counts": audit_summary["violation_kinds"],
        "penetration": {"max_mm": penetration_max[0] * 1000.0,
                        "pair": penetration_max[1], "t": penetration_max[2]},
        "structure_support": {"max_s": structure_max, "pair": structure_pair,
                              "flagged": structure_flagged},
        "events": {"first_front_contact_s": first_contact, "trigger_s": trigger,
                   "platform_support_s": support_start, "brake_s": brake_start,
                   "settle_check_s": settle_start},
        "peak_wheel_force_n": {c.names[row[0]]: float(peak[i]) for i, row in enumerate(c.wheels)},
        "duration_s": float(d.time),
        "budget_s": case["timeout_s"],
        "settle_check": None if judge_counts is None else {
            "support_steps": judge_counts[0], "settled_steps": judge_counts[1],
            "all_inside_footprint": judge_counts[2], "loaded_wheels": judge_counts[3],
            "unloaded_gap_mm": judge_counts[4], "margins": judge_counts[5]},
        "success_criterion": {
            "platform_footprint_m": [platform_start, platform_end],
            "loaded_wheels_min": 5, "loaded_force_n": SUPPORT_N,
            "unloaded_gap_mm_max": UNLOADED_GAP_M * 1000.0,
            "hold_s": 1.0, "settle_s": 1.0,
        },
        "extensions": extensions,
        "final_base_xyz_m": [float(v) for v in d.qpos[:3]],
    }
    return result


def platform_state(m, d, c, forces, x0: float, x1: float, top: float):
    """Return (all six centres inside the platform footprint, loaded count,
    largest gap of a wheel the platform does not load).

    Force on the platform geom alone accepts a wheel hanging on the far edge,
    so the footprint check is what makes "on the platform" mean position.
    """
    inside = True
    gaps = []
    for wi in range(6):
        geom = c.wheels[wi][6]
        cx = float(d.geom_xpos[geom, 0])
        cy = float(d.geom_xpos[geom, 1])
        bottom = float(d.geom_xpos[geom, 2]) - float(m.geom_size[geom, 0])
        inside = inside and x0 <= cx <= x1 and abs(cy) <= WIDTH_HALF
        if forces[wi] <= SUPPORT_N:
            gaps.append(bottom - top)
    loaded = int(sum(1 for wi in range(6) if forces[wi] > SUPPORT_N))
    return inside, loaded, (max(gaps) if gaps else 0.0)


def platform_support(inside: bool, loaded: int, unloaded_gap: float) -> bool:
    return inside and loaded >= 5 and unloaded_gap <= UNLOADED_GAP_M


def _settle_margins(m, d, c) -> dict:
    """Reported when a six-wheel platform stop misses a settle threshold."""
    roll, pitch, yaw = rpy(d.qpos[3:7])
    wheel_dofs = [row[3] for row in c.wheels]
    other_dofs = [row[3] for row in c.steer] + [row[3] for row in c.rockers]
    return {
        "roll_deg": abs(math.degrees(roll)), "pitch_deg": abs(math.degrees(pitch)),
        "heading_deg": abs(math.degrees(yaw)),
        "linear_m_s": float(np.linalg.norm(d.qvel[:3])),
        "angular_rad_s": float(np.linalg.norm(d.qvel[3:6])),
        "wheel_rad_s": max(abs(float(d.qvel[i])) for i in wheel_dofs),
        "joint_rad_s": max(abs(float(d.qvel[i])) for i in other_dofs),
    }


def _settled(m, d, c) -> bool:
    roll, pitch, yaw = rpy(d.qpos[3:7])
    if abs(math.degrees(roll)) > SETTLE["roll_deg"] or abs(math.degrees(pitch)) > SETTLE["pitch_deg"]:
        return False
    if abs(math.degrees(yaw)) > SETTLE["heading_deg"]:
        return False
    if float(np.linalg.norm(d.qvel[:3])) > SETTLE["linear_m_s"]:
        return False
    if float(np.linalg.norm(d.qvel[3:6])) > SETTLE["angular_rad_s"]:
        return False
    wheel_dofs = [row[3] for row in c.wheels]
    if max(abs(float(d.qvel[i])) for i in wheel_dofs) > SETTLE["wheel_rad_s"]:
        return False
    other_dofs = [row[3] for row in c.steer] + [row[3] for row in c.rockers]
    if max(abs(float(d.qvel[i])) for i in other_dofs) > SETTLE["joint_rad_s"]:
        return False
    return True
