"""P1: run the symmetric URDF model on the experiment-one 45 deg stair and
compare the contact-free trajectory with ground_mu6_v6_rocker_SP."""
from __future__ import annotations

import argparse
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

RUN = ROOT / "runs/urdf_slope_20261112/p1"
SCENE = RUN / "scene_mu6.xml"
REFERENCE = ROOT / "runs/final_observation_120s_20261112/cases/ground_mu6_v6_rocker_SP"
DT = 0.001
LIMIT_NO_CONTACT = 20.0
AFTER_CONTACT = 12.0
STALL_WINDOW = 2.0
STALL_PROGRESS = 0.02
SPEED = 6.0
ROCKER_AMPLITUDE_DEG = 18.0
CONTACT_TOL = {
    "progress_m": 0.010,
    "lateral_m": 0.010,
    "height_m": 0.010,
    "roll_deg": 1.0,
    "pitch_deg": 1.0,
    "yaw_deg": 1.0,
    "steer_rocker_deg": 1.0,
    "wheel_speed_rad_s": 0.5,
    "wheel_speed_from_s": 0.2,
}


def rocker_targets(elapsed: float) -> np.ndarray:
    phase = elapsed % 8.0
    if elapsed >= 24.0:
        angle = 0.0
    elif phase < 1.0:
        angle = ROCKER_AMPLITUDE_DEG * phase
    elif phase < 2.0:
        angle = ROCKER_AMPLITUDE_DEG
    elif phase < 4.0:
        angle = ROCKER_AMPLITUDE_DEG - ROCKER_AMPLITUDE_DEG * (phase - 2.0)
    elif phase < 5.0:
        angle = -ROCKER_AMPLITUDE_DEG
    elif phase < 6.0:
        angle = -ROCKER_AMPLITUDE_DEG + ROCKER_AMPLITUDE_DEG * (phase - 5.0)
    else:
        angle = 0.0
    return math.radians(angle) * np.ones(2)


def build_scene() -> None:
    if SCENE.exists():
        return
    source = ROOT / "runs/friction20_t22_20261111/scenes/scene_mu6.xml"
    p0 = ROOT / "runs/urdf_slope_20261112/model/rover_p0.xml"
    terrain = source.read_text().split("<worldbody>", 1)[1].split(
        '<body name="base_link">', 1)[0]
    head, tail = p0.read_text().split("<worldbody>", 1)
    RUN.mkdir(parents=True, exist_ok=True)
    SCENE.write_text(head + "<worldbody>\n" + terrain + tail)


def run() -> dict:
    build_scene()
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = DT
    d = common.initialize(m, 8.0)
    c = common.Controller(m, d)

    initial = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, initial, mujoco.mjtState.mjSTATE_INTEGRATION)

    cap = round((LIMIT_NO_CONTACT + AFTER_CONTACT + 1.0) / DT)
    time_series = np.zeros(cap + 1)
    qpos = np.zeros((cap + 1, m.nq))
    qvel = np.zeros((cap + 1, m.nv))
    ctrl = np.zeros((cap, m.nu))
    wheel_targets = np.zeros((cap, 6))
    rocker_commands = np.zeros((cap, 2))
    contact_force = np.zeros((cap, 6))
    time_series[0] = d.time
    qpos[0] = d.qpos
    qvel[0] = d.qvel

    wheels = {row[-1]: row[4] for row in c.wheels}
    stair_ids = {
        i for i in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("up")
    }
    force = np.zeros(6)
    first_contact = None
    trigger = None
    reason = "no_contact_within_limit"

    for step in range(cap):
        elapsed = 0.0 if trigger is None else float(d.time) - trigger
        rocker = rocker_targets(elapsed)
        if trigger is None:
            rocker[:] = 0.0
        goal = c.update(np.full(3, SPEED))
        c.rocker_commands[:] = rocker
        for i, (_, act, pos, vel) in enumerate(c.rockers):
            desired = float(np.clip(4.0 * (rocker[i] - float(d.qpos[pos])), -0.7, 0.7))
            d.ctrl[act] = np.clip(150.0 * (desired - float(d.qvel[vel])), -100.0, 100.0)

        ctrl[step] = d.ctrl
        wheel_targets[step] = goal
        rocker_commands[step] = rocker
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("nonfinite simulation state")

        if first_contact is None:
            for i in range(d.ncon):
                con = d.contact[i]
                g1, g2 = int(con.geom1), int(con.geom2)
                if con.efc_address < 0:
                    continue
                if ((wheels.get(g1) == 0 and g2 in stair_ids)
                        or (wheels.get(g2) == 0 and g1 in stair_ids)):
                    mujoco.mj_contactForce(m, d, i, force)
                    if force[0] > 0.0:
                        first_contact = float(d.time)
                        break

        _, _, _, support = common.contacts(m, d, c)
        for wi, terrains in enumerate(support):
            contact_force[step, wi] = len([n for n in terrains if n.startswith("up")])

        time_series[step + 1] = d.time
        qpos[step + 1] = d.qpos
        qvel[step + 1] = d.qvel

        if (trigger is None and first_contact is not None
                and d.time - first_contact >= STALL_WINDOW - 1e-9):
            back = step + 1 - round(STALL_WINDOW / DT)
            if d.qpos[0] - qpos[back, 0] < STALL_PROGRESS:
                trigger = float(d.time)

        if first_contact is not None and d.time - first_contact >= AFTER_CONTACT - 1e-9:
            reason = "post_contact_12s"
            break
        if trigger is not None and d.time - trigger >= AFTER_CONTACT - 1e-9:
            reason = "post_trigger_12s"
            break

    n = step + 1
    arrays = {
        "time": time_series[:n + 1],
        "qpos": qpos[:n + 1],
        "qvel": qvel[:n + 1],
        "ctrl": ctrl[:n],
        "wheel_targets": wheel_targets[:n],
        "rocker_commands": rocker_commands[:n],
        "wheel_stair_support_count": contact_force[:n],
    }
    np.savez_compressed(RUN / "trajectory.npz", **arrays,
                        initial_integration_state=initial, simulation_dt=DT,
                        scene_sha256=common.digest(SCENE))

    return {
        "first_contact_s": first_contact,
        "stall_trigger_s": trigger,
        "duration_s": float(d.time),
        "termination_reason": reason,
        "wheel_names": [c.names[row[0]] for row in c.wheels],
    }


def rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ])


def compare(new: dict) -> dict:
    reference = json.loads((REFERENCE / "result.json").read_text())
    ref_first = reference["first_contact_s"]
    new_first = new["first_contact_s"]
    if new_first is None:
        return {"reference_first_contact_s": ref_first, "new_first_contact_s": None,
                "contact_detected": False}

    with np.load(REFERENCE / "trajectory.npz", allow_pickle=False) as a:
        ref_time = a["time"]
        ref_qpos = a["qpos"]
        ref_qvel = a["qvel"]
    with np.load(RUN / "trajectory.npz", allow_pickle=False) as a:
        new_time = a["time"]
        new_qpos = a["qpos"]
        new_qvel = a["qvel"]

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    steer_rocker_qpos, wheel_qpos, wheel_dof = [], [], []
    for joint in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if name.endswith("_wheel_joint"):
            wheel_qpos.append(int(m.jnt_qposadr[joint]))
            wheel_dof.append(int(m.jnt_dofadr[joint]))
        elif name.endswith(("_steer_joint", "_rocker_joint")):
            steer_rocker_qpos.append(int(m.jnt_qposadr[joint]))

    end = min(new_first, ref_first) - DT
    count = min(len(ref_time), len(new_time), int(round(end / DT)) + 1)
    ref_rows, new_rows = ref_qpos[:count], new_qpos[:count]

    progress = new_rows[:, 0] - new_rows[0, 0]
    ref_progress = ref_rows[:, 0] - ref_rows[0, 0]
    lateral = new_rows[:, 1] - new_rows[0, 1]
    ref_lateral = ref_rows[:, 1] - ref_rows[0, 1]
    height = new_rows[:, 2] - new_rows[0, 2]
    ref_height = ref_rows[:, 2] - ref_rows[0, 2]

    ref_rpy = np.array([rpy(q) for q in ref_rows[:, 3:7]])
    new_rpy = np.array([rpy(q) for q in new_rows[:, 3:7]])

    channel_values = {
        "progress_m": np.abs(progress - ref_progress),
        "lateral_m": np.abs(lateral - ref_lateral),
        "height_m": np.abs(height - ref_height),
        "roll_deg": np.abs(np.degrees(new_rpy[:, 0] - ref_rpy[:, 0])),
        "pitch_deg": np.abs(np.degrees(new_rpy[:, 1] - ref_rpy[:, 1])),
        "yaw_deg": np.abs(np.degrees(new_rpy[:, 2] - ref_rpy[:, 2])),
        "steer_rocker_deg": np.degrees(np.max(np.abs(
            new_rows[:, steer_rocker_qpos] - ref_rows[:, steer_rocker_qpos]), axis=1)),
    }
    wheel_angle = np.degrees(np.max(np.abs(
        new_rows[:, wheel_qpos] - ref_rows[:, wheel_qpos]), axis=1))

    moving = new_time[:count] >= CONTACT_TOL["wheel_speed_from_s"]
    wheel_speed = np.abs(
        new_qvel[:count][moving][:, wheel_dof]
        - ref_qvel[:count][moving][:, wheel_dof]).max(axis=1)
    channel_values["wheel_speed_rad_s"] = wheel_speed
    channel_time = {"wheel_speed_rad_s": new_time[:count][moving]}

    summary = {}
    for name, values in channel_values.items():
        index = int(np.argmax(values))
        times = channel_time.get(name, new_time[:count])
        summary[name] = {
            "max": float(values[index]),
            "at_s": float(times[index]),
            "tolerance": CONTACT_TOL[name],
            "pass": bool(values[index] <= CONTACT_TOL[name]),
        }

    diverged = not all(item["pass"] for item in summary.values())

    ref_zero = int(round(ref_first / DT))
    new_zero = int(round(new_first / DT))
    post_count = int(round(AFTER_CONTACT / DT)) + 1
    ref_post = ref_qpos[ref_zero:ref_zero + post_count]
    new_post = new_qpos[new_zero:new_zero + post_count]
    post_count = min(len(ref_post), len(new_post))
    ref_post, new_post = ref_post[:post_count], new_post[:post_count]

    return {
        "reference_first_contact_s": ref_first,
        "new_first_contact_s": new_first,
        "contact_time_delta_s": new_first - ref_first,
        "compared_samples": count,
        "compared_until_s": float(end),
        "pre_contact": summary,
        "wheel_angle_deg_informational": {
            "max": float(wheel_angle.max()),
            "at_s": float(new_time[int(np.argmax(wheel_angle))]),
            "note": "accumulated wheel angle, not a gate; wheel radii differ by up to 0.16 mm",
        },
        "pre_contact_diverged": diverged,
        "post_contact_informational": {
            "max_abs_dx_m": float(np.abs(new_post[:, 0] - ref_post[:, 0]).max()),
            "max_abs_dy_m": float(np.abs(new_post[:, 1] - ref_post[:, 1]).max()),
            "max_abs_dz_m": float(np.abs(new_post[:, 2] - ref_post[:, 2]).max()),
            "final_new_xyz_m": [float(v) for v in new_post[-1, :3]],
            "final_reference_xyz_m": [float(v) for v in ref_post[-1, :3]],
            "note": "rocker amplitude is 18 deg here and 20 deg in the reference",
        },
        "reference_new_final_time_s": [float(ref_time[-1]), float(new_time[-1])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (RUN / "result.json").exists() and not args.force:
        raise SystemExit(f"result already exists: {RUN/'result.json'}")
    started = time.monotonic()
    result = run()
    comparison = compare(result)
    result["comparison"] = comparison
    result["wall_seconds"] = time.monotonic() - started
    (RUN / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
