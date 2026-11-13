"""P2: single- and two-step mount diagnosis for the URDF-constrained model.

Each case builds a 0.20 m riser with one or two treads, drives the rover at a
fixed wheel speed, and starts the rocker waveform either at first front-wheel
contact or after a stall. A case counts as mounted when both front and both
middle wheels carry more than 5 N on the target level for 1 s and the rover
then advances 0.5 m without rolling over.
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
import urdf_slope_model as p0  # noqa: E402

RUN = ROOT / "runs/urdf_slope_20261112/p2"
ROBOT = ROOT / "runs/urdf_slope_20261112/model/rover_p0.xml"
DT = 0.001
STEP_RISE = 0.20
APPROACH_M = 8.0
WIDTH_HALF = 2.5
PLATFORM_HALF = 4.0  # long enough that the far edge cannot cut the 1 s hold short
RISER_X = 1.6
LIMIT_S = 90.0
SUPPORT_N = 5.0
SUPPORT_HOLD_S = 1.0
ADVANCE_M = 0.5
STALL_WINDOW = 2.0
STALL_PROGRESS = 0.02
ROLLOVER_DEG = 60.0

CASES = [
    {"id": "s1_t0746_3rps_contact_18", "steps": 1, "tread_m": 0.746410,
     "speed_rad_s": 3.0, "trigger": "contact", "rocker_amp_deg": 18.0},
    {"id": "s1_t0746_3rps_contact_20", "steps": 1, "tread_m": 0.746410,
     "speed_rad_s": 3.0, "trigger": "contact", "rocker_amp_deg": 20.0},
    {"id": "s1_t0746_12rps_stall_18", "steps": 1, "tread_m": 0.746410,
     "speed_rad_s": 12.0, "trigger": "stall", "rocker_amp_deg": 18.0},
    {"id": "s1_t0746_12rps_stall_20", "steps": 1, "tread_m": 0.746410,
     "speed_rad_s": 12.0, "trigger": "stall", "rocker_amp_deg": 20.0},
    {"id": "s1_t0200_3rps_contact_18", "steps": 1, "tread_m": 0.200000,
     "speed_rad_s": 3.0, "trigger": "contact", "rocker_amp_deg": 18.0},
    {"id": "s1_t0200_3rps_contact_20", "steps": 1, "tread_m": 0.200000,
     "speed_rad_s": 3.0, "trigger": "contact", "rocker_amp_deg": 20.0},
    {"id": "s2_t0746_12rps_stall_20", "steps": 2, "tread_m": 0.746410,
     "speed_rad_s": 12.0, "trigger": "stall", "rocker_amp_deg": 20.0},
    # Added because the stall trigger never fires at 12 rad/s on a short stair,
    # so the stall rows above never exercise the rocker profile at all.
    {"id": "s1_t0746_12rps_contact_18", "steps": 1, "tread_m": 0.746410,
     "speed_rad_s": 12.0, "trigger": "contact", "rocker_amp_deg": 18.0},
    {"id": "s1_t0746_12rps_contact_20", "steps": 1, "tread_m": 0.746410,
     "speed_rad_s": 12.0, "trigger": "contact", "rocker_amp_deg": 20.0},
    {"id": "s2_t0746_12rps_contact_20", "steps": 2, "tread_m": 0.746410,
     "speed_rad_s": 12.0, "trigger": "contact", "rocker_amp_deg": 20.0},
]
FRICTION = 6


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
    for i in range(case["steps"]):
        top = STEP_RISE * (i + 1)
        terrain += box(f"up{i}", x + case["tread_m"] / 2, case["tread_m"] / 2,
                       top, FRICTION)
        x += case["tread_m"]
    terrain += box("platform", x + PLATFORM_HALF, PLATFORM_HALF,
                   STEP_RISE * case["steps"], 4.0)
    head, tail = ROBOT.read_text().split("<worldbody>", 1)
    return head + "<worldbody>\n" + terrain + tail


def scene_path(case: dict) -> Path:
    return RUN / f"{case['id']}.xml"


def rocker_targets(elapsed: float, amplitude_deg: float) -> np.ndarray:
    phase = elapsed % 8.0
    if elapsed >= 24.0:
        angle = 0.0
    elif phase < 1.0:
        angle = amplitude_deg * phase
    elif phase < 2.0:
        angle = amplitude_deg
    elif phase < 4.0:
        angle = amplitude_deg - amplitude_deg * (phase - 2.0)
    elif phase < 5.0:
        angle = -amplitude_deg
    elif phase < 6.0:
        angle = -amplitude_deg + amplitude_deg * (phase - 5.0)
    else:
        angle = 0.0
    return math.radians(angle) * np.ones(2)


def rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ])


class Audit:
    """Strict per-step URDF limit check; tolerance is zero."""

    def __init__(self, m):
        joints = {j["name"]: j for j in p0.parse_urdf()["joints"]}
        self.rows = []
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name not in joints:
                continue
            limit = joints[name]
            self.rows.append({
                "name": name, "qpos": int(m.jnt_qposadr[j]),
                "dof": int(m.jnt_dofadr[j]),
                "lower": limit["lower"], "upper": limit["upper"],
                "velocity": limit["velocity"], "effort": limit["effort"],
            })
        self.first = {}
        self.counts = {}

    def check(self, d, t: float) -> None:
        for row in self.rows:
            value = float(d.qpos[row["qpos"]])
            speed = abs(float(d.qvel[row["dof"]]))
            effort = abs(float(d.qfrc_actuator[row["dof"]]))
            for kind, bound, measured in (
                    ("position_lower", row["lower"], value),
                    ("position_upper", row["upper"], value),
                    ("velocity", row["velocity"], speed),
                    ("effort", row["effort"], effort)):
                if bound is None:
                    continue
                over = (bound - measured) if kind == "position_lower" else (measured - bound)
                if over <= 0.0:
                    continue
                key = f"{kind}:{row['name']}"
                self.counts[key] = self.counts.get(key, 0) + 1
                self.first.setdefault(key, {
                    "joint": row["name"], "kind": kind, "measured": measured,
                    "limit": bound, "qpos": value, "t": t,
                })

    def summary(self) -> dict:
        return {
            "violation_kinds": self.counts,
            "first_violations": list(self.first.values()),
            "compliant": not self.counts,
        }


def wheel_force(m, d, c, geoms: set[int]) -> np.ndarray:
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


def penetration(m, d) -> tuple[float, str]:
    raw = np.zeros(6)
    depth, pair = 0.0, ""
    for i in range(d.ncon):
        ct = d.contact[i]
        a, b = int(ct.geom1), int(ct.geom2)
        a_world = int(m.geom_bodyid[a]) == 0
        b_world = int(m.geom_bodyid[b]) == 0
        if a_world == b_world or float(ct.dist) >= 0.0:
            continue
        mujoco.mj_contactForce(m, d, i, raw)
        if raw[0] <= 0.0:
            continue
        if -float(ct.dist) > depth:
            depth = -float(ct.dist)
            na = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a) or str(a)
            nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b) or str(b)
            pair = f"{na}|{nb}"
    return depth, pair


def run(case: dict, physics_dt: float = DT) -> dict:
    RUN.mkdir(parents=True, exist_ok=True)
    scene_path(case).write_text(scene_text(case))
    m = mujoco.MjModel.from_xml_path(str(scene_path(case)))
    substeps = max(1, int(round(DT / physics_dt)))
    m.opt.timestep = physics_dt
    d = common.initialize(m, APPROACH_M)
    c = common.Controller(m, d)

    wheel_roles = [row[4] for row in c.wheels]
    wheel_sides = [row[5] for row in c.wheels]
    front_middle = [i for i in range(6) if wheel_roles[i] in (0, 1)]
    stair_ids = {g for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("up")}
    top_name = f"up{case['steps'] - 1}"
    level_ids = {g for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") in (top_name, "platform")}
    platform_end = RISER_X + case["steps"] * case["tread_m"] + 2 * PLATFORM_HALF
    initial_z = float(d.qpos[2])
    audit = Audit(m)

    cap = int(round(LIMIT_S / DT))
    time_series = np.zeros(cap + 1)
    qpos = np.zeros((cap + 1, m.nq))
    qvel = np.zeros((cap + 1, m.nv))
    support = np.zeros((cap, 6))
    penetration_mm = np.zeros(cap)
    time_series[0] = d.time
    qpos[0] = d.qpos
    qvel[0] = d.qvel

    first_contact = None
    trigger = None
    hold_start = None
    advance_start_x = None
    advance_s = None
    support_peak = np.zeros(6)
    penetration_max = (0.0, "", 0.0)
    best_hold = 0.0
    best_hold_at = None
    max_base_z = initial_z
    reason = "time_limit"

    for step in range(cap):
        elapsed = 0.0 if trigger is None else float(d.time) - trigger
        rocker = rocker_targets(elapsed, case["rocker_amp_deg"])
        if trigger is None:
            rocker[:] = 0.0
        goal = c.update(np.full(3, case["speed_rad_s"]))
        c.rocker_commands[:] = rocker
        for i, (_, act, pos, vel) in enumerate(c.rockers):
            desired = float(np.clip(4.0 * (rocker[i] - float(d.qpos[pos])), -0.7, 0.7))
            d.ctrl[act] = np.clip(150.0 * (desired - float(d.qvel[vel])), -100.0, 100.0)

        for _ in range(substeps):
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            reason = "nonfinite"
            break

        audit.check(d, float(d.time))

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

        forces = wheel_force(m, d, c, level_ids)
        support[step] = forces
        support_peak = np.maximum(support_peak, forces)
        depth, pair = penetration(m, d)
        penetration_mm[step] = depth * 1000.0
        if depth > penetration_max[0]:
            penetration_max = (depth, pair, float(d.time))

        time_series[step + 1] = d.time
        qpos[step + 1] = d.qpos
        qvel[step + 1] = d.qvel

        max_base_z = max(max_base_z, float(d.qpos[2]))
        roll, pitch, _ = rpy(d.qpos[3:7])
        if hold_start is None and all(forces[i] > SUPPORT_N for i in front_middle):
            hold_start = float(d.time)
        elif hold_start is not None and not all(forces[i] > SUPPORT_N for i in front_middle):
            if d.time - hold_start > best_hold:
                best_hold, best_hold_at = float(d.time - hold_start), hold_start
            hold_start = None

        if hold_start is not None and advance_start_x is None \
                and d.time - hold_start >= SUPPORT_HOLD_S - 1e-9:
            advance_start_x = float(d.qpos[0])
            advance_s = float(d.time)

        if advance_start_x is not None and advance_s is not None \
                and d.qpos[0] - advance_start_x >= ADVANCE_M:
            reason = "mounted"
            break

        if abs(math.degrees(roll)) > ROLLOVER_DEG or abs(math.degrees(pitch)) > ROLLOVER_DEG:
            reason = "rollover"
            break

        if d.qpos[0] > platform_end + 1.0:
            reason = "left_platform"
            break

        if (trigger is None and first_contact is not None
                and d.time - first_contact >= STALL_WINDOW - 1e-9):
            back = step + 1 - int(round(STALL_WINDOW / DT))
            if d.qpos[0] - qpos[back, 0] < STALL_PROGRESS:
                trigger = float(d.time)
        if trigger is None and case["trigger"] == "contact" and first_contact is not None:
            trigger = first_contact

        if trigger is not None and d.time - trigger >= 30.0:
            back = step + 1 - int(round(5.0 / DT))
            if d.qpos[0] - qpos[max(back, 0), 0] < 0.01:
                reason = "stall"
                break

    if hold_start is not None and d.time - hold_start > best_hold:
        best_hold, best_hold_at = float(d.time - hold_start), hold_start

    n = step + 1
    np.savez_compressed(
        RUN / f"{case['id']}_trajectory.npz",
        time=time_series[:n + 1], qpos=qpos[:n + 1], qvel=qvel[:n + 1],
        wheel_level_force=support[:n], penetration_mm=penetration_mm[:n],
        simulation_dt=physics_dt, control_dt=DT,
        scene_sha256=common.digest(scene_path(case)))

    return {
        "run_id": f"urdf_slope_20261112/p2/{case['id']}",
        "phase": "P2",
        "case": case,
        "physics_dt_s": physics_dt,
        "control_dt_s": DT,
        "friction": {"stair": FRICTION, "platform": 4.0, "ground": 4.0},
        "wheel_names": [c.names[row[0]] for row in c.wheels],
        "wheel_roles": wheel_roles,
        "wheel_sides_left": wheel_sides,
        "first_contact_s": first_contact,
        "trigger_s": trigger,
        "duration_s": float(d.time),
        "end_reason": reason,
        "mounted": reason == "mounted",
        "support": {
            "front_middle_sustained_1s": advance_start_x is not None,
            "hold_start_s": advance_s,
            "peak_force_n": {c.names[row[0]]: float(support_peak[i])
                             for i, row in enumerate(c.wheels)},
            "advance_from_support_m": None if advance_start_x is None
            else float(d.qpos[0] - advance_start_x),
            "longest_hold_s": best_hold,
            "longest_hold_start_s": best_hold_at,
        },
        "level_reached": bool(max_base_z > initial_z + 0.15),
        "max_base_z_m": max_base_z,
        "initial_base_z_m": initial_z,
        "final_base_xyz_m": [float(v) for v in d.qpos[:3]],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "penetration_mm": {"max": penetration_max[0] * 1000.0,
                           "pair": penetration_max[1], "t": penetration_max[2]},
        "audit": audit.summary(),
        "sample_count": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--physics-dt", type=float, default=DT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = [c for c in CASES if not args.case or c["id"] in args.case]
    if not selected:
        raise SystemExit(f"unknown case; choose from {[c['id'] for c in CASES]}")
    for case in selected:
        suffix = "" if args.physics_dt == DT else f"_dt{args.physics_dt * 1000:g}ms"
        target = RUN / f"{case['id']}{suffix}.json"
        if target.exists() and not args.force:
            print(f"[skip] {case['id']} already has a result")
            continue
        started = time.monotonic()
        result = run(case, args.physics_dt)
        result["wall_seconds"] = time.monotonic() - started
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"[p2] {case['id']}{suffix}: mounted={result['mounted']} "
              f"end={result['end_reason']} t={result['duration_s']:.2f}s "
              f"contact={result['first_contact_s']} "
              f"pen={result['penetration_mm']['max']:.1f}mm "
              f"compliant={result['audit']['compliant']}")


if __name__ == "__main__":
    main()
