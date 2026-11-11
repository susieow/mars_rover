"""Prepare a deterministic diagnostic branch from unchanged recorded actions."""

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("/home/agilex/wzy/rover")
SOURCE = ROOT / "review_v4/high20_slow_neutral_v2"
OUT = ROOT / "review_v4/dynamic_search20"
TARGET_TIME = 10.484
sys.path.insert(0, str(SOURCE / "sources"))
from episode_audit import EpisodeAudit
from verify_dataset import RouteAudit

OUT.mkdir(exist_ok=True)
scene = OUT / "scene.xml"
shutil.copy2(SOURCE / "scene.xml", scene)
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
with np.load(SOURCE / "trajectory.npz", allow_pickle=False) as archive:
    saved = {name: archive[name] for name in archive.files}
spec = int(saved["integration_state_spec"])
mujoco.mj_setState(model, data, saved["initial_integration_state"], spec)
steps = round((TARGET_TIME - data.time) / model.opt.timestep)
if not np.array_equal(saved["action_steps"][:steps], np.ones(steps, dtype=int)):
    raise ValueError("Seed preparation requires one recorded action per physical step")
if not np.array_equal(saved["physics_ctrl"][:steps], saved["ctrl"][:steps]):
    raise ValueError("Recorded action and physical control rows differ")
audit = EpisodeAudit(model, ROOT / "assets/mars_rover/urdf/mars_rover.urdf")
route = RouteAudit(model, audit)
names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
wheels = {g: n.removesuffix("_col_1") for g, n in enumerate(names) if n.endswith("_wheel_col_1")}
ema = {n: np.zeros(3) for n in wheels.values()}
force = np.zeros(6)
max_errors = {name: 0.0 for name in ("time", "qpos", "qvel", "actuator_force")}
for step in range(steps):
    data.ctrl[:] = saved["ctrl"][step]
    mujoco.mj_step(model, data)
    audit.step(model, data, float(data.time))
    route.step(model, data, float(data.time))
    for name, actual, expected in (
        ("time", data.time, saved["physics_time"][step]),
        ("qpos", data.qpos, saved["physics_qpos"][step]),
        ("qvel", data.qvel, saved["physics_qvel"][step]),
        ("actuator_force", data.qfrc_actuator, saved["physics_qfrc_actuator"][step]),
    ):
        max_errors[name] = max(max_errors[name], float(np.max(np.abs(actual - expected))))
    forces = {n: np.zeros(3) for n in wheels.values()}
    for i in range(data.ncon):
        contact = data.contact[i]
        a, b = int(contact.geom1), int(contact.geom2)
        wheel = wheels.get(b, wheels.get(a))
        if wheel is None:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        world = contact.frame.reshape(3, 3).T @ force[:3]
        if a in wheels:
            world = -world
        forces[wheel] += world
    for name in ema:
        ema[name] += model.opt.timestep / 0.05 * (forces[name] - ema[name])
if max(max_errors.values()) != 0.0:
    raise ValueError(f"Seed prefix is not an exact independent replay: {max_errors}")
branch = np.empty(mujoco.mj_stateSize(model, spec))
mujoco.mj_getState(model, data, branch, spec)
np.savez_compressed(
    OUT / "seed_prefix.npz",
    initial_integration_state=saved["initial_integration_state"],
    integration_state_spec=spec,
    branch_integration_state=branch,
    branch_qpos=data.qpos.copy(),
    branch_qvel=data.qvel.copy(),
    prefix_ctrl=saved["ctrl"][:steps],
    prefix_time=saved["time"][:steps + 1],
    branch_time=float(data.time),
    source_trajectory=str(SOURCE / "trajectory.npz"),
    prefix_qpos=saved["qpos"][:steps + 1],
    prefix_qvel=saved["qvel"][:steps + 1],
    support_ema_wheel_names=np.asarray(list(ema), dtype=np.str_),
    support_ema_force_xyz_n=np.asarray(list(ema.values())),
)
w, x, y, z = data.qpos[3:7]
rpy = [math.atan2(2 * (w * x + y * z), 1 - 2 * (x*x + y*y)),
       math.asin(np.clip(2 * (w*y - z*x), -1, 1)),
       math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))]
up0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "up0")
report = {
    "dataset_accepted": False,
    "purpose": "Exact prefix and diagnostic seed for constrained 20 cm short-horizon search",
    "source_episode": str(SOURCE),
    "source_scene_sha256": hashlib.sha256((SOURCE / "scene.xml").read_bytes()).hexdigest(),
    "copied_scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
    "source_trajectory_sha256": hashlib.sha256((SOURCE / "trajectory.npz").read_bytes()).hexdigest(),
    "branch_time_s": float(data.time),
    "prefix_physics_steps": steps,
    "feedback_dt_s": float(model.opt.timestep),
    "independent_replay_max_error": max_errors,
    "base_position_m": data.qpos[:3].tolist(),
    "base_rpy_deg": np.degrees(rpy).tolist(),
    "wheel_centers_m": {n: data.geom_xpos[g].tolist() for g, n in wheels.items()},
    "wheel_support_50ms_ema_xyz_n": {n: v.tolist() for n, v in ema.items()},
    "contact_cache_phase": "Previous force evaluation, one physical interval before integrated qpos",
    "first_stair_front_x_m": float(model.geom_pos[up0, 0] - model.geom_size[up0, 0]),
    "first_stair_top_z_m": float(model.geom_pos[up0, 2] + model.geom_size[up0, 2]),
    "prefix_audit": audit.result(),
    "prefix_route_audit": route.result(),
    "candidate_budget": {"candidates_per_round": 64, "rounds": 3},
    "held_out_heights_m": [0.17, 0.18],
    "held_out_heights_simulated": False,
    "scoring_intent": [
        "Apply unchanged process constraints at each 1 ms physical step",
        "Prioritize both rear wheels crossing the first upper corner and continued forward progress",
        "Measure front and middle actual support, without assuming a fixed wheel center",
        "Treat rear-wheel lift and reduced pitch as diagnostic progress only",
        "Independently replay initial state plus all selected controls before claiming a result",
    ],
    "backup_design_not_executed": "Slight oblique approach to stagger rear-wheel impact within original yaw and lateral bounds",
}
(OUT / "seed_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("branch_time_s", "independent_replay_max_error", "base_position_m", "base_rpy_deg", "wheel_centers_m", "wheel_support_50ms_ema_xyz_n")}, indent=2))
