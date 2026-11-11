"""Independently replay selected controls from the original episode initial state."""

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources"))
from dataset_io import EpisodeRecorder, replay_check
from episode_audit import EpisodeAudit
from verify_dataset import RouteAudit
import backend


def main():
    backend.init_worker(HERE / "scene.xml", HERE / "seed_prefix.npz")
    with np.load(HERE / "selected_candidate.npz") as archive:
        selected = {key: archive[key] for key in archive.files}
    with np.load(HERE / "seed_prefix.npz") as archive:
        seed = {key: archive[key] for key in archive.files}
    with np.load(str(seed["source_trajectory"].item())) as archive:
        source = {key: archive[key] for key in archive.files}
    reference = backend.evaluate({"id": str(selected["candidate_id"].item()),
                                  "sequence": selected["sequence"].tolist(),
                                  "horizon": 2.0, "record_states": True})
    if not reference.get("valid"):
        raise ValueError(f"Selected candidate was not valid on repeat: {reference.get('reasons')}")
    if not np.array_equal(reference["controls"], selected["controls"]):
        raise ValueError("Selected feedback controls changed on repeat")
    if not np.array_equal(reference["track"], selected["track"]):
        raise ValueError("Selected feedback state metrics changed on repeat")
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, seed["initial_integration_state"], int(seed["integration_state_spec"]))
    wheel_joints = [j for j in range(model.njnt)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("_wheel_joint")]
    dummy = SimpleNamespace(wheels=wheel_joints, target_deg=0., rocker_command=0.,
                            phase="SEARCH_PREFIX", last_contacts={})
    recorder = EpisodeRecorder(control_dt=0.001, record_physics=True)
    audit = EpisodeAudit(model, HERE / "reference.urdf")
    route = RouteAudit(model, audit)
    prefix_steps = len(seed["prefix_ctrl"])
    actions = np.concatenate([seed["prefix_ctrl"], selected["controls"]])
    errors = {key: 0.0 for key in ("time", "qpos", "qvel", "actuator_force", "base_rear_metrics")}
    pair_commands = np.zeros(2)
    desired_pairs = []
    commanded_pairs = []
    pair_indices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, side + "_rocker_joint")
                    for side in ("left", "right")]
    rear_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, side + "_rear_wheel_col_1")
                  for side in ("left", "right")]
    for index, action in enumerate(actions):
        if index < prefix_steps:
            dummy.target_deg = math.degrees(float(source["desired_rocker"][index]))
            dummy.rocker_command = float(source["commanded_rocker"][index])
            pair_targets = np.full(2, math.radians(dummy.target_deg))
            pair_commands[:] = dummy.rocker_command
        else:
            relative = index - prefix_steps
            target = selected["sequence"][min(len(selected["sequence"]) - 1,
                int(relative * len(selected["sequence"]) / len(selected["controls"])))]
            pair_targets = np.asarray([np.clip(math.radians(float(target[3 + i])), *model.jnt_range[j])
                                       for i, j in enumerate(pair_indices)])
            pair_commands += np.clip(pair_targets - pair_commands,
                                     -math.radians(15.) * .001, math.radians(15.) * .001)
            dummy.target_deg = math.degrees(float(np.mean(pair_targets)))
            dummy.rocker_command = float(np.mean(pair_commands))
            dummy.phase = "SEARCH_DIAGNOSTIC"
        desired_pairs.append(pair_targets.copy())
        commanded_pairs.append(pair_commands.copy())
        data.ctrl[:] = action
        recorder.on_control(model, data, float(data.time), dummy)
        mujoco.mj_step(model, data)
        recorder.on_step(model, data, float(data.time))
        audit.step(model, data, float(data.time))
        route.step(model, data, float(data.time))
        if index < prefix_steps:
            expected_time = source["physics_time"][index]
            expected_qpos = source["physics_qpos"][index]
            expected_qvel = source["physics_qvel"][index]
            expected_force = source["physics_qfrc_actuator"][index]
        else:
            relative = index - prefix_steps
            expected_time = reference["track"][relative, 0]
            expected_qpos = reference["record_qpos"][relative]
            expected_qvel = reference["record_qvel"][relative]
            expected_force = reference["record_qfrc_actuator"][relative]
            actual_track = np.asarray([
                float(data.time), float(data.qpos[0]),
                min(float(data.geom_xpos[g, 0]) for g in rear_geoms),
                min(float(data.geom_xpos[g, 2]) for g in rear_geoms),
                min(float(data.geom_xpos[g, 2]) - float(model.geom_size[g, 0]) for g in rear_geoms),
                float(data.qvel[0]),
            ])
            errors["base_rear_metrics"] = max(errors["base_rear_metrics"],
                float(np.max(np.abs(actual_track - reference["track"][relative]))))
        for key, actual, expected in (("time", data.time, expected_time),
                                     ("qpos", data.qpos, expected_qpos),
                                     ("qvel", data.qvel, expected_qvel),
                                     ("actuator_force", data.qfrc_actuator, expected_force)):
            errors[key] = max(errors[key], float(np.max(np.abs(actual - expected))))
    if max(errors.values()) != 0.0:
        raise ValueError(f"Selected independent replay changed the physical trajectory: {errors}")
    trajectory = HERE / "selected_trajectory.npz"
    recorder.save(trajectory, HERE / "scene.xml")
    np.savez_compressed(HERE / "selected_rocker_targets.npz", desired_pair_rad=desired_pairs,
                        commanded_pair_rad=commanded_pairs,
                        pair_names=np.asarray(["left", "right"], dtype=np.str_))
    replay = replay_check(HERE / "scene.xml", trajectory)
    report = {
        "dataset_accepted": False,
        "candidate_id": str(selected["candidate_id"].item()),
        "independent_from_original_initial_state": True,
        "feedback_controls_repeat_exact": True,
        "full_state_max_errors_against_feedback_reference": errors,
        "saved_artifact_replay": replay,
        "audit": audit.result(),
        "route": route.result(),
        "metrics": reference["metrics"],
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "recording_note": "Diagnostic only; scalar rocker target fields contain left/right mean, exact pair targets are saved separately. Ground-contact advisory masks are unset; independently audited wheel support remains authoritative.",
    }
    (HERE / "selected_independent_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    search_path = HERE / "search_report.json"
    search = json.loads(search_path.read_text())
    search["independent_final_replay_completed"] = True
    search["independent_final_replay_report"] = str(HERE / "selected_independent_replay.json")
    search_path.write_text(json.dumps(search, indent=2) + "\n")
    print(json.dumps({"candidate_id": report["candidate_id"], "errors": errors,
                      "saved_artifact_replay": replay, "metrics": reference["metrics"],
                      "dataset_accepted": False}), flush=True)


if __name__ == "__main__":
    main()
