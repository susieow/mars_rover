"""Verify whether the selected endpoint sustains progress under its last target."""

import json
from pathlib import Path

import mujoco
import numpy as np
import backend

HERE = Path(__file__).resolve().parent


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert(v) for v in value]
    return value


def main():
    selection = json.loads((HERE / "selected_candidate.json").read_text())
    with np.load(HERE / "selected_candidate.npz") as archive:
        selected = {k: archive[k] for k in archive.files}
    with np.load(HERE / "seed_prefix.npz") as archive:
        seed = {k: archive[k] for k in archive.files}
    backend.init_worker(HERE / "scene.xml", HERE / "seed_prefix.npz")
    result = backend.evaluate({"id": selection["id"] + "_tail_hold_1s",
        "sequence": [selection["sequence"][-1]], "horizon": 1.0,
        "extra_ctrl": selected["controls"], "controller_state": selection["controller_state"],
        "record_states": True})
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, seed["initial_integration_state"], int(seed["integration_state_spec"]))
    for control in np.concatenate([seed["prefix_ctrl"], selected["controls"]]):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
    errors = {name: 0.0 for name in ("time", "qpos", "qvel", "actuator_force")}
    for index, control in enumerate(result["controls"]):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        for name, actual, expected in (("time", data.time, result["track"][index, 0]),
                ("qpos", data.qpos, result["record_qpos"][index]),
                ("qvel", data.qvel, result["record_qvel"][index]),
                ("actuator_force", data.qfrc_actuator, result["record_qfrc_actuator"][index])):
            errors[name] = max(errors[name], float(np.max(np.abs(actual - expected))))
    if max(errors.values()) != 0.0:
        raise ValueError(f"Tail-hold replay mismatch: {errors}")
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(HERE / "tail_hold_1s.npz", **arrays)
    report = convert({k: v for k, v in result.items() if not isinstance(v, np.ndarray)})
    report["independent_initial_state_replay_max_error"] = errors
    report["purpose"] = "One-second continuation to disambiguate temporary return-to-wall speed from sustained progress; no additional search"
    (HERE / "tail_hold_1s.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"valid": report["valid"], "reasons": report["reasons"],
                      "metrics": report.get("metrics"), "replay_errors": errors}), flush=True)


if __name__ == "__main__":
    main()
