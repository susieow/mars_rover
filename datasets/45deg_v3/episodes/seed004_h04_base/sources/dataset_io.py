"""Synchronized state/action recording and controller-free MuJoCo replay."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import mujoco
import numpy as np


def _names(model, kind, count):
    return np.asarray([mujoco.mj_id2name(model, kind, i) or "" for i in range(count)], dtype=np.str_)


def _finite(value, name):
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _maximum_error(actual, expected):
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


class EpisodeRecorder:
    """Record callbacks after control assignment and after every physical step.

    time/qpos/qvel have N+1 rows. ctrl/phase/contact_mask/desired_rocker
    have N rows and apply over [time[i], time[i+1]). physics_qpos/qvel are
    post-step states; physics_action_index identifies the applied action.
    Contact and actuator-force caches belong to the preceding force evaluation,
    which may precede the integrated qpos/qvel by one physical timestep.
    """

    def __init__(self, control_dt=0.02, record_physics=True, contact_force_threshold=5.0):
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        if not np.isfinite(contact_force_threshold) or contact_force_threshold < 0:
            raise ValueError("contact_force_threshold must be nonnegative and finite")
        self.control_dt = float(control_dt)
        self.record_physics = bool(record_physics)
        self.contact_force_threshold = float(contact_force_threshold)
        self._model = None
        self._times = []
        self._qpos = []
        self._qvel = []
        self._ctrl = []
        self._phase = []
        self._desired = []
        self._commanded = []
        self._contacts = []
        self._ground_contacts = []
        self._contact_source_steps = []
        self._physics_step_start_time = []
        self._physics_time = []
        self._physics_qpos = []
        self._physics_qvel = []
        self._physics_qfrc = []
        self._physics_ctrl = []
        self._physics_action_index = []
        self._terminal = None
        self._steps = 0
        self._initial_state = None
        self._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)

    def _initialize(self, model, data, controller):
        self._model = model
        self.simulation_dt = float(model.opt.timestep)
        if self.simulation_dt <= 0 or not np.isfinite(self.simulation_dt):
            raise ValueError("model timestep must be positive and finite")
        ratio = self.control_dt / self.simulation_dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of simulation_dt")
        self.joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        self.actuator_names = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
        self._wheels = list(getattr(controller, "wheels", [
            i for i, name in enumerate(self.joint_names) if name.endswith("_wheel_joint")
        ]))
        self.wheel_names = self.joint_names[self._wheels]
        wheel_bodies = {int(model.jnt_bodyid[j]): i for i, j in enumerate(self._wheels)}
        self._wheel_geoms = {
            geom: wheel_bodies[int(body)] for geom, body in enumerate(model.geom_bodyid)
            if int(body) in wheel_bodies
        }
        self._initial_state = np.empty(mujoco.mj_stateSize(model, self._state_spec), dtype=np.float64)
        mujoco.mj_getState(model, data, self._initial_state, self._state_spec)

    def _check_model(self, model):
        if model is not self._model:
            raise ValueError("one EpisodeRecorder can only record one model instance")
        if float(model.opt.timestep) != self.simulation_dt:
            raise ValueError("simulation_dt changed during recording")

    def _contact_mask(self, model, data):
        mask = np.zeros(len(self._wheels), dtype=np.bool_)
        force = np.empty(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            for wheel_geom, other_geom in ((first, second), (second, first)):
                wheel_index = self._wheel_geoms.get(wheel_geom)
                if wheel_index is None or int(model.geom_bodyid[other_geom]) != 0:
                    continue
                mujoco.mj_contactForce(model, data, contact_index, force)
                if abs(float(force[0])) >= self.contact_force_threshold:
                    mask[wheel_index] = True
        return mask

    def on_control(self, model, data, t, controller):
        """Capture pre-step state and the newly assigned action; t is ignored."""
        if self._model is None:
            self._initialize(model, data, controller)
        self._check_model(model)
        now = float(data.time)
        _finite(now, "control time")
        if self._times:
            if now <= self._times[-1]:
                raise ValueError("control callbacks must have strictly increasing d.time")
            if self._terminal is None or abs(now - self._terminal[0]) > 1e-12:
                raise ValueError("on_step must record every step before the next control callback")
            if not np.array_equal(data.qpos, self._terminal[1]) or not np.array_equal(data.qvel, self._terminal[2]):
                raise ValueError("state changed between on_step and on_control")
        self._times.append(now)
        self._qpos.append(_finite(data.qpos, "qpos").copy())
        self._qvel.append(_finite(data.qvel, "qvel").copy())
        self._ctrl.append(_finite(data.ctrl, "ctrl").copy())
        self._phase.append(str(getattr(controller, "phase", "UNKNOWN")))
        desired = float(np.deg2rad(getattr(controller, "target_deg", 0.0)))
        commanded = float(getattr(controller, "rocker_command", desired))
        _finite([desired, commanded], "rocker targets")
        self._desired.append(desired)
        self._commanded.append(commanded)
        self._contacts.append(self._contact_mask(model, data))
        self._contact_source_steps.append(self._steps - 1)
        ground_mask = np.zeros(len(self._wheels), dtype=np.bool_)
        for index in getattr(controller, "last_contacts", {}).get("ground_wheels", []):
            if not 0 <= int(index) < len(ground_mask):
                raise ValueError("ground_wheels contains an invalid wheel index")
            ground_mask[int(index)] = True
        self._ground_contacts.append(ground_mask)

    def on_step(self, model, data, t):
        """Capture the state after the action has advanced one physical step."""
        if not self._times:
            raise ValueError("on_control must be called before the first on_step")
        self._check_model(model)
        now = float(data.time)
        previous = self._terminal[0] if self._terminal is not None else self._times[0]
        if not np.isfinite(now) or abs(now - previous - self.simulation_dt) > 1e-10:
            raise ValueError("on_step must be called once after each physical timestep")
        if not np.array_equal(data.ctrl, self._ctrl[-1]):
            raise ValueError("ctrl changed without an on_control callback")
        qpos = _finite(data.qpos, "qpos").copy()
        qvel = _finite(data.qvel, "qvel").copy()
        self._terminal = (now, qpos, qvel)
        self._steps += 1
        if self.record_physics:
            self._physics_step_start_time.append(previous)
            self._physics_time.append(now)
            self._physics_qpos.append(qpos)
            self._physics_qvel.append(qvel)
            self._physics_qfrc.append(_finite(data.qfrc_actuator, "qfrc_actuator").copy())
            self._physics_ctrl.append(data.ctrl.copy())
            self._physics_action_index.append(len(self._ctrl) - 1)

    def save(self, path, scene_path):
        if not self._times or self._terminal is None:
            raise ValueError("cannot save an episode without actions and physical steps")
        if self._terminal[0] <= self._times[-1]:
            raise ValueError("the final action has no observed physical step")
        model = self._model
        self._check_model(model)
        times = np.asarray([*self._times, self._terminal[0]], dtype=np.float64)
        action_dt = np.diff(times)
        action_steps = np.rint(action_dt / self.simulation_dt).astype(np.int64)
        if np.any(action_steps <= 0) or not np.allclose(action_dt, action_steps * self.simulation_dt, atol=1e-10, rtol=0):
            raise ValueError("action durations do not match integer physical steps")
        if int(action_steps.sum()) != self._steps:
            raise ValueError("action transitions do not cover all recorded physical steps")
        scene_path = Path(scene_path).resolve(strict=True)
        destination = Path(path).resolve()
        if destination.suffix != ".npz":
            raise ValueError("trajectory path must end in .npz")
        arrays = {
            "schema_version": np.asarray("rover_state_action_v1"),
            "time": times,
            "qpos": np.stack([*self._qpos, self._terminal[1]]),
            "qvel": np.stack([*self._qvel, self._terminal[2]]),
            "ctrl": np.stack(self._ctrl),
            "action_dt": action_dt,
            "action_steps": action_steps,
            "phase": np.asarray(self._phase, dtype=np.str_),
            "desired_rocker": np.asarray(self._desired, dtype=np.float64),
            "commanded_rocker": np.asarray(self._commanded, dtype=np.float64),
            "rocker_angle_unit": np.asarray("radian"),
            "contact_mask": np.stack(self._contacts),
            "ground_contact_mask": np.stack(self._ground_contacts),
            "contact_cache_source_step": np.asarray(self._contact_source_steps, dtype=np.int64),
            "contact_force_threshold": np.asarray(self.contact_force_threshold),
            "contact_mask_definition": np.asarray("cached wheel-to-world contact with absolute normal force at least threshold; wheel_names order; cache is not recomputed at the integrated qpos"),
            "contact_cache_source_definition": np.asarray("zero-based preceding recorded physics step; -1 means initial cache predates recording and its evaluation time is unknown"),
            "joint_names": self.joint_names,
            "joint_types": np.asarray(model.jnt_type).copy(),
            "joint_qposadr": np.asarray(model.jnt_qposadr).copy(),
            "joint_dofadr": np.asarray(model.jnt_dofadr).copy(),
            "actuator_names": self.actuator_names,
            "actuator_ctrlrange": np.asarray(model.actuator_ctrlrange).copy(),
            "actuator_forcerange": np.asarray(model.actuator_forcerange).copy(),
            "actuator_gear": np.asarray(model.actuator_gear).copy(),
            "wheel_names": self.wheel_names,
            "simulation_dt": np.asarray(self.simulation_dt),
            "control_dt": np.asarray(self.control_dt),
            "mujoco_version": np.asarray(mujoco.__version__),
            "scene_path": np.asarray(str(scene_path)),
            "scene_sha256": np.asarray(hashlib.sha256(scene_path.read_bytes()).hexdigest()),
            "initial_integration_state": self._initial_state,
            "integration_state_spec": np.asarray(self._state_spec, dtype=np.int64),
            "physics_recorded": np.asarray(self.record_physics),
            "physics_step_start_time": np.asarray(self._physics_step_start_time, dtype=np.float64),
            "physics_time": np.asarray(self._physics_time, dtype=np.float64),
            "physics_qpos": np.asarray(self._physics_qpos, dtype=np.float64).reshape(-1, model.nq),
            "physics_qvel": np.asarray(self._physics_qvel, dtype=np.float64).reshape(-1, model.nv),
            "physics_qfrc_actuator": np.asarray(self._physics_qfrc, dtype=np.float64).reshape(-1, model.nv),
            "physics_qfrc_actuator_definition": np.asarray("last generalized actuator force evaluated during [physics_step_start_time, physics_time]; not a new force evaluation at the post-step qpos"),
            "physics_ctrl": np.asarray(self._physics_ctrl, dtype=np.float64).reshape(len(self._physics_time), model.nu),
            "physics_action_index": np.asarray(self._physics_action_index, dtype=np.int64),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return {
            "trajectory_path": str(destination), "actions": len(self._ctrl),
            "states": len(times), "physics_steps": self._steps,
            "duration": float(times[-1] - times[0]),
            "simulation_dt": self.simulation_dt, "control_dt": self.control_dt,
        }


def replay_check(scene_path, npz_path, atol=1e-8):
    """Replay recorded controls in a fresh instance without importing a controller."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be nonnegative and finite")
    if mujoco.get_mjcb_control() is not None:
        raise ValueError("independent replay requires no global MuJoCo control callback")
    scene_path = Path(scene_path).resolve(strict=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    with np.load(npz_path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    times, qpos, qvel, controls = [saved[key] for key in ("time", "qpos", "qvel", "ctrl")]
    count = len(controls)
    if count < 1 or times.shape != (count + 1,) or qpos.shape != (count + 1, model.nq) or qvel.shape != (count + 1, model.nv) or controls.shape != (count, model.nu):
        raise ValueError("trajectory must contain N controls and N+1 compatible states")
    for name in ("time", "qpos", "qvel", "ctrl", "action_dt", "initial_integration_state"):
        _finite(saved[name], name)
    if str(saved["scene_sha256"].item()) != hashlib.sha256(scene_path.read_bytes()).hexdigest():
        raise ValueError("scene XML hash differs from the recorded scene")
    if str(saved["mujoco_version"].item()) != mujoco.__version__:
        raise ValueError("MuJoCo version differs from the recording")
    if abs(float(saved["simulation_dt"]) - float(model.opt.timestep)) > 1e-15:
        raise ValueError("scene timestep differs from the recorded simulation_dt")
    for key, kind, size in (("joint_names", mujoco.mjtObj.mjOBJ_JOINT, model.njnt), ("actuator_names", mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)):
        if not np.array_equal(saved[key], _names(model, kind, size)):
            raise ValueError(f"scene {key} differ from the recording")
    durations = np.diff(times)
    steps = np.rint(durations / model.opt.timestep).astype(np.int64)
    if np.any(steps <= 0) or not np.allclose(durations, steps * model.opt.timestep, atol=1e-10, rtol=0):
        raise ValueError("recorded intervals must be positive integer physical steps")
    if saved["action_dt"].shape != (count,) or not np.allclose(saved["action_dt"], durations, atol=1e-12, rtol=0) or not np.array_equal(saved["action_steps"], steps):
        raise ValueError("action durations are inconsistent with state timestamps")
    spec = int(saved["integration_state_spec"])
    if saved["initial_integration_state"].shape != (mujoco.mj_stateSize(model, spec),):
        raise ValueError("initial integration state is incompatible with the scene")
    physics = bool(saved["physics_recorded"])
    total_steps = int(steps.sum())
    if physics:
        for key, shape in (("physics_step_start_time", (total_steps,)), ("physics_time", (total_steps,)), ("physics_qpos", (total_steps, model.nq)), ("physics_qvel", (total_steps, model.nv)), ("physics_qfrc_actuator", (total_steps, model.nv)), ("physics_ctrl", (total_steps, model.nu)), ("physics_action_index", (total_steps,))):
            if saved[key].shape != shape:
                raise ValueError(f"{key} has incompatible shape")
            _finite(saved[key], key)
        if not np.allclose(saved["physics_time"] - saved["physics_step_start_time"], model.opt.timestep, atol=1e-10, rtol=0):
            raise ValueError("physics force-evaluation intervals do not match simulation_dt")
    # Warmstart, activation and other integration state affect deterministic replay.
    mujoco.mj_setState(model, data, saved["initial_integration_state"], spec)
    errors = {"max_time_error": abs(float(data.time) - float(times[0])),
              "max_qpos_error": _maximum_error(data.qpos, qpos[0]),
              "max_qvel_error": _maximum_error(data.qvel, qvel[0]),
              "max_physics_time_error": 0.0, "max_physics_qpos_error": 0.0,
              "max_physics_qvel_error": 0.0, "max_physics_qfrc_actuator_error": 0.0}
    step_index = 0
    for index, (action, step_count) in enumerate(zip(controls, steps)):
        data.ctrl[:] = action
        for _ in range(int(step_count)):
            mujoco.mj_step(model, data)
            _finite(data.time, "replay time")
            _finite(data.qpos, "replay qpos")
            _finite(data.qvel, "replay qvel")
            _finite(data.qfrc_actuator, "replay qfrc_actuator")
            if physics:
                if int(saved["physics_action_index"][step_index]) != index or not np.array_equal(saved["physics_ctrl"][step_index], action):
                    raise ValueError("physics action alignment is inconsistent")
                for key, actual, expected in (
                    ("max_physics_time_error", np.asarray(data.time), saved["physics_time"][step_index]),
                    ("max_physics_qpos_error", data.qpos, saved["physics_qpos"][step_index]),
                    ("max_physics_qvel_error", data.qvel, saved["physics_qvel"][step_index]),
                    ("max_physics_qfrc_actuator_error", data.qfrc_actuator, saved["physics_qfrc_actuator"][step_index]),
                ):
                    errors[key] = max(errors[key], _maximum_error(actual, expected))
            step_index += 1
        errors["max_time_error"] = max(errors["max_time_error"], abs(float(data.time) - float(times[index + 1])))
        errors["max_qpos_error"] = max(errors["max_qpos_error"], _maximum_error(data.qpos, qpos[index + 1]))
        errors["max_qvel_error"] = max(errors["max_qvel_error"], _maximum_error(data.qvel, qvel[index + 1]))
    return {"replay_ok": all(error <= atol for error in errors.values()),
            "actions": count, "states": count + 1, "physics_steps": step_index,
            "duration": float(times[-1] - times[0]), "atol": float(atol),
            "physics_checked": physics, **errors}
