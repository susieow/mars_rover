"""Run the approved 22-second continuation matrix without altering its parent."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

import mujoco
import numpy as np

import speed20_experiment as common
import recovery20_amp20_experiment as parent_exp
import xml.etree.ElementTree as ET
from overspeed20_stairs import summarize_speed

RUN = common.ROOT / 'runs/rear_steer_lock_20261111'
BASE_SCENE = common.FROZEN / 'scene.xml'
PARENTS = common.ROOT / 'runs/overspeed20_20261111/cases'
SCENE = RUN / 'scene_locked.xml'
START, DURATION, CONTROL_DT = 22.0, 30.0, .001
WHEELS = {'H': (0, 0, 0), 'F': (1, 0, 0), 'M': (0, 1, 0), 'R': (0, 0, 1),
          'FM': (1, 1, 0), 'FR': (1, 0, 1), 'MR': (0, 1, 1), 'FMR': (1, 1, 1)}
ROCKERS = {'N': (0, 0), 'SP': (1, 1), 'SN': (-1, -1), 'DP': (1, -1), 'DN': (-1, 1)}
SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def write(path, value):
    common.write_json(path, value)


def append_event(value):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'invocations.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'wall_time_unix': time.time(), **value}, ensure_ascii=False) + '\n')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checkpoint_dir(dt):
    return parent_exp.checkpoint_dir(dt)


def params(wheel, rocker, dt=.001):
    if wheel not in WHEELS or rocker not in ROCKERS or dt not in (.001, .0005, .00025):
        raise ValueError('condition outside the approved matrix')
    return {'id': f'T22_W{wheel}_R{rocker}_dt{round(dt * 1e6):04d}us',
            'wheel': wheel, 'rocker': rocker, 'physics_dt_s': dt,
            'intervention_s': START, 'max_continuation_s': DURATION,
            'wheel_initial_rad_s': 3., 'wheel_final_rad_s': 6., 'ramp_s': 3.,
            'rocker_amplitude_deg': 20., 'rocker_rate_deg_s': 20.,
            'rocker_period_s': 8., 'rocker_cycles': 3,
            'control_period_s': CONTROL_DT, 'is_control': wheel == 'H'}


def targets(case, elapsed):
    wheel = 3. + np.asarray(WHEELS[case['wheel']]) * min(max(elapsed, 0.), 3.)
    phase = elapsed % 8.
    if elapsed >= 24.:
        angle = 0.
    elif phase < 1.:
        angle = 20. * phase
    elif phase < 2.:
        angle = 20.
    elif phase < 4.:
        angle = 20. - 20. * (phase - 2.)
    elif phase < 5.:
        angle = -20.
    elif phase < 6.:
        angle = -20. + 20. * (phase - 5.)
    else:
        angle = 0.
    return wheel, np.radians(angle * np.asarray(ROCKERS[case['rocker']]))


def update(c, wheel, rocker):
    goals = c.update(wheel)
    # Preserve the parent wheel and steering feedback, replacing only the rocker reference.
    c.rocker_commands[:] = rocker
    for i, (_, act, pos, vel) in enumerate(c.rockers):
        desired = float(np.clip(4 * (rocker[i] - float(c.d.qpos[pos])), -.7, .7))
        c.d.ctrl[act] = np.clip(150 * (desired - float(c.d.qvel[vel])), -100, 100)
    return goals


def source_manifest():
    paths = [Path(__file__), Path(parent_exp.__file__), Path(common.__file__),
             Path(sys.modules['overspeed20_stairs'].__file__),
             Path(sys.modules[common.EpisodeAudit.__module__].__file__),
             Path(sys.modules[common.RouteAudit.__module__].__file__)]
    return {str(p): common.digest(p) for p in paths}


def controller_memory(c):
    return {k: copy.deepcopy(v) for k, v in vars(c).items() if k not in ('m', 'd')}


def save_checkpoint(m, d, c, a, r, dt, parent, prefix_error):
    folder = checkpoint_dir(dt)
    folder.mkdir(parents=True, exist_ok=False)
    state = np.empty(mujoco.mj_stateSize(m, SPEC))
    mujoco.mj_getState(m, d, state, SPEC)
    np.savez_compressed(folder / 'physics.npz', integration_state=state, state_spec=SPEC)
    # This pickle is generated here, hashed, and loaded only from this run's checkpoint directory.
    memory = pickle.dumps((controller_memory(c), a, r), protocol=5)
    (folder / 'memory.pkl').write_bytes(memory)
    manifest = source_manifest()
    sources = RUN / 'sources'
    sources.mkdir(parents=True, exist_ok=True)
    for original, digest in manifest.items():
        destination = sources / (digest[:12] + '_' + Path(original).name)
        if not destination.exists():
            destination.write_bytes(Path(original).read_bytes())
    meta = {'parent_id': parent['id'], 'parent_trajectory': str(PARENTS / parent['id'] / 'trajectory.npz'),
            'parent_trajectory_sha256': parent['trajectory_sha256'], 'parent_step': round(START / dt),
            'time_s': float(d.time), 'physics_dt_s': dt, 'scene_sha256': common.digest(SCENE),
            'urdf_sha256': common.digest(common.URDF), 'physics_sha256': common.digest(folder / 'physics.npz'),
            'memory_sha256': common.digest(folder / 'memory.pkl'), 'source_sha256': manifest,
            'prefix_comparison': prefix_error, 'mujoco_version': mujoco.__version__,
            'numpy_version': np.__version__, 'course': r.result()['geometry']}
    meta['checkpoint_id'] = digest_json(meta)
    write(folder / 'checkpoint.json', meta)
    return meta



orientation, wrap, DT, LIMIT = common.orientation, common.wrap, common.DT, common.LIMIT

class LockedRearController(common.Controller):
    def update(self, target):
        m, d = self.m, self.d
        _, _, yaw = orientation(d)
        if self.yaw_previous is not None:
            self.yaw_rate = .65*self.yaw_rate + .35*wrap(yaw-self.yaw_previous)/DT
        self.yaw_previous = yaw
        path_heading = np.clip(-math.atan(1.5*float(d.qpos[1])+.3*float(d.qvel[1])), -.35, .35)
        error = wrap(path_heading-yaw)
        correction = float(np.clip(4*error-1.2*self.yaw_rate, -1.5, 1.5))
        steer_target = float(np.clip(.45*error-.1*self.yaw_rate, -.15, .15))
        self.wheel_commands += np.clip(np.asarray(target)-self.wheel_commands, -30*DT, 30*DT)
        actual_targets = np.zeros(6)
        for i, (j, a, q, v, role, left, g) in enumerate(self.wheels):
            goal = float(np.clip(self.wheel_commands[role]+(-correction if left else correction), -LIMIT, LIMIT))
            actual_targets[i] = goal
            d.ctrl[a] = np.clip(40*(goal-float(d.qvel[v])), -50, 50)
        for j, a, q, v in self.steer:
            if 'rear' in self.names[j]:
                d.ctrl[a] = 0.
                continue
            goal = steer_target if 'front' in self.names[j] else -steer_target
            e = goal-float(d.qpos[q])
            desired = float(np.clip(8*e, -1.2, 1.2))
            d.ctrl[a] = np.clip(50*(desired-float(d.qvel[v])), -40, 40)
            wheel_name = self.names[j].replace('_steer_joint', '_wheel_joint')
            wheel_a = next(a0 for j0, a0, *_ in self.wheels if self.names[j0] == wheel_name)
            if d.ctrl[wheel_a] > 0:
                d.ctrl[wheel_a] *= np.clip((25-abs(math.degrees(e)))/15, 0, 1)
        for i, (j, a, q, v) in enumerate(self.rockers):
            self.rocker_commands[i] += np.clip(-self.rocker_commands[i], -math.radians(15)*DT, math.radians(15)*DT)
            desired = float(np.clip(4*(self.rocker_commands[i]-float(d.qpos[q])), -.7, .7))
            d.ctrl[a] = np.clip(150*(desired-float(d.qvel[v])), -100, 100)
        return actual_targets

def restore(dt):
    original_model, original_data, controller, audit, route, original_meta = parent_exp.restore(dt)
    if not SCENE.exists():
        raise RuntimeError('locked scene missing')
    model = mujoco.MjModel.from_xml_path(str(SCENE)); model.opt.timestep = dt
    if (model.nq, model.nv, model.nu, model.nbody, model.ngeom) != (original_model.nq, original_model.nv, original_model.nu, original_model.nbody, original_model.ngeom) or model.neq != 2:
        raise RuntimeError('locked model changed robot dimensions')
    data = mujoco.MjData(model)
    data.time = original_data.time
    for name in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'mocap_pos', 'mocap_quat', 'userdata'):
        dst, src = getattr(data, name), getattr(original_data, name)
        if dst.shape != src.shape:
            raise RuntimeError('cannot transfer state field ' + name)
        dst[:] = src
    locks = {}
    for i, name in enumerate(('left_rear_steer_joint', 'right_rear_steer_joint')):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'lock_' + name)
        model.eq_data[eq, 0] = data.qpos[model.jnt_qposadr[joint]]
        data.eq_active[eq] = 1
        locks[name] = float(model.eq_data[eq, 0])
    for name in ('qpos','qvel','ctrl','act'):
        if not np.array_equal(getattr(data,name),getattr(original_data,name)):
            raise RuntimeError('checkpoint state changed: '+name)
    controller.m = model; controller.d = data; controller.__class__ = LockedRearController
    meta = dict(original_meta)
    meta['original_scene_sha256'] = original_meta['scene_sha256']
    meta['scene_sha256'] = common.digest(SCENE)
    meta['rear_lock_positions_rad'] = locks
    return model, data, controller, audit, route, meta


def compare_old(z, d, step, before=False):
    if before:
        return float(np.max(np.abs(z['ctrl'][step] - d.ctrl)))
    return max(float(np.max(np.abs(z['qpos'][step] - d.qpos))),
               float(np.max(np.abs(z['qvel'][step] - d.qvel))))


def prepare(dt):
    folder = checkpoint_dir(dt)
    if (folder / 'gate.json').exists():
        gate = json.loads((folder / 'gate.json').read_text())
        if not gate['passed']:
            raise RuntimeError('existing checkpoint gate failed')
        restore(dt)
        append_event({'kind': 'checkpoint_reused', 'dt': dt})
        return
    if folder.exists():
        raise RuntimeError('incomplete checkpoint exists; preserve it and investigate')
    append_event({'kind': 'prefix_replay', 'event': 'started', 'dt': dt, 'duration_s': START})
    parent_id = f'A03_C03_dt{round(dt * 1e6):04d}us'
    parent_folder = PARENTS / parent_id
    parent = json.loads((parent_folder / 'result.json').read_text())
    if common.digest(parent_folder / 'trajectory.npz') != parent['trajectory_sha256']:
        raise RuntimeError('parent trajectory hash mismatch')
    with np.load(parent_folder / 'trajectory.npz', allow_pickle=False) as old:
        n = round((START + DURATION) / dt)
        old_arrays = {k: old[k][:n + (k in ('qpos', 'qvel'))].copy()
                      for k in ('qpos', 'qvel', 'ctrl', 'qfrc_actuator')}
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = dt
    d = common.initialize(m, 8.)
    c = common.Controller(m, d)
    a = common.EpisodeAudit(m, common.URDF)
    r = common.RouteAudit(m, a, {'height': .2, 'depth': .2, 'stairs': 15})
    substeps = round(CONTROL_DT / dt)
    errors = {'state_max_abs': compare_old(old_arrays, d, 0), 'control_max_abs': 0., 'actuator_max_abs': 0.}
    for step in range(round(START / dt)):
        if step % substeps == 0:
            c.update(np.full(3, 0. if d.time < .5 else 3.))
        errors['control_max_abs'] = max(errors['control_max_abs'], compare_old(old_arrays, d, step, True))
        mujoco.mj_step(m, d)
        a.step(m, d, float(d.time))
        r.step(m, d, float(d.time))
        errors['state_max_abs'] = max(errors['state_max_abs'], compare_old(old_arrays, d, step + 1))
        errors['actuator_max_abs'] = max(errors['actuator_max_abs'], float(np.max(abs(old_arrays['qfrc_actuator'][step] - d.qfrc_actuator))))
    if max(errors.values()) > 1e-10 or common.process_reasons(a, r):
        raise RuntimeError(f'parent prefix differs: {errors}')
    meta = save_checkpoint(m, d, c, a, r, dt, parent, errors)
    append_event({'kind': 'prefix_replay', 'event': 'completed', 'dt': dt, 'errors': errors})
    append_event({'kind': 'restore_validation', 'event': 'started', 'dt': dt, 'duration_s': DURATION})
    m2, d2, c2, a2, r2, _ = restore(dt)
    gate = {'parent_state_max_abs': 0., 'parent_control_max_abs': 0., 'parent_actuator_max_abs': 0.,
            'uninterrupted_state_max_abs': 0., 'uninterrupted_control_max_abs': 0., 'uninterrupted_actuator_max_abs': 0.}
    hold = params('H', 'N', dt)
    for i in range(round(DURATION / dt)):
        step = round(START / dt) + i
        if i % substeps == 0:
            c.update(np.full(3, 3.))
            update(c2, *targets(hold, i * dt))
        gate['parent_control_max_abs'] = max(gate['parent_control_max_abs'], compare_old(old_arrays, d2, step, True))
        gate['uninterrupted_control_max_abs'] = max(gate['uninterrupted_control_max_abs'], float(np.max(abs(d.ctrl - d2.ctrl))))
        mujoco.mj_step(m, d)
        mujoco.mj_step(m2, d2)
        a.step(m, d, float(d.time)); r.step(m, d, float(d.time))
        a2.step(m2, d2, float(d2.time)); r2.step(m2, d2, float(d2.time))
        gate['parent_state_max_abs'] = max(gate['parent_state_max_abs'], compare_old(old_arrays, d2, step + 1))
        gate['parent_actuator_max_abs'] = max(gate['parent_actuator_max_abs'], float(np.max(abs(old_arrays['qfrc_actuator'][step] - d2.qfrc_actuator))))
        gate['uninterrupted_state_max_abs'] = max(gate['uninterrupted_state_max_abs'], float(np.max(abs(d.qpos - d2.qpos))), float(np.max(abs(d.qvel - d2.qvel))))
        gate['uninterrupted_actuator_max_abs'] = max(gate['uninterrupted_actuator_max_abs'], float(np.max(abs(d.qfrc_actuator - d2.qfrc_actuator))))
    gate['audit_equal'] = a.result() == a2.result()
    gate['route_equal'] = r.result() == r2.result()
    gate['passed'] = all(v if isinstance(v, bool) else v <= 1e-10 for v in gate.values())
    gate.update(checkpoint_id=meta['checkpoint_id'], duration_s=DURATION, reference='uninterrupted_and_original_22_to_52')
    write(folder / 'gate.json', gate)
    append_event({'kind': 'restore_validation', 'event': 'completed', 'dt': dt, 'physics_rollouts': 2, 'gate': gate})
    if not gate['passed']:
        raise RuntimeError('restore validation failed')
    print(json.dumps({'checkpoint_dt': dt, 'gate': gate}), flush=True)


def allocate(m, count):
    arrays = {name: np.zeros((count + 1, size)) for name, size in [('qpos', m.nq), ('qvel', m.nv)]}
    arrays['time'] = np.zeros(count + 1)
    for name, size in [('ctrl', m.nu), ('qfrc_actuator', m.nv), ('wheel_targets', 6),
                       ('wheel_group_commands', 3), ('rocker_targets_rad', 2),
                       ('contact_force_n', 6), ('contact_slip_tangent_m_s', 6),
                       ('qacc_step_start', m.nv), ('qfrc_bias_step_start', m.nv),
                       ('qfrc_passive_step_start', m.nv), ('qfrc_constraint_step_start', m.nv)]:
        arrays[name] = np.zeros((count, size))
    arrays['wheel_support_level'] = np.full((count, 6), -1, dtype=np.int8)
    arrays['wheel_centers_step_start'] = np.zeros((count, 6, 3))
    arrays['rear_geometry_level'] = np.zeros(count, dtype=np.int8)
    arrays['rear_supported_level'] = np.zeros(count, dtype=np.int8)
    arrays['contacts_geom_ids'] = np.full((count, 64, 2), -1, dtype=np.int16)
    arrays['contacts_normal_force_n'] = np.zeros((count, 64), dtype=np.float32)
    return arrays


def run_case(case):
    dt = case['physics_dt_s']
    folder = RUN / 'cases' / case['id']
    m, d, c, a, r, meta = restore(dt)
    if not json.loads((checkpoint_dir(dt) / 'gate.json').read_text())['passed']:
        raise RuntimeError('checkpoint gate did not pass')
    identity = {'parameters': {k: v for k, v in case.items() if k != 'id'},
                'checkpoint': meta['checkpoint_id'], 'source_sha256': source_manifest(),
                'audit_policy': 'overspeed_record_only_other_existing_stops'}
    fingerprint = digest_json(identity)
    for existing in (RUN / 'cases').glob('*/result.json'):
        row = json.loads(existing.read_text())
        if row.get('experiment_fingerprint') == fingerprint:
            return {'id': case['id'], 'reused': row['id'], 'duplicate_skipped': True}
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'started.json', {'case': case, 'fingerprint': fingerprint, 'status': 'started'})
    cap = round(DURATION / dt)
    arrays = allocate(m, cap)
    arrays['time'][0], arrays['qpos'][0], arrays['qvel'][0] = d.time, d.qpos, d.qvel
    initial = np.empty(mujoco.mj_stateSize(m, SPEC)); mujoco.mj_getState(m, d, initial, SPEC)
    rear = [i for i, row in enumerate(c.wheels) if row[4] == 2]
    crossing = np.zeros(15); peaks = np.zeros(15); supported_total = np.zeros(15)
    goal = np.zeros(6); rocker = np.zeros(2)
    reason = 'continuation_time_limit_30s'; stop_reasons = []; first_violation = None
    first_crossing = None; started = time.monotonic()
    forces_buf = np.zeros(6)
    for step in range(cap):
        if step % round(CONTROL_DT / dt) == 0:
            raw, rocker = targets(case, step * dt)
            goal = update(c, raw, rocker)
        arrays['ctrl'][step] = d.ctrl
        arrays['wheel_targets'][step] = goal
        arrays['wheel_group_commands'][step] = c.wheel_commands
        arrays['rocker_targets_rad'][step] = rocker
        mujoco.mj_step(m, d)
        for key, source in [('qacc_step_start', d.qacc), ('qfrc_bias_step_start', d.qfrc_bias),
                            ('qfrc_passive_step_start', d.qfrc_passive), ('qfrc_constraint_step_start', d.qfrc_constraint),
                            ('qfrc_actuator', d.qfrc_actuator)]:
            arrays[key][step] = source
        if d.ncon > 64:
            raise RuntimeError('contact record capacity exceeded')
        for ci in range(d.ncon):
            ct = d.contact[ci]
            arrays['contacts_geom_ids'][step, ci] = [ct.geom1, ct.geom2]
            if ct.efc_address >= 0:
                mujoco.mj_contactForce(m, d, ci, forces_buf)
                arrays['contacts_normal_force_n'][step, ci] = forces_buf[0]
        a.step(m, d, float(d.time)); r.step(m, d, float(d.time))
        forces, slip, _, support = common.contacts(m, d, c)
        levels = np.full(6, -1, dtype=np.int8)
        for wi, terrains in enumerate(support):
            values = [int(name[2:]) + 1 for name in terrains if name.startswith('up') and name[2:].isdigit()]
            if 'platform' in terrains:
                values.append(15)
            if values:
                levels[wi] = max(values)
        centers = d.geom_xpos[[row[-1] for row in c.wheels]]
        arrays['wheel_centers_step_start'][step] = centers
        instantaneous_level = 0; support_level = 0
        for k in range(15):
            crossed = all(float(centers[i, 0]) > 1.6 + .2 * k + .005 and
                          float(centers[i, 2] - m.geom_size[c.wheels[i][-1], 0]) > .2 * (k + 1) - .010 for i in rear)
            crossing[k] = crossing[k] + dt if crossed else 0.
            peaks[k] = max(peaks[k], crossing[k])
            if crossed:
                instantaneous_level = k + 1
                if all(levels[i] >= k + 1 for i in rear):
                    supported_total[k] += dt
                    support_level = k + 1
            if k == 0 and first_crossing is None and crossing[k] >= .1 - 1e-9:
                first_crossing = float(d.time)
        arrays['rear_geometry_level'][step] = instantaneous_level
        arrays['rear_supported_level'][step] = support_level
        arrays['time'][step + 1], arrays['qpos'][step + 1], arrays['qvel'][step + 1] = d.time, d.qpos, d.qvel
        arrays['contact_force_n'][step], arrays['contact_slip_tangent_m_s'][step] = forces, slip
        arrays['wheel_support_level'][step] = levels
        reasons = common.process_reasons(a, r)
        if reasons and first_violation is None:
            first_violation = float(d.time)
        stop_reasons = [x for x in reasons if not x.startswith('joint_velocity_limit:')]
        if stop_reasons:
            reason = 'physics_or_route_violation'
            break
        if 'all_six_supported_on_platform_s' in r.events:
            reason = 'ascent_platform_support_observed'
            break
    n = step + 1
    arrays = {k: value[:n + (k in ('time', 'qpos', 'qvel'))] for k, value in arrays.items()}
    np.savez_compressed(folder / 'trajectory.npz', **arrays, initial_integration_state=initial,
                        integration_state_spec=SPEC, simulation_dt=dt, scene_sha256=meta['scene_sha256'])
    overspeed = summarize_speed(arrays['time'], arrays['qvel'], a.velocity_dofs, a.velocity_limits, dt)
    highest = max((k + 1 for k in range(15) if peaks[k] >= .1 - 1e-9), default=0)
    tail = max(1, round(min(1., n * dt) / dt))
    retained = int(np.min(arrays['rear_geometry_level'][-tail:]))
    end_x = float(d.qpos[0]); start_x = float(arrays['qpos'][0, 0])
    result = {'id': case['id'], 'kind': 'recovery22', 'parameters': case,
              'experiment_fingerprint': fingerprint, 'parent_id': meta['parent_id'],
              'parent_step': meta['parent_step'], 'checkpoint_id': meta['checkpoint_id'],
              'parent_trajectory_sha256': meta['parent_trajectory_sha256'],
              'intervention_s': START, 'actual_duration_s': float(d.time),
              'continuation_duration_s': n * dt, 'height_cm': 20, 'stairs': 15,
              'physics_step_s': dt, 'control_period_s': CONTROL_DT,
              'termination_reason': reason, 'diagnostic_stop_reasons': stop_reasons,
              'diagnostic_stop_time_s': float(d.time), 'first_failure_s': first_violation,
              'process_valid': not reasons, 'process_reasons': reasons,
              'rear_highest_100ms_geometry_level': highest, 'first_rear_crossing_s': first_crossing,
              'rear_crossing_peak_s_by_level': peaks.tolist(),
              'rear_crossing_supported_total_s_by_level': supported_total.tolist(),
              'rear_retained_geometry_level_final_1s': retained,
              'true_both_rear_crossing': highest > 0,
              'ascent_platform_support': 'all_six_supported_on_platform_s' in r.events,
              'net_base_x_m': end_x - start_x,
              'max_base_x_progress_m': float(np.max(arrays['qpos'][:, 0]) - start_x),
              'end_base_xyz_m': d.qpos[:3].tolist(),
              'max_rear_min_x_m': float(np.max(np.min(arrays['wheel_centers_step_start'][:, rear, 0], axis=1))),
              'full_route_evaluated': False, 'full_route_success': False, 'dataset_accepted': False,
              'audit': a.result(), 'route': r.result(), 'overspeed': overspeed,
              'wheel_names': [c.names[row[0]] for row in c.wheels], 'geom_names': a.geom_names,
              'trajectory_sha256': common.digest(folder / 'trajectory.npz'),
              'source_sha256': source_manifest(), 'scene_sha256': meta['scene_sha256'], 'original_scene_sha256': meta['original_scene_sha256'],
              'rear_steer_lock_positions_rad': meta['rear_lock_positions_rad'],
              'rear_lock_method': 'MuJoCo joint equality at checkpoint angles; rear steer actuator disabled; constraint reaction is not actuator torque limited',
              'wheel_target_sequence_sha256': hashlib.sha256(arrays['wheel_targets'].tobytes()).hexdigest(),
              'actual_control_sequence_sha256': hashlib.sha256(arrays['ctrl'].tobytes()).hexdigest(),
              'media_provenance': 'original_recorded_parent_and_continuation_states',
              'force_time_alignment': 'step-start dynamics and contact geometry; qpos/qvel at step boundaries',
              'wall_seconds': time.monotonic() - started}
    write(folder / 'result.json', result)
    return {k: result[k] for k in ('id', 'continuation_duration_s', 'termination_reason', 'diagnostic_stop_reasons',
                                  'process_valid', 'rear_highest_100ms_geometry_level', 'ascent_platform_support')}



def create_scene():
    RUN.mkdir(parents=True, exist_ok=True)
    if SCENE.exists():
        raise RuntimeError('locked scene already exists; preserve previous run')
    tree = ET.parse(BASE_SCENE)
    root = tree.getroot()
    if root.find('equality') is not None:
        raise RuntimeError('unexpected existing equality constraints')
    equality = ET.SubElement(root, 'equality')
    for name in ('left_rear_steer_joint', 'right_rear_steer_joint'):
        ET.SubElement(equality, 'joint', {'name':'lock_' + name, 'joint1':name,
                                         'polycoef':'0 0 0 0 0', 'active':'false', 'solref':'0.002 1'})
    tree.write(SCENE, encoding='unicode')


if __name__ == '__main__':
    create_scene()
    case = params('MR', 'SP', .001)
    case['id'] = 'T22_WMR_RSP_rear_lock_dt1000us'
    case['rear_steering'] = 'mechanically_locked_at_22s_checkpoint_angle'
    print(json.dumps({'plan':case, 'parent_scene_sha256':common.digest(BASE_SCENE), 'locked_scene_sha256':common.digest(SCENE)}), flush=True)
    print(json.dumps(run_case(case)), flush=True)
