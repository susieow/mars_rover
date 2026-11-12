"""One physics-step comparison against a preserved 1000 Hz experiment."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np

import speed20_experiment as common
import recovery20_amp20_experiment as wave
from overspeed20_stairs import summarize_speed

ROOT = common.ROOT
RUN = ROOT / 'runs/physics5000_single_20261112'
BASELINE = ROOT / 'runs/ground_mu6_constant_wheels_20261112/cases/ground_mu6_v6_rocker_SP'
SCENE = ROOT / 'runs/friction20_t22_20261111/scenes/scene_mu6.xml'
DT = 0.0002
CONTROL_DT = 0.001
SUBSTEPS = 5
LIMIT = 120.0
AFTER_TRIGGER = 30.0
CASE_ID = 'ground_mu6_v6_rocker_SP_dt0200us_ctrl1000hz'


def main():
    folder = RUN / 'cases' / CASE_ID
    folder.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((BASELINE / 'result.json').read_text(encoding='utf-8'))
    assert abs(common.DT - CONTROL_DT) < 1e-15
    assert abs(SUBSTEPS * DT - CONTROL_DT) < 1e-15
    assert common.digest(SCENE) == baseline['scene_sha256']
    assert common.digest(BASELINE / 'trajectory.npz') == baseline['trajectory_sha256']
    params = dict(baseline['parameters'], id=CASE_ID, physics_dt_s=DT,
                  control_dt_s=CONTROL_DT, event_monitor_dt_s=CONTROL_DT,
                  physics_hz=5000, control_hz=1000,
                  audit_sample_dt_s=DT, substeps_per_control=SUBSTEPS)
    provenance = {
        'baseline_result_path': str(BASELINE / 'result.json'),
        'baseline_result_sha256': common.digest(BASELINE / 'result.json'),
        'baseline_trajectory_sha256': baseline['trajectory_sha256'],
        'scene_sha256': common.digest(SCENE),
        'source_sha256': common.digest(__file__),
        'source_commit': subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'dependencies': {str(Path(p)): common.digest(p) for p in (
            common.__file__, wave.__file__, inspect.getfile(common.EpisodeAudit),
            inspect.getfile(common.RouteAudit))},
        'new_physical_runs_planned': 1,
        'repeatability_assessed': False,
        'decision_before_run': (
            'Compare the full-run maximum using the existing active robot-terrain '
            'contact-distance metric. Report both 0.2 ms and 1 ms sampling. '
            'A lower peak supports only this case; do not change the final suite '
            'or infer a general convergence claim from one run.'),
        'changed_setting': 'physics timestep only; controller and event monitor remain 1 ms',
    }
    common.write_json(folder / 'started.json', {'parameters': params, **provenance})
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    with np.load(BASELINE / 'trajectory.npz', allow_pickle=False) as saved:
        initial = saved['initial_integration_state'].copy()
        original_qpos = saved['qpos'][0].copy()
        original_qvel = saved['qvel'][0].copy()
    mujoco.mj_setState(m, d, initial, spec)
    mujoco.mj_forward(m, d)
    actual_initial = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, actual_initial, spec)
    assert np.array_equal(actual_initial, initial)
    assert np.array_equal(d.qpos, original_qpos)
    assert np.array_equal(d.qvel, original_qvel)
    c = common.Controller(m, d)
    audit = common.EpisodeAudit(m, common.URDF)
    sampled_audit = common.EpisodeAudit(m, common.URDF)
    sampled_audit.dt = CONTROL_DT
    route = common.RouteAudit(m, audit, {'height': .2, 'depth': .2, 'stairs': 15})
    cap = round(LIMIT / DT)
    arrays = {name: np.zeros((cap + 1, size))
              for name, size in [('qpos', m.nq), ('qvel', m.nv)]}
    arrays['time'] = np.zeros(cap + 1)
    for name, size in [('ctrl', m.nu), ('qfrc_actuator', m.nv), ('wheel_targets', 6),
                       ('wheel_group_commands', 3), ('rocker_targets_rad', 2),
                       ('contact_force_n', 6), ('contact_slip_tangent_m_s', 6)]:
        arrays[name] = np.zeros((cap, size))
    arrays['wheel_support_level'] = np.full((cap, 6), -1, dtype=np.int8)
    arrays['rear_geometry_level'] = np.zeros(cap, dtype=np.int8)
    arrays['rear_supported_level'] = np.zeros(cap, dtype=np.int8)
    arrays['time'][0], arrays['qpos'][0], arrays['qvel'][0] = d.time, d.qpos, d.qvel
    rear = [i for i, row in enumerate(c.wheels) if row[4] == 2]
    wheels = {row[-1]: row[4] for row in c.wheels}
    stair_ids = {i for i in range(m.ngeom) if
                 (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('up')}
    first_contact = trigger = first_failure = first_crossing = None
    peak_at_trigger = None
    crossing, peaks, supported = np.zeros(15), np.zeros(15), np.zeros(15)
    contactforce = np.zeros(6)
    reason = 'simulation_time_limit_120s'
    wall_started = time.monotonic()
    for step in range(cap):
        if step % SUBSTEPS == 0:
            elapsed = 0 if trigger is None else float(d.time) - trigger
            _, rocker = wave.targets(wave.params('H', 'SP'), elapsed)
            if trigger is None:
                rocker[:] = 0.0
            goal = c.update(np.full(3, 6.0))
            c.rocker_commands[:] = rocker
            for i, (_, act, pos, vel) in enumerate(c.rockers):
                desired = float(np.clip(4 * (rocker[i] - float(d.qpos[pos])), -.7, .7))
                d.ctrl[act] = np.clip(150 * (desired - float(d.qvel[vel])), -100, 100)
        arrays['ctrl'][step] = d.ctrl
        arrays['wheel_targets'][step] = goal
        arrays['wheel_group_commands'][step] = c.wheel_commands
        arrays['rocker_targets_rad'][step] = rocker
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError('nonfinite simulation state')
        audit.step(m, d, float(d.time))
        route.step(m, d, float(d.time))
        if first_failure is None and common.process_reasons(audit, route):
            first_failure = float(d.time)
        monitor_tick = (step + 1) % SUBSTEPS == 0
        if monitor_tick:
            sampled_audit.step(m, d, float(d.time))
        if monitor_tick and first_contact is None:
            for i in range(d.ncon):
                con = d.contact[i]
                g1, g2 = int(con.geom1), int(con.geom2)
                if con.efc_address >= 0 and ((wheels.get(g1) == 0 and g2 in stair_ids)
                                            or (wheels.get(g2) == 0 and g1 in stair_ids)):
                    mujoco.mj_contactForce(m, d, i, contactforce)
                    if contactforce[0] > 0:
                        first_contact = float(d.time)
                        break
        forces, slip, _, support = common.contacts(m, d, c)
        levels = np.full(6, -1, dtype=np.int8)
        for wi, terrains in enumerate(support):
            values = [int(name[2:]) + 1 for name in terrains
                      if name.startswith('up') and name[2:].isdigit()]
            if 'platform' in terrains:
                values.append(15)
            if values:
                levels[wi] = max(values)
        centers = d.geom_xpos[[row[-1] for row in c.wheels]]
        level = supported_level = 0
        for k in range(15):
            crossed = all(float(centers[i, 0]) > 1.6 + .2 * k + .005
                          and float(centers[i, 2] - m.geom_size[c.wheels[i][-1], 0])
                          > .2 * (k + 1) - .010 for i in rear)
            crossing[k] = crossing[k] + DT if crossed else 0.0
            peaks[k] = max(peaks[k], crossing[k])
            if crossed:
                level = k + 1
                if all(levels[i] >= k + 1 for i in rear):
                    supported[k] += DT
                    supported_level = k + 1
            if k == 0 and first_crossing is None and crossing[k] >= .1 - 1e-9:
                first_crossing = float(d.time)
        arrays['time'][step + 1] = d.time
        arrays['qpos'][step + 1], arrays['qvel'][step + 1] = d.qpos, d.qvel
        arrays['qfrc_actuator'][step] = d.qfrc_actuator
        arrays['contact_force_n'][step] = forces
        arrays['contact_slip_tangent_m_s'][step] = slip
        arrays['wheel_support_level'][step] = levels
        arrays['rear_geometry_level'][step] = level
        arrays['rear_supported_level'][step] = supported_level
        if monitor_tick and trigger is None and first_contact is not None and d.time - first_contact >= 2.0 - 1e-9:
            if d.qpos[0] - arrays['qpos'][step + 1 - round(2 / DT), 0] < .02:
                trigger = float(d.time)
                peak_at_trigger = audit.max_contact_penetration_m
        if (step + 1) % 5000 == 0:
            print(json.dumps({'progress_s': float(d.time), 'trigger_s': trigger,
                              'max_penetration_mm': 1000 * audit.max_contact_penetration_m,
                              'wall_s': time.monotonic() - wall_started}), flush=True)
        if trigger is not None and d.time - trigger >= AFTER_TRIGGER - 1e-9:
            reason = 'post_stall_observation_30s'
            break
    n = step + 1
    arrays = {k: v[:n + 1] if k in ('time', 'qpos', 'qvel') else v[:n]
              for k, v in arrays.items()}
    np.savez_compressed(folder / 'trajectory.npz', **arrays,
                        initial_integration_state=initial, simulation_dt=DT,
                        control_dt=CONTROL_DT, scene_sha256=common.digest(SCENE))
    overspeed = summarize_speed(arrays['time'], arrays['qvel'], audit.velocity_dofs,
                               audit.velocity_limits, DT)
    reasons = common.process_reasons(audit, route)
    highest = max((k + 1 for k in range(15) if peaks[k] >= .1 - 1e-9), default=0)
    data = {
        'id': CASE_ID, 'parameters': params, 'physics_step_s': DT,
        'control_step_s': CONTROL_DT, 'height_cm': 20, 'stairs': 15,
        'actual_duration_s': float(d.time), 'termination_reason': reason,
        'first_contact_s': first_contact, 'stall_trigger_s': trigger,
        'first_failure_s': first_failure, 'process_reasons': reasons,
        'process_valid': not reasons, 'true_both_rear_crossing': highest > 0,
        'first_rear_crossing_s': first_crossing,
        'rear_highest_100ms_geometry_level': highest,
        'rear_retained_geometry_level_final_1s': int(np.min(
            arrays['rear_geometry_level'][-min(n, round(1 / DT)):])),
        'rear_crossing_supported_total_s_by_level': supported.tolist(),
        'ascent_platform_support': 'all_six_supported_on_platform_s' in route.events,
        'full_route_evaluated': False, 'full_route_success': False,
        'dataset_accepted': False, 'audit': audit.result(), 'route': route.result(),
        'overspeed': overspeed, 'audit_sampled_1000hz': sampled_audit.result(),
        'pre_trigger_max_penetration_m': peak_at_trigger,
        'wheel_names': [c.names[x[0]] for x in c.wheels],
        'max_base_x_progress_m': float(np.max(arrays['qpos'][:, 0]) - arrays['qpos'][0, 0]),
        'net_base_x_m': float(arrays['qpos'][-1, 0] - arrays['qpos'][0, 0]),
        'trajectory_sha256': common.digest(folder / 'trajectory.npz'),
        'initial_integration_state_exact_match': True,
        'wall_seconds': time.monotonic() - wall_started, **provenance,
    }
    common.write_json(folder / 'result.json', data)
    print(json.dumps({'completed': True, 'result_path': str(folder / 'result.json'),
                      'max_penetration_mm': 1000 * audit.max_contact_penetration_m,
                      'baseline_max_penetration_mm': 1000 * baseline['audit']['max_contact_penetration_m'],
                      'first_contact_s': first_contact, 'trigger_s': trigger,
                      'highest_level': highest, 'process_reasons': reasons}), flush=True)


if __name__ == '__main__':
    main()
