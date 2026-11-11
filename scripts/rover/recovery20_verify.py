"""Audit saved recovery records and media without running further physics trials."""
from pathlib import Path
import hashlib
import json
import math

import mujoco
import numpy as np

import recovery20_experiment as exp
import render_dataset


def longest(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return int(max(np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1), default=0))


def verify():
    expected = {x['id'] for x in exp.jobs_primary()}
    paths = sorted((exp.RUN / 'cases').glob('*/result.json'))
    actual = {x.parent.name for x in paths}
    assert expected.issubset(actual)
    assert len(actual) <= 46
    checked = []; failures = []; rows = []; wheel_stats = {}
    for path in paths:
        r = json.loads(path.read_text()); rows.append(r); tag = r['id']; folder = path.parent
        try:
            m, d, c, audit, route, meta = exp.restore(r['physics_step_s'])
            assert exp.common.digest(folder / 'trajectory.npz') == r['trajectory_sha256']
            assert r['checkpoint_id'] == meta['checkpoint_id'] and r['parent_step'] == meta['parent_step']
            assert r['source_sha256'] == meta['source_sha256']
            with np.load(folder / 'trajectory.npz', allow_pickle=False) as z:
                t, q, v, ctrl, tau = z['time'], z['qpos'], z['qvel'], z['ctrl'], z['qfrc_actuator']
                dt = float(z['simulation_dt']); n = len(ctrl); sub = round(.001 / dt)
                assert len(t) == len(q) == len(v) == n + 1 and abs(t[0] - 22) < 1e-8
                assert np.array_equal(q[0], d.qpos) and np.array_equal(v[0], d.qvel)
                assert np.max(abs(np.diff(t) - dt)) < 1e-10
                assert abs(t[-1] - r['actual_duration_s']) < 1e-9 and n <= round(30 / dt)
                assert all(np.isfinite(a).all() for a in (q, v, ctrl, tau))
                assert r['dataset_accepted'] is False and r['full_route_evaluated'] is False
                if sub > 1:
                    for offset in range(1, sub):
                        ix = np.arange(offset, n, sub)
                        assert np.array_equal(ctrl[ix], ctrl[ix - offset])
                command = z['wheel_group_commands']; times = np.arange(n) // sub * sub * dt
                for role, flag in enumerate(exp.WHEELS[r['parameters']['wheel']]):
                    assert np.max(abs(command[:, role] - (3 + flag * np.minimum(times, 3)))) < 1e-10
                rocker = z['rocker_targets_rad']
                assert np.max(abs(rocker)) <= math.radians(10) + 1e-12
                assert np.max(abs(np.diff(rocker[::sub], axis=0))) <= math.radians(10) * .001 + 1e-12
                assert np.all(abs(rocker[times >= 24]) < 1e-12)
                by_wheel = []
                for i, (joint, act, qp, vp, role, left, geom) in enumerate(c.wheels):
                    assert np.max(abs(tau[:, vp])) <= 50.001
                    by_wheel.append({'name': c.names[joint], 'group': role,
                        'mean_speed_rad_s': float(v[:-1, vp].mean()), 'mean_torque_nm': float(tau[:, vp].mean()),
                        'saturation_fraction': float((abs(tau[:, vp]) >= 49.99).mean()),
                        'mean_contact_slip_m_s': float(np.nanmean(z['contact_slip_tangent_m_s'][:, i])),
                        'peak_center_z_m': float(np.max(z['wheel_centers_step_start'][:, i, 2]))})
                wheel_stats[tag] = by_wheel
                for name, vp in audit.velocity_dofs.items():
                    values = abs(v[1:, vp]); recorded = r['overspeed'][name]
                    for label, threshold in [('urdf', audit.velocity_limits[name]), ('audit', audit.velocity_limits[name] + .01)]:
                        over = values > threshold
                        assert abs(over.sum() * dt - recorded[label]['total_duration_s']) < 1e-9
                        assert abs(longest(over) * dt - recorded[label]['longest_duration_s']) < 1e-9
                    maximum = max(audit.velocity_peak_by_joint_rad_s[name], float(values.max()))
                    assert abs(maximum - r['audit']['joint_velocity_peak_rad_s'][name]) < 1e-9
                for joint, act, qp, vp in c.rockers:
                    assert np.max(abs(tau[:, vp])) <= 100.001
                for joint, act, qp, vp in c.steer:
                    assert np.max(abs(tau[:, vp])) <= 40.001
                assert hashlib.sha256(ctrl.tobytes()).hexdigest() == r['actual_control_sequence_sha256']
                centers = z['wheel_centers_step_start']; rear = [i for i, row in enumerate(c.wheels) if row[4] == 2]
                highest = 0
                for k in range(15):
                    crossed = np.ones(n, dtype=bool)
                    for i in rear:
                        radius = m.geom_size[c.wheels[i][-1], 0]
                        crossed &= (centers[:, i, 0] > 1.605 + .2 * k) & (centers[:, i, 2] - radius > .2 * (k + 1) - .010)
                    peak = longest(crossed) * dt
                    assert abs(peak - r['rear_crossing_peak_s_by_level'][k]) < 1e-8
                    if peak >= .1 - 1e-9:
                        highest = k + 1
                assert highest == r['rear_highest_100ms_geometry_level']
            media = json.loads((folder / 'media.json').read_text())
            assert media['trajectory_sha256'] == r['trajectory_sha256']
            assert media['parent_trajectory_sha256'] == r['parent_trajectory_sha256']
            assert exp.common.digest(media['video_path']) == media['video_sha256']
            probe = render_dataset._probe(Path(media['video_path']))
            assert probe['frames'] == media['frames'] and probe['width'] == 960 and probe['height'] == 360
            assert abs(media['first_time_s'] - 17) < 1e-8 and abs(media['last_time_s'] - r['actual_duration_s']) < 1e-8
            curves = json.loads((folder / 'curves.json').read_text())
            assert len(curves['joints']) == 12
            for name, joint in curves['joints'].items():
                assert abs(max(abs(row[1]) for row in joint['rows']) - joint['max_abs_velocity_rad_s']) < 1e-6
            checked.append(tag)
        except Exception as error:
            failures.append({'id': tag, 'error': repr(error)})
    report = {'passed': not failures and len(checked) == len(rows), 'checked': len(checked), 'failures': failures,
              'checks': ['parent checkpoint and source hashes', 'start state and timing', 'commands and force limits',
                         'raw velocity events and audit peaks', 'sustained rear crossing', 'video hash, frames and time',
                         '12-joint curves with retained speed extrema'], 'additional_physics_trials': 0}
    exp.write(exp.RUN / 'verification.json', report)
    exp.write(exp.RUN / 'wheel_diagnostics.json', wheel_stats)
    primary = [r for r in rows if r['physics_step_s'] == .001]
    summary = {'completed': len(rows), 'primary_conditions': len(primary), 'new_action_conditions': sum(not r['parameters']['is_control'] for r in primary),
               'controls': sum(r['parameters']['is_control'] for r in primary), 'process_valid': sum(r['process_valid'] for r in primary),
               'speed_violation_cases': sum(any(v['audit']['event_count'] for v in r['overspeed'].values()) for r in primary),
               'sustained_crossing': sum(r['true_both_rear_crossing'] for r in primary),
               'platform': sum(r['ascent_platform_support'] for r in primary),
               'unique_actual_control_sequences': len({r['actual_control_sequence_sha256'] for r in primary}),
               'full_route': 'not_evaluated', 'dataset_accepted': 0,
               'at_time_limit': sum(r['termination_reason'] == 'continuation_time_limit_30s' for r in primary),
               'total_continuation_simulation_s': sum(r['continuation_duration_s'] for r in rows),
               'checkpoint_rebuilds': len(list((exp.RUN / 'checkpoints').glob('*/checkpoint.json'))),
               'restore_reference_continuations': 2 * len(list((exp.RUN / 'checkpoints').glob('*/gate.json'))),
               'execution_errors': len((exp.RUN / 'execution_errors.jsonl').read_text().splitlines()) if (exp.RUN / 'execution_errors.jsonl').exists() else 0,
               'refinement': json.loads((exp.RUN / 'refinement_selection.json').read_text())}
    exp.write(exp.RUN / 'summary.json', summary)
    if not report['passed']:
        raise RuntimeError(json.dumps(report))
    print(json.dumps({'verification': report, 'summary': summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    verify()
