"""Reproducible 20 cm stair speed experiment with preserved per-step data."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path('/home/agilex/wzy/rover')
FROZEN = ROOT / 'review_v4/dynamic_search20'
BASELINE = ROOT / 'review_v4/high20_slow_neutral_v2/trajectory.npz'
URDF = ROOT / 'assets/mars_rover/urdf/mars_rover.urdf'
RUN = ROOT / 'runs/speed20_15steps_20261111'
sys.path.insert(0, str(FROZEN / 'sources'))
from episode_audit import EpisodeAudit
from verify_dataset import RouteAudit

LEVELS = (3., 6., 12., 20., 30., 40.)
LIMIT = 52.359878
DT = .001
RADIUS = .163


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def orientation(d):
    w, x, y, z = map(float, d.qpos[3:7])
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(np.clip(2*(w*y-z*x), -1, 1)),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


class Controller:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.wheel_commands = np.zeros(3)
        self.rocker_commands = np.zeros(2)
        self.yaw_previous = None
        self.yaw_rate = 0.
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '' for j in range(m.njnt)]
        acts = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.wheels, self.steer, self.rockers = [], [], []
        for j, name in enumerate(self.names):
            row = (j, acts[j], int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])) if j in acts else None
            if name.endswith('_wheel_joint'):
                role = next(i for i, r in enumerate(('front', 'middle', 'rear')) if r in name)
                geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name.replace('_joint', '') + '_col_1')
                self.wheels.append((*row, role, 'left' in name, geom))
            elif name.endswith('_steer_joint'):
                self.steer.append(row)
            elif name.endswith('_rocker_joint'):
                self.rockers.append(row)
        if (len(self.wheels), len(self.steer), len(self.rockers)) != (6, 4, 2):
            raise ValueError('unexpected movable joint mapping')
        self.wheel_geom = {row[-1]: i for i, row in enumerate(self.wheels)}

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


def initialize(m, distance):
    d = mujoco.MjData(m)
    with np.load(BASELINE, allow_pickle=False) as a:
        d.qpos[:] = a['qpos'][0]
        d.qvel[:] = a['qvel'][0]
    mujoco.mj_forward(m, d)
    front = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '').endswith('front_wheel_col_1')]
    if len(front) != 2:
        raise ValueError('expected two front wheel geoms')
    front_x = max(float(d.geom_xpos[g, 0]+m.geom_size[g, 0]) for g in front)
    d.qpos[0] += 1.6 - distance - front_x
    mujoco.mj_forward(m, d)
    return d


def process_reasons(audit, route=None):
    result = [x for x in audit.result()['quality_rejection_reasons'] if not x.startswith('final_')]
    if route:
        result += [x for x in route.result()['rejection_reasons'] if not x.startswith(('route_missing:', 'route_final_'))]
    return result


def flat_route_reasons(m, d, controller):
    roll, pitch, yaw = orientation(d)
    reasons = []
    if abs(float(d.qpos[1])) > .5:
        reasons.append('route_outside_corridor')
    if abs(yaw) > math.radians(30):
        reasons.append('route_yaw_over_30deg')
    if abs(roll) > math.radians(15):
        reasons.append('route_roll_over_15deg')
    if abs(pitch) > math.radians(60):
        reasons.append('route_pitch_over_60deg')
    if any(abs(float(d.qpos[row[2]])) > math.radians(30) for row in controller.steer):
        reasons.append('route_steer_over_30deg')
    return reasons


def contacts(m, d, controller):
    forces = np.zeros(6)
    slip_sum = np.zeros(6)
    slip_weight = np.zeros(6)
    force = np.zeros(6)
    velocity = np.zeros(6)
    front_force = 0.
    wheel_support = [set() for _ in range(6)]
    for i in range(d.ncon):
        ct = d.contact[i]
        if int(ct.efc_address) < 0:
            continue
        a, b = int(ct.geom1), int(ct.geom2)
        if a in controller.wheel_geom:
            geom, terrain = a, b
        elif b in controller.wheel_geom:
            geom, terrain = b, a
        else:
            continue
        if int(m.geom_bodyid[terrain]) != 0:
            continue
        wi = controller.wheel_geom[geom]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, terrain) or ''
        mujoco.mj_contactForce(m, d, i, force)
        fn = max(0., float(force[0]))
        forces[wi] += fn
        if fn > 5 and name.startswith('up') and controller.wheels[wi][4] == 0:
            front_force += fn
        if fn > 5 and abs(float(ct.frame[2])) > .7:
            wheel_support[wi].add(name)
        if fn > 0:
            body = int(m.geom_bodyid[geom])
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            point_vel = velocity[3:] + np.cross(velocity[:3], ct.pos - d.xpos[body])
            frame = ct.frame.reshape(3, 3)
            tangential = float(np.linalg.norm((frame @ point_vel)[1:]))
            slip_sum[wi] += fn*tangential
            slip_weight[wi] += fn
    slip = np.full(6, np.nan)
    np.divide(slip_sum, slip_weight, out=slip, where=slip_weight > 0)
    return forces, slip, front_force, wheel_support


def flat_scene():
    tree = ET.parse(FROZEN / 'scene.xml')
    world = tree.getroot().find('worldbody')
    for geom in list(world.findall('geom')):
        name = geom.get('name', '')
        if name == 'platform' or name.startswith(('up', 'down')):
            world.remove(geom)
    target = RUN / 'flat_scene.xml'
    tree.write(target, encoding='unicode')
    return target


def calibrate(speed):
    name = f'flat_{int(speed):02d}'
    folder = RUN / 'cases' / name
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    scene = RUN / 'flat_scene.xml'
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = initialize(m, 8.)
    c = Controller(m, d)
    audit = EpisodeAudit(m, URDF)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    initial = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, initial, spec)
    names = [c.names[row[0]] for row in c.wheels]
    dofs = [row[3] for row in c.wheels]
    acts = [row[1] for row in c.wheels]
    cap = 180000
    arrays = {'time': np.zeros(cap+1), 'qpos': np.zeros((cap+1, m.nq)),
              'qvel': np.zeros((cap+1, m.nv)), 'ctrl': np.zeros((cap, m.nu)),
              'qfrc_actuator': np.zeros((cap, m.nv)), 'wheel_targets': np.zeros((cap, 6)),
              'wheel_group_commands': np.zeros((cap, 3)), 'contact_force_n': np.zeros((cap, 6)),
              'contact_slip_tangent_m_s': np.full((cap, 6), np.nan)}
    arrays['qpos'][0], arrays['qvel'][0] = d.qpos, d.qvel
    history = deque(maxlen=300)
    virtual_crossing = None
    brake_to_one_x = None
    stopped_x = None
    brake_count = 0
    stop_count = 0
    reason = 'simulation_time_limit_180s'
    reasons = []
    phase = 'settle'
    samples_at_crossing = []
    for step in range(cap):
        if d.time < .5:
            target = 0.
        elif virtual_crossing is None:
            phase, target = 'approach', speed
        elif brake_to_one_x is None:
            phase, target = 'brake_to_one', 1.
        else:
            phase, target = 'stop', 0.
        targets = c.update([target]*3)
        arrays['ctrl'][step] = d.ctrl
        arrays['wheel_targets'][step] = targets
        arrays['wheel_group_commands'][step] = c.wheel_commands
        mujoco.mj_step(m, d)
        audit.step(m, d, float(d.time))
        forces, slips, _, _ = contacts(m, d, c)
        actual = d.qvel[dofs].copy()
        body = float(d.qvel[0])
        rolling = RADIUS * float(np.mean(actual))
        good = bool(np.all((actual >= .9*speed) & (actual <= 1.1*speed)) and abs(body-rolling) <= .15*abs(rolling))
        history.append((good, *actual.tolist(), body, rolling))
        arrays['time'][step+1], arrays['qpos'][step+1], arrays['qvel'][step+1] = d.time, d.qpos, d.qvel
        arrays['qfrc_actuator'][step] = d.qfrc_actuator
        arrays['contact_force_n'][step] = forces
        arrays['contact_slip_tangent_m_s'][step] = slips
        front_x = max(float(d.geom_xpos[row[-1], 0]+m.geom_size[row[-1], 0]) for row in c.wheels if row[4] == 0)
        if virtual_crossing is None and front_x >= 1.6:
            virtual_crossing = {'time_s': float(d.time), 'front_x_m': front_x, 'base_x_m': float(d.qpos[0]),
                                'wheel_speeds_rad_s': actual.tolist(), 'body_speed_m_s': body,
                                'window_0_3s_all_good': len(history) == 300 and all(x[0] for x in history)}
            samples_at_crossing = list(history)
        elif virtual_crossing is not None and brake_to_one_x is None:
            brake_count = brake_count+1 if np.all(np.abs(actual-1.) <= .2) and abs(body-RADIUS) <= .1 else 0
            if brake_count >= 300:
                brake_to_one_x = float(d.qpos[0])
        elif brake_to_one_x is not None:
            stop_count = stop_count+1 if np.all(np.abs(actual) < .2) and abs(body) < .03 else 0
            if stop_count >= 1000:
                stopped_x = float(d.qpos[0])
                reason = 'flat_calibration_complete'
                break
        reasons = process_reasons(audit) + flat_route_reasons(m, d, c)
        if reasons:
            reason = 'physics_violation'
            break
        if virtual_crossing is not None and d.time-virtual_crossing['time_s'] > 20.:
            reason = 'braking_or_stopping_not_completed_within_20s'
            break
    count = step+1
    trimmed = {k: v[:count+1] if k in ('time', 'qpos', 'qvel') else v[:count] for k, v in arrays.items()}
    np.savez_compressed(folder/'trajectory.npz', **trimmed, initial_integration_state=initial,
                        integration_state_spec=spec, wheel_names=np.asarray(names), simulation_dt=DT,
                        scene_sha256=digest(scene))
    window = np.asarray(samples_at_crossing, dtype=float)
    outcome = {'id': name, 'kind': 'flat_calibration', 'target_rad_s': speed, 'runup_m': 8.,
               'actual_duration_s': float(d.time), 'termination_reason': reason,
               'process_valid': not reasons, 'process_reasons': reasons,
               'virtual_crossing': virtual_crossing,
               'attained': bool(virtual_crossing and virtual_crossing['window_0_3s_all_good']),
               'brake_to_1_rad_s_distance_m': None if brake_to_one_x is None else brake_to_one_x-virtual_crossing['base_x_m'],
               'stop_distance_m': None if stopped_x is None else stopped_x-virtual_crossing['base_x_m'],
               'braking_completed': reason == 'flat_calibration_complete',
               'crossing_window_wheel_min_rad_s': window[:, 1:7].min(axis=0).tolist() if len(window) else None,
               'crossing_window_wheel_max_rad_s': window[:, 1:7].max(axis=0).tolist() if len(window) else None,
               'audit': audit.result(), 'dataset_accepted': False, 'scene_sha256': digest(scene),
               'trajectory_sha256': digest(folder/'trajectory.npz'), 'wall_seconds': time.monotonic()-started,
               'wheel_names': names, 'target_max_rad_s': speed, 'torque_limit_nm': 50.,
               'speed_slew_rad_s2': 30., 'physics_dt_s': DT}
    write_json(folder/'result.json', outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'calibrate'])
    args = parser.parse_args()
    if args.command == 'prepare':
        RUN.mkdir(parents=True, exist_ok=False)
        flat = flat_scene()
        write_json(RUN/'plan.json', {'height_m': .2, 'depth_m': .2, 'slope_deg': 45,
            'ascending_steps': 15, 'descending_steps': 15, 'runup_m': 8., 'levels_rad_s': LEVELS,
            'physics_dt_s': DT, 'wheel_torque_limit_nm': 50., 'wheel_gain': 40.,
            'wheel_slew_rad_s2': 30., 'frozen_scene_sha256': digest(FROZEN/'scene.xml'),
            'flat_scene_sha256': digest(flat), 'frozen_backend_sha256': digest(FROZEN/'backend.py'),
            'urdf_sha256': digest(URDF), 'source_sha256': digest(__file__),
            'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
            'status': 'prepared_no_simulation'})
        print(json.dumps({'height_cm':20, 'stairs':15, 'phase':'prepared', 'simulation_started':False}), flush=True)
    else:
        if not (RUN/'plan.json').is_file():
            raise RuntimeError('run prepare before calibration')
        if list((RUN/'cases').glob('flat_*')) if (RUN/'cases').exists() else False:
            raise RuntimeError('calibration already started; refusing to repeat')
        with mp.get_context('spawn').Pool(processes=6) as pool:
            rows = pool.map(calibrate, LEVELS)
        write_json(RUN/'calibration.json', {'results':rows, 'calls':len(rows),
            'eligible_approach_levels_rad_s':[r['target_rad_s'] for r in rows if r['attained'] and r['braking_completed'] and r['process_valid']]})
        print(json.dumps({'phase':'calibration_complete','results':[{k:r[k] for k in ('id','attained','braking_completed','termination_reason','process_reasons')} for r in rows]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
