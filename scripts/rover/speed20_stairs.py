"""Run the approved 20 cm, fifteen-step wheel-speed comparison."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import mujoco
import numpy as np

import speed20_experiment as common

RUN = common.RUN
LEVELS = common.LEVELS
RATIOS = ((1.25, 1., .75), (.75, 1., 1.25), (1., 1.25, .75), (1., .75, 1.25))


def eligible():
    path = RUN/'calibration.json'
    if not path.is_file():
        raise RuntimeError('calibration results are missing')
    data = json.loads(path.read_text())
    return set(map(float, data['eligible_approach_levels_rad_s']))


def save_case(case, arrays, result, initial, spec, scene):
    folder = RUN/'cases'/case['id']
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'trajectory.npz', **arrays, initial_integration_state=initial,
                        integration_state_spec=spec, simulation_dt=float(case['physics_dt_s']),
                        scene_sha256=common.digest(scene))
    result.update(trajectory_sha256=common.digest(folder/'trajectory.npz'),
                  scene_sha256=common.digest(scene), source_sha256=common.digest(__file__),
                  dataset_accepted=False, media_provenance='original_recorded_state')
    common.write_json(folder/'result.json', result)
    return result


def run_case(case):
    started = time.monotonic()
    folder = RUN/'cases'/case['id']
    folder.mkdir(parents=True, exist_ok=False)
    common.write_json(folder/'started.json', {'parameters':case, 'status':'started'})
    scene = common.FROZEN/'scene.xml'
    m = mujoco.MjModel.from_xml_path(str(scene))
    dt = float(case['physics_dt_s'])
    if dt != .001:
        raise ValueError('only 1 ms primary runs are supported')
    d = common.initialize(m, float(case.get('runup_m', 8.)))
    if case.get('yaw_deg', 0.):
        angle = math.radians(case['yaw_deg'])
        d.qpos[3:7] = [math.cos(angle/2), 0., 0., math.sin(angle/2)]
        mujoco.mj_forward(m, d)
    c = common.Controller(m, d)
    a = common.EpisodeAudit(m, common.URDF)
    r = common.RouteAudit(m, a, {'height':.2, 'depth':.2, 'stairs':15})
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    initial = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, initial, spec)
    cap = 180000
    arrays = {'time':np.zeros(cap+1), 'qpos':np.zeros((cap+1,m.nq)),
              'qvel':np.zeros((cap+1,m.nv)), 'ctrl':np.zeros((cap,m.nu)),
              'qfrc_actuator':np.zeros((cap,m.nv)), 'wheel_targets':np.zeros((cap,6)),
              'wheel_group_commands':np.zeros((cap,3)), 'contact_force_n':np.zeros((cap,6)),
              'contact_slip_tangent_m_s':np.full((cap,6),np.nan),
              'wheel_support_level':np.full((cap,6),-1,dtype=np.int8),
              'phase':np.zeros(cap,dtype=np.int8)}
    arrays['qpos'][0], arrays['qvel'][0] = d.qpos, d.qvel
    dofs = [row[3] for row in c.wheels]
    rear = [i for i,row in enumerate(c.wheels) if row[4] == 2]
    ratios = np.asarray(case.get('ratios',(1.,1.,1.)))
    contact_count = 0
    contact_time = None
    contact_x = None
    contact_wheels = None
    max_rear_x = -1e9
    peak_center_z = np.full(6, -1e9)
    crossing_time = np.zeros(15)
    crossing_peak = np.zeros(15)
    supported_crossing = np.zeros(15)
    brake_trigger = None
    brake_outside_range = False
    reasons = []
    reason = 'simulation_time_limit_180s'
    previous_events = {}
    front_event = 0
    route_braking = False
    route_stopping = False
    calibration = json.loads((RUN/'calibration.json').read_text())
    brake_table = sorted((float(v['virtual_crossing']['body_speed_m_s']),
                          float(v['brake_to_1_rad_s_distance_m']))
                         for v in calibration['results']
                         if v['attained'] and v['braking_completed'] and v['process_valid'])
    for step in range(cap):
        if d.time < .5:
            raw = np.zeros(3)
            phase = 0
        elif contact_time is None:
            raw = np.full(3, case['approach_rad_s'])
            phase = 1
        else:
            raw = case['climb_rad_s']*ratios
            phase = 2
        if case['kind'] == 'full_route' and contact_time is not None:
            front_x = max(float(d.geom_xpos[row[-1],0]+m.geom_size[row[-1],0]) for row in c.wheels if row[4] == 0)
            remaining = r.course['platform_end_x'] - front_x
            forward_v = max(0., float(d.qvel[0]))
            covering = next((distance for velocity,distance in brake_table if velocity >= forward_v), None)
            if covering is None:
                brake_outside_range = True
                route_braking = True
            elif remaining <= 1.25*covering + .5:
                route_braking = True
            if route_braking and brake_trigger is None:
                brake_trigger = {'time_s':float(d.time), 'base_x_m':float(d.qpos[0]),
                                 'remaining_m':float(remaining), 'actual_speed_m_s':forward_v,
                                 'outside_calibration':brake_outside_range}
            if route_braking:
                raw = np.ones(3)
                phase = 3
            if ('all_six_ground_after_stairs_s' in r.events and
                    float(d.qpos[0]) > r.course['end_x']+.3):
                route_stopping = True
            if route_stopping:
                raw = np.zeros(3)
                phase = 4
        goals = c.update(raw)
        arrays['ctrl'][step] = d.ctrl
        arrays['wheel_targets'][step] = goals
        arrays['wheel_group_commands'][step] = c.wheel_commands
        arrays['phase'][step] = phase
        mujoco.mj_step(m, d)
        a.step(m, d, float(d.time))
        r.step(m, d, float(d.time))
        forces, slip, front_force, support = common.contacts(m,d,c)
        if contact_time is None:
            contact_count = contact_count+1 if front_force > 5. else 0
            if contact_count >= 3:
                contact_time, contact_x = float(d.time), float(d.qpos[0])
                contact_wheels = d.qvel[dofs].tolist()
        levels = np.full(6,-1,dtype=np.int8)
        for wi, terrains in enumerate(support):
            ints = [int(name[2:])+1 for name in terrains if name.startswith('up') and name[2:].isdigit()]
            if 'platform' in terrains:
                ints.append(15)
            if ints:
                levels[wi] = max(ints)
        rear_x = min(float(d.geom_xpos[c.wheels[i][-1],0]) for i in rear)
        max_rear_x = max(max_rear_x,rear_x)
        for i,row in enumerate(c.wheels):
            peak_center_z[i] = max(peak_center_z[i],float(d.geom_xpos[row[-1],2]))
        for k in range(15):
            edge = 1.6 + .2*k
            height = .2*(k+1)
            crossed = all(float(d.geom_xpos[c.wheels[i][-1],0]) > edge+.005 and
                          float(d.geom_xpos[c.wheels[i][-1],2]-m.geom_size[c.wheels[i][-1],0]) > height-.010 for i in rear)
            crossing_time[k] = crossing_time[k]+dt if crossed else 0.
            crossing_peak[k] = max(crossing_peak[k],crossing_time[k])
            if crossed and all(levels[i] >= k+1 for i in rear):
                supported_crossing[k] += dt
        arrays['time'][step+1], arrays['qpos'][step+1], arrays['qvel'][step+1] = d.time,d.qpos,d.qvel
        arrays['qfrc_actuator'][step] = d.qfrc_actuator
        arrays['contact_force_n'][step], arrays['contact_slip_tangent_m_s'][step] = forces,slip
        arrays['wheel_support_level'][step] = levels
        reasons = common.process_reasons(a,r)
        if reasons:
            reason = 'physics_or_route_violation'
            break
        if case['kind'] != 'full_route' and 'all_six_supported_on_platform_s' in r.events:
            reason = 'ascent_platform_support_observed'
            break
        if case['kind'] == 'full_route' and route_stopping and a.continuous_final_stability_s >= 1.:
            reason = 'end_ground_stability_observed'
            break
    count = step+1
    trimmed = {k:v[:count+1] if k in ('time','qpos','qvel') else v[:count] for k,v in arrays.items()}
    audit,route = a.result(),r.result()
    complete = bool(case['kind'] == 'full_route' and reason == 'end_ground_stability_observed' and
                    not audit['quality_rejection_reasons'] and not route['rejection_reasons'] and not brake_outside_range)
    max_level = max((k+1 for k in range(15) if crossing_peak[k] >= .1-1e-9),default=0)
    names = [c.names[row[0]] for row in c.wheels]
    result = {'id':case['id'], 'kind':case['kind'], 'parameters':case,
              'height_cm':20, 'stairs':15, 'actual_duration_s':float(d.time),
              'termination_reason':reason, 'process_valid':not reasons, 'process_reasons':reasons,
              'first_failure_s':float(d.time) if reasons else None, 'wheel_names':names,
              'first_contact_time_s':contact_time, 'first_contact_base_x_m':contact_x,
              'first_contact_wheel_speeds_rad_s':contact_wheels,
              'max_rear_min_x_m':max_rear_x, 'max_wheel_center_z_m':dict(zip(names,peak_center_z.tolist())),
              'rear_crossing_peak_s_by_level':crossing_peak.tolist(),
              'rear_crossing_supported_total_s_by_level':supported_crossing.tolist(),
              'rear_highest_100ms_geometry_level':max_level,
              'true_both_rear_crossing':bool(crossing_peak[0]>=.1-1e-9),
              'ascent_platform_support': 'all_six_supported_on_platform_s' in route['events'],
              'full_route_success':complete, 'full_route_evaluated':case['kind']=='full_route',
              'brake_trigger':brake_trigger, 'brake_outside_calibration':brake_outside_range,
              'audit':audit, 'route':route, 'wall_seconds':time.monotonic()-started}
    return save_case(case,trimmed,result,initial,spec,scene)


def read_results(kind):
    rows = []
    for path in sorted((RUN/'cases').glob('*/result.json')):
        data = json.loads(path.read_text())
        if data.get('kind') == kind:
            rows.append(data)
    return rows


def rank(row):
    p = row['parameters']
    return (-int(row['ascent_platform_support']),-int(row['rear_highest_100ms_geometry_level']),
            -row['max_rear_min_x_m'],p['approach_rad_s'],p['climb_rad_s'],row['id'])


def execute(jobs, path):
    if path.exists():
        raise RuntimeError(f'{path} already exists; refusing to repeat')
    for job in jobs:
        if (RUN/'cases'/job['id']).exists():
            raise RuntimeError(f"case already exists: {job['id']}")
    common.write_json(path.with_suffix('.plan.json'),{'jobs':jobs,'requested':len(jobs)})
    if jobs:
        with mp.get_context('spawn').Pool(processes=min(6,len(jobs))) as pool:
            rows = []
            for result in pool.imap_unordered(run_case,jobs):
                rows.append(result)
                print(json.dumps({'completed':result['id'],'elapsed':result['actual_duration_s'],
                                  'termination':result['termination_reason'],'valid':result['process_valid'],
                                  'platform':result['ascent_platform_support'],'rears':result['rear_highest_100ms_geometry_level']},ensure_ascii=False),flush=True)
    else:
        rows = []
    rows.sort(key=lambda x:x['id'])
    common.write_json(path,{'results':rows,'actual_evaluations':len(rows),
                            'process_valid':sum(row['process_valid'] for row in rows),
                            'ascent_platform_support':sum(row['ascent_platform_support'] and row['process_valid'] for row in rows),
                            'full_route_success':sum(row['full_route_success'] for row in rows)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('main','distribution','routes'))
    args = parser.parse_args()
    if args.phase == 'main':
        allowed=eligible()
        jobs=[{'id':f'A{int(a):02d}_C{int(c):02d}','kind':'speed_matrix',
               'approach_rad_s':a,'climb_rad_s':c,'ratios':[1.,1.,1.],
               'runup_m':8.,'physics_dt_s':.001}
              for a in LEVELS for c in LEVELS if a in allowed and c in allowed]
        skipped=[f'A{int(a):02d}_C{int(c):02d}' for a in LEVELS for c in LEVELS if a not in allowed or c not in allowed]
        common.write_json(RUN/'main_skipped.json',{'ids':skipped,'reason':'flat calibration did not pass'})
        execute(jobs,RUN/'main_results.json')
    elif args.phase == 'distribution':
        if not (RUN/'main_results.json').exists():
            raise RuntimeError('main matrix has not completed')
        base=sorted((r for r in read_results('speed_matrix') if r['process_valid']),key=rank)[:3]
        jobs=[]
        for row in base:
            p=row['parameters']
            for i,ratios in enumerate(RATIOS,1):
                jobs.append({'id':f"{row['id']}_D{i}",'kind':'wheel_distribution',
                    'approach_rad_s':p['approach_rad_s'],'climb_rad_s':p['climb_rad_s'],
                    'ratios':ratios,'parent_id':row['id'],'runup_m':8.,'physics_dt_s':.001})
        execute(jobs,RUN/'distribution_results.json')
    else:
        if not (RUN/'distribution_results.json').exists():
            raise RuntimeError('distribution stage has not completed')
        base=sorted((r for r in read_results('speed_matrix')+read_results('wheel_distribution')
                     if r['process_valid'] and r['ascent_platform_support']),
                    key=lambda row:(row['parameters']['approach_rad_s'],row['parameters']['climb_rad_s'],row['id']))[:3]
        jobs=[]
        for row in base:
            p=row['parameters']
            for tag,runup,yaw in (('base',8.,0.),('runup4',4.,0.),('runup2',2.,0.),('yaw1',8.,1.)):
                jobs.append({'id':f"{row['id']}_{tag}",'kind':'full_route',
                    'approach_rad_s':p['approach_rad_s'],'climb_rad_s':p['climb_rad_s'],
                    'ratios':p['ratios'],'parent_id':row['id'],'runup_m':runup,
                    'yaw_deg':yaw,'physics_dt_s':.001})
        execute(jobs,RUN/'routes_results.json')


if __name__ == '__main__':
    main()
