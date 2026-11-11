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

RUN = common.ROOT/'runs/overspeed20_20261111'
ORIGINAL_RUN = common.RUN
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
    substeps = round(.001 / dt)
    if dt not in (.001, .0005, .00025) or abs(substeps * dt - .001) > 1e-12:
        raise ValueError('unsupported physics timestep')
    m.opt.timestep = dt
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
    cap = int(round(180. / dt))
    arrays = {'time':np.zeros(cap+1), 'qpos':np.zeros((cap+1,m.nq)),
              'qvel':np.zeros((cap+1,m.nv)), 'ctrl':np.zeros((cap,m.nu)),
              'qfrc_actuator':np.zeros((cap,m.nv)), 'wheel_targets':np.zeros((cap,6)),
              'wheel_group_commands':np.zeros((cap,3)), 'contact_force_n':np.zeros((cap,6)),
              'contact_slip_tangent_m_s':np.full((cap,6),np.nan),
              'wheel_support_level':np.full((cap,6),-1,dtype=np.int8),
              'phase':np.zeros(cap,dtype=np.int8),
              'qacc_step_start':np.zeros((cap,m.nv)),
              'qfrc_bias_step_start':np.zeros((cap,m.nv)),
              'qfrc_passive_step_start':np.zeros((cap,m.nv)),
              'qfrc_constraint_step_start':np.zeros((cap,m.nv)),
              'contacts_geom_ids':np.full((cap,64,2),-1,dtype=np.int16),
              'contacts_normal_force_n':np.zeros((cap,64),dtype=np.float32)}
    arrays['qpos'][0], arrays['qvel'][0] = d.qpos, d.qvel
    dofs = [row[3] for row in c.wheels]
    rear = [i for i,row in enumerate(c.wheels) if row[4] == 2]
    ratios = np.asarray(case.get('ratios',(1.,1.,1.)))
    contact_count = 0
    contact_steps = int(round(.003/dt))
    gate_checked = False
    last_goals = np.zeros(6)
    phase = 0
    raw = np.zeros(3)
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
    calibration = json.loads((ORIGINAL_RUN/'calibration.json').read_text())
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
        if step % substeps == 0:
            last_goals = c.update(raw)
        goals = last_goals
        arrays['ctrl'][step] = d.ctrl
        arrays['wheel_targets'][step] = goals
        arrays['wheel_group_commands'][step] = c.wheel_commands
        arrays['phase'][step] = phase
        mujoco.mj_step(m, d)
        arrays['qacc_step_start'][step] = d.qacc
        arrays['qfrc_bias_step_start'][step] = d.qfrc_bias
        arrays['qfrc_passive_step_start'][step] = d.qfrc_passive
        arrays['qfrc_constraint_step_start'][step] = d.qfrc_constraint
        if d.ncon > 64:
            raise RuntimeError('contact record capacity exceeded')
        cf = np.zeros(6)
        for ci in range(d.ncon):
            con = d.contact[ci]
            arrays['contacts_geom_ids'][step,ci] = [con.geom1,con.geom2]
            if con.efc_address >= 0:
                mujoco.mj_contactForce(m,d,ci,cf)
                arrays['contacts_normal_force_n'][step,ci] = cf[0]
        a.step(m, d, float(d.time))
        r.step(m, d, float(d.time))
        forces, slip, front_force, support = common.contacts(m,d,c)
        if contact_time is None:
            contact_count = contact_count+1 if front_force > 5. else 0
            if contact_count >= contact_steps:
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
        if case['id'] == 'A03_C06_dt1000us' and not gate_checked and d.time >= 17.750999999 - 1e-9:
            old = ORIGINAL_RUN/'cases/A03_C06/trajectory.npz'
            with np.load(old,allow_pickle=False) as previous:
                pos_err = float(np.max(np.abs(previous['qpos'][:step+2] - arrays['qpos'][:step+2])))
                vel_err = float(np.max(np.abs(previous['qvel'][:step+2] - arrays['qvel'][:step+2])))
                ctrl_err = float(np.max(np.abs(previous['ctrl'][:step+1] - arrays['ctrl'][:step+1])))
            gate = {'passed':max(pos_err,vel_err,ctrl_err) < 1e-10,'position_max_abs':pos_err,'velocity_max_abs':vel_err,'control_max_abs':ctrl_err,'time_s':float(d.time)}
            common.write_json(RUN/'prefix_gate.json',gate)
            if not gate['passed']:
                raise RuntimeError(f'prefix gate failed: {gate}')
            gate_checked = True
        all_reasons = common.process_reasons(a,r)
        reasons = [x for x in all_reasons if not x.startswith('joint_velocity_limit:')]
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
    overspeed = summarize_speed(trimmed['time'], trimmed['qvel'], a.velocity_dofs, a.velocity_limits, dt)
    speed_first = min((entry['audit']['events'][0]['first_s'] for entry in overspeed.values() if entry['audit']['events']), default=None)
    first_failure = min((x for x in (speed_first, float(d.time) if reasons else None) if x is not None), default=None)
    complete = bool(case['kind'] == 'full_route' and reason == 'end_ground_stability_observed' and
                    not audit['quality_rejection_reasons'] and not route['rejection_reasons'] and not brake_outside_range)
    max_level = max((k+1 for k in range(15) if crossing_peak[k] >= .1-1e-9),default=0)
    names = [c.names[row[0]] for row in c.wheels]
    result = {'id':case['id'], 'kind':case['kind'], 'parameters':case,
              'height_cm':20, 'stairs':15, 'actual_duration_s':float(d.time),
              'termination_reason':reason, 'process_valid':not all_reasons, 'process_reasons':all_reasons, 'diagnostic_stop_reasons':reasons,
              'first_failure_s':first_failure, 'diagnostic_stop_time_s':float(d.time), 'wheel_names':names,
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
              'audit':audit, 'route':route, 'overspeed':overspeed,
              'physics_step_s':dt,'control_period_s':.001,
              'force_time_alignment':'step-start dynamics from mj_step; state samples at step boundaries',
              'geom_names':a.geom_names,
              'wall_seconds':time.monotonic()-started}
    return save_case(case,trimmed,result,initial,spec,scene)



def summarize_speed(times, qvel, dofs, limits, dt):
    out = {}
    for name, idx in dofs.items():
        vel = qvel[:,idx]
        entries = {}
        for label, threshold in (("urdf",limits[name]),("audit",limits[name]+.01)):
            mask = np.abs(vel[1:]) > threshold
            changes = np.diff(np.r_[False,mask,False].astype(np.int8))
            starts = np.flatnonzero(changes == 1)+1
            ends = np.flatnonzero(changes == -1)+1
            events = []
            for start,end in zip(starts,ends):
                section=vel[start:end]
                peak_idx=start+int(np.argmax(np.abs(section)))
                events.append({"first_s":float(times[start]),"first_signed_rad_s":float(vel[start]),
                    "peak_signed_rad_s":float(vel[peak_idx]),"peak_time_s":float(times[peak_idx]),
                    "duration_s":float((end-start)*dt),
                    "recovered_s":float(times[end]) if end < len(times) else None,
                    "censored":bool(end >= len(times))})
            entries[label] = {"threshold_rad_s":threshold,"event_count":len(events),
                "total_duration_s":float(mask.sum()*dt),
                "longest_duration_s":max((x['duration_s'] for x in events),default=0.),
                "integrated_excess_rad":float(np.maximum(np.abs(vel[1:])-threshold,0).sum()*dt),
                "events":events}
        out[name]=entries
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--dt', type=float, required=True)
    args=parser.parse_args()
    permitted={'A03_C03':(3.,3.),'A03_C06':(3.,6.),'A03_C20':(3.,20.),'A20_C03':(20.,3.),'A20_C20':(20.,20.),'A40_C03':(40.,3.)}
    if args.case not in permitted or args.dt not in (.001,.0005,.00025):
        raise ValueError('case or timestep outside approved batch')
    a,c=permitted[args.case]
    case={'id':f'{args.case}_dt{round(args.dt*1e6):04d}us','kind':'speed_matrix',
          'approach_rad_s':a,'climb_rad_s':c,'ratios':[1.,1.,1.],
          'runup_m':8.,'physics_dt_s':args.dt}
    result=run_case(case)
    print(json.dumps({'id':case['id'],'time':result['actual_duration_s'],
          'termination':result['termination_reason'],'reasons':result['diagnostic_stop_reasons'],
          'overspeed':not result['process_valid'],'platform':result['ascent_platform_support']}),flush=True)

if __name__ == '__main__':
    main()
