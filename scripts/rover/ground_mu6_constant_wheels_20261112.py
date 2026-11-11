from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import mujoco
import numpy as np
import speed20_experiment as common
import recovery20_amp20_experiment as wave
from overspeed20_stairs import summarize_speed

ROOT = common.ROOT
RUN = ROOT / 'runs/ground_mu6_constant_wheels_20261112'
SCENE = ROOT / 'runs/friction20_t22_20261111/scenes/scene_mu6.xml'
DT = .001
LIMIT = 120.
AFTER_TRIGGER = 30.

def main(speed, direction):
    case_id=f'ground_mu6_v{speed}_rocker_{direction}'
    folder=RUN/'cases'/case_id
    if folder.exists():
        raise RuntimeError(f'case folder already exists: {folder}')
    folder.mkdir(parents=True)
    params={'id':case_id,'wheel_base_target_rad_s':speed,'rocker':direction,'stair_friction_slide':6.,
            'physics_dt_s':DT,'stall_window_s':2.,'stall_forward_net_m_less_than':.02,
            'max_simulation_s':LIMIT,'post_trigger_s':AFTER_TRIGGER,'audit_violations_stop':False,
            'stair_contact_trigger':'first positive normal contact force on front wheel',
            'direction_correction_preserved':True,'wheel_target_slew_rad_s2':30.,'actuator_force_limits_preserved':True}
    common.write_json(folder/'started.json',params)
    m=mujoco.MjModel.from_xml_path(str(SCENE)); m.opt.timestep=DT
    d=common.initialize(m,8.); c=common.Controller(m,d)
    a=common.EpisodeAudit(m,common.URDF)
    route=common.RouteAudit(m,a,{'height':.2,'depth':.2,'stairs':15})
    initial=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m,d,initial,mujoco.mjtState.mjSTATE_INTEGRATION)
    cap=round(LIMIT/DT)
    arrays={name:np.zeros((cap+1,size)) for name,size in [('qpos',m.nq),('qvel',m.nv)]}
    arrays['time']=np.zeros(cap+1)
    for name,size in [('ctrl',m.nu),('qfrc_actuator',m.nv),('wheel_targets',6),('wheel_group_commands',3),('rocker_targets_rad',2),('contact_force_n',6),('contact_slip_tangent_m_s',6)]:
        arrays[name]=np.zeros((cap,size))
    arrays['wheel_support_level']=np.full((cap,6),-1,dtype=np.int8)
    arrays['rear_geometry_level']=np.zeros(cap,dtype=np.int8)
    arrays['rear_supported_level']=np.zeros(cap,dtype=np.int8)
    arrays['time'][0]=d.time; arrays['qpos'][0]=d.qpos; arrays['qvel'][0]=d.qvel
    rear=[i for i,row in enumerate(c.wheels) if row[4]==2]
    wheels={row[-1]:row[4] for row in c.wheels}
    stair_ids={i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('up')}
    first_contact=None;trigger=None;first_failure=None;crossing=np.zeros(15);peaks=np.zeros(15);supported=np.zeros(15);first_crossing=None
    contactforce=np.zeros(6); reason='simulation_time_limit_120s'; started=time.monotonic()
    for step in range(cap):
        elapsed=0 if trigger is None else float(d.time)-trigger
        _, rocker=wave.targets(wave.params('H',direction),elapsed)
        if trigger is None: rocker[:]=0.
        goal=c.update(np.full(3,speed))
        c.rocker_commands[:]=rocker
        for i,(_,act,pos,vel) in enumerate(c.rockers):
            desired=float(np.clip(4*(rocker[i]-float(d.qpos[pos])),-.7,.7))
            d.ctrl[act]=np.clip(150*(desired-float(d.qvel[vel])),-100,100)
        arrays['ctrl'][step]=d.ctrl; arrays['wheel_targets'][step]=goal
        arrays['wheel_group_commands'][step]=c.wheel_commands; arrays['rocker_targets_rad'][step]=rocker
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError('nonfinite simulation state')
        a.step(m,d,float(d.time));route.step(m,d,float(d.time))
        if first_failure is None and common.process_reasons(a,route): first_failure=float(d.time)
        if first_contact is None:
            for i in range(d.ncon):
                con=d.contact[i]; g1,g2=int(con.geom1),int(con.geom2)
                if con.efc_address>=0 and ((wheels.get(g1)==0 and g2 in stair_ids) or (wheels.get(g2)==0 and g1 in stair_ids)):
                    mujoco.mj_contactForce(m,d,i,contactforce)
                    if contactforce[0]>0:
                        first_contact=float(d.time);break
        forces,slip,_,support=common.contacts(m,d,c)
        levels=np.full(6,-1,dtype=np.int8)
        for wi,terrains in enumerate(support):
            vals=[int(name[2:])+1 for name in terrains if name.startswith('up') and name[2:].isdigit()]
            if 'platform' in terrains: vals.append(15)
            if vals: levels[wi]=max(vals)
        centers=d.geom_xpos[[row[-1] for row in c.wheels]]
        level=0;slevel=0
        for k in range(15):
            crossed=all(float(centers[i,0])>1.6+.2*k+.005 and float(centers[i,2]-m.geom_size[c.wheels[i][-1],0])>.2*(k+1)-.010 for i in rear)
            crossing[k]=crossing[k]+DT if crossed else 0.;peaks[k]=max(peaks[k],crossing[k])
            if crossed:
                level=k+1
                if all(levels[i]>=k+1 for i in rear): supported[k]+=DT;slevel=k+1
            if k==0 and first_crossing is None and crossing[k]>=.1-1e-9: first_crossing=float(d.time)
        arrays['time'][step+1]=d.time; arrays['qpos'][step+1]=d.qpos;arrays['qvel'][step+1]=d.qvel
        arrays['qfrc_actuator'][step]=d.qfrc_actuator;arrays['contact_force_n'][step]=forces;arrays['contact_slip_tangent_m_s'][step]=slip
        arrays['wheel_support_level'][step]=levels;arrays['rear_geometry_level'][step]=level;arrays['rear_supported_level'][step]=slevel
        if trigger is None and first_contact is not None and d.time-first_contact>=2.-1e-9:
            if d.qpos[0]-arrays['qpos'][step+1-round(2/DT),0]<.02:
                trigger=float(d.time)
        if trigger is not None and d.time-trigger>=AFTER_TRIGGER-1e-9:
            reason='post_stall_observation_30s';break
    n=step+1
    arrays={k:v[:n+1] if k in ('time','qpos','qvel') else v[:n] for k,v in arrays.items()}
    np.savez_compressed(folder/'trajectory.npz',**arrays,initial_integration_state=initial,simulation_dt=DT,scene_sha256=common.digest(SCENE))
    overspeed=summarize_speed(arrays['time'],arrays['qvel'],a.velocity_dofs,a.velocity_limits,DT)
    reasons=common.process_reasons(a,route)
    highest=max((k+1 for k in range(15) if peaks[k]>=.1-1e-9),default=0)
    data={'id':case_id,'parameters':params,'physics_step_s':DT,'height_cm':20,'stairs':15,
          'actual_duration_s':float(d.time),'termination_reason':reason,'first_contact_s':first_contact,
          'stall_trigger_s':trigger,'first_failure_s':first_failure,'process_reasons':reasons,'process_valid':not reasons,
          'true_both_rear_crossing':highest>0,'first_rear_crossing_s':first_crossing,
          'rear_highest_100ms_geometry_level':highest,'rear_retained_geometry_level_final_1s':int(np.min(arrays['rear_geometry_level'][-min(n,1000):])),
          'rear_crossing_supported_total_s_by_level':supported.tolist(),'ascent_platform_support':'all_six_supported_on_platform_s' in route.events,
          'full_route_evaluated':False,'full_route_success':False,'dataset_accepted':False,'audit':a.result(),'route':route.result(),
          'overspeed':overspeed,'wheel_names':[c.names[x[0]] for x in c.wheels],
          'max_base_x_progress_m':float(np.max(arrays['qpos'][:,0])-arrays['qpos'][0,0]),
          'net_base_x_m':float(arrays['qpos'][-1,0]-arrays['qpos'][0,0]),
          'trajectory_sha256':common.digest(folder/'trajectory.npz'),'scene_sha256':common.digest(SCENE),
          'source_sha256':common.digest(__file__),'wall_seconds':time.monotonic()-started}
    common.write_json(folder/'result.json',data)
    print(json.dumps({k:data[k] for k in ('id','first_contact_s','stall_trigger_s','actual_duration_s','termination_reason','process_reasons','true_both_rear_crossing','ascent_platform_support')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--speed',type=int,choices=(3,4,5,6),required=True);parser.add_argument('--direction',choices=('SN','SP'),required=True)
    args=parser.parse_args();main(args.speed,args.direction)
