"""Build bounded-size review curves from recorded per-step states."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np

root=Path('/home/agilex/wzy/rover')
folder=root/'runs/overspeed20_A40_C40_20261111/cases'/sys.argv[1]
r=json.loads((folder/'result.json').read_text())
m=mujoco.MjModel.from_xml_path(str(root/'review_v4/dynamic_search20/scene.xml'))
with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
    time=z['time']; pos=z['qpos']; vel=z['qvel']; ctrl=z['ctrl']; wheels=z['wheel_targets']; forces=z['contact_force_n']; dt=float(z['simulation_dt'])
    n=len(ctrl); sub=round(.001/dt); tick=np.arange(0,n,sub)
    quat=pos[tick,3:7];w,x,y,zz=quat.T
    yaw=np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz))
    yaw_rate=np.zeros(len(tick))
    delta=(np.diff(yaw)+np.pi)%(2*np.pi)-np.pi
    for i in range(1,len(tick)):
        yaw_rate[i]=.65*yaw_rate[i-1]+.35*delta[i-1]/.001
    heading=np.clip(-np.arctan(1.5*pos[tick,1]+.3*vel[tick,1]),-.35,.35)
    error=(heading-yaw+np.pi)%(2*np.pi)-np.pi
    steer_target=np.clip(.45*error-.1*yaw_rate,-.15,.15)
    states=pos[1:]; w,x,y,zz=states[:,3:7].T
    roll=np.degrees(np.arctan2(2*(w*x+y*zz),1-2*(x*x+y*y)))
    pitch=np.degrees(np.arcsin(np.clip(2*(w*y-zz*x),-1,1)))
    contacts=np.sum(forces,axis=1)
    out={'id':r['id'],'timestep_s':dt,'sample_bin_s':.05,'note':'Within each bin retain first, last and velocity extrema. Force and target belong to the preceding integration interval.', 'joints':{}}
    samples=max(1,round(.05/dt)); breaks=np.arange(0,n,samples); end=np.minimum(breaks+samples,n)
    for name,limit in r['audit']['joint_velocity_limit_rad_s'].items():
        jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
        dof=int(m.jnt_dofadr[jid]); q=int(m.jnt_qposadr[jid])
        act=next(a for a in range(m.nu) if int(m.actuator_trnid[a,0])==jid)
        values=vel[1:,dof]
        # Extremes are preserved per joint so short spikes remain visible.
        selected=[0,n-1]
        for start,stop in zip(breaks,end):
            local=values[start:stop]
            selected.extend((int(start),int(stop-1),int(start+np.argmin(local)),int(start+np.argmax(local))))
        inds=np.unique(selected); control_i=(inds//sub)*sub; tick_i=inds//sub
        if name.endswith('_wheel_joint'):
            wi=r['wheel_names'].index(name)
            target=wheels[inds,wi]
        elif name.endswith('_rocker_joint'):
            target=np.clip(-4*pos[control_i,q],-.7,.7)
        else:
            goal=steer_target[tick_i]*(1 if 'front' in name else -1)
            target=np.clip(8*(goal-pos[control_i,q]),-1.2,1.2)
        rows=np.column_stack((time[inds+1],values[inds],target,ctrl[inds,act],roll[inds],pitch[inds],contacts[inds]))
        out['joints'][name]={'limit':limit,'audit_threshold':limit+.01,'columns':['time_s','actual_rad_s','target_rad_s','torque_nm','roll_deg','pitch_deg','wheel_contact_force_n'], 'rows':np.round(rows,6).tolist()}
(folder/'curves.json').write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n')
print(json.dumps({'id':r['id'],'curves':len(out['joints'])}),flush=True)
