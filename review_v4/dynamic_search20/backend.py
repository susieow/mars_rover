"""Small 1 ms closed-loop candidate evaluator with original per-step audits."""
import sys,time,math
from pathlib import Path
import numpy as np
import mujoco
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"sources"))
from episode_audit import EpisodeAudit
from verify_dataset import RouteAudit

M=None
SEED=None

def init_worker(scene,seed_npz):
    global M,SEED
    M=mujoco.MjModel.from_xml_path(str(scene))
    if abs(float(M.opt.timestep)-.001)>1e-12:raise ValueError("Search requires original 1 ms physics timestep")
    with np.load(seed_npz) as z:SEED={k:z[k] for k in z.files}

def angles(d):
    w,x,y,z=map(float,d.qpos[3:7])
    return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1.,min(1.,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))

def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi

def clamp(x,lo,hi):return max(lo,min(hi,x))

def hard_reasons(a,r):
    excluded = {
        "final_six_wheels_not_supported_on_ground",
        "final_stability_under_1s",
        "route_final_base_not_past_end",
        "route_final_wheels_not_past_end",
        "route_final_ground_stability_under_1s",
    }
    original = a.result()["quality_rejection_reasons"] + r.result()["rejection_reasons"]
    return [reason for reason in original if reason not in excluded and not reason.startswith("route_missing:")]

class Feedback:
    def __init__(self,m,d,state=None,yaw_rate=0.,yaw_previous=None):
        state=state or {};self.m=m;self.d=d;self.wheel_commands=list(state.get("wheel_commands",[2.,2.,2.]));self.rocker_commands=list(state.get("rocker_commands_rad",[0.,0.]));self.yaw_rate=yaw_rate;self.yaw_previous=yaw_previous
        self.names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or "" for j in range(m.njnt)]
        acts={int(m.actuator_trnid[a,0]):a for a in range(m.nu)}
        self.wheels=[];self.steer=[];self.rockers=[];self.geoms={};self.geom_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or "" for g in range(m.ngeom)]
        for j,n in enumerate(self.names):
            if n.endswith("_wheel_joint"):
                role=next(i for i,r in enumerate(["front","middle","rear"]) if r in n);g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n.replace("_joint","")+"_col_1")
                row=(j,acts[j],int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]),role,"left" in n,g)
                self.wheels.append(row);self.geoms[g]=len(self.wheels)-1
            elif n.endswith("_steer_joint"):self.steer.append((j,acts[j],int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j])))
            elif n.endswith("_rocker_joint"):self.rockers.append((j,acts[j],int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j])))
        self.force=np.zeros(6)
    def update(self,target):
        m,d=self.m,self.d;_,_,yaw=angles(d)
        if self.yaw_previous is not None:self.yaw_rate=.65*self.yaw_rate+.35*wrap(yaw-self.yaw_previous)/.001
        self.yaw_previous=yaw
        path_heading=clamp(-math.atan(1.5*float(d.qpos[1])+.3*float(d.qvel[1])),-.35,.35)
        error=wrap(path_heading-yaw);correction=clamp(4.*error-1.2*self.yaw_rate,-1.5,1.5);starget=clamp(.45*error-.1*self.yaw_rate,-.15,.15)
        for k in range(3):self.wheel_commands[k]+=clamp(clamp(float(target[k]),-2.,3.)-self.wheel_commands[k],-.004,.004)
        for j,a,q,v,role,left,g in self.wheels:
            goal=clamp(self.wheel_commands[role]+(-correction if left else correction),-2.,3.)
            d.ctrl[a]=clamp(40.*(goal-float(d.qvel[v])),-50.,50.)
        for j,a,q,v in self.steer:
            goal=starget if "front" in self.names[j] else -starget;e=goal-float(d.qpos[q]);desired=clamp(8.*e,-1.2,1.2);d.ctrl[a]=clamp(50.*(desired-float(d.qvel[v])),-40.,40.)
            wheelname=self.names[j].replace("_steer_joint","_wheel_joint")
            wa=next(a0 for j0,a0,*_ in self.wheels if self.names[j0]==wheelname)
            if d.ctrl[wa]>0:d.ctrl[wa]*=clamp((25.-abs(math.degrees(e)))/15.,0.,1.)
        for i,(j,a,q,v) in enumerate(self.rockers):
            lo,hi=map(float,m.jnt_range[j]);goal=clamp(math.radians(float(target[3+i])),lo,hi);step=math.radians(15.)*.001
            self.rocker_commands[i]+=clamp(goal-self.rocker_commands[i],-step,step)
            desired=clamp(4.*(self.rocker_commands[i]-float(d.qpos[q])),-.7,.7);d.ctrl[a]=clamp(150.*(desired-float(d.qvel[v])),-100.,100.)
    def support(self):
        forces=np.zeros((len(self.wheels),3));d=self.d
        for i in range(d.ncon):
            ct=d.contact[i];a,b=int(ct.geom1),int(ct.geom2)
            if a in self.geoms:wg,tg,sign=a,b,-1.
            elif b in self.geoms:wg,tg,sign=b,a,1.
            else:continue
            if not self.geom_names[tg].startswith(("ground","up","down","platform")):continue
            mujoco.mj_contactForce(self.m,d,i,self.force);forces[self.geoms[wg]]+=sign*(np.asarray(ct.frame).reshape(3,3).T@self.force[:3])
        return forces
    def state(self):return {"wheel_commands":list(self.wheel_commands),"rocker_commands_rad":list(self.rocker_commands),"yaw_rate":self.yaw_rate,"yaw_previous":self.yaw_previous}

def evaluate(job):
    started=time.perf_counter();m=M;d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,SEED["initial_integration_state"],int(SEED["integration_state_spec"]))
    a=EpisodeAudit(m,HERE/"reference.urdf");r=RouteAudit(m,a);yaw_rate=0.;yaw_previous=None
    prefix=[SEED["prefix_ctrl"],np.asarray(job.get("extra_ctrl",[]),dtype=float).reshape(-1,m.nu)]
    for chunk in prefix:
        for ctrl in chunk:
            yaw=angles(d)[2]
            if yaw_previous is not None:yaw_rate=.65*yaw_rate+.35*wrap(yaw-yaw_previous)/.001
            yaw_previous=yaw;d.ctrl[:]=ctrl;mujoco.mj_step(m,d);a.step(m,d,float(d.time));r.step(m,d,float(d.time))
            reasons=hard_reasons(a,r)
            if reasons:return {"id":job.get("id"),"valid":False,"reasons":["prefix:"+v for v in reasons],"elapsed_s":time.perf_counter()-started}
        if chunk is prefix[0] and "branch_qpos" in SEED:
            err=float(np.max(np.abs(d.qpos-SEED["branch_qpos"])))
            if err!=0.:return {"id":job.get("id"),"valid":False,"reasons":["prefix_replay_mismatch"],"prefix_error":err}
    c=Feedback(m,d,job.get("controller_state"),yaw_rate,yaw_previous);start_time=float(d.time);x0=float(d.qpos[0]);rear=[row for row in c.wheels if row[4]==2]
    rear0=np.array([[float(d.geom_xpos[g,0]),float(d.geom_xpos[g,2])] for *_,g in rear]);height=float(r.course["height_m"]);edge=float(r.course["start_x"])
    horizon=float(job.get("horizon",2.));steps=round(horizon/.001);sequence=np.asarray(job["sequence"],dtype=float)
    if sequence.ndim!=2 or sequence.shape[1]!=5 or not np.isfinite(sequence).all():raise ValueError("sequence must be finite Kx5")
    controls=np.empty((steps,m.nu));track=[];force_sum=np.zeros((6,3));all_support=[];cross_time=0.;cross_peak=0.;reasons=[];first_failure=None;record_states=bool(job.get("record_states",False));record_qpos=[];record_qvel=[];record_force=[]
    for i in range(steps):
        target=sequence[min(len(sequence)-1,int(i*len(sequence)/steps))];c.update(target);controls[i]=d.ctrl;mujoco.mj_step(m,d);a.step(m,d,float(d.time));r.step(m,d,float(d.time))
        reasons=hard_reasons(a,r)
        f=c.support();force_sum+=f;loads=[sum(float(f[k,2]) for k,row in enumerate(c.wheels) if row[4]==role) for role in range(3)];all_support.append(loads)
        rearx=min(float(d.geom_xpos[g,0]) for *_,g in rear);rearz=min(float(d.geom_xpos[g,2]) for *_,g in rear);rearbottom=min(float(d.geom_xpos[g,2])-float(m.geom_size[g,0]) for *_,g in rear)
        crossed=rearx>edge+.005 and rearbottom>height-.010;cross_time=cross_time+.001 if crossed else 0.;cross_peak=max(cross_peak,cross_time)
        track.append([float(d.time),float(d.qpos[0]),rearx,rearz,rearbottom,float(d.qvel[0])])
        if record_states:record_qpos.append(d.qpos.copy());record_qvel.append(d.qvel.copy());record_force.append(d.qfrc_actuator.copy())
        if reasons:first_failure=float(d.time);break
    used=len(track);t=np.asarray(track);support=np.asarray(all_support);tail=t[-min(500,used):];tails=support[-min(500,used):]
    metrics={"start_time_s":start_time,"end_time_s":float(d.time),"base_x_start":x0,"base_x_final":float(d.qpos[0]),"base_x_delta":float(d.qpos[0])-x0,"rear_min_x_start":float(np.min(rear0[:,0])),"rear_min_x_final":float(t[-1,2]),"rear_min_x_tail_mean":float(np.mean(tail[:,2])),"rear_min_x_delta":float(t[-1,2]-np.min(rear0[:,0])),"rear_min_z_start":float(np.min(rear0[:,1])),"rear_min_z_final":float(t[-1,3]),"rear_min_z_max":float(np.max(t[:,3])),"rear_bottom_min_final":float(t[-1,4]),"rear_x_change_last_500ms":float(tail[-1,2]-tail[0,2]),"rear_x_slope_last_500ms":float(np.polyfit(tail[:,0]-tail[0,0],tail[:,2],1)[0]) if len(tail)>1 else 0.,"rear_both_crossed_continuous_peak_s":cross_peak,"rear_both_crossed_final_s":cross_time,"front_middle_both_supported_tail_fraction":float(np.mean((tails[:,0]>5.)&(tails[:,1]>5.))),"mean_support_world_force_n":{c.geom_names[row[6]]:(force_sum[k]/used).tolist() for k,row in enumerate(c.wheels)},"tail_mean_upward_force_by_role_n":dict(zip(["front","middle","rear"],np.mean(tails,axis=0).tolist())),"max_roll_deg":r.max_abs_roll,"max_pitch_deg":r.max_abs_pitch,"max_steer_deg":math.degrees(r.max_abs_steer_rad)}
    result={"id":job.get("id"),"valid":not reasons,"reasons":reasons,"first_failure_s":first_failure,"metrics":metrics,"controller_state":c.state(),"controls":controls[:used].copy(),"track":t,"support":support,"audit":a.result(),"route":r.result(),"elapsed_s":time.perf_counter()-started,"sequence":sequence.tolist(),"terminal_qpos":d.qpos.copy(),"terminal_qvel":d.qvel.copy(),"record_qpos":np.asarray(record_qpos),"record_qvel":np.asarray(record_qvel),"record_qfrc_actuator":np.asarray(record_force),"dataset_accepted":False}
    return result
