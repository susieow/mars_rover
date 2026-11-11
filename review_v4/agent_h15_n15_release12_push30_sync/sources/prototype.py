"""Diagnose higher, uniform stair flights without changing the accepted seed."""
import sys,json,hashlib,argparse,os,math,types
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path("/home/agilex/wzy/rover")
sys.path.insert(0,str(ROOT/"scripts/rover"))
import course
from dataset_io import EpisodeRecorder,replay_check
from episode_audit import EpisodeAudit
from verify_dataset import RouteAudit
parser=argparse.ArgumentParser()
parser.add_argument("--height",type=float,required=True)
parser.add_argument("--tag",required=True)
parser.add_argument("--stairs",type=int,default=15)
parser.add_argument("--middle-angle",type=float,default=12.)
parser.add_argument("--unload",action="store_true")
parser.add_argument("--speed",type=float,default=2.)
parser.add_argument("--wheel-gain",type=float,default=40.)
parser.add_argument("--disable-assist",action="store_true")
parser.add_argument("--traverse-angle",type=float)
parser.add_argument("--anchor-cycle",action="store_true")
parser.add_argument("--anchor-angle",type=float,default=12.)
parser.add_argument("--entry-angle",type=float)
parser.add_argument("--sync",action="store_true")
parser.add_argument("--leading-drive-scale",type=float,default=1.)
args=parser.parse_args()
folder=ROOT/"review_v4"/args.tag
folder.mkdir(exist_ok=False)
sources=folder/"sources";sources.mkdir()
source_hash={}
for name in ["course.py","closed_loop_controller.py","episode_audit.py","dataset_io.py","verify_dataset.py","climb.py"]:
    b=(ROOT/"scripts/rover"/name).read_bytes()
    (sources/name).write_bytes(b)
    source_hash[name]=hashlib.sha256(b).hexdigest()
for module_name in ["climb","closed_loop_controller","episode_audit","dataset_io","verify_dataset","course"]:
    module=types.ModuleType(module_name)
    module.__file__=str(ROOT/"scripts/rover"/(module_name+".py"))
    sys.modules[module_name]=module
    exec(compile((sources/(module_name+".py")).read_text(),module.__file__,"exec"),module.__dict__)
import course
import closed_loop_controller as cc
from dataset_io import EpisodeRecorder,replay_check
from episode_audit import EpisodeAudit
from verify_dataset import RouteAudit
prototype=Path(__file__).read_bytes()
(sources/"prototype.py").write_bytes(prototype)
source_hash["prototype.py"]=hashlib.sha256(prototype).hexdigest()
original_target=cc.ClosedLoopController._update_target
original_wheels=cc.ClosedLoopController._command_wheels
original_rockers=cc.ClosedLoopController._command_rockers
assist_events=[]

def observe(self):
    rows={k:{"wall_force":0.,"top_force":0.,"up_force":0.,"load":0.,"edges":[],"platform":False} for k in range(len(self.wheels))}
    force=np.zeros(6)
    for i in range(self.d.ncon):
        con=self.d.contact[i];a,b=int(con.geom1),int(con.geom2)
        if a in self.wheel_geom:wg,tg=a,b
        elif b in self.wheel_geom:wg,tg=b,a
        else:continue
        name=self.geom_names[tg]
        if not name.startswith(("up","platform","ground","down")):continue
        mujoco.mj_contactForce(self.m,self.d,i,force)
        fn=max(0.,float(force[0]))
        if fn<5:continue
        row=rows[self.wheel_geom[wg]];row["load"]+=fn
        world=np.asarray(con.frame).reshape(3,3).T@force[:3]
        row["up_force"]+=float(world[2])*(-1. if wg==a else 1.)
        if abs(float(con.frame[2]))>.7:
            row["top_force"]+=fn
            row["platform"]|=name=="platform"
        if (name.startswith("up") or name=="platform") and abs(float(con.frame[0]))>.7 and abs(float(con.frame[2]))<.7:
            row["wall_force"]+=fn
            row["edges"].append({"name":name,"x":float(self.m.geom_pos[tg,0]-self.m.geom_size[tg,0]),"z":float(self.m.geom_pos[tg,2]+self.m.geom_size[tg,2])})
    return rows

def transition(self,mode,t,reason):
    self._assist_mode=mode;self._assist_since=t;self._assist_stable=0.
    self._change_phase("ACTIVE_"+mode,t)
    assist_events.append({"t":float(t),"mode":mode,"reason":reason,"x":float(self.d.qpos[0]),"pitch_deg":math.degrees(self._rpy()[1]),"rocker_deg":[math.degrees(float(self.d.qpos[self.m.jnt_qposadr[j]])) for j in self.rockers],"edges":self._assist_edges.copy()})

def target(self,t,c):
    original_target(self,t,c)
    if not hasattr(self,"_assist_mode"):
        self._assist_mode="MONITOR";self._assist_debounce=0.;self._assist_stable=0.;self._assist_hold=0.;self._assist_edges={};self._assist_start_x={};self._support_ema={k:0. for k in range(len(self.wheels))};self._entry_active=False
    if self.finishing or (self.platform_seen and self.down_seen):
        self.target_deg=0.;return
    rows=observe(self)
    for k,row in rows.items():self._support_ema[k]+=(row["up_force"]-self._support_ema[k])*self.dt/.05
    geom_by_wheel={k:g for g,k in self.wheel_geom.items()}
    if all(rows[k]["platform"] for k in rows):
        self._assist_mode="MONITOR";self._assist_hold=0.;self.target_deg=0.;return
    if self._assist_mode=="MONITOR":
        self.target_deg=self._assist_hold
        if args.entry_angle is not None and not self.platform_seen:
            up0=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,"up0")
            edge=float(self.m.geom_pos[up0,0]-self.m.geom_size[up0,0])
            if max(float(self.d.geom_xpos[g,0]) for g,k in self.wheel_geom.items() if k in self.front)>edge-.45:
                self.target_deg=args.entry_angle
                if not self._entry_active:
                    self._entry_active=True
                    assist_events.append({"t":float(t),"mode":"ENTRY_POSE","reason":"front_center_within_45cm_of_first_edge","x":float(self.d.qpos[0]),"target_deg":args.entry_angle})
        blocked=any(rows[k]["wall_force"]>5 for k in self.middle)
        supported=all(rows[k]["load"]>5 for k in self.rear) and self._rpy()[1]<math.radians(-15.) and not args.disable_assist
        self._assist_debounce=self._assist_debounce+self.dt if blocked and supported else 0.
        if self._assist_debounce>=.020:
            available=[edge for k in self.middle for edge in rows[k]["edges"]]
            if available:
                fallback=max(available,key=lambda e:e["x"])
                self._assist_edges={k:max(rows[k]["edges"],key=lambda e:e["x"]) if rows[k]["edges"] else dict(fallback) for k in self.middle}
                self._assist_start_x={k:float(self.d.geom_xpos[geom_by_wheel[k],0]) for k in self.middle}
                transition(self,"RELEASE" if args.unload else "TRAVERSE",t,"middle_wall_20ms_rear_loaded_nose_up")
                self.target_deg=args.middle_angle
    elif self._assist_mode=="RELEASE":
        self.target_deg=args.middle_angle
        free=all(rows[k]["wall_force"]<5 and float(self.d.geom_xpos[geom_by_wheel[k],0])<self._assist_start_x[k]-.01 for k in self.middle)
        if args.traverse_angle is not None:
            free=free and all(abs(math.degrees(float(self.d.qpos[self.m.jnt_qposadr[j]]))-args.middle_angle)<2. and abs(float(self.d.qvel[self.m.jnt_dofadr[j]]))<.15 for j in self.rockers)
        self._assist_stable=self._assist_stable+self.dt if free else 0.
        if self._assist_stable>=.020:transition(self,"TRAVERSE",t,"middle_wall_released_and_retreat_observed")
    elif self._assist_mode=="TRAVERSE":
        self.target_deg=args.middle_angle if args.traverse_angle is None else args.traverse_angle
        passed=all(float(self.d.geom_xpos[geom_by_wheel[k],0])>self._assist_edges[k]["x"]+.005 and float(self.d.geom_xpos[geom_by_wheel[k],2])-float(self.m.geom_size[geom_by_wheel[k],0])>self._assist_edges[k]["z"]-.010 and rows[k]["top_force"]>5 for k in self.middle)
        self._assist_stable=self._assist_stable+self.dt if passed else 0.
        if self._assist_stable>=.020:transition(self,"RESEAT",t,"both_middle_centers_passed_edge_and_supported")
        if args.anchor_cycle and all(self._support_ema[k]>80. for k in self.middle) and all(math.degrees(float(self.d.qpos[self.m.jnt_qposadr[j]]))<-12. for j in self.rockers):
            transition(self,"ANCHOR",t,"both_middle_support_force_above_80N_and_negative_rocker")
    elif self._assist_mode=="ANCHOR":
        self.target_deg=args.anchor_angle
        passed=all(float(self.d.geom_xpos[geom_by_wheel[k],0])>1.605 and float(self.d.geom_xpos[geom_by_wheel[k],2])-self.radius>args.height-.01 for k in self.rear)
        self._assist_stable=self._assist_stable+self.dt if passed else 0.
        if self._assist_stable>=.05:
            self._assist_hold=math.degrees(self.rocker_command)
            transition(self,"MONITOR",t,"rear_centers_passed_first_edge_and_raised")
            self._assist_debounce=0.;self.target_deg=self._assist_hold
    else:
        self.target_deg=0.
        seated=all(rows[k]["top_force"]>5 for k in self.middle) and all(abs(float(self.d.qvel[self.m.jnt_dofadr[j]]))<.1 and abs(float(self.d.qpos[self.m.jnt_qposadr[j]])-self.rocker_command)<math.radians(2) for j in self.rockers)
        self._assist_stable=self._assist_stable+self.dt if seated else 0.
        if self._assist_stable>=.020:
            self._assist_hold=math.degrees(self.rocker_command)
            transition(self,"MONITOR",t,"both_middle_loaded_and_actual_rocker_settled")
            self._assist_debounce=0.;self.target_deg=self._assist_hold

def wheels(self,speed,yaw,yaw_rate):
    original_wheels(self,speed,yaw,yaw_rate)
    if args.unload and getattr(self,"_assist_mode",None)=="RELEASE" and not self.finishing:
        for k in self.middle:
            act=self.wheel_act[k];self.d.ctrl[act]=min(float(self.d.ctrl[act]),0.)
    if getattr(self,"_assist_mode",None)=="ANCHOR" and not self.finishing:
        for k in self.middle:
            vel=float(self.d.qvel[self.m.jnt_dofadr[self.wheels[k]]])
            self.d.ctrl[self.wheel_act[k]]=float(np.clip(-40.*vel,-self.tau_max,self.tau_max))
    if args.sync and not self.finishing:
        positions=[float(self.d.qpos[self.m.jnt_qposadr[j]]) for j in self.rockers]
        target=math.radians(self.target_deg);tolerance=math.radians(3.)
        for j,q in zip(self.rockers,positions):
            leading=(target<min(positions) and q<max(positions)-tolerance) or (target>max(positions) and q>min(positions)+tolerance)
            if leading:
                side="left" if "left" in self.joint_names[j] else "right"
                for k in self.front+self.middle:
                    if side in self.joint_names[self.wheels[k]] and self.d.ctrl[self.wheel_act[k]]>0:
                        self.d.ctrl[self.wheel_act[k]]*=args.leading_drive_scale

def rockers(self):
    if not args.sync:return original_rockers(self)
    target=math.radians(self.target_deg)
    self.rocker_command+=float(np.clip(target-self.rocker_command,-self.rocker_step,self.rocker_step))
    positions=[float(self.d.qpos[self.m.jnt_qposadr[j]]) for j in self.rockers]
    tolerance=math.radians(3.)
    for joint,actuator,q in zip(self.rockers,self.rocker_act,positions):
        goal=self.rocker_command
        if goal<min(positions) and q<max(positions)-tolerance:goal=max(positions)-tolerance
        elif goal>max(positions) and q>min(positions)+tolerance:goal=min(positions)+tolerance
        velocity=float(self.d.qvel[self.m.jnt_dofadr[joint]])
        desired=float(np.clip(self.rocker_gain/self.rocker_damp*(goal-q),-.7,.7))
        self.d.ctrl[actuator]=float(np.clip(self.rocker_damp*(desired-velocity),-100.,100.))
cc.ClosedLoopController._command_rockers=rockers
cc.ClosedLoopController._update_target=target
cc.ClosedLoopController._command_wheels=wheels

params=dict(seconds=120.,speed=args.speed,friction=4.,tau_max=50.,nosing=0.,
            front_lift_deg=0.,middle_lift_deg=0.,rocker_gain=600.,rocker_damp=150.,
            wheel_gain=args.wheel_gain,descent_wheel_gain=40.,steer_damp=50.,
            descent_speed=1.,settle_time=1.2,quality_pitch_limit=60.,
            quality_roll_limit=15.,contact_timeconst=.004,init_x=0.,init_y=0.,init_yaw_deg=0.)
rec=EpisodeRecorder(control_dt=course.CTRL_DT,record_physics=True)
h={"sample_next":0.,"samples":[],"progress_x":0.,"progress_t":0.}
def control(m,d,t,c):
    h["controller"]=c
    if "audit" not in h:
        h["audit"]=EpisodeAudit(m);h["route"]=RouteAudit(m,h["audit"])
    rec.on_control(m,d,t,c)
def step(m,d,t):
    rec.on_step(m,d,t)
    h["audit"].step(m,d,t);h["route"].step(m,d,t)
    c=h["controller"]
    if d.qpos[0]>h["progress_x"]+.03:
        h["progress_x"]=float(d.qpos[0]);h["progress_t"]=t
    if t-h["progress_t"]>12 and not c.finishing:
        c.abort_reason="diagnostic_no_forward_progress_12s"
    if t>=h["sample_next"]:
        force=np.zeros(6);contacts=[]
        for i in range(d.ncon):
            ct=d.contact[i];a,b=int(ct.geom1),int(ct.geom2)
            if a not in c.wheel_geom and b not in c.wheel_geom:continue
            mujoco.mj_contactForce(m,d,i,force)
            if force[0]<5:continue
            contacts.append({"pair":[c.geom_names[a],c.geom_names[b]],"fn":float(force[0]),"normal":list(map(float,ct.frame[:3]))})
        h["samples"].append({"t":float(t),"x":float(d.qpos[0]),"z":float(d.qpos[2]),"rpy_deg":list(map(float,np.degrees(c._rpy()))),"phase":c.phase,"target":c.target_deg,"rocker_deg":[float(np.degrees(d.qpos[m.jnt_qposadr[j]])) for j in c.rockers],"contacts":contacts,"wheel_centers":{c.geom_names[g]:list(map(float,d.geom_xpos[g])) for g in c.wheel_geom},"qvel":d.qvel.tolist(),"qfrc_actuator":d.qfrc_actuator.tolist()})
        h["sample_next"]+=1.
r=course.simulate(args.height,args.stairs,args.height,on_control=control,on_step=step,**params)
scene=folder/"scene.xml"
xml=(course.ROOT/f"scene_course_{os.getpid()}.xml").read_text().replace('file="meshes/','file="'+str(course.ROOT)+'/meshes/')
scene.write_text(xml)
record=rec.save(folder/"trajectory.npz",scene)
audit=h["audit"].result();route=h["route"].result()
result={"parameters":{"height":args.height,"stairs":args.stairs,"depth":args.height,**params},"result":r,"audit":audit,"route":route,"record":record,"source_sha256":source_hash,"prototype_parameters":{"middle_angle":args.middle_angle,"traverse_angle":args.traverse_angle,"anchor_cycle":args.anchor_cycle,"anchor_angle":args.anchor_angle,"entry_angle":args.entry_angle,"sync":args.sync,"leading_drive_scale":args.leading_drive_scale,"middle_positive_drive_unload":args.unload,"speed":args.speed,"wheel_gain":args.wheel_gain,"disable_assist":args.disable_assist,"activation_count":sum(e["mode"] in ("RELEASE","TRAVERSE") and e["reason"]=="middle_wall_20ms_rear_loaded_nose_up" for e in assist_events)},"assist_events":assist_events,"samples":h["samples"],"dataset_accepted":False}
if r["success"] and audit["audit_quality_ok"] and route["route_ok"]:
    result["replay"]=replay_check(scene,folder/"trajectory.npz")
result["automated_checks_passed"]=bool(r["success"] and audit["audit_quality_ok"] and route["route_ok"] and result.get("replay",{}).get("replay_ok",False))
(folder/"diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"tag":args.tag,"success":r["success"],"duration":r["duration_s"],"x":r["x"],"phase":r["phase"],"abort":r["abort_reason"],"physics_rejections":audit["quality_rejection_reasons"],"route_rejections":route["rejection_reasons"],"peak_steer_deg":route["max_abs_steer_deg"]}),flush=True)
