"""Explore slow contact-driven support transfer with unchanged model and audit."""
import sys,os,json,argparse,math,hashlib,shutil
from pathlib import Path
import numpy as np
import mujoco
LAB=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/"sources"))
import closed_loop_controller as controller_module
BaseController=controller_module.ClosedLoopController
parser=argparse.ArgumentParser()
parser.add_argument("--mode",choices=["front_clear","front_free"],required=True)
parser.add_argument("--front-angle",type=float,default=35.)
parser.add_argument("--sync",action="store_true")
parser.add_argument("--tag",required=True)
args=parser.parse_args()
folder=LAB.parent/args.tag
folder.mkdir(exist_ok=False)
shutil.copytree(LAB/"sources",folder/"sources")
shutil.copy2(__file__,folder/"sources/experiment.py")
class Candidate(BaseController):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.front_stage=None
        self.front_edges={}
        self.front_support_since=None
        self.riser_contact_since=None
        self.front_geoms={k:g for g,k in self.wheel_geom.items() if k in self.front}
        self.risers=[]
        for g,name in enumerate(self.geom_names):
            if name.startswith("up"):
                self.risers.append((float(self.m.geom_pos[g,0]-self.m.geom_size[g,0]),
                                    float(self.m.geom_pos[g,2]+self.m.geom_size[g,2]),name))
        self.risers.sort()
    def _front_support(self):
        supported=set()
        for i in range(self.d.ncon):
            ct=self.d.contact[i]
            a,b=int(ct.geom1),int(ct.geom2)
            wheel=self.wheel_geom.get(a,self.wheel_geom.get(b))
            if wheel not in self.front or abs(float(ct.frame[2]))<.7:continue
            mujoco.mj_contactForce(self.m,self.d,i,self.force)
            if self.force[0]>5:supported.add(wheel)
        return len(supported)==len(self.front)
    def _update_target(self,t,c):
        super()._update_target(t,c)
        if self.finishing or self.down_seen or self.platform_seen:
            self.front_stage=None
            self.target_deg=0.
            return
        if self.front_stage is None:
            blocked=any(name.startswith("up") for name in c["vertical"]["front"])
            if blocked:
                if self.riser_contact_since is None:self.riser_contact_since=t
            else:self.riser_contact_since=None
            if self.riser_contact_since is not None and t-self.riser_contact_since>=.02:
                self.front_edges={}
                for k,g in self.front_geoms.items():
                    pos=self.d.geom_xpos[g]
                    ahead=[e for e in self.risers if e[0]>pos[0]-.02 and e[1]>pos[2]-self.radius+.01]
                    if ahead:self.front_edges[k]=ahead[0]
                if len(self.front_edges)==len(self.front):
                    self.front_stage="FRONT_CLEAR_GEOMETRY"
                    self._change_phase(self.front_stage,t)
            self.target_deg=0.
            return
        if self.front_stage=="FRONT_CLEAR_GEOMETRY":
            self.target_deg=-args.front_angle
            crossed=all(float(self.d.geom_xpos[self.front_geoms[k],0])>e[0]+.015
                        and float(self.d.geom_xpos[self.front_geoms[k],2])>e[1]+self.radius-.01
                        for k,e in self.front_edges.items())
            if crossed:
                self.front_stage="FRONT_RESTORE_SUPPORT"
                self._change_phase(self.front_stage,t)
        elif self.front_stage=="FRONT_RESTORE_SUPPORT":
            self.target_deg=0.
            if self._front_support():
                if self.front_support_since is None:self.front_support_since=t
                if t-self.front_support_since>=.05:
                    self.front_stage=None
                    self.front_support_since=None
                    self.riser_contact_since=None
                    self._change_phase("CRUISE",t)
            else:self.front_support_since=None
    def _command_wheels(self,speed_target,yaw,yaw_rate):
        super()._command_wheels(speed_target,yaw,yaw_rate)
        if args.mode=="front_free" and self.front_stage=="FRONT_CLEAR_GEOMETRY":
            for index in self.front:self.d.ctrl[self.wheel_act[index]]=0.
    def _command_rockers(self):
        if not args.sync:
            return super()._command_rockers()
        target=math.radians(self.target_deg)
        self.rocker_command+=float(np.clip(target-self.rocker_command,-self.rocker_step,self.rocker_step))
        positions=[float(self.d.qpos[self.m.jnt_qposadr[j]]) for j in self.rockers]
        tolerance=math.radians(3.)
        for joint,actuator,q in zip(self.rockers,self.rocker_act,positions):
            goal=self.rocker_command
            if goal<min(positions) and q<max(positions)-tolerance:
                goal=max(positions)-tolerance
            elif goal>max(positions) and q>min(positions)+tolerance:
                goal=min(positions)+tolerance
            velocity=float(self.d.qvel[self.m.jnt_dofadr[joint]])
            desired=float(np.clip(self.rocker_gain/self.rocker_damp*(goal-q),-.7,.7))
            self.d.ctrl[actuator]=float(np.clip(self.rocker_damp*(desired-velocity),-100.,100.))
controller_module.ClosedLoopController=Candidate
import climb
climb.ROOT=Path("/home/agilex/wzy/rover/assets/mars_rover")
climb.URDF=climb.ROOT/"urdf/mars_rover.urdf"
climb.ROBOT_XML=climb.ROOT/"rover_actuated.xml"
import course
from dataset_io import EpisodeRecorder
from episode_audit import EpisodeAudit
from verify_dataset import RouteAudit
params=dict(seconds=90.,speed=2.,friction=4.,tau_max=50.,nosing=0.,
            front_lift_deg=0.,middle_lift_deg=0.,rocker_gain=600.,rocker_damp=150.,
            wheel_gain=40.,descent_wheel_gain=40.,steer_damp=50.,slew=15.,
            descent_speed=1.,settle_time=1.2,quality_pitch_limit=60.,
            quality_roll_limit=15.,contact_timeconst=.004,init_x=0.,init_y=0.,init_yaw_deg=0.)
rec=EpisodeRecorder(control_dt=course.CTRL_DT,record_physics=True)
h={"sample_next":0.,"samples":[],"progress_x":0.,"progress_t":0.,"audit_next":0.}
def control(m,d,t,c):
    h["controller"]=c
    if "audit" not in h:
        h["audit"]=EpisodeAudit(m,climb.URDF);h["route"]=RouteAudit(m,h["audit"])
    rec.on_control(m,d,t,c)
def step(m,d,t):
    rec.on_step(m,d,t)
    h["audit"].step(m,d,t);h["route"].step(m,d,t)
    c=h["controller"]
    if d.qpos[0]>h["progress_x"]+.03:
        h["progress_x"]=float(d.qpos[0]);h["progress_t"]=t
    if t-h["progress_t"]>14 and not c.finishing:
        c.abort_reason="diagnostic_no_forward_progress_14s"
    if t>=h["audit_next"]:
        failures=[r for r in h["audit"].result()["quality_rejection_reasons"] if not r.startswith("final_")]
        routefail=[r for r in h["route"].result()["rejection_reasons"] if not r.startswith(("route_missing:","route_final_"))]
        if failures or routefail:c.abort_reason="quality:"+",".join(failures+routefail)
        h["audit_next"]+=.05
    if t>=h["sample_next"]:
        force=np.zeros(6);contacts=[]
        for i in range(d.ncon):
            ct=d.contact[i];a,b=int(ct.geom1),int(ct.geom2)
            if a not in c.wheel_geom and b not in c.wheel_geom:continue
            mujoco.mj_contactForce(m,d,i,force)
            if force[0]<5:continue
            contacts.append({"pair":[c.geom_names[a],c.geom_names[b]],"fn":float(force[0]),"normal":list(map(float,ct.frame[:3]))})
        h["samples"].append({"t":float(t),"x":float(d.qpos[0]),"z":float(d.qpos[2]),"rpy_deg":list(map(float,np.degrees(c._rpy()))),"phase":c.phase,"target":c.target_deg,"rocker_deg":[float(np.degrees(d.qpos[m.jnt_qposadr[j]])) for j in c.rockers],"contacts":contacts,"wheel_centers":{c.geom_names[g]:list(map(float,d.geom_xpos[g])) for g in c.wheel_geom}})
        h["sample_next"]+=.5
r=course.simulate(.20,15,.20,on_control=control,on_step=step,**params)
scene=folder/"scene.xml"
xml=(course.ROOT/f"scene_course_{os.getpid()}.xml").read_text().replace('file="meshes/','file="'+str(course.ROOT)+'/meshes/')
scene.write_text(xml)
record=rec.save(folder/"trajectory.npz",scene)
audit=h["audit"].result();route=h["route"].result()
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/"sources").glob("*.py")}
result={"parameters":{"height":.20,"stairs":15,"depth":.20,**params},"mode":args.mode,"front_angle":args.front_angle,"sync":args.sync,"result":r,"audit":audit,"route":route,"record":record,"source_sha256":hashes,"samples":h["samples"],"dataset_accepted":False}
(folder/"diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"tag":args.tag,"success":r["success"],"duration":r["duration_s"],"x":r["x"],"phase":r["phase"],"abort":r["abort_reason"],"physics_rejections":audit["quality_rejection_reasons"],"route_rejections":route["rejection_reasons"],"peak_steer_deg":route["max_abs_steer_deg"]}),flush=True)
