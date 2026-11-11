"""Diagnose higher, uniform stair flights without changing the accepted seed."""
import sys,json,hashlib,argparse,os
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
args=parser.parse_args()
folder=ROOT/"review_v4"/args.tag
folder.mkdir(exist_ok=False)
sources=folder/"sources";sources.mkdir()
source_hash={}
for name in ["course.py","closed_loop_controller.py","episode_audit.py","dataset_io.py","verify_dataset.py","climb.py"]:
    b=(ROOT/"scripts/rover"/name).read_bytes()
    (sources/name).write_bytes(b)
    source_hash[name]=hashlib.sha256(b).hexdigest()
params=dict(seconds=90.,speed=6.,friction=4.,tau_max=50.,nosing=0.,
            front_lift_deg=0.,middle_lift_deg=0.,rocker_gain=600.,rocker_damp=150.,
            wheel_gain=10.,descent_wheel_gain=40.,steer_damp=50.,
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
        h["samples"].append({"t":float(t),"x":float(d.qpos[0]),"z":float(d.qpos[2]),"rpy_deg":list(map(float,np.degrees(c._rpy()))),"phase":c.phase,"target":c.target_deg,"rocker_deg":[float(np.degrees(d.qpos[m.jnt_qposadr[j]])) for j in c.rockers],"contacts":contacts})
        h["sample_next"]+=1.
r=course.simulate(args.height,12,args.height,on_control=control,on_step=step,**params)
scene=folder/"scene.xml"
xml=(course.ROOT/f"scene_course_{os.getpid()}.xml").read_text().replace('file="meshes/','file="'+str(course.ROOT)+'/meshes/')
scene.write_text(xml)
record=rec.save(folder/"trajectory.npz",scene)
audit=h["audit"].result();route=h["route"].result()
result={"parameters":{"height":args.height,"stairs":12,"depth":args.height,**params},"result":r,"audit":audit,"route":route,"record":record,"source_sha256":source_hash,"samples":h["samples"],"dataset_accepted":False}
(folder/"diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"tag":args.tag,"success":r["success"],"duration":r["duration_s"],"x":r["x"],"phase":r["phase"],"abort":r["abort_reason"],"physics_rejections":audit["quality_rejection_reasons"],"route_rejections":route["rejection_reasons"],"peak_steer_deg":route["max_abs_steer_deg"]}),flush=True)
