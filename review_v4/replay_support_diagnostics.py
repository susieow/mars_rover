"""Independently replay saved physics controls and derive terrain support forces."""
import argparse,json,math
from pathlib import Path
import numpy as np
import mujoco
parser=argparse.ArgumentParser();parser.add_argument("folder");args=parser.parse_args();p=Path(args.folder)
z=np.load(p/"trajectory.npz");source=json.loads((p/"diagnostic.json").read_text())
m=mujoco.MjModel.from_xml_path(str(p/"scene.xml"));d=mujoco.MjData(m)
mujoco.mj_setState(m,d,z["initial_integration_state"],int(z["integration_state_spec"]));mujoco.mj_forward(m,d)
names={g:mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or "" for g in range(m.ngeom)}
wheels={g:n for g,n in names.items() if "_wheel_col_1" in n};wheel_names=list(wheels.values())
events=[e for e in source.get("assist_events",[]) if e["mode"].startswith("SIDE_")]
windows=[{"stage":e["mode"],"start_s":e["t"],"end_s":events[i+1]["t"] if i+1<len(events) else float(z["physics_time"][-1]),"sum":{n:np.zeros(3) for n in wheel_names},"count":0,"base_x_min":float("inf"),"base_x_max":-float("inf")} for i,e in enumerate(events)]
windows.append({"stage":"FINAL_1S","start_s":float(z["physics_time"][-1])-1.,"end_s":float(z["physics_time"][-1])+1e-6,"sum":{n:np.zeros(3) for n in wheel_names},"count":0,"base_x_min":float("inf"),"base_x_max":-float("inf")})
contact=np.zeros(6);qpos_error=0.;qvel_error=0.
physics_qpos=z["physics_qpos"];physics_qvel=z["physics_qvel"];physics_ctrl=z["physics_ctrl"]
for i,ctrl in enumerate(physics_ctrl):
 d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
 qpos_error=max(qpos_error,float(np.max(np.abs(d.qpos-physics_qpos[i]))))
 qvel_error=max(qvel_error,float(np.max(np.abs(d.qvel-physics_qvel[i]))))
 active=[w for w in windows if w["start_s"]<=d.time<w["end_s"]]
 if not active:continue
 forces={n:np.zeros(3) for n in wheel_names}
 for j in range(d.ncon):
  ct=d.contact[j];a,b=int(ct.geom1),int(ct.geom2)
  if a in wheels:wg,tg,sign=a,b,-1.
  elif b in wheels:wg,tg,sign=b,a,1.
  else:continue
  if not names[tg].startswith(("ground","up","down","platform")):continue
  mujoco.mj_contactForce(m,d,j,contact)
  forces[wheels[wg]]+=sign*(np.asarray(ct.frame).reshape(3,3).T@contact[:3])
 for w in active:
  w["count"]+=1;w["base_x_min"]=min(w["base_x_min"],float(d.qpos[0]));w["base_x_max"]=max(w["base_x_max"],float(d.qpos[0]))
  for name in wheel_names:w["sum"][name]+=forces[name]
out_windows=[]
for w in windows:
 n=w.pop("count");s=w.pop("sum");w["physics_samples"]=n;w["mean_terrain_force_world_n"]={k:(v/n).tolist() for k,v in s.items()} if n else {};out_windows.append(w)
g=mujoco.MjData(m);g.qpos[:]=d.qpos;g.qvel[:]=0.;g.ctrl[:]=0.;mujoco.mj_forward(m,g)
gravity={};final_act={}
for side in ["left","right"]:
 name=side+"_rocker_joint";j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);a=m.jnt_dofadr[j];gravity[name]=float(g.qfrc_bias[a]);final_act[name]=float(z["physics_qfrc_actuator"][-1,a])
result={"source":"Saved initial integration state and every physics_ctrl action, independently stepped through the saved scene.","contact_force_convention":"mj_contactForce local force rotated by contact frame transpose into world, reversed for geom1 wheel. Only enabled wheel-terrain contacts are summed.","max_qpos_replay_error":qpos_error,"max_qvel_replay_error":qvel_error,"replay_ok":qpos_error==0. and qvel_error==0.,"weight_n":float(np.sum(m.body_mass)*abs(m.opt.gravity[2])),"windows":out_windows,"final_zero_velocity_gravity_bias_nm":gravity,"final_recorded_actuator_force_nm":final_act,"dataset_accepted":False}
(p/"support_replay_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"folder":p.name,"replay_ok":result["replay_ok"],"max_qpos_error":qpos_error,"max_qvel_error":qvel_error,"windows":out_windows,"gravity_bias":gravity}),flush=True)
