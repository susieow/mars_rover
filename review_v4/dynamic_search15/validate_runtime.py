"""Check every physical step against an independent replay from the initial state."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import mujoco
import backend
HERE=Path(__file__).resolve().parent
backend.init_worker(HERE/"scene.xml",HERE/"seed_prefix.npz")
result=backend.evaluate({"id":"runtime_neutral","sequence":[[2,2,2,0,0]]*2,"record_states":True})
with np.load(HERE/"seed_prefix.npz") as z:
    seed={k:z[k] for k in z.files}
with np.load(str(seed["source_trajectory"])) as z:
    source={k:z[k] for k in z.files}
m=mujoco.MjModel.from_xml_path(str(HERE/"scene.xml"))
d=mujoco.MjData(m)
mujoco.mj_setState(m,d,seed["initial_integration_state"],int(seed["integration_state_spec"]))
a=backend.EpisodeAudit(m,HERE/"reference.urdf"); r=backend.RouteAudit(m,a)
errors={"prefix_time":0.,"prefix_qpos":0.,"prefix_qvel":0.,"prefix_qfrc_actuator":0.,"candidate_time":0.,"candidate_qpos":0.,"candidate_qvel":0.,"candidate_qfrc_actuator":0.}
for i,ctrl in enumerate(seed["prefix_ctrl"]):
    d.ctrl[:]=ctrl;mujoco.mj_step(m,d);a.step(m,d,float(d.time));r.step(m,d,float(d.time))
    errors["prefix_time"]=max(errors["prefix_time"],abs(float(d.time)-float(seed["prefix_time"][i])))
    errors["prefix_qpos"]=max(errors["prefix_qpos"],float(np.max(np.abs(d.qpos-seed["prefix_qpos"][i]))))
    errors["prefix_qvel"]=max(errors["prefix_qvel"],float(np.max(np.abs(d.qvel-seed["prefix_qvel"][i]))))
    errors["prefix_qfrc_actuator"]=max(errors["prefix_qfrc_actuator"],float(np.max(np.abs(d.qfrc_actuator-source["physics_qfrc_actuator"][i]))))
for i,ctrl in enumerate(result["controls"]):
    d.ctrl[:]=ctrl;mujoco.mj_step(m,d);a.step(m,d,float(d.time));r.step(m,d,float(d.time))
    errors["candidate_time"]=max(errors["candidate_time"],abs(float(d.time)-float(result["track"][i,0])))
    for key,expected in (("qpos",result["record_qpos"][i]),("qvel",result["record_qvel"][i]),("qfrc_actuator",result["record_qfrc_actuator"][i])):
        errors["candidate_"+key]=max(errors["candidate_"+key],float(np.max(np.abs(getattr(d,key)-expected))))
independent_reasons=backend.hard_reasons(a,r)
report={"errors":errors,"consistent":all(x==0 for x in errors.values()),"runtime_s":result["elapsed_s"],"candidate_valid":result["valid"],"candidate_metrics":result["metrics"],"independent_reasons":independent_reasons,"independent_audit_matches":a.result()==result["audit"],"independent_route_matches":r.result()==result["route"],"candidate_metrics":result["metrics"],"audit":result["audit"],"route":result["route"],"backend_sha256":hashlib.sha256((HERE/"backend.py").read_bytes()).hexdigest(),"hard_reason_policy":"Each physics step calls original audit.result and route.result; excludes only final stabilization and incomplete route events.","startup_limitation":"Seed 10.362 s is only 21 ms before first sustained rear-wall contact. This is dead-point recovery, not enough preparation time for a 15 deg/s rocker target.","dataset_accepted":False}
(HERE/"runtime_consistency.json").write_text(json.dumps(report,indent=2,allow_nan=False))
np.savez_compressed(HERE/"baseline_candidate.npz",ctrl=result["controls"],time=result["track"][:,0],qpos=result["record_qpos"],qvel=result["record_qvel"],qfrc_actuator=result["record_qfrc_actuator"])
print(json.dumps({k:v for k,v in report.items() if k in ("errors","consistent","runtime_s","candidate_valid","independent_reasons","independent_audit_matches","independent_route_matches","backend_sha256")}),flush=True)
assert report["consistent"] and report["candidate_valid"] and not independent_reasons and report["independent_audit_matches"] and report["independent_route_matches"]
