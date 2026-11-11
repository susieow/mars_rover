"""Build an earlier prefix only by replaying actual original actions."""
import json
from pathlib import Path
import mujoco,numpy as np
import backend
HERE=Path(__file__).resolve().parent
with np.load(HERE/"seed_prefix.npz") as z:prior={k:z[k] for k in z.files}
with np.load(str(prior["source_trajectory"])) as z:source={k:z[k] for k in z.files}
count=9383
m=mujoco.MjModel.from_xml_path(str(HERE/"scene.xml"));d=mujoco.MjData(m)
mujoco.mj_setState(m,d,prior["initial_integration_state"],int(prior["integration_state_spec"]))
a=backend.EpisodeAudit(m,HERE/"reference.urdf");r=backend.RouteAudit(m,a)
errors={"time":0.,"qpos":0.,"qvel":0.,"qfrc_actuator":0.}
controls=source["physics_ctrl"][:count].copy()
for i,ctrl in enumerate(controls):
    d.ctrl[:]=ctrl;mujoco.mj_step(m,d);a.step(m,d,float(d.time));r.step(m,d,float(d.time))
    for key,actual in (("time",d.time),("qpos",d.qpos),("qvel",d.qvel),("qfrc_actuator",d.qfrc_actuator)):
        errors[key]=max(errors[key],float(np.max(np.abs(np.asarray(actual)-source["physics_"+key][i]))))
    if backend.hard_reasons(a,r):raise RuntimeError(backend.hard_reasons(a,r))
state=np.empty(mujoco.mj_stateSize(m,int(prior["integration_state_spec"])))
mujoco.mj_getState(m,d,state,int(prior["integration_state_spec"]))
np.savez_compressed(HERE/"early_seed_prefix.npz",initial_integration_state=prior["initial_integration_state"],integration_state_spec=prior["integration_state_spec"],branch_integration_state=state,branch_qpos=d.qpos.copy(),branch_qvel=d.qvel.copy(),branch_time=float(d.time),prefix_ctrl=controls,prefix_qpos=source["physics_qpos"][:count],prefix_qvel=source["physics_qvel"][:count],prefix_time=source["physics_time"][:count],source_trajectory=prior["source_trajectory"])
result={"branch_time_s":float(d.time),"replay_steps":count,"errors":errors,"hard_reasons":backend.hard_reasons(a,r),"source_trajectory":str(prior["source_trajectory"]),"dataset_accepted":False}
(HERE/"early_seed_consistency.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
assert all(x==0 for x in errors.values())
