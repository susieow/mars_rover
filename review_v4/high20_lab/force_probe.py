"""Measure world contact forces during exact stored-action replay."""
import sys,json,argparse
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0,"/home/agilex/wzy/rover/scripts/rover")
from dataset_io import replay_check
parser=argparse.ArgumentParser()
parser.add_argument("episode")
args=parser.parse_args()
folder=Path(args.episode)
with np.load(folder/"trajectory.npz",allow_pickle=False) as a:
    end=float(a["time"][-1])
rows=[]; setup={}
def observe(m,d,t):
    if t<end-1.:return
    if not setup:
        setup["names"]=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or "" for g in range(m.ngeom)]
        setup["wheels"]={g:n.replace("_col_1","") for g,n in enumerate(setup["names"]) if n.endswith("_wheel_col_1")}
        setup["joints"]={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j):j for j in range(m.njnt)}
        setup["weight_n"]=float(m.body_mass.sum())*abs(float(m.opt.gravity[2]))
    forces={n:np.zeros(3) for n in setup["wheels"].values()}
    terrain={};f=np.zeros(6)
    for i in range(d.ncon):
        c=d.contact[i];a,b=int(c.geom1),int(c.geom2)
        wheel=setup["wheels"].get(b,setup["wheels"].get(a))
        if wheel is None:continue
        mujoco.mj_contactForce(m,d,i,f)
        world=c.frame.reshape(3,3).T@f[:3]
        if a in setup["wheels"]:world=-world
        forces[wheel]+=world
        other=setup["names"][a if b in setup["wheels"] else b]
        terrain.setdefault(other,np.zeros(3))
        terrain[other]+=world
    values={}
    for name,force in forces.items():
        j=setup["joints"][name+"_joint"]
        values[name]={"force":force.tolist(),"torque":float(d.qfrc_actuator[m.jnt_dofadr[j]]),
                     "speed":float(d.qvel[m.jnt_dofadr[j]])}
    rows.append({"t":t,"wheel":values,"terrain":{k:v.tolist() for k,v in terrain.items()}})
replay=replay_check(folder/"scene.xml",folder/"trajectory.npz",on_step=observe)
report={"episode":str(folder),"window_s":[end-1,end],"samples":len(rows),"weight_n":setup["weight_n"],
        "force_frame":"world, full normal+tangential force on wheel; contact geom1 side negated",
        "contact_cache_phase":"preceding integrated physical interval","replay":replay,"wheels":{},"roles":{},"terrain":{}}
for name in setup["wheels"].values():
    forces=np.array([r["wheel"][name]["force"] for r in rows])
    report["wheels"][name]={"mean_force_xyz_n":forces.mean(axis=0).tolist(),
         "min_up_force_n":float(forces[:,2].min()),"max_up_force_n":float(forces[:,2].max()),
         "up_support_fraction_over_5n":float(np.mean(forces[:,2]>5)),
         "mean_torque_nm":float(np.mean([r["wheel"][name]["torque"] for r in rows])),
         "mean_speed_rad_s":float(np.mean([r["wheel"][name]["speed"] for r in rows]))}
for role in ("front","middle","rear"):
    names=[n for n in report["wheels"] if "_"+role+"_" in n]
    report["roles"][role]={"mean_force_xyz_n":np.sum([report["wheels"][n]["mean_force_xyz_n"] for n in names],axis=0).tolist()}
for name in {k for row in rows for k in row["terrain"]}:
    report["terrain"][name]=np.mean([row["terrain"].get(name,[0,0,0]) for row in rows],axis=0).tolist()
report["total_mean_force_xyz_n"]=np.sum([r["mean_force_xyz_n"] for r in report["wheels"].values()],axis=0).tolist()
(folder/"world_force_last_second.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
