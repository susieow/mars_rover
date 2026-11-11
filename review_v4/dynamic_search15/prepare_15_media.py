"""Extract visual poses from saved candidate torque replay, without changing controls."""
import hashlib,json,multiprocessing as mp,time
from pathlib import Path
import mujoco,numpy as np
import backend
HERE=Path(__file__).resolve().parent
CACHE={}
def seed_state(seed_path,scene):
    if seed_path in CACHE:return CACHE[seed_path]
    with np.load(seed_path) as z:s={k:z[k] for k in z.files}
    m=mujoco.MjModel.from_xml_path(scene);d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,s["initial_integration_state"],int(s["integration_state_spec"]))
    for ctrl in s["prefix_ctrl"]:d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
    if not np.array_equal(d.qpos,s["branch_qpos"]) or not np.array_equal(d.qvel,s["branch_qvel"]):raise ValueError("Independent prefix mismatch")
    state=np.empty(mujoco.mj_stateSize(m,int(s["integration_state_spec"])))
    mujoco.mj_getState(m,d,state,int(s["integration_state_spec"]))
    CACHE[seed_path]=(m,state,int(s["integration_state_spec"]))
    return CACHE[seed_path]

def extract(job):
    m,state,spec=seed_state(job["seed_npz"],job["scene_path"])
    path=job["trace_npz"]
    if path not in CACHE:
        with np.load(path) as z:CACHE[path]={k:z[k] for k in z.files}
    traces=CACHE[path];ctrl=traces[job["controls_key"]];expected=traces[job["track_key"]]
    d=mujoco.MjData(m);mujoco.mj_setState(m,d,state,spec);observer=backend.Feedback(m,d)
    qpos=[];qvel=[];times=[];error=0.
    rear=[row for row in observer.wheels if row[4]==2]
    for index,action in enumerate(ctrl):
        d.ctrl[:]=action;mujoco.mj_step(m,d)
        track=np.array([float(d.time),float(d.qpos[0]),min(float(d.geom_xpos[g,0]) for *_,g in rear),min(float(d.geom_xpos[g,2]) for *_,g in rear),min(float(d.geom_xpos[g,2])-float(m.geom_size[g,0]) for *_,g in rear),float(d.qvel[0])])
        error=max(error,float(np.max(np.abs(track-expected[index]))))
        times.append(float(d.time));qpos.append(d.qpos.copy());qvel.append(d.qvel.copy())
    results_path=job["result_json"]
    if results_path not in CACHE:CACHE[results_path]={r["id"]:r for r in json.loads(Path(results_path).read_text())}
    original=CACHE[results_path][job["candidate_id"]]
    terminal_errors={key:float(np.max(np.abs(getattr(d,key)-np.asarray(original["terminal_"+key])))) for key in ("qpos","qvel")}
    if error!=0 or any(terminal_errors.values()):raise ValueError({"track_error":error,**terminal_errors})
    output=HERE/"render_data"/"h15"/job["batch"]/(job["candidate_id"]+".npz")
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,time=np.asarray(times),qpos=np.asarray(qpos),qvel=np.asarray(qvel))
    return {"id":job["id"],"poses_npz":str(output),"media_provenance":"saved_actual_torque_replay_from_true_prefix","candidate_track_max_error":error,"terminal_errors":terminal_errors,"pose_steps":len(times)}

def main():
    start=time.perf_counter();manifest=json.loads((HERE/"candidate_manifest.json").read_text())
    unique={};aliases={}
    for row in manifest["candidates"]:
        key=(row["seed_npz"],row["control_sha256"])
        if key in unique:aliases[row["id"]]=unique[key]["id"]
        else:unique[key]=row
    with mp.get_context("spawn").Pool(8) as pool:rows=pool.map(extract,list(unique.values()))
    mapping={row["id"]:row for row in rows}
    for alias,canonical in aliases.items():mapping[alias]={**mapping[canonical],"id":alias,"shared_media_with":canonical}
    report={"count":len(mapping),"unique_control_replays":len(rows),"wall_s":time.perf_counter()-start,"entries":list(mapping.values())}
    (HERE/"media_pose_manifest.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="entries"}),flush=True)
if __name__=="__main__":main()
