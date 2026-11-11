"""Verify selected feedback, record full actions, and independently replay from initial state."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import backend
from run_search import jsonable
from dataset_io import EpisodeRecorder,replay_check
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=HERE)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(HERE):raise ValueError("Output must stay in dynamic_search15")
    report=json.loads((out/"search_report.json").read_text())
    seed_path=Path(report["seed_npz"])
    with np.load(seed_path) as z:seed={k:z[k] for k in z.files}
    with np.load(out/"selected_candidate.npz") as z:selected={k:z[k] for k in z.files}
    with np.load(str(seed["source_trajectory"])) as z:source={k:z[k] for k in z.files}
    backend.init_worker(HERE/"scene.xml",seed_path)
    best=backend.evaluate({"id":str(selected["candidate_id"]),"sequence":selected["sequence"],"record_states":True})
    control_error=float(np.max(np.abs(best["controls"]-selected["controls"])))
    m=mujoco.MjModel.from_xml_path(str(HERE/"scene.xml"));d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,seed["initial_integration_state"],int(seed["integration_state_spec"]))
    a=backend.EpisodeAudit(m,HERE/"reference.urdf");r=backend.RouteAudit(m,a)
    recorder=EpisodeRecorder(control_dt=.001,record_physics=True)
    marker=SimpleNamespace(wheels=[j for j in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or "").endswith("_wheel_joint")],phase="LOCAL_PREFIX")
    observer=backend.Feedback(m,d)
    errors={key:0. for key in ("prefix_time","prefix_qpos","prefix_qvel","prefix_qfrc_actuator","candidate_time","candidate_qpos","candidate_qvel","candidate_qfrc_actuator","candidate_track","candidate_support")}
    reasons=[]
    for part,controls in (("prefix",seed["prefix_ctrl"]),("candidate",selected["controls"])):
        marker.phase="LOCAL_"+part.upper()
        for i,ctrl in enumerate(controls):
            d.ctrl[:]=ctrl;recorder.on_control(m,d,float(d.time),marker)
            mujoco.mj_step(m,d);recorder.on_step(m,d,float(d.time))
            a.step(m,d,float(d.time));r.step(m,d,float(d.time))
            reasons=backend.hard_reasons(a,r)
            if part=="prefix":
                expected=[seed["prefix_time"][i],seed["prefix_qpos"][i],seed["prefix_qvel"][i],source["physics_qfrc_actuator"][i]]
            else:
                expected=[best["track"][i,0],best["record_qpos"][i],best["record_qvel"][i],best["record_qfrc_actuator"][i]]
                rear=[row for row in observer.wheels if row[4]==2]
                track=np.array([float(d.time),float(d.qpos[0]),min(float(d.geom_xpos[g,0]) for *_,g in rear),min(float(d.geom_xpos[g,2]) for *_,g in rear),min(float(d.geom_xpos[g,2])-float(m.geom_size[g,0]) for *_,g in rear),float(d.qvel[0])])
                f=observer.support();support=np.array([sum(float(f[k,2]) for k,row in enumerate(observer.wheels) if row[4]==role) for role in range(3)])
                errors["candidate_track"]=max(errors["candidate_track"],float(np.max(np.abs(track-selected["track"][i]))))
                errors["candidate_support"]=max(errors["candidate_support"],float(np.max(np.abs(support-selected["support"][i]))))
            for key,actual,want in zip(("time","qpos","qvel","qfrc_actuator"),(float(d.time),d.qpos,d.qvel,d.qfrc_actuator),expected):
                errors[part+"_"+key]=max(errors[part+"_"+key],float(np.max(np.abs(np.asarray(actual)-want))))
    recorded=recorder.save(out/"trajectory.npz",HERE/"scene.xml")
    independent={}
    def on_step(model,data,t):
        if not independent:
            independent["audit"]=backend.EpisodeAudit(model,HERE/"reference.urdf")
            independent["route"]=backend.RouteAudit(model,independent["audit"])
        independent["audit"].step(model,data,t);independent["route"].step(model,data,t)
    replay=replay_check(HERE/"scene.xml",out/"trajectory.npz",atol=0.,on_step=on_step)
    independent_reasons=backend.hard_reasons(independent["audit"],independent["route"])
    checked={"dataset_accepted":False,"candidate_id":str(selected["candidate_id"]),"feedback_rerun_controls_max_error":control_error,"errors":errors,"actual_controls_independent_audit_match":a.result()==best["audit"],"actual_controls_independent_route_match":r.result()==best["route"],"original_replay_check":replay,"hard_rejection_reasons":independent_reasons,"recording":recorded,"candidate_metrics":best["metrics"],"audit":independent["audit"].result(),"route":independent["route"].result(),"backend_sha256":hashlib.sha256((HERE/"backend.py").read_bytes()).hexdigest(),"scope":"Only a locally selected diagnostic prefix; scalar desired/commanded rocker fields in EpisodeRecorder are placeholders. Full two-side target sequence and recorded controls are in selected_candidate.npz."}
    checked["consistent"]=control_error==0. and all(v==0. for v in errors.values()) and replay["replay_ok"] and replay["physics_checked"] and checked["actual_controls_independent_audit_match"] and checked["actual_controls_independent_route_match"]
    (out/"independent_replay.json").write_text(json.dumps(jsonable(checked),indent=2,allow_nan=False)+"\n")
    np.savez_compressed(out/"selected_full_states.npz",time=best["track"][:,0],qpos=best["record_qpos"],qvel=best["record_qvel"],qfrc_actuator=best["record_qfrc_actuator"],controls=best["controls"])
    report["independent_final_replay_completed"]=bool(checked["consistent"])
    report["selected_process_valid"]=not independent_reasons
    (out/"search_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in checked.items() if k in ("consistent","errors","feedback_rerun_controls_max_error","actual_controls_independent_audit_match","actual_controls_independent_route_match","original_replay_check","hard_rejection_reasons")}),flush=True)
    assert checked["consistent"] and not independent_reasons
if __name__=="__main__":main()
