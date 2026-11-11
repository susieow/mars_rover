"""Create a complete media catalog before rendering; no candidate is omitted."""
import hashlib,json
from pathlib import Path
import mujoco,numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=Path("/home/agilex/wzy/videos/review_high_stairs")
manifest={"scope":"Diagnostic review only; stopped candidate search. No media is accepted success data.","candidates":[],"historical":[],"prefixes":[],"render_jobs_completed":0}
jobs=[]
poses15={r["id"]:r for r in json.loads((HERE/"media_pose_manifest.json").read_text())["entries"]}
for height in (15,20):
    base=ROOT/f"dynamic_search{height}"
    for batch,folder in (("near_contact",base),("early_entry",base/"early_entry")):
        if height==15:seed=base/("seed_prefix.npz" if batch=="near_contact" else "early_seed_prefix.npz")
        else:seed=folder/"seed_prefix.npz"
        with np.load(seed) as z:s={k:z[k] for k in z.files}
        scene=base/"scene.xml"
        initial_model=mujoco.MjModel.from_xml_path(str(scene));initial_data=mujoco.MjData(initial_model)
        mujoco.mj_setState(initial_model,initial_data,s["initial_integration_state"],int(s["integration_state_spec"]))
        prefix_npz=HERE/"render_data"/f"h{height}"/batch/"prefix.npz"
        prefix_npz.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(prefix_npz,time=np.concatenate(([float(initial_data.time)],s["prefix_time"])),qpos=np.concatenate((initial_data.qpos[None,:],s["prefix_qpos"]),axis=0))
        directory=OUT/f"h{height}"/batch
        prefix={"id":f"h{height}_{batch}_prefix","kind":"prefix","height_cm":height,"batch":batch,"seed_time_s":float(s["branch_time"]),"poses_npz":str(prefix_npz),"scene_path":str(scene),"source_start_s":float(initial_data.time),"source_end_s":float(s["branch_time"]),"video_path":str(directory/"prefix.mp4"),"poster_path":str(directory/"prefix.jpg"),"video_label":f"{height} cm | {batch} | shared actual prefix","status_text":"SHARED PREFIX | actual saved states before candidate | DIAGNOSTIC ONLY","render_status":"pending","media_provenance":"saved_verified_prefix_qpos"}
        manifest["prefixes"].append(prefix);jobs.append(prefix)
        seen={}
        for round_index in range(3):
            records=json.loads((folder/f"round{round_index}_results.json").read_text())
            for index,row in enumerate(records):
                cid=row["id"];global_id=f"h{height}_{batch}_{cid}"
                sequence=np.asarray(row["sequence"])
                sequence_key=sequence.tobytes()
                canonical=seen.setdefault(sequence_key,cid)
                item={"id":global_id,"kind":"candidate","height_cm":height,"batch":batch,"seed_time_s":float(s["branch_time"]),"round":round_index,"index":index,"candidate_id":cid,"sequence":row["sequence"],"process_valid":row["valid"],"reasons":row["reasons"],"first_failure_s":row.get("first_failure_s"),"metrics":row["metrics"],"source_start_s":float(row["metrics"]["start_time_s"]),"source_end_s":float(row["metrics"]["end_time_s"]),"duration_s":float(row["metrics"]["end_time_s"]-row["metrics"]["start_time_s"]),"local_crossing":False,"full_route_success":False,"dataset_accepted":False,"prefix_video_path":prefix["video_path"],"scene_path":str(scene),"seed_npz":str(seed),"video_label":f"{height} cm | {batch} | recorded candidate motion","status_text":("PROCESS VALID | no rear crossing | full route incomplete" if row["valid"] else "PROCESS REJECTED | "+"; ".join(row["reasons"]))+" | DIAGNOSTIC ONLY"}
                if height==15:
                    pose=poses15[global_id];item.update(pose);canonical=pose.get("shared_media_with",global_id).split(f"h{height}_{batch}_",1)[1]
                    item["time_key"]="time";item["qpos_key"]="qpos";item["render_status"]="pending"
                else:item["render_status"]="waiting_for_pose_export"
                item["video_path"]=str(directory/(canonical+".mp4"));item["poster_path"]=str(directory/(canonical+".jpg"))
                item["canonical_media_id"]=f"h{height}_{batch}_{canonical}"
                manifest["candidates"].append(item)
                if height==15 and canonical==cid:jobs.append(item)
for path in sorted(ROOT.glob("*/diagnostic.json")):
    d=json.loads(path.read_text());params=d.get("parameters",{})
    if params.get("height") not in (.15,.20) or params.get("stairs")!=15:continue
    height=round(params["height"]*100);name=path.parent.name
    trajectory=Path(d["record"]["trajectory_path"])
    with np.load(trajectory) as z:
        times=z["time"];source_start=float(times[0]);source_end=float(times[-1])
        if not bool(z["physics_recorded"]):raise ValueError("Historical physics states missing")
    notes_path=path.parent/"diagnostic_notes.json"
    notes=json.loads(notes_path.read_text()) if notes_path.exists() else None
    warnings=[]
    if name in ("agent_h15_n15_positive12","agent_h15_n15_positive12_unload"):warnings.append("Requested assist action did not activate; not an effective independent control comparison.")
    if notes is not None:warnings.append("Historical diagnostic notes apply; inspect diagnostic_notes. Trajectory poses are used directly, not historical EMA fields.")
    reasons=d.get("audit",{}).get("quality_rejection_reasons",[])+d.get("route",{}).get("rejection_reasons",[])
    abort=d.get("result",{}).get("abort_reason")
    if abort:reasons=[abort]+reasons
    item={"id":"historical_"+name,"kind":"historical_diagnostic","height_cm":height,"batch":"historical","candidate_id":name,"poses_npz":str(trajectory),"time_key":"physics_time","qpos_key":"physics_qpos","scene_path":str(path.parent/"scene.xml"),"source_start_s":source_start,"source_end_s":source_end,"duration_s":source_end-source_start,"video_path":str(OUT/"historical"/(name+".mp4")),"poster_path":str(OUT/"historical"/(name+".jpg")),"video_label":name,"status_text":"HISTORICAL DIAGNOSTIC | "+str(abort or "Full route incomplete")+" | NOT SUCCESS DATA","render_status":"pending","dataset_accepted":False,"full_route_success":bool(d.get("result",{}).get("success",False)),"reasons":reasons,"warnings":warnings,"diagnostic_notes":notes,"diagnostic_path":str(path),"media_provenance":"saved_physics_qpos_offline_render"}
    manifest["historical"].append(item);jobs.append(item)
manifest["counts"]={"candidate_evaluations":len(manifest["candidates"]),"historical_diagnostics":len(manifest["historical"]),"shared_prefixes":len(manifest["prefixes"]),"initial_unique_render_jobs":len(jobs)}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"media_manifest.json").write_text(json.dumps(manifest,indent=2))
(HERE/"render_jobs_initial.json").write_text(json.dumps(jobs,indent=2))
print(json.dumps(manifest["counts"]),flush=True)
