"""Index every attempted candidate for review and deterministic action playback."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
entries=[];batches=[]
for batch,folder,seed_path in (("near_contact",HERE,HERE/"seed_prefix.npz"),("early_entry",HERE/"early_entry",HERE/"early_seed_prefix.npz")):
    report=json.loads((folder/"search_report.json").read_text())
    targets_seen=set();controls_seen=set();batch_rows=[];missing=[];all_max_x=[];all_max_bottom=[]
    for round_index in range(3):
        result_path=folder/f"round{round_index}_results.json"
        target_path=folder/f"round{round_index}_targets.npz"
        trace_path=folder/f"round{round_index}_traces.npz"
        results=json.loads(result_path.read_text())
        with np.load(target_path) as z:targets=z["targets"].copy()
        with np.load(trace_path) as z:traces={k:z[k] for k in z.files}
        for i in range(64):
            expected=f"round{round_index}_candidate{i:02d}"
            found=[row for row in results if row.get("id")==expected]
            if len(found)!=1:
                missing.append({"id":expected,"matching_result_count":len(found)})
                continue
            row=found[0]
            ctrl=traces[f"candidate{i:02d}_controls"];track=traces[f"candidate{i:02d}_track"];support=traces[f"candidate{i:02d}_support"]
            target=targets[i]
            assert ctrl.shape==(len(track),12) and support.shape==(len(track),3)
            assert np.isfinite(ctrl).all() and np.isfinite(track).all() and np.isfinite(support).all()
            assert np.array_equal(target,np.asarray(row["sequence"]))
            ctrl_sha=hashlib.sha256(np.ascontiguousarray(ctrl).tobytes()).hexdigest()
            targets_seen.add(np.ascontiguousarray(target).tobytes());controls_seen.add(ctrl_sha)
            max_x=float(track[:,2].max());max_bottom=float(track[:,4].max())
            all_max_x.append(max_x);all_max_bottom.append(max_bottom)
            center_crossed=bool(np.any(track[:,2]>1.605))
            assert not center_crossed
            result={"id":f"h15_{batch}_{expected}","batch":batch,"candidate_id":expected,"label":row["label"],"round":round_index,"candidate_index":i,"seed_time_s":report["seed_time_s"],"horizon_requested_s":2.0,"actual_physics_steps":len(ctrl),"actual_duration_s":len(ctrl)*.001,"attempt_completed":len(ctrl)==2000,"process_valid":row["valid"],"process_reasons":row["reasons"],"first_failure_s":row["first_failure_s"],"true_both_rear_crossing":False,"crossing_evidence":"Worst rear wheel-center x remains before the first riser at x=1.6 m throughout; no approximate-bottom claim is needed.","max_worst_rear_x_m":max_x,"max_approx_rear_bottom_m":max_bottom,"full_route_success":bool(row["audit"]["audit_quality_ok"] and row["route"]["route_ok"]),"dataset_accepted":False,"sequence":row["sequence"],"seed_npz":str(seed_path),"scene_path":str(HERE/"scene.xml"),"backend_path":str(HERE/"backend.py"),"trace_npz":str(trace_path),"controls_key":f"candidate{i:02d}_controls","track_key":f"candidate{i:02d}_track","support_key":f"candidate{i:02d}_support","control_sha256":ctrl_sha,"result_json":str(result_path),"metrics":row["metrics"],"audit":row["audit"],"route":row["route"]}
            entries.append(result);batch_rows.append(result)
    batches.append({"batch":batch,"planned_attempts":192,"recorded_attempts":len(batch_rows),"missing_or_ambiguous":missing,"completed_horizons":sum(row["attempt_completed"] for row in batch_rows),"process_valid":sum(row["process_valid"] for row in batch_rows),"true_local_crossing":sum(row["true_both_rear_crossing"] for row in batch_rows),"full_route_success":sum(row["full_route_success"] for row in batch_rows),"unique_target_sequences":len(targets_seen),"unique_actual_candidate_controls":len(controls_seen),"seed_time_s":report["seed_time_s"],"seed_npz":str(seed_path),"seed_sha256":sha(seed_path),"results_dir":str(folder),"max_worst_rear_x_m":max(all_max_x),"max_approx_rear_bottom_m":max(all_max_bottom),"wall_elapsed_s":report["wall_elapsed_s"],"search_report":str(folder/"search_report.json")})
manifest={"scope":"Diagnostic action searches on 15 cm, 45 deg stairs; not full-route trials or accepted dataset episodes.","stop_requested_after_current_batch":True,"further_search_started":False,"candidate_evaluations":len(entries),"all_candidates_visible_in_manifest":True,"scene_path":str(HERE/"scene.xml"),"scene_sha256":sha(HERE/"scene.xml"),"backend_path":str(HERE/"backend.py"),"backend_sha256":sha(HERE/"backend.py"),"simulation_dt_s":.001,"control_period_s":.001,"target_columns":["front_rad_s","middle_rad_s","rear_rad_s","rocker_left_deg","rocker_right_deg"],"track_columns":["time_s","base_x_m","worst_rear_x_m","worst_rear_z_m","approx_worst_rear_bottom_m","base_vx_m_s"],"support_columns":["front_world_z_force_n","middle_world_z_force_n","rear_world_z_force_n"],"replay_instructions":"Load original scene and initial_integration_state from seed NPZ, mj_setState once at original initial time, then replay every prefix_ctrl row and every indexed candidate controls row with one mj_step per row. Do not initialize candidate by branch_qpos. Use prefix_time and candidate track times for scene cursor. Candidate controls are already clipped actual torques, not speed targets.","geometry_limitations":"approx_worst_rear_bottom=z-radius omits tilted cylinder half-length extent. Any future crossing requires independent exact cylinder extent and terminal sustained support checks. All current candidates are unambiguously non-crossing because worst rear x remains <1.6m.","batches":batches,"candidates":entries}
(HERE/"candidate_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
print(json.dumps({"manifest":str(HERE/"candidate_manifest.json"),"candidate_evaluations":len(entries),"batches":batches}),flush=True)
