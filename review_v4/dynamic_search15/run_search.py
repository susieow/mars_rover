"""Run a bounded local search; results are diagnostic and never dataset successes."""
import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import backend
from candidates import first_round,next_round
HERE=Path(__file__).resolve().parent

def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:jsonable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(item) for item in value]
    return value

def compact(result):
    excluded={"controls","track","support","record_qpos","record_qvel","record_qfrc_actuator"}
    return jsonable({key:value for key,value in result.items() if key not in excluded})

def score(result):
    if not result.get("valid",False):return -1e9
    m=result["metrics"]
    return float(100.*(m["rear_both_crossed_continuous_peak_s"]>=.1)
        +10.*m["rear_both_crossed_final_s"]+8.*m["rear_min_x_delta"]
        +3.*(m["rear_min_z_final"]-m["rear_min_z_start"])
        +4.*m["rear_x_change_last_500ms"]+.25*m["base_x_delta"]
        +.003*m["front_middle_both_supported_tail_fraction"])

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed-npz",type=Path,default=HERE/"seed_prefix.npz")
    parser.add_argument("--output-dir",type=Path,default=HERE)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(HERE):raise ValueError("Output must stay in dynamic_search15")
    out.mkdir(parents=True,exist_ok=True)
    with np.load(args.seed_npz) as z:seed_time=float(z["branch_time"])
    started=time.time()
    frozen={str(path.relative_to(HERE)):digest(path) for path in [HERE/"backend.py",HERE/"reference.urdf",HERE/"scene.xml",*sorted((HERE/"sources").glob("*.py"))]}
    candidates,labels=first_round()
    rounds=[];best=None;baseline=None;total_crossed=0;total_valid=0
    with mp.get_context("spawn").Pool(processes=8,initializer=backend.init_worker,initargs=(str(HERE/"scene.xml"),str(args.seed_npz))) as pool:
        for round_index in range(3):
            jobs=[{"id":f"round{round_index}_candidate{i:02d}","sequence":target.tolist(),"horizon":2.0} for i,target in enumerate(candidates)]
            results=pool.map(backend.evaluate,jobs)
            for result,label in zip(results,labels):
                result["label"]=label;result["score"]=score(result)
            ranking=sorted(range(len(results)),key=lambda i:results[i]["score"],reverse=True)
            summaries=[compact(result) for result in results]
            (out/f"round{round_index}_results.json").write_text(json.dumps(summaries,indent=2,allow_nan=False)+"\n")
            np.savez_compressed(out/f"round{round_index}_targets.npz",targets=candidates,labels=np.asarray(labels,dtype=np.str_))
            traces={}
            for i,result in enumerate(results):
                for key in ("controls","track","support"):
                    traces[f"candidate{i:02d}_{key}"]=result.get(key,np.empty((0,0)))
            np.savez_compressed(out/f"round{round_index}_traces.npz",**traces)
            if round_index==0:baseline=summaries[0]
            selected=results[ranking[0]]
            if best is None or selected["score"]>best["score"]:
                best=selected
                np.savez_compressed(out/"selected_candidate.npz",controls=best.get("controls",[]),track=best.get("track",[]),support=best.get("support",[]),sequence=np.asarray(best.get("sequence",jobs[ranking[0]]["sequence"])),candidate_id=str(best["id"]))
                (out/"selected_candidate.json").write_text(json.dumps(compact(best),indent=2,allow_nan=False)+"\n")
            counts={}
            for result in results:
                for reason in result.get("reasons",[]):counts[reason]=counts.get(reason,0)+1
            valid=sum(bool(result.get("valid")) for result in results)
            crossed=sum(bool(result.get("valid")) and result.get("metrics",{}).get("rear_both_crossed_continuous_peak_s",0)>=.1 for result in results)
            total_crossed+=crossed;total_valid+=valid
            summary={"round_index":round_index,"evaluated_candidates":len(results),"process_valid_candidates":valid,"both_rear_crossed_100ms_candidates":crossed,"failure_counts":counts,"best_id":selected["id"],"best_score":selected["score"],"best_metrics":selected.get("metrics"),"wall_elapsed_s":time.time()-started}
            rounds.append(summary)
            print(json.dumps(summary),flush=True)
            if round_index<2:candidates,labels=next_round(candidates,ranking,round_index+1)
    if any(digest(HERE/name)!=sha for name,sha in frozen.items()):raise ValueError("Frozen inputs changed during search")
    report={"dataset_accepted":False,"frozen_sha256":frozen,"candidate_generator_sha256":digest(HERE/"candidates.py"),"driver_sha256":digest(HERE/"run_search.py"),"seed_npz":str(args.seed_npz.resolve()),"seed_sha256":digest(args.seed_npz),"candidate_budget":{"rounds":3,"per_round":64,"evaluated":192,"workers":8},"target_units":["rad/s","rad/s","rad/s","deg","deg"],"target_bounds":[[-2,-2,-2,-30,-30],[3,3,3,15,15]],"sequence_rows":2,"target_hold_s":1.0,"wheel_slew_rad_s2":4.0,"rocker_slew_deg_s":15.0,"horizon_s":2.0,"feedback_period_s":.001,"seed_time_s":seed_time,"scope":"True initial-state prefix replay, short-horizon local search; early-entry comparison only if its explicit seed is selected.","score_definition":"100*(both rear crossing >=100ms) +10*final crossing duration +8*worst-rear dx +3*worst-rear dz +4*rear tail dx +.25*base dx +.003*front/middle tail support fraction","crossing_is_local_diagnostic_only":True,"independent_final_replay_completed":False,"baseline":baseline,"best":compact(best),"rounds":rounds,"valid_candidate_evaluations":total_valid,"both_rear_crossed_100ms_candidate_evaluations":total_crossed,"wall_elapsed_s":time.time()-started}
    (out/"search_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"finished":True,"best_id":best["id"],"best_label":best["label"],"best_metrics":best.get("metrics"),"dataset_accepted":False}),flush=True)
if __name__=="__main__":main()
