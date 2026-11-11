"""Render existing pose jobs with bounded parallelism and update the live catalog."""
import argparse,json,multiprocessing as mp,time,traceback
from pathlib import Path
from render_review_media import render
HERE=Path(__file__).resolve().parent
OUT=Path("/home/agilex/wzy/videos/review_high_stairs")
def safe_render(job):
    try:return render(job)
    except Exception as exc:return {**job,"render_status":"error","render_error":str(exc),"traceback":traceback.format_exc()}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("jobs",type=Path);parser.add_argument("--workers",type=int,default=4);args=parser.parse_args()
    jobs=json.loads(args.jobs.read_text());started=time.time()
    manifest=json.loads((OUT/"media_manifest.json").read_text());completed=0;errors=[]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(safe_render,jobs,chunksize=1):
            completed+=1
            if result["render_status"]=="error":errors.append({"id":result["id"],"error":result["render_error"]})
            for key in ("candidates","historical","prefixes","additional"):
                for row in manifest[key]:
                    if row.get("video_path")==result["video_path"]:
                        for field in ("render_status","render_error","source_duration_s","video_duration_s","frame_count","fps","sample_error_s","probe","render_wall_s","exact_terminal_poster","terminal_poster_time_s","terminal_poster_error_s"):
                            if field in result:row[field]=result[field]
            manifest["render_jobs_completed"]=manifest.get("render_jobs_completed",0)+1
            temporary=OUT/"media_manifest.tmp.json";temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(OUT/"media_manifest.json")
            if completed%20==0 or completed==len(jobs):print(json.dumps({"completed":completed,"total":len(jobs),"errors":len(errors),"elapsed_s":time.time()-started}),flush=True)
    report={"jobs":len(jobs),"completed":completed,"errors":errors,"elapsed_s":time.time()-started}
    (HERE/(args.jobs.stem+"_report.json")).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()
