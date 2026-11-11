"""Independently probe every unique media file and verify all catalog references."""
import concurrent.futures,json,math,subprocess
from collections import Counter
from pathlib import Path
import sys
sys.path.append("/usr/lib/python3/dist-packages")
from PIL import Image
OUT=Path("/home/agilex/wzy/videos/review_high_stairs")
def probe(item):
    path=Path(item["video_path"]);poster=Path(item["poster_path"]);errors=[]
    if not path.is_file():return {"video_path":str(path),"errors":["missing_video"]}
    if not poster.is_file():errors.append("missing_poster")
    data=json.loads(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,nb_frames,r_frame_rate,duration:format=duration","-of","json",str(path)],capture_output=True,text=True,check=True).stdout)
    stream=data["streams"][0];width=int(stream["width"]);height=int(stream["height"]);frames=int(stream["nb_frames"])
    n,d=stream["r_frame_rate"].split("/");fps=float(n)/float(d)
    duration=float(stream.get("duration",data["format"]["duration"]))
    source_duration=float(item["source_end_s"]-item["source_start_s"])
    expected=max(1,int(math.ceil(source_duration*20-1e-9)))
    if (width,height)!=(960,360):errors.append("wrong_dimensions")
    if fps!=20:errors.append("wrong_fps")
    if frames!=expected:errors.append("wrong_frame_count")
    if abs(duration-source_duration)>.050001:errors.append("wrong_duration")
    if abs(item.get("terminal_poster_time_s",-1)-item["source_end_s"])>.00050001:errors.append("terminal_poster_time_mismatch")
    if poster.is_file():
        with Image.open(poster) as image:
            if image.size!=(960,360):errors.append("wrong_poster_dimensions")
            image.verify()
    return {"video_path":str(path),"poster_path":str(poster),"width":width,"height":height,"fps":fps,"frames":frames,"source_duration_s":source_duration,"video_duration_s":duration,"duration_difference_s":duration-source_duration,"bytes":path.stat().st_size,"errors":errors}
def main():
    manifest=json.loads((OUT/"media_manifest.json").read_text())
    rows=manifest["candidates"]+manifest["historical"]+manifest["prefixes"]+manifest.get("additional",[])
    unique={row["video_path"]:row for row in rows}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:results=list(pool.map(probe,unique.values()))
    errors=[row for row in results if row["errors"]]
    statuses={key:dict(Counter(row.get("render_status") for row in manifest[key])) for key in ("candidates","historical","prefixes","additional")}
    counts={"candidate_entries":len(manifest["candidates"]),"historical_entries":len(manifest["historical"]),"shared_prefix_entries":len(manifest["prefixes"]),"additional_validation_entries":len(manifest.get("additional",[])),"all_catalog_entries":len(rows),"unique_mp4_files":len(unique),"unique_candidate_mp4_files":len(set(row["video_path"] for row in manifest["candidates"])),"unique_posters":len(set(row["poster_path"] for row in rows)),"checked_with_ffprobe":len(results)}
    report={"media_complete":not errors and all(row.get("render_status")=="complete" for row in rows),"counts":counts,"statuses":statuses,"missing_or_invalid":errors,"total_unique_video_bytes":sum(row.get("bytes",0) for row in results),"max_duration_difference_s":max(abs(row.get("duration_difference_s",0)) for row in results),"scope":"Every unique encoded video independently ffprobed; every catalog entry and poster checked. Does not certify rover success.","files":results}
    (OUT/"media_validation.json").write_text(json.dumps(report,indent=2))
    manifest["validation_path"]=str(OUT/"media_validation.json");manifest["media_complete"]=report["media_complete"];manifest["final_media_counts"]=counts
    temporary=OUT/"media_manifest.tmp.json";temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(OUT/"media_manifest.json")
    print(json.dumps({k:v for k,v in report.items() if k!="files"}),flush=True)
    assert report["media_complete"]
if __name__=="__main__":main()
