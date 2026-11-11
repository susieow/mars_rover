"""Render saved poses with original URDF visuals, without advancing dynamics."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
import argparse,hashlib,json,math,subprocess,sys,time
from pathlib import Path
import mujoco,numpy as np
sys.path.append("/usr/lib/python3/dist-packages")
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,"/home/agilex/wzy/rover/scripts/rover")
import render_dataset as rd
rd.WIDTH=960;rd.HEIGHT=360;rd.SHEET_CROP_TOP=0
WIDTH=960;HEIGHT=360;FPS=20
HERE=Path(__file__).resolve().parent
OUTPUT=Path("/home/agilex/wzy/videos/review_high_stairs")
CACHE={}
FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
URDF=Path("/home/agilex/wzy/rover/assets/mars_rover/urdf/mars_rover.urdf")

def visual(scene):
    key=str(Path(scene).resolve())
    if key in CACHE:return CACHE[key]
    if len(CACHE)>=4:
        CACHE.pop(next(iter(CACHE)))[2].close()
    folder=HERE/"render_work"/str(os.getpid())/hashlib.sha256(key.encode()).hexdigest()[:12]
    folder.mkdir(parents=True,exist_ok=True)
    model,option,info=rd._visual_render_model(Path(key),folder,{"audit":{"velocity_limit_source":{"path":str(URDF)}},"urdf_sha256":hashlib.sha256(URDF.read_bytes()).hexdigest()})
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=HEIGHT,width=WIDTH//2)
    cameras=[mujoco.MjvCamera(),mujoco.MjvCamera()]
    for c in cameras:mujoco.mjv_defaultFreeCamera(model,c)
    cameras[0].azimuth=90.;cameras[0].elevation=-18.;cameras[1].azimuth=45.;cameras[1].elevation=-24.
    geoms=np.flatnonzero((model.geom_bodyid!=0)&(model.geom_group==1))
    platform=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"platform")
    result=(model,data,renderer,option,cameras,geoms,platform,info)
    CACHE[key]=result
    return result

def image_at(state,qpos,t,title,status):
    model,data,renderer,option,cameras,geoms,platform,info=state
    data.qpos[:]=qpos;data.qvel[:]=0;data.time=float(t);mujoco.mj_forward(model,data)
    centers=data.geom_xpos[geoms];radii=model.geom_rbound[geoms]
    lower=np.min(centers-radii[:,None],axis=0);upper=np.max(centers+radii[:,None],axis=0);focus=(lower+upper)/2
    left=float(model.geom_pos[platform,0]-model.geom_size[platform,0]);right=float(model.geom_pos[platform,0]+model.geom_size[platform,0])
    cameras[1].azimuth=45.+90.*float(np.clip((focus[0]-left)/(right-left),0,1))
    images=[]
    for camera,min_distance in zip(cameras,(4.0,4.2)):
        rd._fit_review_camera(camera,lower,upper,model.vis.global_.fovy,min_distance)
        renderer.update_scene(data,camera=camera,scene_option=option)
        visible={int(g.objid) for g in renderer.scene.geoms[:renderer.scene.ngeom] if int(g.objtype)==int(mujoco.mjtObj.mjOBJ_GEOM)}
        if any(model.geom_bodyid[g]!=0 and model.geom_group[g]!=1 for g in visible):raise RuntimeError("Collision geometry visible")
        if not set(map(int,geoms)).issubset(visible):raise RuntimeError("URDF visual geometry missing")
        images.append(renderer.render().copy())
    image=Image.fromarray(np.concatenate(images,axis=1))
    draw=ImageDraw.Draw(image);draw.rectangle((0,0,WIDTH,38),fill=(15,23,35))
    draw.text((8,2),f"{title} | physical t={t:.3f}s",font=FONT,fill=(235,240,245))
    draw.text((8,20),status[:132],font=FONT,fill=(255,195,105))
    draw.rectangle((0,HEIGHT-20,WIDTH,HEIGHT),fill=(15,23,35))
    draw.text((8,HEIGHT-20),"SIDE",font=FONT,fill=(240,240,240))
    draw.text((WIDTH//2+8,HEIGHT-20),"THREE-QUARTER",font=FONT,fill=(240,240,240))
    return np.asarray(image)

def render(job):
    begun=time.perf_counter()
    with np.load(job["poses_npz"]) as z:
        times=z[job.get("time_key","time")].copy()
        poses=z[job.get("qpos_key","qpos")].copy()
    start=float(job.get("source_start_s",times[0]));end=float(job.get("source_end_s",times[-1]));duration=end-start
    if duration<=0:raise ValueError("Nonpositive source duration")
    frames=max(1,int(math.ceil(duration*FPS-1e-9)))
    requested=np.minimum(start+(np.arange(frames)+1)/FPS,end)
    indices=rd._nearest(times,requested)
    sample_error=float(np.max(np.abs(times[indices]-requested)))
    if sample_error>.00050001:raise ValueError(f"Pose sample error exceeds 0.5 ms: {sample_error}")
    path=Path(job["video_path"]);poster=Path(job["poster_path"]);path.parent.mkdir(parents=True,exist_ok=True)
    state=visual(job["scene_path"])
    partial=path.with_name("."+path.stem+f".{os.getpid()}.tmp.mp4")
    process=subprocess.Popen(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","20","-threads","1","-pix_fmt","yuv420p","-movflags","+faststart",str(partial)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    for index in indices:
        frame=image_at(state,poses[index],float(times[index]),job.get("video_label",job["id"]),job.get("status_text","DIAGNOSTIC ONLY | not a successful dataset episode"))
        process.stdin.write(frame.tobytes())
    process.stdin.close();error=process.stderr.read().decode();code=process.wait()
    if code:raise RuntimeError(f"ffmpeg exited {code}: {error}")
    terminal_index=int(rd._nearest(times,np.asarray([end]))[0])
    terminal_error=abs(float(times[terminal_index])-end)
    if terminal_error>.00050001:raise ValueError("Terminal poster is not aligned with source end")
    terminal=image_at(state,poses[terminal_index],float(times[terminal_index]),job.get("video_label",job["id"]),job.get("status_text","DIAGNOSTIC ONLY"))
    Image.fromarray(terminal).save(poster,quality=92)
    probe=rd._probe(partial)
    if probe["frames"]!=frames or probe["width"]!=WIDTH or probe["height"]!=HEIGHT or abs(probe["fps"]-FPS)>1e-9:raise ValueError("Video probe mismatch")
    if abs(probe["video_duration_s"]-duration)>1./FPS+1e-6:raise ValueError("Video duration differs by more than one frame")
    partial.replace(path)
    result={**job,"render_status":"complete","source_start_s":start,"source_end_s":end,"source_duration_s":duration,"video_duration_s":probe["video_duration_s"],"frame_count":frames,"fps":FPS,"sample_error_s":sample_error,"probe":probe,"render_wall_s":time.perf_counter()-begun,"visual_geometry":state[-1],"exact_terminal_poster":terminal_error<1e-9,"terminal_poster_time_s":float(times[terminal_index]),"terminal_poster_error_s":terminal_error,"sampling":"20 fps post-bin samples; last frame is exact source terminal. Constant-rate final duration may exceed source by less than 50 ms."}
    path.with_suffix(".json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"id":job["id"],"render_wall_s":result["render_wall_s"],"source_duration_s":duration,"frames":frames,"video_path":str(path)}),flush=True)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("job",type=Path);args=parser.parse_args()
    render(json.loads(args.job.read_text()))
