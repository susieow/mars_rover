"""Render recorded 20 cm experiment states using the original URDF visuals."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np

import speed20_experiment as common
import render_dataset

MEDIA = Path('/home/agilex/wzy/videos/review_high_stairs/speed20_20261111')
WIDTH, HEIGHT, FPS = 480, 360, 20


def render(case_id):
    folder = common.RUN/'cases'/case_id
    result = json.loads((folder/'result.json').read_text())
    existing = folder/'media.json'
    if existing.exists():
        report = json.loads(existing.read_text())
        if Path(report['video_path']).is_file() and common.digest(report['video_path']) == report['video_sha256']:
            print(json.dumps({'id':case_id,'status':'already_rendered'}),flush=True)
            return
        raise RuntimeError('existing media record does not match its video')
    scene = common.RUN/'flat_scene.xml' if result['kind']=='flat_calibration' else common.FROZEN/'scene.xml'
    path = folder/'trajectory.npz'
    if common.digest(path) != result['trajectory_sha256'] or common.digest(scene) != result['scene_sha256']:
        raise ValueError('scene or recorded trajectory hash differs from result')
    with np.load(path,allow_pickle=False) as archive:
        times = archive['time'].copy()
        poses = archive['qpos'].copy()
    if len(times) != len(poses) or len(times)<2 or not np.all(np.diff(times)>0):
        raise ValueError('invalid saved trajectory')
    metadata = {'audit':{'velocity_limit_source':{'path':str(common.URDF)}},'urdf_sha256':common.digest(common.URDF)}
    model, option, visual_info = render_dataset._visual_render_model(scene,folder,metadata)
    if model.nq != poses.shape[1]:
        raise ValueError('visual model state dimensions differ')
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data=mujoco.MjData(model)
    camera1,camera2=mujoco.MjvCamera(),mujoco.MjvCamera()
    for camera in (camera1,camera2):
        mujoco.mjv_defaultFreeCamera(model,camera)
        camera.distance=4.8
    camera1.azimuth,camera1.elevation=90.,-17.
    camera2.azimuth,camera2.elevation=45.,-22.
    requested=np.arange(int(math.floor(float(times[-1])*FPS))+1)/FPS
    if abs(requested[-1]-times[-1])>1e-9:
        requested=np.append(requested,times[-1])
    selected=render_dataset._nearest(times,requested)
    if selected[0] != 0 or selected[-1] != len(times)-1:
        raise ValueError('render must include both original endpoint states')
    if np.max(abs(times[selected]-requested)) > float(np.max(np.diff(times)))/2+1e-8:
        raise ValueError('recorded states are too sparse')
    MEDIA.mkdir(parents=True,exist_ok=True)
    destination=MEDIA/f'{case_id}.mp4'
    temporary=MEDIA/f'.{case_id}.{os.getpid()}.tmp.mp4'
    renderer=mujoco.Renderer(model,height=HEIGHT,width=WIDTH)
    cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{WIDTH*2}x{HEIGHT}','-r',str(FPS),'-i','-','-an',
         '-c:v','libx264','-preset','ultrafast','-threads','1','-crf','23',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for index in selected:
            data.time=float(times[index])
            data.qpos[:]=poses[index]
            data.qvel[:]=0.
            mujoco.mj_forward(model,data)
            for camera in (camera1,camera2):
                camera.lookat[:]=[float(data.qpos[0]),float(data.qpos[1]),float(data.qpos[2])]
            renderer.update_scene(data,camera=camera1,scene_option=option)
            side=renderer.render().copy()
            renderer.update_scene(data,camera=camera2,scene_option=option)
            quarter=renderer.render().copy()
            proc.stdin.write(np.concatenate((side,quarter),axis=1).tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        errors=proc.stderr.read().decode('utf-8',errors='replace')
        code=proc.wait()
        renderer.close()
    if code:
        raise RuntimeError(f'ffmpeg failed ({code}): {errors}')
    info=render_dataset._probe(temporary)
    if info['frames'] != len(selected) or info['width'] != WIDTH*2 or info['height'] != HEIGHT:
        raise RuntimeError('rendered video frame count or geometry differs')
    temporary.replace(destination)
    report={'id':case_id,'render_status':'complete','video_path':str(destination),
            'video_sha256':common.digest(destination),'render_source':'original_recorded_state',
            'scene_sha256':result['scene_sha256'],'trajectory_sha256':result['trajectory_sha256'],
            'frames':len(selected),'fps':FPS,'duration_s':float(times[-1]),
            'first_time_s':float(times[selected[0]]),'last_time_s':float(times[selected[-1]]),
            'max_sample_error_s':float(np.max(abs(times[selected]-requested))),
            'visual_mode':visual_info['mode'],'render_scene_sha256':visual_info['render_scene_sha256']}
    common.write_json(existing,report)
    print(json.dumps({'id':case_id,'status':'rendered','frames':len(selected),'duration_s':times[-1]}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('shard must be between 0 and shards-1')
    paths=[common.RUN/'cases'/args.case/'result.json'] if args.case else sorted((common.RUN/'cases').glob('*/result.json'))
    paths.sort(key=lambda p:(float(json.loads(p.read_text()).get('actual_duration_s',0)),p.parent.name))
    for path in paths[args.shard::args.shards]:
        render(path.parent.name)


if __name__ == '__main__':
    main()
