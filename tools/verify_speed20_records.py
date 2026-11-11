"""Verify stored speed-experiment traces without advancing the simulator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

ROOT = Path('/home/agilex/wzy/rover')
RUN = ROOT/'runs/speed20_15steps_20261111'
MEDIA = Path('/home/agilex/wzy/videos/review_high_stairs/speed20_20261111')
sys.path.insert(0,str(ROOT/'scripts/rover'))
import speed20_experiment as common


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    failures=[]
    counts={}
    controls=set()
    phase_never=0
    phase_short=0
    paths=sorted((RUN/'cases').glob('*/result.json'))
    for path in paths:
        r=json.loads(path.read_text())
        identifier=r['id']
        kind=r['kind']
        counts[kind]=counts.get(kind,0)+1
        media=json.loads((path.parent/'media.json').read_text())
        scene=RUN/'flat_scene.xml' if kind=='flat_calibration' else common.FROZEN/'scene.xml'
        try:
            assert sha(path.parent/'trajectory.npz')==r['trajectory_sha256']
            assert sha(scene)==r['scene_sha256']==media['scene_sha256']
            video=MEDIA/(identifier+'.mp4')
            assert sha(video)==media['video_sha256']
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(video)],text=True))
            stream=info['streams'][0]
            assert int(stream['width'])==960 and int(stream['height'])==360
            assert int(stream['nb_frames'])==media['frames'] and stream['r_frame_rate']=='20/1'
            assert media['visual_mode']=='original_urdf_visual_meshes_only'
            assert abs(media['first_time_s'])<1e-12
            assert abs(media['last_time_s']-r['actual_duration_s'])<1e-9
            model=mujoco.MjModel.from_xml_path(str(scene))
            with np.load(path.parent/'trajectory.npz',allow_pickle=False) as z:
                t,qpos,qvel,ctrl,force=z['time'],z['qpos'],z['qvel'],z['ctrl'],z['qfrc_actuator']
                n=len(ctrl)
                assert len(t)==len(qpos)==len(qvel)==n+1 and len(force)==n
                assert np.all(np.diff(t)>0) and abs(t[-1]-r['actual_duration_s'])<1e-9
                assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(ctrl).all() and np.isfinite(force).all()
                for j in range(model.njnt):
                    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
                    if name in r['audit']['joint_velocity_peak_rad_s']:
                        dof=int(model.jnt_dofadr[j])
                        peak=float(np.max(abs(qvel[1:,dof])))
                        assert abs(peak-r['audit']['joint_velocity_peak_rad_s'][name])<1e-9,(name,peak)
                    if name in r['audit']['actuator_force_peak_by_joint_nm']:
                        dof=int(model.jnt_dofadr[j])
                        peak=float(np.max(abs(force[:,dof])))
                        assert abs(peak-r['audit']['actuator_force_peak_by_joint_nm'][name])<1e-9,(name,peak)
                for act in range(model.nu):
                    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,act) or ''
                    if '_wheel_' in name:
                        assert np.max(abs(ctrl[:,act]))<=50.0000001
                if kind=='speed_matrix':
                    controls.add(hashlib.sha256(ctrl.tobytes()).hexdigest())
                    phase=float(np.sum(z['phase']==2))*.001
                    phase_never+=phase==0
                    phase_short+=phase<=.010
                if r.get('first_failure_s') is not None:
                    assert abs(float(r['first_failure_s'])-t[-1])<1e-9
        except Exception as exc:
            failures.append({'id':identifier,'error':repr(exc)})
    analysis=json.loads((RUN/'analysis.json').read_text())
    if counts!={'flat_calibration':6,'speed_matrix':36,'wheel_distribution':4}:
        failures.append({'id':'counts','error':repr(counts)})
    if (len(controls),phase_never,phase_short)!=(analysis['unique_main_actual_control_sequences'],
            analysis['main_climb_phase_never_started'],analysis['main_climb_phase_10ms_or_less']):
        failures.append({'id':'analysis','error':'saved control counts do not match'})
    report={'cases_checked':len(paths),'counts':counts,'unique_main_controls':len(controls),
            'no_climb_phase':phase_never,'climb_phase_at_most_10ms':phase_short,
            'simulation_steps_executed':0,'failures':failures,'passed':not failures}
    common.write_json(RUN/'records_verification.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if not failures else 1


if __name__=='__main__':
    raise SystemExit(check())
