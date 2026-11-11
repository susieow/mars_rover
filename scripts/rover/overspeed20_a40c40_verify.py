"""Verify diagnostic trajectories, audit metrics and rendered media."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

root=Path('/home/agilex/wzy/rover');run=root/'runs/overspeed20_A40_C40_20261111'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
results=[]
for p in sorted((run/'cases').glob('*/result.json')):
    folder=p.parent;d=json.loads(p.read_text()); media=json.loads((folder/'media.json').read_text()); curves=json.loads((folder/'curves.json').read_text())
    assert digest(folder/'trajectory.npz')==d['trajectory_sha256'],d['id']
    assert digest(Path(media['video_path']))==media['video_sha256'],d['id']
    probe=subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','default=noprint_wrappers=1:nokey=1',media['video_path']],text=True).strip()
    assert int(probe)==media['frames'],(d['id'],probe,media['frames'])
    assert media['first_time_s']==0 and abs(media['last_time_s']-d['actual_duration_s'])<1e-8
    with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
        t=z['time'];vel=z['qvel'];dt=float(z['simulation_dt']);ctrl=z['ctrl']; assert len(t)==len(ctrl)+1
        assert np.max(np.abs(np.diff(t)-dt))<1e-9
        step=round(.001/dt)
        for offset in range(1,step):
            assert np.array_equal(ctrl[offset::step],ctrl[:len(ctrl)-offset:step]),(d['id'],'hold',offset)
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(root/'review_v4/dynamic_search20/scene.xml'))
        for name,limit in d['audit']['joint_velocity_limit_rad_s'].items():
            jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);idx=int(m.jnt_dofadr[jid]);actual=vel[1:,idx]; peak=float(np.max(np.abs(actual)))
            assert abs(peak-d['audit']['joint_velocity_peak_rad_s'][name])<1e-9,(d['id'],name,'peak')
            for label,threshold in [('urdf',limit),('audit',limit+.01)]:
                o=d['overspeed'][name][label];mask=np.abs(actual)>threshold
                assert abs(float(np.sum(mask)*dt)-o['total_duration_s'])<1e-8,(d['id'],name,label,'duration')
                if mask.any():
                    first=int(np.flatnonzero(mask)[0])+1
                    assert abs(t[first]-o['events'][0]['first_s'])<1e-9
            a=curves['joints'][name]
            assert a['limit']==limit
            if name.endswith(('_rocker_joint','_steer_joint')):
                q=int(m.jnt_qposadr[jid]);act=next(i for i in range(m.nu) if int(m.actuator_trnid[i,0])==jid)
                rows=np.array(a['rows']);inds=np.rint(rows[:,0]/dt).astype(int)-1
                start=(inds//step)*step
                gain,force=(150.,100.) if name.endswith('_rocker_joint') else (50.,40.)
                expected=np.clip(gain*(rows[:,2]-vel[start,idx]),-force,force)
                assert np.max(np.abs(expected-ctrl[inds,act]))<.001,(d['id'],name,'reconstructed_target',np.max(np.abs(expected-ctrl[inds,act])))
    results.append({'id':d['id'],'trajectory_sha256':d['trajectory_sha256'],'video_sha256':media['video_sha256'],'video_frames':media['frames'],'ok':True})
    print(json.dumps(results[-1]),flush=True)
(run/'verification.json').write_text(json.dumps({'count':len(results),'results':results},indent=2)+'\n')
