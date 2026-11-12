from pathlib import Path
import hashlib,json,sys
import numpy as np
import mujoco
sys.path.insert(0,'/home/agilex/wzy/rover/scripts/rover')
import ground_mu4_contact_v6to15_20261112 as exp
import recovery20_amp20_experiment as wave
import render_dataset

run=exp.RUN
rows=[]
for speed in range(6,16):
 for direction in ('SN','SP'):
  tag=f'ground_mu4_contact_v{speed}_rocker_{direction}'
  p=run/'cases'/tag
  r=json.loads((p/'result.json').read_text());media=json.loads((p/'media.json').read_text());curves=json.loads((p/'curves.json').read_text())
  assert r['parameters']['wheel_base_target_rad_s']==speed and r['parameters']['rocker']==direction and r['parameters']['stair_friction_slide']==4
  assert r['source_sha256']==exp.common.digest(exp.__file__) and r['scene_sha256']==exp.common.digest(exp.SCENE)
  assert r['trajectory_sha256']==exp.common.digest(p/'trajectory.npz') and media['video_sha256']==exp.common.digest(media['video_path'])
  m=mujoco.MjModel.from_xml_path(str(exp.SCENE))
  stair=[i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(('up','down'))]
  assert len(stair)==29 and np.all(m.geom_friction[stair,0]==4)
  with np.load(p/'trajectory.npz',allow_pickle=False) as z:
   t=z['time'];n=len(t)-1;assert abs(t[0])<1e-12 and np.isfinite(z['qpos']).all() and np.isfinite(z['qvel']).all() and np.allclose(np.diff(t),.001)
   expected=np.minimum((np.arange(n)+1)*.03,speed)
   assert np.max(abs(z['wheel_group_commands']-expected[:,None]))<1e-8
   trig=r['stall_trigger_s'];contact=r['first_contact_s']
   if trig is not None:
    i=round(trig/.001);ic=round(contact/.001)
    assert i==ic and abs(trig-contact)<1e-9
    assert np.max(abs(z['rocker_targets_rad'][:i]))==0
    for offset in (0,500,1000,2000,3000,4000,5000,6000,8000,24000):
     if i+offset<n:
      _,want=wave.targets(wave.params('H',direction),offset*.001)
      assert np.max(abs(z['rocker_targets_rad'][i+offset]-want))<1e-8
    assert abs(t[-1]-trig-30)<1e-7
   else:
    assert abs(t[-1]-120)<1e-7 and np.max(abs(z['rocker_targets_rad']))==0
   assert len(curves['joints'])==12 and abs(media['first_time_s'])<1e-9 and abs(media['last_time_s']-t[-1])<1e-8
   assert not r['audit']['nonfinite_seen'] and not r['audit']['time_nonmonotonic_seen'] and not r['audit']['mujoco_warning_counts']
   info=render_dataset._probe(Path(media['video_path']));assert info['frames']==media['frames']
  rows.append({'id':tag,'speed':speed,'direction':direction,'contact_s':contact,'trigger_s':trig,'end_s':r['actual_duration_s'],'reasons':r['process_reasons'],'crossing':r['true_both_rear_crossing'],'retained':r['rear_retained_geometry_level_final_1s'],'platform':r['ascent_platform_support'],'penetration_mm':r['audit']['max_contact_penetration_m']*1000,'result_sha256':exp.common.digest(p/'result.json'),'trajectory_sha256':r['trajectory_sha256'],'video_sha256':media['video_sha256'],'video_path':media['video_path'],'dataset_accepted':False,'do_not_repeat_identical_exploration':True})
summary={'completed':20,'passed':True,'checks':['start from ground','wheel base targets','stair friction','first front contact trigger','rocker target sequence','finite simulation and warnings','saved media and trajectory hashes'],'results':rows}
exp.common.write_json(run/'verification.json',summary)
exp.common.write_json(exp.ROOT/'docs/ground_mu4_contact_v6to15_20261112.json',summary)
(exp.ROOT/'docs/diagnostic_registry20_ground_mu4_contact_v6to15_20261112.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
