from pathlib import Path
import hashlib,json,math,sys
import numpy as np
sys.path.insert(0,'/home/agilex/wzy/rover/scripts/rover')
import recovery20_friction as exp
import render_dataset
root=Path('/home/agilex/wzy/rover');run=exp.RUN
_, original, _, _, _, parent_meta=exp.parent_exp.restore(.001)
summary={'completed':2,'dataset_accepted':0,'full_route':'not_evaluated','results':{},'checks':['same full state at 22 s','only stair sliding friction changed','effective wheel-stair friction','wheel and rocker commands','force limits','saved trajectory and video hashes','video and curve structure'],'additional_physics_trials':0}
for mu in (6,8):
 exp.MU=mu;exp.SCENE=run/'scenes'/f'scene_mu{mu}.xml'
 tag=f'T22_WMR_RSP_stair_mu{mu}_dt1000us';folder=run/'cases'/tag
 r=json.loads((folder/'result.json').read_text());media=json.loads((folder/'media.json').read_text());curves=json.loads((folder/'curves.json').read_text())
 m,d,c,a,route,meta=exp.restore(.001)
 assert meta['stair_geom_count']==29 and r['stair_friction_slide']==mu and r['original_scene_sha256']==parent_meta['scene_sha256']
 assert r['scene_sha256']==exp.common.digest(exp.SCENE) and r['source_sha256']==exp.source_manifest()
 assert exp.common.digest(folder/'trajectory.npz')==r['trajectory_sha256'] and exp.common.digest(media['video_path'])==media['video_sha256']
 assert media['trajectory_sha256']==r['trajectory_sha256'] and media['parent_trajectory_sha256']==r['parent_trajectory_sha256']
 assert all(abs(x['effective_friction'][0]-mu)<1e-10 and abs(x['effective_friction'][1]-mu)<1e-10 for x in r['wheel_stair_contact_probe'])
 assert np.array_equal(d.qpos,original.qpos) and np.array_equal(d.qvel,original.qvel) and np.array_equal(d.ctrl,original.ctrl)
 with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
  t=z['time']; ctrl=z['ctrl']; tau=z['qfrc_actuator']; n=len(ctrl); times=np.arange(n)*.001
  assert np.array_equal(z['qpos'][0],original.qpos) and np.array_equal(z['qvel'][0],original.qvel)
  assert abs(t[0]-22)<1e-8 and abs(t[-1]-r['actual_duration_s'])<1e-8 and np.max(abs(np.diff(t)-.001))<1e-10
  assert all(np.isfinite(z[k]).all() for k in ('qpos','qvel','ctrl','qfrc_actuator'))
  for i,flag in enumerate((0,1,1)):
   assert np.max(abs(z['wheel_group_commands'][:,i]-(3+flag*np.minimum(times,3))))<1e-10
  phase=np.mod(times,8.); angle=np.interp(phase,[0,1,2,4,5,6,8],[0,20,20,-20,-20,0,0]);angle[times>=24]=0
  assert np.max(abs(z['rocker_targets_rad']-np.radians(angle[:,None])))<1e-10
  for _,act,qp,v,role,left,g in c.wheels: assert np.max(abs(tau[:,v]))<=50.001
  for _,act,qp,v in c.rockers: assert np.max(abs(tau[:,v]))<=100.001
  for _,act,qp,v in c.steer: assert np.max(abs(tau[:,v]))<=40.001
  assert hashlib.sha256(ctrl.tobytes()).hexdigest()==r['actual_control_sequence_sha256']
 assert r['diagnostic_stop_reasons']==['contact_penetration_over_20mm'] and r['audit']['max_contact_penetration_m']>.02
 assert not r['true_both_rear_crossing'] and not r['ascent_platform_support'] and not r['dataset_accepted']
 assert len(curves['joints'])==12
 info=render_dataset._probe(Path(media['video_path']))
 assert info['frames']==media['frames'] and info['width']==960 and info['height']==360
 assert abs(media['first_time_s']-17)<1e-8 and abs(media['last_time_s']-r['actual_duration_s'])<1e-8
 summary['results'][str(mu)]={'id':tag,'friction':mu,'stop_s':r['actual_duration_s'],'continuation_s':r['continuation_duration_s'],'reasons':r['process_reasons'],'worst_contact':r['audit']['worst_contact'],'max_progress_m':r['max_base_x_progress_m'],'net_progress_m':r['net_base_x_m'],'source_result':str(folder/'result.json'),'source_sha256':r['source_sha256'],'trajectory_sha256':r['trajectory_sha256'],'scene_sha256':r['scene_sha256'],'video_sha256':media['video_sha256'],'video_path':media['video_path'],'initial_contact_probe':r['wheel_stair_contact_probe']}
summary['passed']=True
(run/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(run/'experiment_complete.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'docs/recovery20_friction_comparison_20261111.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'results':summary['results']},ensure_ascii=False))
