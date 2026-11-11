from pathlib import Path
import hashlib,json,math,sys
import mujoco,numpy as np
sys.path.insert(0,'/home/agilex/wzy/rover/scripts/rover')
import recovery20_amp20_experiment as parent
import recovery20_rear_lock as rear
import recovery20_all_lock as all_lock
root=Path('/home/agilex/wzy/rover')
_, original, _, _, _, parent_meta=parent.restore(.001)
summary={'parent_id':'T22_WMR_RSP_dt1000us','intervention_s':22,'parent_checkpoint_id':parent_meta['checkpoint_id'],'runs':{}}
for exp,tag in [(rear,'T22_WMR_RSP_rear_lock_dt1000us'),(all_lock,'T22_WMR_RSP_all_lock_dt1000us')]:
 folder=exp.RUN/'cases'/tag;r=json.loads((folder/'result.json').read_text());media=json.loads((folder/'media.json').read_text());curves=json.loads((folder/'curves.json').read_text())
 assert exp.common.digest(folder/'trajectory.npz')==r['trajectory_sha256'] and exp.common.digest(media['video_path'])==media['video_sha256']
 assert exp.common.digest(exp.SCENE)==r['scene_sha256'] and r['original_scene_sha256']==parent_meta['scene_sha256']
 assert media['trajectory_sha256']==r['trajectory_sha256'] and media['frames']==701 and media['visual_mode']=='original_urdf_visual_meshes_only'
 m,d,c,a,route,meta=exp.restore(.001)
 assert len(r['rear_steer_lock_positions_rad'])==m.neq and set(curves['joints'])==set(r['audit']['joint_velocity_limit_rad_s'])
 with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
  assert np.array_equal(z['qpos'][0],original.qpos) and np.array_equal(z['qvel'][0],original.qvel)
  t=z['time'];assert abs(t[0]-22)<1e-8 and abs(t[-1]-52)<1e-8 and np.max(abs(np.diff(t)-.001))<1e-9
  assert all(np.isfinite(z[k]).all() for k in ('qpos','qvel','ctrl','qfrc_actuator'))
  commands=z['wheel_group_commands'];sec=np.arange(len(commands))*.001
  for i,flag in enumerate((0,1,1)):
   assert np.max(abs(commands[:,i]-(3+flag*np.minimum(sec,3))))<1e-10
  rocker=z['rocker_targets_rad'];phase=np.mod(sec,8.)
  target=np.interp(phase,[0,1,2,4,5,6,8],[0,20,20,-20,-20,0,0]);target[sec>=24]=0
  assert np.max(abs(rocker-np.radians(target[:,None])))<1e-10
  deviations={}
  for name,angle in r['rear_steer_lock_positions_rad'].items():
   j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);q=m.jnt_qposadr[j];v=m.jnt_dofadr[j]
   assert abs(z['qpos'][0,q]-angle)<1e-12
   assert np.max(abs(z['qfrc_actuator'][:,v]))<1e-10
   deviations[name]={'lock_angle_deg':math.degrees(angle),'max_deviation_deg':float(np.degrees(np.max(abs(z['qpos'][:,q]-angle)))),'max_speed_rad_s':float(np.max(abs(z['qvel'][1:,v])))}
  for _,act,qp,v,role,left,g in c.wheels: assert np.max(abs(z['qfrc_actuator'][:,v]))<=50.001
  for _,act,qp,v in c.rockers: assert np.max(abs(z['qfrc_actuator'][:,v]))<=100.001
  for _,act,qp,v in c.steer: assert np.max(abs(z['qfrc_actuator'][:,v]))<=40.001
  assert hashlib.sha256(z['ctrl'].tobytes()).hexdigest()==r['actual_control_sequence_sha256']
 summary['runs'][tag]={'process_valid':r['process_valid'],'process_reasons':r['process_reasons'],'duration_s':r['continuation_duration_s'],'rear_crossing':r['true_both_rear_crossing'],'platform':r['ascent_platform_support'],'max_progress_m':r['max_base_x_progress_m'],'net_progress_m':r['net_base_x_m'],'lock_deviations':deviations,'virtual_lock_note':r['rear_lock_method'],'trajectory_sha256':r['trajectory_sha256'],'scene_sha256':r['scene_sha256'],'video_sha256':media['video_sha256'],'video_path':media['video_path'],'dataset_accepted':False}
print(json.dumps({'passed':True,'cases':summary['runs']},ensure_ascii=False))
(root/'docs/recovery20_steering_lock_comparison_20261111.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
