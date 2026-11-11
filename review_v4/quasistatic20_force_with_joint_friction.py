"""Force balance using existing contact geometry and bounded joint dry friction."""
import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages")
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
import mujoco
ROOT = Path('/home/agilex/wzy/rover/review_v4')
source = json.loads((ROOT/'quasistatic20_force_followup.json').read_text())
m = mujoco.MjModel.from_xml_path(source['scene'])
d = mujoco.MjData(m)
an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
B = np.zeros((m.nv, m.nu))
caps, friction = [], []
for i, name in enumerate(an):
    joint = int(m.actuator_trnid[i, 0])
    dof = int(m.jnt_dofadr[joint])
    assert m.actuator_gear[i, 0] == 1.0
    B[dof, i] = 1.0
    caps.append(50.0 if '_wheel_' in name else 100.0 if '_rocker_' in name else 40.0)
    friction.append(float(m.dof_frictionloss[dof]))
caps, friction = np.array(caps), np.array(friction)


def analyze(pose):
    d.qpos[:] = pose['qpos']
    d.qvel[:] = 0
    d.ctrl[:] = 0
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    mujoco.mj_forward(m, d)
    rhs = (d.qfrc_bias - d.qfrc_passive).copy()
    columns, contacts = [], []
    for record in pose['contacts']:
        c = d.contact[record['index']]
        assert c.efc_address >= 0
        j1, j2 = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        mujoco.mj_jac(m,d,j1,None,c.pos,int(m.geom_bodyid[c.geom1]))
        mujoco.mj_jac(m,d,j2,None,c.pos,int(m.geom_bodyid[c.geom2]))
        frame = np.array(c.frame).reshape(3,3)
        mu1,mu2 = c.friction[:2]
        local = np.array([[1,mu1,0],[1,-mu1,0],[1,0,mu2],[1,0,-mu2]],float)
        if c.dim == 1:
            local = np.array([[1,0,0]],float)
        worlds = [frame.T@ray for ray in local]
        start=len(columns)
        columns.extend([(j2-j1).T@world for world in worlds])
        sign = 1.0 if int(m.geom_bodyid[c.geom1]) == 0 else -1.0
        contacts.append({'index':record['index'],'geoms':record['geoms'],'start':start,'count':len(local),
                         'local_rays':local,'world_rays':np.array(worlds),'rover_sign':sign})
    G=np.column_stack(columns)
    A=np.column_stack([B,G,B])
    n=A.shape[1]
    friction_start=m.nu+G.shape[1]
    joint_bounds=[(-float(x),float(x)) for x in friction]
    free_bounds=[(None,None)]*m.nu+[(0,None)]*G.shape[1]+joint_bounds
    rated_bounds=[(-float(x),float(x)) for x in caps]+[(0,None)]*G.shape[1]+joint_bounds
    eq=np.column_stack([A,np.zeros(m.nv)])
    ub=np.zeros((2*m.nu,n+1))
    for i,cap in enumerate(caps):
        ub[2*i,i]=1
        ub[2*i+1,i]=-1
        ub[2*i:2*i+2,-1]=-cap
    peak_cost=np.zeros(n+1);peak_cost[-1]=1.0
    peak=linprog(peak_cost,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=np.zeros(2*m.nu),bounds=free_bounds+[(0,None)],method='highs')

    def solution(x):
        loads=x[m.nu:friction_start]
        out=[]
        for c in contacts:
            lam=loads[c['start']:c['start']+c['count']]
            out.append({'geoms':c['geoms'],'force_contact_n_t1_t2':(lam@c['local_rays']).tolist(),
                        'force_on_rover_world':(c['rover_sign']*(lam@c['world_rays'])).tolist()})
        return {'actuator_torque_nm':dict(zip(an,x[:m.nu].tolist())),
                'joint_dry_friction_reaction_nm':dict(zip(an,x[friction_start:].tolist())),
                'total_contact_normal_n':float(sum(loads)),
                'peak_actuator_ratio':float(np.max(abs(x[:m.nu])/caps)),
                'max_equilibrium_residual':float(np.max(abs(A@x-rhs))),
                'contacts':out}
    output={'label':pose['label'],'provenance':pose['provenance'],'contact_count':len(contacts),
            'min_peak_status':int(peak.status),'minimum_peak_torque_ratio':float(peak.x[-1]) if peak.success else None,
            'rated_feasible':bool(peak.success and peak.x[-1] <= 1+1e-7)}
    if peak.success:
        output['min_peak_solution']=solution(peak.x[:-1])
    normal_cost=np.zeros(n);normal_cost[m.nu:friction_start]=1
    scenarios={'rated_min_normal':rated_bounds}
    nearzero=list(rated_bounds)
    rear50=list(rated_bounds)
    frontzero=list(rated_bounds)
    for i,name in enumerate(an):
        if 'middle_wheel' in name:
            nearzero[i]=(-1.0,1.0)
        if 'rear_wheel' in name:
            rear50[i]=(50.0,50.0)
    for c in contacts:
        if any('front_wheel' in name for name in c['geoms']):
            for i in range(c['start'],c['start']+c['count']):
                frontzero[m.nu+i]=(0.0,0.0)
    scenarios.update(rated_middle_within_1nm=nearzero,rated_both_rear_plus50nm=rear50,rated_front_contact_forces_zero=frontzero)
    output['scenarios']={}
    for name,bounds in scenarios.items():
        check=linprog(normal_cost,A_eq=A,b_eq=rhs,bounds=bounds,method='highs')
        entry={'feasible':bool(check.success),'status':int(check.status)}
        if check.success:
            entry['min_total_contact_normal_n']=float(check.fun)
            normal_cap=check.fun+max(1e-5,check.fun*1e-7)
            sub=linprog(peak_cost,A_eq=eq,b_eq=rhs,
                        A_ub=np.vstack([ub,np.r_[normal_cost,0.0]]),b_ub=np.r_[np.zeros(2*m.nu),normal_cap],
                        bounds=bounds+[(0.0,1.0)],method='highs')
            entry['solution']=solution(sub.x[:-1] if sub.success else check.x)
        output['scenarios'][name]=entry
    print(output['label'],output['rated_feasible'],output['minimum_peak_torque_ratio'],
          [(k,v['feasible'],v.get('min_total_contact_normal_n')) for k,v in output['scenarios'].items()],flush=True)
    return output

result={
 'analysis_only':True,'dynamics_stepped':False,'source_without_joint_friction':str(ROOT/'quasistatic20_force_followup.json'),
 'scene':source['scene'],'scene_sha256':source['scene_sha256'],
 'actuator_caps_nm':dict(zip(an,caps.tolist())),
 'joint_dry_friction_bounds_nm':dict(zip(an,friction.tolist())),
 'balance':'qfrc_bias - qfrc_passive = B*tau + G*nonnegative_contact_rays + B*joint_dry_friction',
 'limitations':['Only saved exact qpos and actual active contacts are used; no changes to model, margin, or geometry.',
                'The recorded velocity is deliberately zeroed for static analysis; this is not a dynamic reproduction.',
                'Joint dry friction can take any reaction within the original model limits at zero velocity.',
                'Contact force equilibrium does not enforce soft contact constitutive response or dynamic reachability.',
                'Minimum-normal solutions may lie on friction boundaries; they are not open-loop commands.'],
 'poses':[analyze(pose) for pose in source['poses']]}
path=ROOT/'quasistatic20_force_with_joint_friction.json'
path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(path,flush=True)
