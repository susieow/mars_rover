"""Run A40/C40 with the same diagnostic rules as the existing A20/C20 runs."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import overspeed20_stairs as diagnostic

RUN=Path('/home/agilex/wzy/rover/runs/overspeed20_A40_C40_20261111')

def execute(dt):
    diagnostic.RUN=RUN
    case={'id':f'A40_C40_dt{round(dt*1e6):04d}us','kind':'speed_matrix',
          'approach_rad_s':40.,'climb_rad_s':40.,'ratios':[1.,1.,1.],
          'runup_m':8.,'physics_dt_s':dt}
    result=diagnostic.run_case(case)
    print(json.dumps({'id':case['id'],'duration_s':result['actual_duration_s'],
       'contact_s':result['first_contact_time_s'],'stop':result['diagnostic_stop_reasons'],
       'platform':result['ascent_platform_support']}),flush=True)
    return result['id']

if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=False)
    diagnostic.common.write_json(RUN/'plan.json',{'case':'A40_C40','physics_steps_s':[.001,.0005,.00025],
       'control_period_s':.001,'runup_m':8.,'prior_case':'A20_C20','purpose':'diagnostic comparison','dataset_accepted':False})
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(execute,(.001,.0005,.00025)))
