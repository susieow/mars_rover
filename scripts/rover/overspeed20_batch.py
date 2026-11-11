"""Run the approved fixed diagnostic set with bounded concurrency."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import subprocess
import sys

root=Path('/home/agilex/wzy/rover')
run=root/'runs/overspeed20_20261111'
cases=('A03_C03','A03_C06','A03_C20','A20_C03','A20_C20','A40_C03')
gate=json.loads((run/'prefix_gate.json').read_text())
if not gate.get('passed'):
    raise RuntimeError('prefix comparison did not pass')
jobs=[(c,dt) for dt in (.001,.0005,.00025) for c in cases if not (c=='A03_C06' and dt==.001)]
for c,dt in jobs:
    tag=f'{c}_dt{round(dt*1e6):04d}us'
    if (run/'cases'/tag).exists():
        raise RuntimeError(f'refusing to overwrite existing {tag}')
(run/'batch_plan.json').write_text(json.dumps({'jobs':jobs,'prefix_gate':gate,'concurrency':3},indent=2)+'\n')
def execute(job):
    c,dt=job
    tag=f'{c}_dt{round(dt*1e6):04d}us'
    with (run/f'{tag}.log').open('w') as log:
        r=subprocess.run([sys.executable,str(root/'scripts/rover/overspeed20_stairs.py'),'--case',c,'--dt',str(dt)],stdout=log,stderr=subprocess.STDOUT,cwd=root)
    return {'id':tag,'exit_code':r.returncode}
with ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(execute,job) for job in jobs]
    for future in as_completed(futures):
        print(json.dumps(future.result()),flush=True)
