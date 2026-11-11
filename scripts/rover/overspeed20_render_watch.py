"""Render each completed approved diagnostic once."""
from pathlib import Path
import json
import subprocess
import sys
import time
root=Path('/home/agilex/wzy/rover')
run=root/'runs/overspeed20_20261111'
cases=('A03_C03','A03_C06','A03_C20','A20_C03','A20_C20','A40_C03')
ids=[f'{c}_dt{round(dt*1e6):04d}us' for dt in (.001,.0005,.00025) for c in cases]
seen=set()
while True:
    for tag in ids:
        folder=run/'cases'/tag
        if tag in seen or not (folder/'result.json').exists():
            continue
        seen.add(tag)
        with (run/f'{tag}.render.log').open('w') as log:
            p=subprocess.run([sys.executable,str(root/'scripts/rover/overspeed20_render.py'),'--case',tag],stdout=log,stderr=subprocess.STDOUT,cwd=root)
        print(json.dumps({'id':tag,'exit_code':p.returncode}),flush=True)
    if len(seen)==len(ids):
        break
    batch_log=run/'batch.log'
    if batch_log.exists():
        lines=batch_log.read_text().splitlines()
        if len(lines)>=17 and len(seen)>=1+sum('"exit_code": 0' in line for line in lines):
            break
    time.sleep(5)
