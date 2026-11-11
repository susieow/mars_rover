"""Generate review curves for completed approved diagnostics."""
from pathlib import Path
import json
import subprocess
import sys
import time
root=Path('/home/agilex/wzy/rover'); run=root/'runs/overspeed20_20261111'
seen=set()
while True:
    for path in sorted((run/'cases').glob('*/result.json')):
        tag=path.parent.name
        if tag in seen: continue
        seen.add(tag)
        with (run/f'{tag}.curves.log').open('w') as log:
            p=subprocess.run([sys.executable,str(root/'scripts/rover/overspeed20_curves.py'),tag],stdout=log,stderr=subprocess.STDOUT,cwd=root)
        print(json.dumps({'id':tag,'exit_code':p.returncode}),flush=True)
    if len(seen)>=18: break
    batch=run/'batch.log'
    if batch.exists():
        lines=batch.read_text().splitlines()
        if len(lines)>=17 and len(seen)>=1+sum('"exit_code": 0' in line for line in lines): break
    time.sleep(5)
