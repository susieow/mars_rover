"""Preserve original metadata and correct first-failure timestamps from recorded events."""
from pathlib import Path
import hashlib
import json
root=Path('/home/agilex/wzy/rover/runs/overspeed20_20261111/cases')
for path in sorted(root.glob('*/result.json')):
    original=path.read_bytes();d=json.loads(original)
    backup=path.with_name('result.original.json')
    if backup.exists(): raise RuntimeError(f'backup exists: {backup}')
    first=min((v['audit']['events'][0]['first_s'] for v in d['overspeed'].values() if v['audit']['events']),default=None)
    first=min((x for x in (first, d['actual_duration_s'] if d['diagnostic_stop_reasons'] else None) if x is not None),default=None)
    backup.write_bytes(original)
    d['first_failure_s']=first
    d['diagnostic_stop_time_s']=d['actual_duration_s']
    d['metadata_correction']={'field':'first_failure_s','reason':'include recorded speed violations before diagnostic stop','original_result_sha256':hashlib.sha256(original).hexdigest(),'original_file':'result.original.json'}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
    print(d['id'],first,flush=True)
