"""Read-only catalog for the approved overspeed diagnostics."""
from pathlib import Path
import json
import re
from urllib.parse import quote

ROOT=Path('/home/agilex/wzy/rover/runs/overspeed20_20261111')
MEDIA=Path('/home/agilex/wzy/videos/review_high_stairs/overspeed20_20261111')
EXTRA_ROOT=Path('/home/agilex/wzy/rover/runs/overspeed20_A40_C40_20261111')
EXTRA_MEDIA=Path('/home/agilex/wzy/videos/review_high_stairs/overspeed20_A40_C40_20261111')
PAGE=Path(__file__).with_name('overspeed20_review.html')
PATTERN=re.compile(r'A(?:03|20|40)_C(?:03|06|20|40)_dt(?:1000|0500|0250)us')
def valid(tag): return bool(PATTERN.fullmatch(tag))
def read(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError): return None

def catalog():
    rows=[]
    files=list((ROOT/'cases').glob('*/result.json'))+list((EXTRA_ROOT/'cases').glob('*/result.json'))
    for f in sorted(files):
        tag=f.parent.name
        if not valid(tag): continue
        r=read(f)
        if not isinstance(r,dict) or r.get('id')!=tag: continue
        media=read(f.parent/'media.json')
        video=(EXTRA_MEDIA if tag.startswith('A40_C40_') else MEDIA)/f'{tag}.mp4'
        r['video_url']=f'/overspeed20-media/{quote(tag)}.mp4' if isinstance(media,dict) and media.get('render_status')=='complete' and media.get('video_path')==str(video) and video.is_file() else None
        r['curves_url']=f'/overspeed20-curves/{quote(tag)}.json' if (f.parent/'curves.json').is_file() else None
        r['video_info']=media
        rows.append(r)
    return {'planned':21,'cases':rows,'prefix_gate':read(ROOT/'prefix_gate.json'),'hardware_limit':'URDF records 1.57 rad/s; hardware source not verified','full_route':'not_evaluated','dataset_accepted':0}

def media(name):
    if not name.endswith('.mp4') or not valid(name[:-4]): return None
    tag=name[:-4]; root=EXTRA_ROOT if tag.startswith('A40_C40_') else ROOT; media_root=EXTRA_MEDIA if tag.startswith('A40_C40_') else MEDIA
    p=(media_root/name).resolve(); meta=read(root/'cases'/tag/'media.json')
    if not isinstance(meta,dict) or meta.get('video_path')!=str(p) or meta.get('render_status')!='complete': return None
    return p if p.is_file() else None

def curves(name):
    if not name.endswith('.json') or not valid(name[:-5]): return None
    tag=name[:-5]; root=EXTRA_ROOT if tag.startswith('A40_C40_') else ROOT
    p=root/'cases'/tag/'curves.json'
    return p if p.is_file() else None
