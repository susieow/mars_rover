"""Read-only access to recorded ground-start experiments."""
import json
import re
from pathlib import Path

ROOT=Path('/home/agilex/wzy/rover/runs/ground_mu4_contact_v6to15_20261112')
MEDIA=Path('/home/agilex/wzy/videos/review_high_stairs/ground_mu4_contact_v6to15_20261112')
PAGE=Path(__file__).with_name('ground_mu4_contact_v6to15_review.html')
TAG=re.compile(r'ground_mu4_contact_v(?:[6-9]|1[0-5])_rocker_(?:SN|SP)')

def read(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError): return None

def catalog():
    rows=[]
    for p in sorted((ROOT/'cases').glob('*/result.json')):
        tag=p.parent.name
        if not TAG.fullmatch(tag): continue
        row=read(p)
        if not row or row.get('id')!=tag: continue
        row['video_info']=read(p.parent/'media.json')
        row['video_url']=f'/groundmu4contact-media/{tag}.mp4' if media(tag+'.mp4') else None
        row['curves_url']=f'/groundmu4contact-curves/{tag}.json' if curves(tag+'.json') else None
        row['result_url']=f'/groundmu4contact-data/{tag}/result.json'
        row['trajectory_url']=f'/groundmu4contact-data/{tag}/trajectory.npz'
        rows.append(row)
    return {'planned':20,'cases':rows}

def media(name):
    if not name.endswith('.mp4') or not TAG.fullmatch(name[:-4]): return None
    p=MEDIA/name; meta=read(ROOT/'cases'/name[:-4]/'media.json')
    return p if meta and meta.get('video_path')==str(p) and p.is_file() else None

def curves(name):
    if not name.endswith('.json') or not TAG.fullmatch(name[:-5]): return None
    p=ROOT/'cases'/name[:-5]/'curves.json'
    return p if p.is_file() else None

def data(name):
    parts=name.split('/')
    if len(parts)!=2 or not TAG.fullmatch(parts[0]) or parts[1] not in ('result.json','trajectory.npz'): return None
    p=ROOT/'cases'/parts[0]/parts[1]
    return p if p.is_file() else None
