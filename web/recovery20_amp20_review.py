"""Read-only catalog and registered files for the 22-second recovery experiment."""
from pathlib import Path
import json
import re

ROOT = Path('/home/agilex/wzy/rover/runs/recovery20_t22_amp20_20261111')
MEDIA = Path('/home/agilex/wzy/videos/review_high_stairs/recovery20_t22_amp20_20261111')
PAGE = Path(__file__).with_name('recovery20_amp20_review.html')
TAG = re.compile(r'T22_W(?:H|F|M|R|FM|FR|MR|FMR)_R(?:N|SP|SN|DP|DN)_dt(?:1000|0500|0250)us')


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def catalog():
    rows = []
    for p in sorted((ROOT / 'cases').glob('*/result.json')):
        tag = p.parent.name
        if not TAG.fullmatch(tag):
            continue
        row = read(p)
        if not row or row.get('id') != tag:
            continue
        meta = read(p.parent / 'media.json')
        row['video_info'] = meta
        row['video_url'] = f'/recovery20amp20-media/{tag}.mp4' if media(tag + '.mp4') else None
        row['curves_url'] = f'/recovery20amp20-curves/{tag}.json' if (p.parent / 'curves.json').is_file() else None
        row['result_url'] = f'/recovery20amp20-data/{tag}/result.json'
        row['trajectory_url'] = f'/recovery20amp20-data/{tag}/trajectory.npz'
        row['parent_url'] = f'/overspeed20?case={row["parent_id"]}'
        rows.append(row)
    return {'primary_planned': 32, 'cases': rows, 'gate': read(ROOT / 'checkpoints/dt1000us/gate.json'),
            'completion': read(ROOT / 'experiment_complete.json'), 'verification': read(ROOT / 'verification.json'),
            'refinement': read(ROOT / 'refinement_selection.json'), 'dataset_accepted': 0}


def media(name):
    if not name.endswith('.mp4') or not TAG.fullmatch(name[:-4]):
        return None
    p = MEDIA / name
    meta = read(ROOT / 'cases' / name[:-4] / 'media.json')
    return p if meta and meta.get('render_status') == 'complete' and meta.get('video_path') == str(p) and p.is_file() else None


def curves(name):
    if not name.endswith('.json') or not TAG.fullmatch(name[:-5]):
        return None
    p = ROOT / 'cases' / name[:-5] / 'curves.json'
    return p if p.is_file() else None


def data(name):
    parts = name.split('/')
    if len(parts) != 2 or not TAG.fullmatch(parts[0]) or parts[1] not in ('result.json', 'trajectory.npz'):
        return None
    p = ROOT / 'cases' / parts[0] / parts[1]
    return p if (p.parent / 'result.json').is_file() and p.is_file() else None
