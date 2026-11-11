"""Read-only catalog for the 20 cm speed comparison."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import quote

ROOT = Path.home()/'wzy/rover/runs/speed20_15steps_20261111'
MEDIA = Path.home()/'wzy/videos/review_high_stairs/speed20_20261111'
PAGE = Path(__file__).with_name('speed20_review.html')


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def case_record(case_id):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,100}',case_id):
        return None
    folder = ROOT/'cases'/case_id
    row = read_json(folder/'result.json')
    if not isinstance(row,dict) or row.get('id') != case_id:
        return None
    media = read_json(folder/'media.json')
    row['video_url'] = None
    row['media_status'] = 'pending'
    if isinstance(media,dict) and media.get('id') == case_id and media.get('render_status') == 'complete':
        video = safe_media(f'{case_id}.mp4')
        if video and str(video) == media.get('video_path'):
            row['video_url'] = '/speed20-media/'+quote(video.name)
            row['media_status'] = 'ready'
            row['video_info'] = {k:media.get(k) for k in ('fps','frames','duration_s','visual_mode','render_source','max_sample_error_s')}
    return row


def safe_media(name):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,100}\.mp4',name):
        return None
    path = (MEDIA/name).resolve()
    try:
        path.relative_to(MEDIA.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def registered_media(name):
    path = safe_media(name)
    if path is None:
        return None
    case_id = path.stem
    record = read_json(ROOT/'cases'/case_id/'result.json')
    media = read_json(ROOT/'cases'/case_id/'media.json')
    if not isinstance(record,dict) or record.get('id') != case_id or not isinstance(media,dict):
        return None
    if media.get('id') != case_id or media.get('render_status') != 'complete' or media.get('video_path') != str(path):
        return None
    return path


def catalog():
    cases = []
    analysis = read_json(ROOT/'analysis.json') or {}
    per_case = {row['id']:row for row in analysis.get('per_case', [])}
    duplicates = {case_id:group for group in analysis.get('main_duplicate_control_groups', []) for case_id in group}
    for path in sorted((ROOT/'cases').glob('*/result.json')):
        row = case_record(path.parent.name)
        if row:
            if row['id'] in per_case:
                row['climb_phase_s'] = per_case[row['id']]['climb_phase_s']
                row['same_actual_control_as'] = [x for x in duplicates.get(row['id'],[]) if x != row['id']]
            cases.append(row)
    summary = {}
    for kind in ('flat_calibration','speed_matrix','wheel_distribution','full_route'):
        group=[row for row in cases if row.get('kind') == kind]
        summary[kind]={'executed':len(group),'process_valid':sum(row.get('process_valid') is True for row in group),
                       'ascent_platform_support':sum(row.get('ascent_platform_support') is True and row.get('process_valid') is True for row in group),
                       'full_route_success':sum(row.get('full_route_success') is True for row in group),
                       'videos_ready':sum(bool(row.get('video_url')) for row in group)}
    summary['flat_calibration']['attained']=sum(row.get('attained') is True and row.get('braking_completed') is True and row.get('process_valid') is True for row in cases if row.get('kind')=='flat_calibration')
    return {'scope':{'height_cm':20,'stairs':15,'slope_deg':45},'summary':summary,'cases':cases,
            'analysis':{k:analysis.get(k) for k in ('main_evaluations','unique_main_actual_control_sequences',
                        'main_climb_phase_never_started','main_climb_phase_10ms_or_less')},
            'planned':{'flat_calibration':6,'speed_matrix':36,'wheel_distribution':12,'full_route':12,'replay':3,'half_step':3},
            'dataset_accepted':0}
