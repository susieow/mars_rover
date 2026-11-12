"""Extend selected final experiments to absolute simulation time 120 seconds."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/final_observation_120s_20261112'
MANIFEST = RUN / 'manifest.json'
MEDIA = ROOT.parent / 'videos/review_high_stairs/final_observation_120s_20261112'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    if MANIFEST.exists():
        raise RuntimeError('extension manifest already exists; inspect it before resuming')
    catalog_path = ROOT / 'docs/final_experiments.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    rows = sorted((r for r in catalog['cases'] if r['highest_level'] in (2, 3)),
                  key=lambda r: (-r['highest_level'], r['speed'], r['id']))
    assert len(rows) == 17 and sum(r['highest_level'] == 3 for r in rows) == 2
    prior = [json.loads(line) for line in (ROOT / 'docs/experiment_index.jsonl').read_text().splitlines() if line]
    if any(r.get('batch') == RUN.name for r in prior):
        raise RuntimeError('this extension is already registered in the experiment index')
    (RUN / 'sources/original').mkdir(parents=True, exist_ok=False)
    (RUN / 'sources/extended').mkdir()
    (RUN / 'cases').mkdir()
    (RUN / 'logs').mkdir()
    (RUN / 'parent_catalog.json').write_bytes(catalog_path.read_bytes())
    sources = {}
    for batch in sorted({r['batch'] for r in rows}):
        source_path = ROOT / 'scripts/rover' / (batch + '.py')
        source = source_path.read_text(encoding='utf-8')
        for row in (r for r in rows if r['batch'] == batch):
            old = json.loads(Path(row['result_path']).read_text())
            assert digest(source_path) == old['source_sha256'], row['id']
            assert digest(row['result_path']) == row['result_sha256'], row['id']
            assert digest(row['trajectory_path']) == row['trajectory_sha256'], row['id']
        frozen = RUN / 'sources/original' / source_path.name
        frozen.write_bytes(source_path.read_bytes())
        extended, count = re.subn(r'^RUN = .*$', 'RUN = Path(' + repr(str(RUN)) + ')', source, count=1, flags=re.M)
        assert count == 1
        extended = extended.replace("'post_trigger_s':AFTER_TRIGGER", "'post_trigger_s':None,'observation_end_s':LIMIT,'observation_policy':'absolute_simulation_time'")
        gate = "        if step + 1 == _parent_steps:\n            _verify_parent_prefix(arrays, step + 1)"
        extended, count = re.subn(r"        if trigger is not None and d.time-trigger>=AFTER_TRIGGER-1e-9:\n            reason='post_(?:contact|stall)_observation_30s';break", gate, extended, count=1)
        assert count == 1
        compile(extended, str(frozen), 'exec')
        target = RUN / 'sources/extended' / source_path.name
        target.write_text(extended, encoding='utf-8')
        sources[batch] = {'original_path': str(source_path), 'original_sha256': digest(frozen),
                          'extended_path': str(target), 'extended_sha256': digest(target)}
    media_source = ROOT / 'scripts/rover/ground_mu6_contact_v6to15_media_20261112.py'
    frozen_media = RUN / 'sources/media.py'
    frozen_media.write_bytes(media_source.read_bytes())
    write_json(MANIFEST, {'batch': RUN.name, 'absolute_end_s': 120.0, 'planned': len(rows),
                         'parent_catalog_sha256': digest(catalog_path), 'cases': rows, 'sources': sources,
                         'media_source_sha256': digest(frozen_media), 'driver_sha256': digest(__file__),
                         'policy': 'Only replace the post-trigger stop with the existing absolute 120 s cap; preserve all controls, actuator limits and physical settings.'})
    print(json.dumps({'prepared': len(rows), 'batch': RUN.name}), flush=True)


def case_context(identifier):
    manifest = json.loads(MANIFEST.read_text())
    row = next(r for r in manifest['cases'] if r['id'] == identifier)
    info = manifest['sources'][row['batch']]
    assert digest(info['extended_path']) == info['extended_sha256']
    module = load_module(info['extended_path'], 'extension_' + row['batch'])
    return manifest, row, module


def run_case(identifier):
    import numpy as np
    manifest, row, module = case_context(identifier)
    parent_path = Path(row['trajectory_path'])
    assert digest(parent_path) == row['trajectory_sha256']
    folder = RUN / 'cases' / identifier
    with np.load(parent_path, allow_pickle=False) as parent:
        parent_steps = len(parent['ctrl'])
    module._parent_steps = parent_steps

    def verify_prefix(arrays, n):
        errors = {}
        with np.load(parent_path, allow_pickle=False) as parent:
            fields = sorted(set(arrays) & set(parent.files))
            assert {'time', 'qpos', 'qvel', 'ctrl', 'qfrc_actuator'} <= set(fields)
            for key in fields:
                expected = parent[key]
                actual = arrays[key][:n + 1 if key in ('time', 'qpos', 'qvel') else n]
                finite = np.isfinite(actual) & np.isfinite(expected)
                errors[key] = {'equal': bool(np.array_equal(actual, expected, equal_nan=True)),
                               'matching_missing_values': int(np.sum(np.isnan(actual) & np.isnan(expected))),
                               'max_abs_error': float(np.max(np.abs(actual[finite].astype(float) - expected[finite].astype(float)))) if np.any(finite) else 0.0}
        report = {'passed': all(v['equal'] for v in errors.values()), 'parent_steps': n,
                  'parent_end_s': row['end_s'], 'parent_trajectory_sha256': row['trajectory_sha256'], 'fields': errors}
        write_json(folder / 'prefix_verification.json', report)
        if not report['passed']:
            raise RuntimeError('original prefix differs; extension is not comparable')
        print(json.dumps({'id': identifier, 'prefix_verified_at_s': row['end_s']}), flush=True)

    module._verify_parent_prefix = verify_prefix
    module.main(row['speed'], row['direction'])
    path = folder / 'result.json'
    result = json.loads(path.read_text())
    assert json.loads((folder / 'prefix_verification.json').read_text())['passed']
    result['observation_extension'] = {
        'parent_key': row['key'], 'parent_result_path': row['result_path'],
        'parent_result_sha256': row['result_sha256'], 'parent_trajectory_path': row['trajectory_path'],
        'parent_trajectory_sha256': row['trajectory_sha256'], 'parent_end_s': row['end_s'],
        'parent_highest_level': row['highest_level'], 'absolute_end_s': 120.0,
        'prefix_bitwise_equal': True, 'driver_sha256': digest(__file__),
        'original_source_sha256': manifest['sources'][row['batch']]['original_sha256'],
        'rocker_cycles': 3, 'rocker_period_s': 8.0, 'rocker_returns_to_neutral_after_s': 24.0}
    dependencies = [Path(module.common.__file__), Path(module.wave.__file__),
                    Path(sys.modules['overspeed20_stairs'].__file__),
                    Path(sys.modules[module.common.EpisodeAudit.__module__].__file__),
                    Path(sys.modules[module.common.RouteAudit.__module__].__file__)]
    result['dependency_sha256'] = {str(p): digest(p) for p in dependencies}
    write_json(path, result)
    verify_case(identifier)


def verify_case(identifier):
    import numpy as np
    _, row, _ = case_context(identifier)
    folder = RUN / 'cases' / identifier
    result = json.loads((folder / 'result.json').read_text())
    assert result['id'] == identifier and result['termination_reason'] == 'simulation_time_limit_120s'
    assert digest(folder / 'trajectory.npz') == result['trajectory_sha256']
    assert json.loads((folder / 'prefix_verification.json').read_text())['passed']
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as z:
        t, q = z['time'], z['qpos']
        assert len(t) == 120001 and abs(t[-1] - 120.0) < 1e-7
        assert t[0] == 0 and np.all(np.diff(t) > 0)
        assert np.max(np.abs(np.diff(t) - .001)) < 1e-10
        for key in ('qpos', 'qvel', 'ctrl', 'qfrc_actuator', 'contact_force_n', 'wheel_targets', 'rocker_targets_rad'):
            assert np.isfinite(z[key]).all(), key
        levels = z['rear_geometry_level']
        independent_highest = 0
        for level in range(1, 16):
            mask = levels >= level
            boundaries = np.diff(np.r_[False, mask, False].astype(np.int8))
            starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
            if len(starts) and np.max(ends - starts) >= 100:
                independent_highest = level
        assert independent_highest == result['rear_highest_100ms_geometry_level']
        assert int(np.min(levels[-1000:])) == result['rear_retained_geometry_level_final_1s']
        assert independent_highest >= row['highest_level']
        tail = {}
        for seconds in (5, 10, 30):
            i = int(np.searchsorted(t, t[-1] - seconds))
            tail[str(seconds) + 's'] = {'dx_m': float(q[-1, 0] - q[i, 0]), 'dz_m': float(q[-1, 2] - q[i, 2])}
        extra_start = int(np.searchsorted(t, row['end_s']))
        extra = {'dx_m': float(q[-1, 0] - q[extra_start, 0]), 'dz_m': float(q[-1, 2] - q[extra_start, 2])}
    force = result['audit']['actuator_force_peak_by_kind_nm']
    assert all(force[k] <= limit + .001 for k, limit in [('wheel', 50), ('steer', 40), ('rocker', 100)])
    report = {'passed': True, 'id': identifier, 'end_s': result['actual_duration_s'],
              'parent_end_s': row['end_s'], 'parent_highest_level': row['highest_level'],
              'highest_level': independent_highest, 'retained_level': result['rear_retained_geometry_level_final_1s'],
              'platform': result['ascent_platform_support'], 'tail_body_displacement': tail,
              'additional_body_displacement': extra, 'actuator_force_peak_by_kind_nm': force,
              'trajectory_sha256': result['trajectory_sha256'], 'result_sha256': digest(folder / 'result.json')}
    write_json(folder / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def media_case(identifier):
    _, row, module = case_context(identifier)
    media = load_module(RUN / 'sources/media.py', 'extension_media_' + identifier)
    media.exp = SimpleNamespace(RUN=RUN, SCENE=module.SCENE, common=module.common)
    media.MEDIA = MEDIA
    media.curves(identifier)
    media.render(identifier)
    folder = RUN / 'cases' / identifier
    info = json.loads((folder / 'media.json').read_text())
    assert abs(info['last_time_s'] - 120) < 1e-7
    assert info['frames'] == 2401 and digest(info['video_path']) == info['video_sha256']
    curves = json.loads((folder / 'curves.json').read_text())
    assert abs(curves['last_time_s'] - 120) < 1e-7 and len(curves['joints']) == 12
    write_json(folder / 'media_verification.json', {'passed': True, 'id': identifier,
               'frames': info['frames'], 'last_time_s': info['last_time_s'],
               'video_sha256': info['video_sha256'], 'parent_video_path': row['video_path']})
    print(json.dumps({'id': identifier, 'media_verified': True}), flush=True)


def batch():
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest['cases']
    status = {'planned': len(rows), 'simulation_completed': [], 'media_completed': [], 'failed': {}, 'active': []}
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')

    def execute(stage, identifier):
        with (RUN / 'logs' / (identifier + '.' + stage + '.log')).open('a') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), stage, '--case', identifier],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        return identifier

    def checkpoint():
        status['updated_unix_s'] = time.time()
        write_json(RUN / 'progress.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    checkpoint()
    with ThreadPoolExecutor(max_workers=2) as media_pool, ThreadPoolExecutor(max_workers=3) as simulation_pool:
        simulation = {}
        media_jobs = {}
        for row in rows:
            identifier = row['id']
            folder = RUN / 'cases' / identifier
            if (folder / 'verification.json').exists():
                status['simulation_completed'].append(identifier)
                if (folder / 'media_verification.json').exists():
                    status['media_completed'].append(identifier)
                else:
                    media_jobs[media_pool.submit(execute, 'media', identifier)] = identifier
            else:
                simulation[simulation_pool.submit(execute, 'run', identifier)] = identifier
        for future in as_completed(simulation):
            identifier = simulation[future]
            try:
                future.result()
                status['simulation_completed'].append(identifier)
                media_jobs[media_pool.submit(execute, 'media', identifier)] = identifier
            except Exception as error:
                status['failed'][identifier] = {'stage': 'run', 'error': str(error)}
            checkpoint()
        for future in as_completed(media_jobs):
            identifier = media_jobs[future]
            try:
                future.result()
                status['media_completed'].append(identifier)
            except Exception as error:
                status['failed'][identifier] = {'stage': 'media', 'error': str(error)}
            checkpoint()
    status['complete'] = len(status['media_completed']) == len(rows) and not status['failed']
    checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run', 'media', 'verify', 'batch'))
    parser.add_argument('--case')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'scripts/rover'))
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'batch':
        batch()
    else:
        if not args.case:
            parser.error('--case is required')
        {'run': run_case, 'media': media_case, 'verify': verify_case}[args.stage](args.case)


if __name__ == '__main__':
    main()
