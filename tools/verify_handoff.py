"""Verify the handoff against saved files and HTTP responses without simulation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import http.client
import itertools
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8701')
    parser.add_argument('--hash-media', action='store_true')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'docs/final_experiments.json').read_text(encoding='utf-8'))
    rows = catalog['cases']
    assert len(rows) == len({r['id'] for r in rows}) == 112
    assert sum(r['crossing'] for r in rows) == 68
    coordinates = {(r['friction'], r['trigger'], r['speed'], r['direction']) for r in rows}
    expected = set(itertools.product((4, 6, 8), ('contact', 'stall'), range(3, 16), ('SN', 'SP')))
    missing = {(r['friction'], r['trigger'], r['speed'], r['direction']) for r in catalog['untested']}
    assert not coordinates & missing and coordinates | missing == expected and len(missing) == 44
    expected_md = {'AGENTS.md', 'README.md', 'EXPERIENCE.md', 'docs/references/supermimic_nonhumanoid_code_survey.md'}
    tracked_md = set()
    import subprocess
    for name in subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines():
        if name.lower().endswith('.md') and (ROOT / name).is_file():
            tracked_md.add(name)
    assert tracked_md == expected_md, tracked_md
    for name in expected_md:
        data = (ROOT / name).read_bytes()
        text = data.decode('utf-8')
        assert not data.startswith(b'\xef\xbb\xbf') and '\ufffd' not in text
        assert any('\u4e00' <= ch <= '\u9fff' for ch in text)
        assert not re.search(r'\?{5,}', text)
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in target and not target.startswith('#'):
                assert (ROOT / name).parent.joinpath(target.split('#')[0]).exists(), (name, target)
    reference = ROOT / 'docs/references/supermimic_nonhumanoid_code_survey.md'
    assert digest(reference) == '6673b288360370a664181a3cc7f74ad65212f06eaede5099e2277a5454f22b7d'
    original = Path('/home/agilex/wzy/docs/supermimic_nonhumanoid_code_survey.md')
    assert digest(original) == digest(reference)
    for path in [*ROOT.joinpath('scripts/rover').glob('*.py'), *ROOT.joinpath('tools').glob('*.py'), *ROOT.joinpath('web').glob('*.py')]:
        ast.parse(path.read_text(encoding='utf-8'), str(path))
    cleanup = json.loads((ROOT / 'docs/cleanup_manifest_20261112.json').read_text())
    for entry in cleanup['removed_files']:
        assert not (ROOT / entry['path']).exists(), entry['path']
    for name in cleanup['kept_active_simulation_modules']:
        path = ROOT / 'scripts/rover' / (name + '.py')
        original_bytes = subprocess.check_output(['git', '-C', str(ROOT), 'show', cleanup['base_commit'] + ':' + str(path.relative_to(ROOT))])
        assert path.read_bytes() == original_bytes, name
    for name, expected_hash in cleanup['pinned_audit_evidence'].items():
        assert digest(ROOT / name) == expected_hash, name
    index = [json.loads(line) for line in (ROOT / 'docs/experiment_index.jsonl').read_text().splitlines()]
    assert len(index) == len({r['key'] for r in index}) == 1198
    final_index = {r['id']: r for r in index if r['scope'] == 'final'}
    assert len(final_index) == 112
    for row in rows:
        result_path = Path(row['result_path'])
        result = json.loads(result_path.read_text())
        assert result['id'] == row['id'] and result['parameters'] == row['parameters']
        assert digest(result_path) == row['result_sha256']
        assert row['crossing'] == result['true_both_rear_crossing']
        assert final_index[row['id']]['local_crossing'] == row['crossing']
        for field in ('trajectory_path', 'curves_path', 'video_path'):
            assert Path(row[field]).is_file(), row[field]
        curves = json.loads(Path(row['curves_path']).read_text())
        assert curves['id'] == row['id'] and len(curves['joints']) == 12
        if args.hash_media:
            assert digest(row['video_path']) == row['video_sha256'], row['id']
            assert digest(row['trajectory_path']) == row['trajectory_sha256'], row['id']
    parsed = urlsplit(args.url)

    def request(path, method='GET', headers=None):
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=20)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        status, response_headers, body = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return status, response_headers, body

    status, _, body = request('/api/experiments')
    assert status == 200
    public = json.loads(body)
    assert {r['id'] for r in public['cases']} == {r['id'] for r in rows}
    assert len(body) < 250000, 'catalog response must exclude per-event overspeed payloads'
    for row in public['cases']:
        assert request(row['video_url'], 'HEAD')[0] == 200
        assert request(row['curves_url'], 'HEAD')[0] == 200
        assert request(row['result_url'], 'HEAD')[0] == 200
    video = public['cases'][0]['video_url']
    for spec in ('bytes=0-31', 'bytes=-32'):
        status, headers, body = request(video, headers={'Range': spec})
        assert status == 206 and len(body) == 32 and headers['Content-Range'].startswith('bytes ')
    assert request(video, headers={'Range': 'bytes=999999999999-'} )[0] == 416
    for path in ('/experiments', '/speed20', '/overspeed20', '/recovery20', '/recovery20amp20',
                 '/recovery20asym', '/recovery20friction', '/api/high-stairs',
                 '/media/T22_WMR_RSP_dt1000us.mp4', '/media/%2e%2e/AGENTS.md', '/data/nope/result.json'):
        assert request(path)[0] == 404, path
    assert request('/groundmu4contact?case=ground_mu4_contact_v7_rocker_SN')[0] == 302
    assert request('/')[0] == 200
    print(json.dumps({'passed': True, 'final_cases': 112, 'local_crossing': 68,
                      'untested': 44, 'indexed_records': 1198, 'markdown_files': sorted(expected_md),
                      'reference_unchanged': True, 'simulation_sources_unchanged': len(cleanup['kept_active_simulation_modules']),
                      'http_catalog_bytes': len(json.dumps(public, ensure_ascii=False, separators=(',', ':')).encode()),
                      'all_final_media_routes_checked': True, 'old_routes_hidden': True,
                      'media_and_trajectory_hashes_checked': args.hash_media, 'additional_physics_trials': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
