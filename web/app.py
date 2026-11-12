"""Serve experiment entry pages and explicitly registered review files."""
from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

PAGE = Path(__file__).with_name('index.html')
STAIRS_PAGE = Path(__file__).with_name('stairs45.html')
SLOPE_PAGE = Path(__file__).with_name('urdf_slope.html')
STAIRS_ROUTE = '/experiments/stairs-45'
FINAL_ALIASES = {'/groundmu4contact', '/groundmu6contact', '/groundmu8contact',
                 '/groundmu6', '/groundmu8', '/groundmu6high', '/groundmu8high'}


class Catalog:
    def __init__(self, path):
        self.document = json.loads(Path(path).read_text(encoding='utf-8'))
        self.rows = {r['id']: r for r in self.document['cases']}
        if len(self.rows) != self.document['completed'] or any(r.get('dataset_accepted') for r in self.rows.values()):
            raise ValueError('invalid final experiment catalog')
        self.public = {k: v for k, v in self.document.items() if k != 'cases'}
        self.public['cases'] = []
        for row in self.rows.values():
            visible = {k: v for k, v in row.items()
                       if not k.endswith(('_path', '_sha256')) and k != 'overspeed'}
            version = '?v=' + row['video_sha256'][:12] if row.get('extended_to_120s') else ''
            visible.update(video_url=f"/media/{row['id']}.mp4" + version,
                           curves_url=f"/curves/{row['id']}.json",
                           result_url=f"/data/{row['id']}/result.json",
                           trajectory_url=f"/data/{row['id']}/trajectory.npz")
            self.public['cases'].append(visible)

    def asset(self, path):
        for prefix, suffix, field in (('/media/', '.mp4', 'video_path'),
                                      ('/curves/', '.json', 'curves_path')):
            if path.startswith(prefix) and path.endswith(suffix):
                row = self.rows.get(path[len(prefix):-len(suffix)])
                return Path(row[field]) if row else None
        if path.startswith('/data/'):
            parts = path[len('/data/'):].split('/')
            if len(parts) == 2 and parts[0] in self.rows:
                field = {'result.json': 'result_path', 'trajectory.npz': 'trajectory_path'}.get(parts[1])
                if field:
                    return Path(self.rows[parts[0]][field])
        return None


class ReviewCatalog:
    def __init__(self, path):
        path = Path(path) if path else None
        if path is not None and path.is_file():
            self.document = json.loads(path.read_text(encoding='utf-8'))
        else:
            self.document = {'completed': 0, 'cases': []}
        self.rows = {row['id']: row for row in self.document.get('cases', [])}
        self.public = {key: value for key, value in self.document.items() if key != 'cases'}
        self.public['cases'] = []
        for row in self.rows.values():
            visible = {
                key: value for key, value in row.items()
                if not key.endswith(('_path', '_sha256'))
            }
            visible.update(
                video_url=f"/slope-media/{row['id']}.mp4?v={row['video_sha256'][:12]}",
                result_url=f"/slope-data/{row['id']}/result.json",
                trajectory_url=f"/slope-data/{row['id']}/trajectory.npz",
            )
            self.public['cases'].append(visible)

    def asset(self, path):
        if path.startswith('/slope-media/') and path.endswith('.mp4'):
            row = self.rows.get(path[len('/slope-media/'):-len('.mp4')])
            return Path(row['video_path']) if row else None
        if path.startswith('/slope-data/'):
            parts = path[len('/slope-data/'):].split('/')
            if len(parts) == 2 and parts[0] in self.rows:
                field = {
                    'result.json': 'result_path',
                    'trajectory.npz': 'trajectory_path',
                }.get(parts[1])
                return Path(self.rows[parts[0]][field]) if field else None
        return None


def handler_for(catalog, review):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, kind='application/json; charset=utf-8', headers=None):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                self.route()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def route(self):
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path in FINAL_ALIASES:
                query = parse_qs(parsed.query)
                params = {'case': query['case'][0]} if query.get('case') else {}
                mu = re.search(r'mu([468])', path)
                if mu:
                    params['mu'] = mu.group(1)
                params['trigger'] = 'contact' if path.endswith('contact') else 'stall'
                return self.reply(302, b'', headers={'Location': STAIRS_ROUTE + '?' + urlencode(params)})
            if path in ('/', '/index.html'):
                query = parse_qs(parsed.query)
                if set(query) & {'case', 'mu', 'trigger', 'speed', 'direction', 'level', 'crossed'}:
                    return self.reply(302, b'', headers={'Location': STAIRS_ROUTE + '?' + parsed.query})
                return self.serve_file(PAGE)
            if path == STAIRS_ROUTE:
                return self.serve_file(STAIRS_PAGE)
            if path == '/experiments/urdf-slope':
                return self.serve_file(SLOPE_PAGE)
            if path == '/api/experiments':
                return self.reply(200, json.dumps(catalog.public, ensure_ascii=False, separators=(',', ':')).encode())
            if path == '/api/urdf-slope':
                return self.reply(200, json.dumps(review.public, ensure_ascii=False, separators=(',', ':')).encode())
            asset = catalog.asset(path)
            if asset is None:
                asset = review.asset(path)
            if asset is not None and asset.is_file():
                return self.serve_file(asset)
            self.reply(404, '页面或文件未登记在实验目录。历史视频请按 EXPERIENCE.md 的索引自行查找。'.encode(), 'text/plain; charset=utf-8')

        def serve_file(self, path):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            request_range = self.headers.get('Range')
            if request_range:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', request_range)
                if match is None or not any(match.groups()):
                    return self.reply(416, b'invalid range', 'text/plain', {'Content-Range': f'bytes */{size}'})
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), size - 1) if right else size - 1
                else:
                    start = max(0, size - int(right))
                if start >= size or start > end:
                    return self.reply(416, b'invalid range', 'text/plain', {'Content-Range': f'bytes */{size}'})
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store' if path.suffix != '.mp4' else 'private, max-age=3600')
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = stream.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('127.0.0.1',), default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8701)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--review-catalog', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.catalog or root / 'docs/final_experiments.json'
    if args.catalog is None and not path.exists():
        path = root / 'rover/docs/final_experiments.json'
    review_path = args.review_catalog or root / 'docs/urdf_slope_review.json'
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_for(Catalog(path), ReviewCatalog(review_path))
    )
    print(f'Rover experiments: http://{args.host}:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
