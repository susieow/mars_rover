"""Serve only the final experiment catalog and its explicitly registered files."""
from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

PAGE = Path(__file__).with_name('index.html')
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
            visible.update(video_url=f"/media/{row['id']}.mp4",
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


def handler_for(catalog):
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
                return self.reply(302, b'', headers={'Location': '/?' + urlencode(params)})
            if path in ('/', '/index.html'):
                return self.serve_file(PAGE)
            if path == '/api/experiments':
                return self.reply(200, json.dumps(catalog.public, ensure_ascii=False, separators=(',', ':')).encode())
            asset = catalog.asset(path)
            if asset is not None and asset.is_file():
                return self.serve_file(asset)
            self.reply(404, '页面或文件未列入最终测试目录。历史视频请按 EXPERIENCE.md 的索引自行查找。'.encode(), 'text/plain; charset=utf-8')

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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.catalog or root / 'docs/final_experiments.json'
    if args.catalog is None and not path.exists():
        path = root / 'rover/docs/final_experiments.json'
    server = ThreadingHTTPServer((args.host, args.port), handler_for(Catalog(path)))
    print(f'Final experiment review: http://{args.host}:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
