"""Read-only review and download page for accepted rover trajectories."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import high_stairs
import speed20_review
import overspeed20_review
import recovery20_review
import recovery20_amp20_review

BASE = Path.home() / "wzy"
VIDEOS = BASE / "videos"
RUNS = VIDEOS / "runs.jsonl"
DATASETS = BASE / "rover" / "datasets" / "45deg_v3"
DATASET_FIELDS = ("trajectory_path", "metadata_path")

INDEX = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>火星车 · 成功轨迹核验</title>
<style>
  :root { color-scheme:dark; --bg:#0f1319; --card:#171d26; --line:#26303d;
    --text:#e6edf5; --muted:#9aaabc; --ok:#66cf82; --bad:#ff817b; --accent:#88bcff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif; }
  header { padding:20px 24px; border-bottom:1px solid var(--line); }
  h1 { font-size:21px; margin:0 0 4px; }
  header p { color:var(--muted); margin:0; }
  main { padding:24px; max-width:1500px; margin:auto; }
  .notice { margin:0 0 22px; color:var(--muted); }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,410px),1fr)); gap:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
  video { width:100%; display:block; background:#000; aspect-ratio:16/9; }
  .body { padding:16px; }
  h2 { font-size:17px; margin:0 0 12px; display:flex; justify-content:space-between; gap:10px; }
  .badge { font-size:12px; padding:2px 8px; border-radius:6px; color:var(--ok); background:#1d3926; white-space:nowrap; }
  dl { display:grid; grid-template-columns:145px 1fr; gap:6px 12px; margin:0; font-size:13px; }
  dt { color:var(--muted); }
  dd { margin:0; text-align:right; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
  .ok { color:var(--ok); } .bad { color:var(--bad); } .muted { color:var(--muted); }
  .downloads { display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }
  a { color:var(--accent); }
  .downloads a { padding:6px 10px; border:1px solid var(--line); border-radius:6px; text-decoration:none; }
  .empty { padding:20px; border:1px solid var(--line); border-radius:10px; }
  .error { color:var(--bad); }
  details { margin-top:12px; color:var(--muted); font-size:12px; }
  details p { overflow-wrap:anywhere; }
</style>
</head>
<body>
<header>
  <h1>火星车 · 成功轨迹核验</h1>
  <p><a href="/recovery20amp20">查看20 cm · 摇臂 ±20° 续跑实验 →</a> · <a href="/recovery20">上一批 ±10°</a></p>
  <p>只读核验 · 仿真视频、终态与 state/action 轨迹</p>
  <p style="margin-top:10px"><a href="/experiments">查看高台阶试验审阅：15 cm / 20 cm 全部候选与早期诊断 →</a></p>
</header>
<main>
  <p id="status" class="notice" role="status">正在读取已接纳清单…</p>
  <section id="runs" class="grid" aria-label="已接纳轨迹"></section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
}[char]));
function first(...values) { return values.find((v) => v !== null && v !== undefined); }
function fmt(value, unit = '') {
  if (value === null || value === undefined) return '—';
  return esc(typeof value === 'number' ? Number(value.toFixed(3)) : value) + esc(unit);
}
function flag(value) {
  if (value === true) return '<span class="ok">通过</span>';
  if (value === false) return '<span class="bad">未通过</span>';
  return '<span class="muted">未记录</span>';
}
function row(label, value) { return `<dt>${esc(label)}</dt><dd>${value}</dd>`; }
function contacts(value) {
  if (value === null || value === undefined) return '未记录';
  if (typeof value === 'number') return `${fmt(value)} / 6`;
  if (Array.isArray(value)) {
    if (value.every((v) => typeof v === 'boolean' || typeof v === 'number')) {
      return `${value.filter((v) => v === true || v > 0).length} / ${value.length}`;
    }
    return esc(value.join('、'));
  }
  if (typeof value === 'object') {
    return Object.entries(value).map(([name, on]) => `${esc(name)}: ${on ? '有' : '无'}`).join(' · ');
  }
  return esc(value === true ? '六轮接触' : value === false ? '未满足六轮接触' : value);
}
function reasons(run) {
  const value = first(run.quality_rejection_reasons, run.rejection_reasons, run.reject_reasons, run.quality_reasons);
  if (value === undefined) return '未记录';
  if (Array.isArray(value)) return value.length ? esc(value.join('；')) : '无';
  return esc(typeof value === 'object' ? JSON.stringify(value) : value || '无');
}
function card(run) {
  const pose = run.final_pose || {};
  const pitch = first(run.final_pitch_deg, pose.pitch_deg, run.pitch_final_deg);
  const roll = first(run.final_roll_deg, pose.roll_deg, run.roll_final_deg);
  const yaw = first(run.final_yaw_deg, pose.yaw_deg, run.yaw_final_deg);
  const support = first(run.final_wheel_contacts, run.six_wheel_contacts, run.final_contacts,
    run.final_contact_wheels, run.final_ground_contacts, run.final_six_wheel_contact);
  const title = first(run.case, run.run_id, `台阶 ${run.height ?? '—'} m × ${run.stairs ?? '—'} 级`);
  const duration = first(run.duration, run.duration_s, run.simulation_duration, run.elapsed_sim_time, run.sim_time, run.time);
  const video = run.video_url ? `<video src="${esc(run.video_url)}" controls preload="metadata" playsinline></video>` : '';
  const links = [
    run.trajectory_url ? `<a href="${esc(run.trajectory_url)}" download>下载 state/action 轨迹</a>` : '<span class="muted">轨迹文件未关联</span>',
    run.metadata_url ? `<a href="${esc(run.metadata_url)}" download>下载元数据</a>` : '<span class="muted">元数据未关联</span>',
  ].join('');
  return `<article class="card">${video}<div class="body">
    <h2><span>${esc(title)}</span><span class="badge">已接纳</span></h2>
    <dl>
      ${row('控制器', fmt(first(run.controller, run.controller_version)))}
      ${row('台阶高度 / 深度', `${fmt(run.height, ' m')} / ${fmt(run.depth, ' m')}`)}
      ${row('级数 / 摩擦系数 μ', `${fmt(run.stairs)} / ${fmt(run.friction)}`)}
      ${row('执行器力矩上限', fmt(first(run.tau_max_actual, run.tau_max), ' N·m'))}
      ${row('物理仿真时长', fmt(duration, ' s'))}
      ${row('任务成功', flag(first(run.simulation_success, run.success)))}
      ${row('终态车形态', flag(run.final_shape_ok))}
      ${row('稳定停车', flag(run.settled))}
      ${row('全程物理质量', flag(first(run.physics_valid, run.quality_ok)))}
      ${row('视频审核状态', fmt(run.visual_review_status))}
      ${row('质量拒绝原因', reasons(run))}
      ${row('最终俯仰 / 侧倾 / 航向', `${fmt(pitch, '°')} / ${fmt(roll, '°')} / ${fmt(yaw, '°')}`)}
      ${row('最终轮地接触', contacts(support))}
      ${row('最大俯仰 / 侧倾', `${fmt(run.max_pitch_deg, '°')} / ${fmt(run.max_roll_deg, '°')}`)}
    </dl>
    <div class="downloads">${links}</div>
    <details><summary>清单与评价协议</summary><p>${esc(first(run.dataset, run.dataset_name, '数据集未记录'))}
      · ${esc(first(run.quality_protocol, run.quality_protocol_version, '评价协议未记录'))}</p></details>
  </div></article>`;
}
async function loadRuns() {
  try {
    const response = await fetch('/api/runs', {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const runs = await response.json();
    $('status').textContent = `当前清单共 ${runs.length} 条已接纳轨迹。各项审核状态以记录为准。`;
    $('runs').innerHTML = runs.length ? runs.map(card).join('') :
      '<p class="empty">当前没有已接纳轨迹。旧版仅标记 success 的视频不会自动进入新数据集。</p>';
  } catch (error) {
    $('status').textContent = `读取清单失败：${error.message}`;
    $('status').className = 'notice error';
  }
}
loadRuns();
</script>
</body>
</html>
"""


def safe_file(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    candidate = Path(value)
    try:
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
        target.relative_to(root.resolve())
        return target if target.is_file() else None
    except (OSError, ValueError, RuntimeError):
        return None


def file_url(root: Path, value: object, prefix: str) -> str | None:
    target = safe_file(root, value)
    if target is None:
        return None
    return prefix + quote(target.relative_to(root.resolve()).as_posix(), safe="/")


def read_runs() -> list[dict]:
    if not RUNS.is_file():
        return []
    latest: dict[str, dict] = {}
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        try:
            run = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(run, dict):
            continue
        key = run.get("run_id") or run.get("out") or run.get("trajectory_path")
        if isinstance(key, str) and key:
            latest[key] = run
    accepted = []
    for run in latest.values():
        # Legacy success alone does not establish dataset acceptance.
        if run.get("dataset_accepted", run.get("accepted")) is not True:
            continue
        item = {key: value for key, value in run.items() if key != "events"}
        item["video_url"] = file_url(VIDEOS, run.get("out"), "/videos/")
        item["trajectory_url"] = file_url(DATASETS, run.get("trajectory_path"), "/datasets/")
        item["metadata_url"] = file_url(DATASETS, run.get("metadata_path"), "/datasets/")
        accepted.append(item)
    accepted.sort(key=lambda run: (str(run.get("height", "")), str(run.get("stairs", "")), str(run.get("run_id", ""))))
    return accepted


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def _send(self, code: int, body: bytes, content_type: str = "application/json; charset=utf-8", extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return self._send(200, INDEX.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/runs":
            return self._send(200, json.dumps(read_runs(), ensure_ascii=False).encode("utf-8"), extra={"Cache-Control": "no-store"})
        if path in ("/recovery20amp20", "/recovery20amp20/"):
            return self._send(200, recovery20_amp20_review.PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/recovery20amp20":
            return self._send(200, json.dumps(recovery20_amp20_review.catalog(), ensure_ascii=False).encode("utf-8"), extra={"Cache-Control": "no-store"})
        if path.startswith("/recovery20amp20-media/"):
            target = recovery20_amp20_review.media(path[len("/recovery20amp20-media/"):])
            return self._serve_file(target) if target else self._send(404, b"not found", "text/plain")
        if path.startswith("/recovery20amp20-curves/"):
            target = recovery20_amp20_review.curves(path[len("/recovery20amp20-curves/"):])
            return self._serve_file(target) if target else self._send(404, b"not found", "text/plain")
        if path.startswith("/recovery20amp20-data/"):
            target = recovery20_amp20_review.data(path[len("/recovery20amp20-data/"):])
            return self._serve_file(target, download=True) if target else self._send(404, b"not found", "text/plain")
        if path in ("/recovery20", "/recovery20/"):
            return self._send(200, recovery20_review.PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/recovery20":
            return self._send(200, json.dumps(recovery20_review.catalog(), ensure_ascii=False).encode("utf-8"), extra={"Cache-Control": "no-store"})
        if path.startswith("/recovery20-media/"):
            target = recovery20_review.media(path[len("/recovery20-media/"):])
            return self._serve_file(target) if target else self._send(404, b"not found", "text/plain")
        if path.startswith("/recovery20-curves/"):
            target = recovery20_review.curves(path[len("/recovery20-curves/"):])
            return self._serve_file(target) if target else self._send(404, b"not found", "text/plain")
        if path.startswith("/recovery20-data/"):
            target = recovery20_review.data(path[len("/recovery20-data/"):])
            return self._serve_file(target, download=True) if target else self._send(404, b"not found", "text/plain")
        if path in ("/overspeed20", "/overspeed20/"):
            return self._send(200, overspeed20_review.PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/overspeed20":
            return self._send(200, json.dumps(overspeed20_review.catalog(), ensure_ascii=False).encode("utf-8"), extra={"Cache-Control": "no-store"})
        if path.startswith("/overspeed20-media/"):
            target = overspeed20_review.media(path[len("/overspeed20-media/"):])
            return self._serve_file(target) if target else self._send(404, b"not found", "text/plain")
        if path.startswith("/overspeed20-curves/"):
            target = overspeed20_review.curves(path[len("/overspeed20-curves/"):])
            return self._serve_file(target) if target else self._send(404, b"not found", "text/plain")
        if path in ("/speed20", "/speed20/"):
            return self._send(200, speed20_review.PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/speed20":
            return self._send(200, json.dumps(speed20_review.catalog(), ensure_ascii=False).encode("utf-8"), extra={"Cache-Control": "no-store"})
        if path.startswith("/speed20-media/"):
            target = speed20_review.registered_media(path[len("/speed20-media/"):])
            if target is None:
                return self._send(404, b"not found", "text/plain")
            return self._serve_file(target)
        if path in ("/experiments", "/experiments/"):
            return self._send(200, high_stairs.PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/experiments":
            return self._send(200, json.dumps(high_stairs.list_payload(), ensure_ascii=False).encode("utf-8"), extra={"Cache-Control": "no-store"})
        if path.startswith("/api/experiments/"):
            record = high_stairs.find_record(path[len("/api/experiments/"):])
            if record is None:
                return self._send(404, b'{"error":"unknown experiment"}')
            return self._send(200, json.dumps(record, ensure_ascii=False).encode("utf-8"), extra={"Cache-Control": "no-store"})
        if path.startswith("/review-media/"):
            target = high_stairs.safe_media(path[len("/review-media/"):])
            if target is None or target not in high_stairs.registered_media():
                return self._send(404, b"not found", "text/plain")
            return self._serve_file(target)
        for prefix, root, fields in (("/videos/", VIDEOS, ("out",)), ("/datasets/", DATASETS, DATASET_FIELDS)):
            if not path.startswith(prefix):
                continue
            target = safe_file(root, path[len(prefix):])
            registered = {safe_file(root, run.get(field)) for run in read_runs() for field in fields}
            if target is None or target not in registered:
                return self._send(404, b"not found", "text/plain")
            return self._serve_file(target, download=prefix == "/datasets/")
        return self._send(404, b"not found", "text/plain")

    def _serve_file(self, target: Path, download: bool = False) -> None:
        size = target.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match is None or not any(match.groups()) or size == 0:
                return self._send(416, b"invalid range", "text/plain", {"Content-Range": f"bytes */{size}"})
            lower, upper = match.groups()
            if lower:
                start = int(lower)
                end = min(int(upper), size - 1) if upper else size - 1
            else:
                start = max(size - int(upper), 0)
            if start >= size or start > end:
                return self._send(416, b"invalid range", "text/plain", {"Content-Range": f"bytes */{size}"})
            status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(target.name, safe=""))
        self.end_headers()
        if self.command == "HEAD":
            return
        with target.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self) -> None:
        self._send(405, b'{"error":"read-only review server"}', extra={"Allow": "GET, HEAD"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8701)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"serving accepted rover runs on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
