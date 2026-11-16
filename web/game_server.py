"""Keyboard-driven stair climbing game on the experiment-two model.

One MuJoCo session per player; the browser polls rendered JPEG frames and posts
key state. A climb that ends with six wheels parked on the platform is saved as
a LAFAN1-shaped episode (root_pos, root_rot, dof_pos, fps) for later training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rover"))
import speed20_experiment as common  # noqa: E402
import urdf_slope_runtime_20261113 as pb  # noqa: E402
import render_dataset  # noqa: E402

PAGE = Path(__file__).with_name("game.html")
SAVE_DIR = ROOT / "runs/rover_game"
SCENE_DIR = ROOT / "runs/rover_game/scenes"
DT = 0.001
TICK = 0.02
RENDER_EVERY = 0.1
WIDTH, HEIGHT = 480, 360
MAX_SESSIONS = 3
IDLE_EXPIRE_S = 600.0
START_DISTANCE_M = 2.0
SPEED_MAX = 4.0
ROCKER_MAX = math.radians(20.0)
SAMPLE_DT = 0.01
SAVE_FPS = 30
ROLLOVER_DEG = 60.0
LATERAL_LIMIT_M = 2.5
BOOST_EFFORT = 160.0
RISE_CHOICES_CM = (5, 10, 15, 20, 25, 30, 35)
DEFAULT_RISE_CM = 20

SESSIONS: dict[str, "Session"] = {}
LOCK = threading.RLock()


def orientation(d):
    w, x, y, z = map(float, d.qpos[3:7])
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(np.clip(2 * (w * y - z * x), -1, 1)),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rise_cm_of(body):
    value = body.get("rise_cm", DEFAULT_RISE_CM)
    try:
        rise_cm = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return rise_cm if rise_cm in RISE_CHOICES_CM else None


class Session:
    def __init__(self, angle, friction, name, mode="stock", rise_cm=DEFAULT_RISE_CM):
        self.id = uuid.uuid4().hex[:10]
        self.name = name
        self.angle = angle
        self.friction = friction
        self.mode = "boost" if mode == "boost" else "stock"
        self.created = time.time()
        self.touched = time.time()
        # The player picks the riser height; the tread follows from the angle so
        # the staircase keeps the requested slope.
        self.rise_cm = int(rise_cm)
        self.rise = self.rise_cm / 100.0
        self.tread = self.rise / math.tan(math.radians(angle))
        # Every riser height builds the same 3 m staircase, so a low riser means
        # many small steps instead of a shorter climb.
        self.steps = pb.steps_for(self.rise)
        self.timeout = pb.budget(angle, self.rise, self.steps)
        case = {"angle_deg": angle, "tread_m": self.tread, "rise_m": self.rise,
                "friction": friction, "steps": self.steps, "timeout_s": self.timeout}
        SCENE_DIR.mkdir(parents=True, exist_ok=True)
        scene = SCENE_DIR / f"{self.id}.xml"
        scene.write_text(pb.scene_text(case), encoding="utf-8")
        self.scene_path = scene
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = DT
        if self.mode == "boost":
            for j in range(self.model.njnt):
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("rocker_joint"):
                    self.model.jnt_actfrcrange[j] = (-BOOST_EFFORT, BOOST_EFFORT)
            for a in range(self.model.nu):
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").endswith("rocker_joint_act"):
                    self.model.actuator_ctrlrange[a] = (-BOOST_EFFORT, BOOST_EFFORT)
        self.model.vis.global_.offwidth = WIDTH
        self.model.vis.global_.offheight = HEIGHT
        self.data = common.initialize(self.model, START_DISTANCE_M)
        self.ctl = common.Controller(self.model, self.data)
        self.render_model, self.option, _ = render_dataset._visual_render_model(
            scene, SCENE_DIR,
            {"audit": {"velocity_limit_source": {"path": str(pb.URDF)}},
             "urdf_sha256": hashlib.sha256(pb.URDF.read_bytes()).hexdigest()})
        self.render_model.vis.global_.offwidth = WIDTH
        self.render_model.vis.global_.offheight = HEIGHT
        self.render_data = mujoco.MjData(self.render_model)
        self.renderer = mujoco.Renderer(self.render_model, height=HEIGHT, width=WIDTH)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.render_model, self.camera)
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 4.2, 90.0, -14.0
        self.platform_ids = {g for g in range(self.model.ngeom)
                             if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == "platform"}
        self.platform_start = pb.RISER_X + self.steps * self.tread
        self.platform_end = self.platform_start + pb.PLATFORM_LEN
        self.platform_top = self.rise * self.steps
        self.throttle = 0.0
        self.rocker = 0.0
        self.rocker_cmd = 0.0
        self.auto_brake = False
        self.brake = False
        self.status = "running"
        self.message = ""
        self.saved = None
        self.frame = None
        self.last_render = 0.0
        self.support_since = None
        self.settle_since = None
        self.times = []
        self.states = []
        self.last_sample = -1.0

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass
        try:
            self.scene_path.unlink(missing_ok=True)
        except OSError:
            pass

    def input(self, throttle, rocker, brake):
        self.touched = time.time()
        if self.status != "running":
            return
        self.throttle = float(np.clip(throttle, -1.0, 1.0))
        self.rocker = float(np.clip(rocker, -1.0, 1.0))
        self.brake = bool(brake)

    def advance(self, seconds):
        steps = int(round(seconds / DT))
        for _ in range(steps):
            if self.status != "running":
                return
            speed = 0.0 if (self.brake or self.auto_brake) else self.throttle * SPEED_MAX
            self.ctl.update(np.full(3, speed))
            target = self.rocker * ROCKER_MAX
            self.rocker_cmd += float(np.clip(target - self.rocker_cmd,
                                             -math.radians(20.0) * DT, math.radians(20.0) * DT))
            rocker_target = self.rocker_cmd
            # Stock mode leaves the velocity-form loop's torque at the gravity
            # load, so the joint stops short of the command. Boost raises the
            # effort cap and pays that load with a feedforward instead.
            gain, cap, feedforward_gain, effort = (
                (12.0, 1.57, 78.0, BOOST_EFFORT) if self.mode == "boost"
                else (4.0, 0.7, 0.0, 100.0))
            for _, actuator, qpos, dof in self.ctl.rockers:
                error = rocker_target - float(self.data.qpos[qpos])
                desired = float(np.clip(gain * error, -cap, cap))
                feedforward = feedforward_gain * float(np.clip(error / 0.05, 0.0, 1.0))
                self.data.ctrl[actuator] = np.clip(
                    150.0 * (desired - float(self.data.qvel[dof])) + feedforward, -effort, effort)
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                self.fail("仿真数值异常")
                return
            if self.data.time - self.last_sample >= SAMPLE_DT - 1e-9:
                self.times.append(float(self.data.time))
                self.states.append(self.data.qpos.copy())
                self.last_sample = float(self.data.time)
            self.judge()

    def judge(self):
        d = self.data
        if abs(float(d.qpos[1])) > LATERAL_LIMIT_M:
            return self.fail("偏出车道")
        roll, _, _ = orientation(d)
        if abs(math.degrees(roll)) > ROLLOVER_DEG:
            return self.fail("翻侧")
        if float(d.time) > self.timeout:
            return self.fail("超时")
        forces = pb.wheel_forces(self.model, d, self.ctl, self.platform_ids)
        inside, loaded, gap = pb.platform_state(
            self.model, d, self.ctl, forces, self.platform_start, self.platform_end, self.platform_top)
        support = pb.platform_support(inside, loaded, gap)
        if support:
            self.auto_brake = True
        now = float(d.time)
        if support:
            if self.support_since is None:
                self.support_since = now
        else:
            self.support_since = None
        settled = support and pb._settled(self.model, d, self.ctl)
        if not settled:
            self.settle_since = None
        elif self.settle_since is None:
            self.settle_since = now
        if settled and self.settle_since is not None and now - self.settle_since >= 1.0:
            self.status = "success"
            self.message = "六轮停在平台，已保存"
            self.save()
        elif support and self.support_since is not None and now - self.support_since >= 1.0:
            self.message = "已上平台，正在停稳"
        elif float(d.qpos[0]) > self.platform_end + 1.5:
            self.fail("冲出平台")

    def fail(self, reason):
        self.status = "failed"
        self.message = reason

    def steps_climbed(self):
        return int(np.clip((float(self.data.qpos[0]) - pb.RISER_X) / self.tread, 0, self.steps))

    def render(self):
        self.render_data.qpos[:] = self.data.qpos
        self.render_data.qvel[:] = 0.0
        mujoco.mj_forward(self.render_model, self.render_data)
        self.camera.lookat[:] = self.data.qpos[:3] + np.array([1.2, 0.0, 0.2])
        self.renderer.update_scene(self.render_data, camera=self.camera, scene_option=self.option)
        rgb = self.renderer.render()
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG", quality=72)
        self.frame = buffer.getvalue()
        self.last_render = time.time()

    def save(self):
        if self.saved or not self.states:
            return
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        times = np.asarray(self.times)
        poses = np.asarray(self.states)
        index = np.searchsorted(times, np.arange(times[0], times[-1], 1.0 / SAVE_FPS))
        index = np.clip(index, 0, len(times) - 1)
        poses = poses[index]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"game_{self.angle:g}deg_{self.rise_cm}cm_{stamp}_{self.id}"
        np.savez(
            SAVE_DIR / f"{stem}.npz",
            root_pos=poses[:, :3].astype(np.float64),
            root_rot=poses[:, [4, 5, 6, 3]].astype(np.float64),
            dof_pos=poses[:, 7:].astype(np.float64),
            fps=np.float64(SAVE_FPS),
        )
        meta = {
            "id": self.id, "name": self.name, "angle_deg": self.angle, "tread_m": self.tread,
            "rise_cm": self.rise_cm, "steps_total": self.steps,
            "platform_top_m": round(self.platform_top, 3),
            "friction": self.friction, "duration_s": float(times[-1]), "fps": SAVE_FPS,
            "frames": int(len(poses)), "quaternion_order": "xyzw",
            "joint_names": [self.model.joint(j).name for j in range(self.model.njnt)
                            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE],
            "status": self.status, "saved_at": stamp,
            "rocker_mode": self.mode,
            "rocker_effort_nm": BOOST_EFFORT if self.mode == "boost" else 100.0,
        }
        (SAVE_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        self.saved = stem
        self.message = f"已保存 {stem}"

    def state(self):
        return {
            "id": self.id, "angle_deg": self.angle, "friction": self.friction,
            "rise_cm": self.rise_cm, "steps_total": self.steps,
            "t": round(float(self.data.time), 2), "x": round(float(self.data.qpos[0]), 2),
            "speed": round(self.throttle * SPEED_MAX, 2),
            "rocker_deg": round(math.degrees(float(self.data.qpos[self.ctl.rockers[0][2]])), 1),
            "mode": self.mode,
            "steps": self.steps_climbed(), "status": self.status, "message": self.message,
            "saved": self.saved, "timeout_s": self.timeout,
        }


def loop(stop):
    while not stop.is_set():
        started = time.time()
        with LOCK:
            expired = [sid for sid, s in SESSIONS.items() if time.time() - s.touched > IDLE_EXPIRE_S]
            for sid in expired:
                SESSIONS.pop(sid).close()
            for session in list(SESSIONS.values()):
                try:
                    session.advance(TICK)
                    if time.time() - session.last_render >= RENDER_EVERY:
                        session.render()
                except Exception as error:
                    session.status = "failed"
                    session.message = f"仿真错误 {error}"
        stop.wait(max(0.0, TICK - (time.time() - started)))


class Handler(BaseHTTPRequestHandler):
    server_version = "RoverGame/1.0"

    def reply(self, status, body, kind="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def payload(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def route(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/join":
            body = self.payload()
            angle = float(body.get("angle", 25))
            friction = int(body.get("friction", 6))
            rise_cm = rise_cm_of(body)
            if angle < 10 or angle > 50:
                return self.reply(400, b'{"error":"angle out of range"}')
            if rise_cm is None:
                return self.reply(400, json.dumps(
                    {"error": "每级高度只能选 5/10/15/20/25/30/35 cm"}).encode())
            with LOCK:
                active = [s for s in SESSIONS.values() if s.status == "running"]
                if len(active) >= MAX_SESSIONS:
                    return self.reply(429, json.dumps({"error": "上楼梯的人已有 3 位，稍等再开"}).encode())
                session = Session(angle, friction, str(body.get("name", ""))[:20],
                                  body.get("mode", "stock"), rise_cm)
                SESSIONS[session.id] = session
            return self.reply(200, json.dumps({"id": session.id, "angle_deg": angle,
                                               "friction": friction, "mode": session.mode,
                                               "rise_cm": session.rise_cm}).encode())
        if path == "/api/state":
            sid = self.query().get("id", "")
            with LOCK:
                session = SESSIONS.get(sid)
                if session is None:
                    return self.reply(404, b'{"error":"session gone"}')
                session.touched = time.time()
                return self.reply(200, json.dumps(session.state(), ensure_ascii=False).encode())
        if path == "/api/frame":
            sid = self.query().get("id", "")
            with LOCK:
                session = SESSIONS.get(sid)
                if session is None or session.frame is None:
                    return self.reply(404, b"no frame")
                frame = session.frame
            return self.reply(200, frame, "image/jpeg")
        if path == "/api/input":
            body = self.payload()
            with LOCK:
                session = SESSIONS.get(body.get("id", ""))
                if session is None:
                    return self.reply(404, b'{"error":"session gone"}')
                session.input(body.get("throttle", 0), body.get("rocker", 0), body.get("brake", False))
            return self.reply(200, b'{"ok":true}')
        if path == "/api/leave":
            body = self.payload()
            with LOCK:
                old = SESSIONS.pop(body.get("id", ""), None)
                if old is not None:
                    old.close()
            return self.reply(200, b'{"ok":true}')
        if path == "/api/reset":
            body = self.payload()
            with LOCK:
                old = SESSIONS.pop(body.get("id", ""), None)
                if old is not None:
                    old.close()
                session = Session(float(body.get("angle", 25)), int(body.get("friction", 6)),
                                  str(body.get("name", ""))[:20], body.get("mode", "stock"),
                                  rise_cm_of(body) or DEFAULT_RISE_CM)
                SESSIONS[session.id] = session
            return self.reply(200, json.dumps({"id": session.id, "mode": session.mode,
                                               "rise_cm": session.rise_cm}).encode())
        if path == "/api/episodes":
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            rows = sorted(SAVE_DIR.glob("game_*.json"), reverse=True)[:50]
            return self.reply(200, json.dumps([json.loads(p.read_text(encoding="utf-8")) for p in rows],
                                              ensure_ascii=False).encode())
        self.reply(404, b"not found", "text/plain")

    def query(self):
        from urllib.parse import parse_qs, urlsplit
        return {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8702)
    args = parser.parse_args()
    stop = threading.Event()
    thread = threading.Thread(target=loop, args=(stop,), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Rover game: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
