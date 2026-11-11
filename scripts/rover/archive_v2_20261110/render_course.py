"""Render one 45-degree stair course with the closed-loop controller."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from climb import build_robot
from course import build_course, simulate

W, H, FPS = 1280, 720, 50


def main(a) -> None:
    build_robot()
    if a.stair_angle is not None:
        depth = a.height / math.tan(math.radians(a.stair_angle))
    elif a.tread_from_height:
        depth = a.height
    else:
        depth = a.depth

    scene = build_course(
        a.height,
        a.stairs,
        depth,
        a.friction,
        nosing=a.nosing,
    )
    xml = scene.read_text()
    if "offwidth" not in xml:
        xml = xml.replace(
            "</mujoco>",
            f'  <visual><global offwidth="{W}" offheight="{H}"/></visual>\n</mujoco>',
        )
        scene.write_text(xml)

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            a.out,
        ],
        stdin=subprocess.PIPE,
    )
    state = {"renderer": None, "cam": None, "frames": 0}

    def on_frame(m, d, t):
        if state["renderer"] is None:
            state["renderer"] = mujoco.Renderer(m, height=H, width=W)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(m, cam)
            cam.distance, cam.azimuth, cam.elevation = 4.0, 90.0, -10.0
            state["cam"] = cam
        cam = state["cam"]
        cam.lookat[:] = [float(d.qpos[0]) + 0.8, 0.0, 0.5 + a.height]
        state["renderer"].update_scene(d, camera=cam)
        proc.stdin.write(state["renderer"].render().tobytes())
        state["frames"] += 1

    result = simulate(
        a.height,
        a.stairs,
        depth,
        seconds=a.seconds,
        speed=a.speed,
        friction=a.friction,
        tau_max=a.tau_max,
        deg=a.deg,
        assist=not a.no_assist,
        on_frame=on_frame,
        nosing=a.nosing,
        quality_pitch_limit=a.quality_pitch_limit,
        front_lift_deg=a.front_lift_deg,
        middle_lift_deg=a.middle_lift_deg,
        rocker_gain=a.rocker_gain,
        rocker_damp=a.rocker_damp,
        stop_when_done=not a.no_stop,
    )

    if proc.stdin is not None:
        proc.stdin.close()
    proc.wait()
    if state["renderer"] is not None:
        state["renderer"].close()

    result["out"] = a.out
    result["frames"] = state["frames"]
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--stairs", type=int, default=12)
    p.add_argument("--depth", type=float, default=0.15)
    p.add_argument("--seconds", type=float, default=150.0)
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--friction", type=float, default=4.0)
    p.add_argument("--tau-max", type=float, default=50.0)
    p.add_argument("--deg", type=float, default=12.0)
    p.add_argument("--tread-from-height", action="store_true")
    p.add_argument("--stair-angle", type=float, default=None)
    p.add_argument("--front-lift-deg", type=float, default=14.0)
    p.add_argument("--middle-lift-deg", type=float, default=17.0)
    p.add_argument("--quality-pitch-limit", type=float, default=38.0)
    p.add_argument("--rocker-gain", type=float, default=80.0)
    p.add_argument("--rocker-damp", type=float, default=3.0)
    p.add_argument("--nosing", type=float, default=0.0)
    p.add_argument("--no-assist", action="store_true")
    p.add_argument("--no-stop", action="store_true")
    p.add_argument("--out", default="/home/agilex/wzy/videos/course.mp4")
    main(p.parse_args())
