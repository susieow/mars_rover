"""Render a scripted climb to mp4 (offscreen, EGL backend).

Usage:
  MUJOCO_GL=egl python scripts/rover/render.py --height 0.10 --stairs 3 --out climb.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from climb import build_robot, build_scene, make_controller, standing_height

W, H, FPS = 1280, 720, 50
SIM_DT = 0.005
CTRL_DT = 0.02
RENDER_EVERY = 4


def main(args) -> None:
    build_robot()
    scene = build_scene(args.height, args.stairs, args.depth, args.friction)
    xml = scene.read_text()
    if "offwidth" not in xml:
        xml = xml.replace("</mujoco>", f'  <visual><global offwidth="{W}" offheight="{H}"/></visual>\n</mujoco>')
        scene.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]
    h0 = standing_height(m, wb)
    d.qpos[:] = m.qpos0
    d.qpos[2] = h0
    mujoco.mj_forward(m, d)

    step = make_controller(m, d, speed=args.speed, tau_max=args.tau_max,
                           rocker_deg=args.rocker_deg, rocker_free=not args.hold_rocker)

    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance = 3.2
    cam.azimuth = 90.0
    cam.elevation = -12.0

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", args.out],
        stdin=subprocess.PIPE,
    )

    t, next_ctrl, frame_i = 0.0, 0.0, 0
    done_at = None
    max_pitch, min_dist = 0.0, 0.0
    top_x = 1.6 + args.stairs * args.depth
    target_z = h0 + args.height * args.stairs

    for n in range(int(args.seconds / SIM_DT)):
        if t >= next_ctrl:
            step()
            next_ctrl += CTRL_DT
        mujoco.mj_step(m, d)
        t += SIM_DT

        w, xx, yy, zz = d.qpos[3:7]
        max_pitch = max(max_pitch, abs(float(np.degrees(np.arcsin(np.clip(2 * (w * yy - zz * xx), -1, 1))))))
        for i in range(d.ncon):
            min_dist = min(min_dist, float(d.contact[i].dist))

        if n % RENDER_EVERY == 0:
            cam.lookat[:] = [float(d.qpos[0]) + 0.6, 0.0, 0.45 + args.height * args.stairs * 0.5]
            renderer.update_scene(d, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
            frame_i += 1

        if done_at is None and float(d.qpos[0]) > top_x + 1.0:
            done_at = round(t, 2)

    proc.stdin.close()
    proc.wait()
    renderer.close()

    x, z = float(d.qpos[0]), float(d.qpos[2])
    on_top = abs(z - target_z) < 0.06 and x > top_x + 0.3
    print(json.dumps({
        "out": args.out,
        "height": args.height,
        "stairs": args.stairs,
        "depth": args.depth,
        "friction": args.friction,
        "tau_max": args.tau_max,
        "speed": args.speed,
        "seconds": args.seconds,
        "frames": frame_i,
        "final_x": round(x, 2),
        "final_y": round(float(d.qpos[1]), 2),
        "final_z": round(z, 3),
        "target_z": round(target_z, 3),
        "max_pitch_deg": round(max_pitch, 1),
        "max_penetration_mm": round(-min_dist * 1000, 1),
        "crossed_at_s": done_at,
        "success": bool(on_top),
    }))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.10)
    p.add_argument("--stairs", type=int, default=3)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--tau-max", type=float, default=50.0)
    p.add_argument("--friction", type=float, default=1.0)
    p.add_argument("--rocker-deg", type=float, default=0.0)
    p.add_argument("--hold-rocker", action="store_true")
    p.add_argument("--out", type=str, default="climb.mp4")
    main(p.parse_args())
