"""Render the rocker-assist climb and dump a PNG at the moment of deepest contact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

W, H, FPS = 1280, 720, 50
SIM_DT, CTRL_DT, EVERY = 0.005, 0.02, 4


def main(a) -> None:
    build_robot()
    scene = build_scene(a.height, a.stairs, a.depth, a.friction)
    xml = scene.read_text()
    if "offwidth" not in xml:
        xml = xml.replace("</mujoco>", f'  <visual><global offwidth="{W}" offheight="{H}"/></visual>\n</mujoco>')
        scene.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
    steer = [i for i, n in enumerate(jn) if n.endswith("_steer_joint")]
    rockers = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
    wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]
    act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")  # noqa: E731
    w_act = [act(jn[j]) for j in wheels]
    s_act = [act(jn[j]) for j in steer]
    r_act = [act(jn[j]) for j in rockers]

    d.qpos[:] = m.qpos0
    d.qpos[2] = standing_height(m, wb)
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.azimuth, cam.elevation = 3.4, 90.0, -10.0

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", a.out],
        stdin=subprocess.PIPE,
    )

    worst = (0.0, None, 0.0)
    rocker_deg, t, next_ctrl = 0.0, 0.0, 0.0
    snap_t = None

    for n in range(int(a.seconds / SIM_DT)):
        if t >= next_ctrl:
            x = float(d.qpos[0])
            wanted = 0.0
            for i in range(a.stairs):
                r = 1.6 + i * a.depth
                if r - 0.95 < x < r - 0.10:
                    wanted = -a.deg
                elif r - 0.10 <= x < r + 0.70:
                    wanted = a.deg
            rocker_deg += float(np.clip(wanted - rocker_deg, -1.8, 1.8))
            for j, ai in zip(wheels, w_act):
                d.ctrl[ai] = float(np.clip(25 * (a.speed - d.qvel[m.jnt_dofadr[j]]), -a.tau_max, a.tau_max))
            for j, ai in zip(steer, s_act):
                d.ctrl[ai] = float(np.clip(-20 * d.qpos[m.jnt_qposadr[j]] - d.qvel[m.jnt_dofadr[j]], -40, 40))
            for j, ai in zip(rockers, r_act):
                d.ctrl[ai] = float(np.clip(80 * (np.deg2rad(rocker_deg) - d.qpos[m.jnt_qposadr[j]])
                                           - 3 * d.qvel[m.jnt_dofadr[j]], -100, 100))
            next_ctrl += CTRL_DT

        mujoco.mj_step(m, d)
        t += SIM_DT

        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < worst[0]:
                worst = (float(c.dist), (i, t))

        if n % EVERY == 0:
            cam.lookat[:] = [float(d.qpos[0]) + 0.6, 0.0, 0.5 + a.height]
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            if worst[1] is not None and snap_t is None and t > worst[1][1]:
                snap_t = t
            proc.stdin.write(img.tobytes())

    proc.stdin.close()
    proc.wait()
    renderer.close()
    if snap_t is not None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(snap_t),
                        "-i", a.out, "-frames:v", "1", a.snapshot], check=False)
    print(json.dumps({"out": a.out, "snapshot": a.snapshot, "snap_t": snap_t,
                      "worst_penetration_mm": round(-worst[0] * 1000, 1),
                      "worst_at_s": round(worst[1][1], 2) if worst[1] else None,
                      "final_x": round(float(d.qpos[0]), 2),
                      "final_z": round(float(d.qpos[2]), 3)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--stairs", type=int, default=3)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--friction", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=50.0)
    p.add_argument("--deg", type=float, default=12.0)
    p.add_argument("--out", default="/home/agilex/wzy/videos/assist_015.mp4")
    p.add_argument("--snapshot", default="/home/agilex/wzy/videos/assist_015_penetration.png")
    main(p.parse_args())
