"""Build the rover MJCF (with visual meshes), a stair scene, and run the scripted controller.

Usage:
  python scripts/rover/climb.py --height 0.15 --stairs 3 --friction 1.0 --tau-max 50 --speed 6

The URDF's <visual> meshes are re-injected as collision-free geoms so the model looks
like the CAD. Terrain is N steps followed by a long top platform, so "success" means the
rover is standing on the platform, not merely somewhere past the first riser.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "assets" / "mars_rover"
URDF = ROOT / "urdf" / "mars_rover.urdf"
ROBOT_XML = ROOT / "rover_actuated.xml"

WHEEL_RADIUS = 0.163
SIM_DT = 0.005
CTRL_DT = 0.02
ROBOT_COLLIDE = 'contype="1" conaffinity="2"'


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def visual_geoms() -> dict[str, dict]:
    """link name -> mesh file, origin, colour, all taken from the URDF <visual> blocks."""
    root = ET.parse(URDF).getroot()
    out = {}
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        origin = vis.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else [0, 0, 0])]
        color = vis.find("material/color")
        rgba = color.get("rgba", "0.6 0.6 0.6 1") if color is not None else "0.6 0.6 0.6 1"
        name = Path(mesh.get("filename")).name
        slim = Path(name).with_name(Path(name).stem + "_slim.stl")
        if (ROOT / "meshes" / slim).exists():
            name = slim.name
        out[link.get("name")] = {
            "file": name,
            "xyz": xyz,
            "quat": rpy_to_quat(*rpy),
            "rgba": rgba,
        }
    return out


def build_robot() -> Path:
    if ROBOT_XML.exists():
        return ROBOT_XML

    text = URDF.read_text()
    text = text.replace("</robot>", '  <mujoco><compiler fusestatic="false"/></mujoco>\n</robot>')
    tmp = ROOT / "urdf" / "_float.urdf"
    tmp.write_text(text)
    m = mujoco.MjModel.from_xml_path(str(tmp))
    mujoco.mj_saveLastXML(str(ROBOT_XML), m)

    xml = ROBOT_XML.read_text()
    xml, n = re.subn(r'(<body name="base_link"[^>]*>)', r'\1\n      <freejoint name="root"/>', xml, count=1)
    if n != 1:
        raise RuntimeError("base_link not found")

    joints = re.findall(r'<joint name="([^"]+)"', xml)
    wheels = [j for j in joints if j.endswith("_wheel_joint")]
    steers = [j for j in joints if j.endswith("_steer_joint")]
    rockers = [j for j in joints if j.endswith("_rocker_joint")]
    if (len(wheels), len(steers), len(rockers)) != (6, 4, 2):
        raise RuntimeError(f"unexpected joints: {wheels} {steers} {rockers}")

    acts = [f'    <motor name="{j}_act" joint="{j}" gear="1" ctrlrange="-120 120"/>' for j in wheels]
    acts += [f'    <motor name="{j}_act" joint="{j}" gear="1" ctrlrange="-40 40"/>' for j in steers]
    acts += [f'    <motor name="{j}_act" joint="{j}" gear="1" ctrlrange="-100 100"/>' for j in rockers]
    xml = xml.replace("</mujoco>", "  <actuator>\n" + "\n".join(acts) + "\n  </actuator>\n</mujoco>")

    vis = visual_geoms()
    assets = []
    for link, v in vis.items():
        assets.append(f'    <mesh name="vis_{link}" file="meshes/{v["file"]}"/>')
    if "<asset>" in xml:
        xml = xml.replace("<asset>", "<asset>\n" + "\n".join(assets), 1)
    else:
        xml = xml.replace("<worldbody>", "  <asset>\n" + "\n".join(assets) + "\n  </asset>\n  <worldbody>", 1)

    for link, v in vis.items():
        pos = " ".join(f"{x:.6g}" for x in v["xyz"])
        quat = " ".join(f"{x:.6g}" for x in v["quat"])
        geom = (
            f'      <geom name="vis_{link}" type="mesh" mesh="vis_{link}" pos="{pos}" quat="{quat}" '
            f'rgba="{v["rgba"]}" contype="0" conaffinity="0" group="1" density="0"/>'
        )
        xml, k = re.subn(rf'(<body name="{re.escape(link)}"[^>]*>)', rf"\1\n{geom}", xml, count=1)
        if k != 1:
            print(f"[warn] no body for {link}")

    ROBOT_XML.write_text(xml)
    print(f"[build] {ROBOT_XML}  joints={len(joints)} actuators={len(acts)} visuals={len(vis)}")
    return ROBOT_XML


def build_scene(height: float, stairs: int, depth: float, friction: float,
                start_x: float = 1.6, top_len: float = 30.0) -> Path:
    xml = ROBOT_XML.read_text()
    xml = re.sub(r"<geom (?![^>]*contype)", f"<geom {ROBOT_COLLIDE} ", xml)

    mu = f'friction="{friction} 0.02 0.001"'
    geoms = f'    <geom name="ground" type="plane" size="60 60 0.1" contype="2" conaffinity="1" {mu} rgba=".72 .72 .72 1"/>\n'
    for i in range(stairs):
        top = height * (i + 1)
        x0 = start_x + i * depth
        geoms += (
            f'    <geom name="step{i}" type="box" pos="{x0 + depth / 2} 0 {top / 2}" '
            f'size="{depth / 2} 2.5 {top / 2}" contype="2" conaffinity="1" {mu} rgba=".55 .45 .35 1"/>\n'
        )
    top_x = start_x + stairs * depth
    top_h = height * stairs
    geoms += (
        f'    <geom name="platform" type="box" pos="{top_x + top_len / 2} 0 {top_h / 2}" '
        f'size="{top_len / 2} 2.5 {top_h / 2}" contype="2" conaffinity="1" {mu} rgba=".62 .52 .42 1"/>\n'
    )
    opt = '  <option iterations="200" solver="Newton" cone="elliptic" impratio="10"/>\n'
    if "<option" not in xml:
        xml = xml.replace("</mujoco>", opt + "</mujoco>")
    light = '    <light name="sun" pos="0 0 12" dir="0 0 -1" directional="true" diffuse=".9 .9 .85" specular=".2 .2 .2"/>\n'
    xml = xml.replace("<worldbody>\n", "<worldbody>\n" + light + geoms, 1)
    scene = ROOT / "scene_current.xml"
    scene.write_text(xml)
    return scene


def standing_height(model: mujoco.MjModel, wheel_bodies: list[int]) -> float:
    d = mujoco.MjData(model)
    lo, hi = 0.2, 1.4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d.qpos[:] = model.qpos0
        d.qpos[2] = mid
        mujoco.mj_kinematics(model, d)
        lowest = min(d.xpos[b][2] for b in wheel_bodies) - WHEEL_RADIUS
        if lowest < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_controller(model, data, speed=6.0, kp=25.0, tau_max=50.0, rocker_deg=0.0,
                    rocker_free=True, k_psi=4.0, k_dpsi=0.6, k_y=0.4):
    """Returns a closure advancing one control step, shared by run() and the renderer."""
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    wheel_j = [i for i, n in enumerate(jnames) if n.endswith("_wheel_joint")]
    steer_j = [i for i, n in enumerate(jnames) if n.endswith("_steer_joint")]
    rocker_j = [i for i, n in enumerate(jnames) if n.endswith("_rocker_joint")]

    def act_id(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_act")

    wheel_act = [act_id(jnames[j]) for j in wheel_j]
    steer_act = [act_id(jnames[j]) for j in steer_j]
    rocker_act = [act_id(jnames[j]) for j in rocker_j]
    left = [k for k, j in enumerate(wheel_j) if "left" in jnames[j]]
    rocker_target = np.deg2rad(rocker_deg)

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def step():
        qw, qx, qy, qz = data.qpos[3:7]
        yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        yaw_rate = float(data.qvel[5])
        y_ref = float(np.clip(-k_y * data.qpos[1], -0.25, 0.25))
        corr = k_psi * wrap(y_ref - yaw) + k_dpsi * yaw_rate
        for k, (j, ai) in enumerate(zip(wheel_j, wheel_act)):
            target = speed - corr if k in left else speed + corr
            err = target - data.qvel[model.jnt_dofadr[j]]
            data.ctrl[ai] = float(np.clip(kp * err, -tau_max, tau_max))
        for j, ai in zip(steer_j, steer_act):
            q = data.qpos[model.jnt_qposadr[j]]
            data.ctrl[ai] = float(np.clip(-20.0 * q - 1.0 * data.qvel[model.jnt_dofadr[j]], -40, 40))
        for j, ai in zip(rocker_j, rocker_act):
            if rocker_free:
                data.ctrl[ai] = 0.0
            else:
                q = data.qpos[model.jnt_qposadr[j]]
                data.ctrl[ai] = float(
                    np.clip(50.0 * (rocker_target - q) - 2.0 * data.qvel[model.jnt_dofadr[j]], -100, 100)
                )

    return step


def run(
    height: float,
    stairs: int,
    depth: float,
    seconds: float = 40.0,
    speed: float = 6.0,
    kp: float = 25.0,
    tau_max: float = 50.0,
    friction: float = 1.0,
    rocker_deg: float = 0.0,
    rocker_free: bool = True,
    k_psi: float = 4.0,
    k_dpsi: float = 0.6,
    k_y: float = 0.4,
) -> dict:
    build_robot()
    scene = build_scene(height, stairs, depth, friction)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    bnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    wheel_j = [i for i, n in enumerate(jnames) if n.endswith("_wheel_joint")]
    steer_j = [i for i, n in enumerate(jnames) if n.endswith("_steer_joint")]
    rocker_j = [i for i, n in enumerate(jnames) if n.endswith("_rocker_joint")]
    wheel_bodies = [i for i, n in enumerate(bnames) if n.endswith("_wheel")]

    h0 = standing_height(model, wheel_bodies)
    data.qpos[:] = model.qpos0
    data.qpos[2] = h0
    mujoco.mj_forward(model, data)

    def act_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_act")

    wheel_act = [act_id(jnames[j]) for j in wheel_j]
    steer_act = [act_id(jnames[j]) for j in steer_j]
    rocker_act = [act_id(jnames[j]) for j in rocker_j]

    rocker_target = np.deg2rad(rocker_deg)
    t, next_ctrl = 0.0, 0.0
    max_pitch, min_dist = 0.0, 0.0

    def wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    left_wheels = [k for k, j in enumerate(wheel_j) if "left" in jnames[j]]
    right_wheels = [k for k, j in enumerate(wheel_j) if "right" in jnames[j]]

    for _ in range(int(seconds / SIM_DT)):
        if t >= next_ctrl:
            qw, qx, qy, qz = data.qpos[3:7]
            yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
            yaw_rate = float(data.qvel[5])
            y_ref = float(np.clip(-k_y * data.qpos[1], -0.25, 0.25))
            corr = k_psi * wrap(y_ref - yaw) + k_dpsi * yaw_rate
            for k, (j, ai) in enumerate(zip(wheel_j, wheel_act)):
                target = speed - corr if k in left_wheels else speed + corr
                err = target - data.qvel[model.jnt_dofadr[j]]
                data.ctrl[ai] = float(np.clip(kp * err, -tau_max, tau_max))
            for j, ai in zip(steer_j, steer_act):
                q = data.qpos[model.jnt_qposadr[j]]
                data.ctrl[ai] = float(np.clip(-20.0 * q - 1.0 * data.qvel[model.jnt_dofadr[j]], -40, 40))
            for j, ai in zip(rocker_j, rocker_act):
                if rocker_free:
                    data.ctrl[ai] = 0.0
                    continue
                q = data.qpos[model.jnt_qposadr[j]]
                data.ctrl[ai] = float(
                    np.clip(50.0 * (rocker_target - q) - 2.0 * data.qvel[model.jnt_dofadr[j]], -100, 100)
                )
            next_ctrl += CTRL_DT

        mujoco.mj_step(model, data)
        t += SIM_DT
        w, xx, yy, zz = data.qpos[3:7]
        max_pitch = max(max_pitch, abs(float(np.degrees(np.arcsin(np.clip(2 * (w * yy - zz * xx), -1, 1))))))
        for i in range(data.ncon):
            min_dist = min(min_dist, float(data.contact[i].dist))

    x, z = float(data.qpos[0]), float(data.qpos[2])
    top_x = 1.6 + stairs * depth
    target_z = h0 + height * stairs
    on_top = abs(z - target_z) < 0.06 and x > top_x + 0.3
    return {
        "height": height,
        "stairs": stairs,
        "depth": depth,
        "friction": friction,
        "tau_max": tau_max,
        "speed": speed,
        "seconds": seconds,
        "x": round(x, 2),
        "y": round(float(data.qpos[1]), 2),
        "z": round(z, 3),
        "target_z": round(target_z, 3),
        "max_pitch_deg": round(max_pitch, 1),
        "max_penetration_mm": round(-min_dist * 1000, 1),
        "success": bool(on_top),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--stairs", type=int, default=3)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--kp", type=float, default=25.0)
    p.add_argument("--tau-max", type=float, default=50.0)
    p.add_argument("--friction", type=float, default=1.0)
    p.add_argument("--rocker-deg", type=float, default=0.0)
    p.add_argument("--hold-rocker", action="store_true")
    a = p.parse_args()
    print(run(a.height, a.stairs, a.depth, a.seconds, a.speed, a.kp, a.tau_max,
              a.friction, a.rocker_deg, not a.hold_rocker))
