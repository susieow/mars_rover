"""Symmetric stair course: N steps up, a platform, N steps down.

Rocker assist is scheduled by base position for both directions. Success means the
rover reached platform height and came back down to ground level on the far side.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np

from climb import ROBOT_XML, ROOT, build_robot, standing_height

SIM_DT = 0.005
CTRL_DT = 0.02


def build_course(height: float, stairs: int, depth: float, friction: float,
                 start_x: float = 1.6, top_len: float = 2.0, nosing: float = 0.0,
                 wheel_radius: float = 0.163, middle_collision: bool = True) -> Path:
    xml = ROBOT_XML.read_text()
    xml = re.sub(r"<geom (?![^>]*contype)", '<geom contype="1" conaffinity="2" ', xml)
    if not middle_collision:
        def _off(m):
            tag = m.group(0)
            tag = re.sub(r'contype="[^"]*"', 'contype="0"', tag)
            tag = re.sub(r'conaffinity="[^"]*"', 'conaffinity="0"', tag)
            return tag
        xml = re.sub(r"<geom[^>]*middle_wheel_col[^>]*>", _off, xml)
    if abs(wheel_radius - 0.163) > 1e-9:
        def _scale(m):
            return f'{m.group(1)}size="{wheel_radius:.6f} {m.group(2)}"'
        xml = re.sub(r'(name="[a-z_]+_wheel_col_\d+"[^>]*?)size="[0-9.]+ ([0-9.]+)"', _scale, xml)
    mu = f'friction="{friction} 0.02 0.001"'
    plat_top_c = height * stairs

    def shade(top: float) -> str:
        return 'rgba=".58 .47 .36 1"'

    geoms = (f'    <geom name="ground" type="plane" size="80 80 0.1" contype="2" conaffinity="1" '
             f'{mu} rgba=".72 .72 .72 1"/>\n')
    x = start_x
    for i in range(stairs):
        top = height * (i + 1)
        geoms += (f'    <geom name="up{i}" type="box" pos="{x + depth / 2} 0 {top / 2}" '
                  f'size="{depth / 2} 2.5 {top / 2}" contype="2" conaffinity="1" {mu} {shade(top)}/>\n')
        x += depth
    plat_top = height * stairs
    geoms += (f'    <geom name="platform" type="box" pos="{x + top_len / 2} 0 {plat_top / 2}" '
              f'size="{top_len / 2} 2.5 {plat_top / 2}" contype="2" conaffinity="1" {mu} {shade(plat_top)}/>\n')
    x += top_len
    for i in range(stairs - 1):
        top = height * (stairs - 1 - i)
        geoms += (f'    <geom name="down{i}" type="box" pos="{x + depth / 2} 0 {top / 2}" '
                  f'size="{depth / 2} 2.5 {top / 2}" contype="2" conaffinity="1" {mu} {shade(top)}/>\n')
        x += depth
    if nosing > 0:
        for i in range(stairs):
            geoms += _chamfer_geom(start_x + i * depth, height * (i + 1), nosing, mu)
        plat_end = start_x + stairs * depth
        geoms += _chamfer_geom(plat_end, height * stairs, nosing, mu)
        xd = plat_end + top_len
        for i in range(stairs - 1):
            geoms += _chamfer_geom(xd + i * depth, height * (stairs - 1 - i), nosing, mu)
    light = '    <light name="sun" pos="0 0 14" dir="0 0 -1" directional="true" diffuse=".9 .9 .85"/>\n'
    xml = xml.replace("<worldbody>\n", "<worldbody>\n" + light + geoms, 1)
    if "offwidth" not in xml:
        xml = xml.replace("</mujoco>", '  <visual><global offwidth="1280" offheight="720"/></visual>\n</mujoco>')
    scene = ROOT / "scene_course.xml"
    scene.write_text(xml)
    return scene


def _nosing_geom(x_edge: float, top: float, radius: float, mu: str = '1.4 0.02 0.001') -> str:
    return (f'    <geom name="nose{abs(int(x_edge * 100))}_{int(top * 1000)}" type="cylinder" '
            f'pos="{x_edge} 0 {top - radius}" size="{radius} 2.5" '
            f'quat="0.707107 0.707107 0 0" contype="2" conaffinity="1" '
            f'{mu} rgba=".58 .47 .36 1"/>\n')


def _chamfer_geom(x_edge: float, top: float, c: float, mu: str = '1.4 0.02 0.001') -> str:
    """45-degree chamfer at a riser corner: a thin plate from (x_edge-c, top-c) to (x_edge, top)."""
    half = c * 0.70710678
    # box centred on the chamfer surface, rotated -45 deg about y
    cx = x_edge - c / 2
    cz = top - c / 2
    return (f'    <geom name="cham{abs(int(x_edge * 100))}_{int(top * 1000)}" type="box" '
            f'pos="{cx} 0 {cz}" size="{half} 2.5 0.004" '
            f'quat="0.92388 0 0.38268 0" contype="2" conaffinity="1" '
            f'{mu} rgba=".58 .47 .36 1"/>\n')


def edges(height: float, stairs: int, depth: float, start_x: float = 1.6, top_len: float = 2.0):
    """Riser x-positions, ascending then descending."""
    up = [start_x + i * depth for i in range(stairs)]
    plat_end = start_x + stairs * depth + top_len
    down = [plat_end + i * depth for i in range(stairs - 1)]
    return up + down, plat_end + (stairs - 1) * depth


def simulate(height: float, stairs: int, depth: float, seconds: float = 90.0, speed: float = 6.0,
             friction: float = 1.4, tau_max: float = 50.0, deg: float = 12.0,
             front_lead: float = 0.95, mid_lead: float = 0.10, mid_trail: float = 0.70,
             slew: float = 90.0, assist: bool = True, on_frame=None, on_step=None,
             pitch_limit: float = 12.0, pitch_fade: float = 20.0, const_deg=None,
             rocker_free: bool = False,
             init_yaw_deg: float = 0.0, init_x: float = 0.0, stop_when_done: bool = True,
             mid_deg: float | None = None, mode: str = "roll", nosing: float = 0.0,
             wheel_radius: float = 0.163, middle_collision: bool = True,
             lift_deg: float = 35.0, lift_lead: float = 0.90, lift_hold: float = 0.20,
             land_trail: float = 0.70, rocker_gain: float = 80.0,
             rocker_damp: float = 3.0, alt_amp: float = 0.0,
             alt_freq: float = 1.0, alt_bias: float = 0.0,
             heading_hold: bool = True, alt_front_scale: float = 1.0,
             recovery: bool = False, stall_dx: float = 0.01,
             stall_time: float = 1.0, recovery_time: float = 2.0,
             recovery_freq: float = 1.5, recovery_amp: float = 20.0,
             dwell_amp: float = 0.0, dwell_hold: float = 1.5) -> dict:
    build_robot()
    scene = build_course(height, stairs, depth, friction, nosing=nosing,
                          wheel_radius=wheel_radius, middle_collision=middle_collision)
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
    left = [k for k, j in enumerate(wheels) if "left" in jn[j]]
    rear_k = [k for k, j in enumerate(wheels) if "rear" in jn[j]]
    mid_k = [k for k, j in enumerate(wheels) if "middle" in jn[j]]
    front_k = [k for k, j in enumerate(wheels) if "front" in jn[j]]

    h0 = standing_height(m, wb) + (wheel_radius - 0.163)
    d.qpos[:] = m.qpos0
    d.qpos[2] = h0
    d.qpos[0] = init_x
    yaw0 = np.deg2rad(init_yaw_deg)
    d.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
    mujoco.mj_forward(m, d)

    risers, course_end = edges(height, stairs, depth)
    plat_top = height * stairs

    def wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def schedule(x: float) -> float:
        if const_deg is not None:
            return float(const_deg)
        if mode == "lift":
            for r in risers:
                if r - lift_lead < x < r + lift_hold:
                    return -lift_deg
                if r + lift_hold <= x < r + land_trail:
                    return 0.0
            return 0.0
        if not assist:
            return 0.0
        for r in risers:
            if r - front_lead < x < r - mid_lead:
                return -deg
            if r - mid_lead <= x < r + mid_trail:
                return deg if mid_deg is None else mid_deg
        return 0.0

    rocker_deg, t, next_ctrl = 0.0, 0.0, 0.0
    max_z, max_pitch = -1.0, 0.0
    stopped_at = None
    x_ref, t_ref = 0.0, 0.0
    recovery_until, recovery_start = -1.0, 0.0
    recovery_sign, recovery_count = 1.0, 0

    for n in range(int(seconds / SIM_DT)):
        if t >= next_ctrl:
            x = float(d.qpos[0])
            qw, qx, qy, qz = d.qpos[3:7]
            yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
            yaw_rate = float(d.qvel[5])
            y_ref = yaw0 + float(np.clip(-0.4 * d.qpos[1], -0.25, 0.25))
            if recovery and t > 2.0:
                if abs(x - x_ref) >= stall_dx:
                    x_ref, t_ref = x, t
                elif t - t_ref >= stall_time and t >= recovery_until:
                    recovery_start = t
                    recovery_until = t + recovery_time
                    recovery_sign = -recovery_sign
                    recovery_count += 1
                    x_ref, t_ref = x, t
            recovering = recovery and t < recovery_until
            corr = 4.0 * wrap(y_ref - yaw) + 0.6 * yaw_rate if heading_hold and not recovering else 0.0
            pitch_deg = abs(float(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))))
            rear_scale = 1.0
            if pitch_limit > 0 and pitch_deg > pitch_limit:
                rear_scale = float(np.clip(1.0 - (pitch_deg - pitch_limit) / pitch_fade, 0.15, 1.0))
            for k, (j, ai) in enumerate(zip(wheels, w_act)):
                base = speed - corr if k in left else speed + corr
                if k in rear_k:
                    base *= rear_scale
                elif k in front_k:
                    base *= 1.0 + 0.5 * (1.0 - rear_scale)
                if alt_amp > 0.0 and (k in front_k or k in mid_k):
                    base *= alt_front_scale
                d.ctrl[ai] = float(np.clip(25.0 * (base - d.qvel[m.jnt_dofadr[j]]), -tau_max, tau_max))
            for j, ai in zip(steer, s_act):
                q = d.qpos[m.jnt_qposadr[j]]
                d.ctrl[ai] = float(np.clip(-20.0 * q - d.qvel[m.jnt_dofadr[j]], -40, 40))
            stand_dwell = dwell_amp > 0.0 and (not recovery or recovering)
            if stand_dwell:
                t0 = recovery_start if (recovery and recovering) else 0.0
                up_left = int((t - t0) / dwell_hold) % 2 == 0
                for j, ai in zip(rockers, r_act):
                    target = dwell_amp if (jn[j].startswith("left") == up_left) else 0.0
                    q = d.qpos[m.jnt_qposadr[j]]
                    d.ctrl[ai] = 0.0 if rocker_free else float(np.clip(
                        rocker_gain * (np.deg2rad(target) - q) - rocker_damp * d.qvel[m.jnt_dofadr[j]],
                        -100, 100))
            elif recovering or alt_amp > 0.0:
                if recovering:
                    swing = recovery_sign * recovery_amp * np.sin(
                        2.0 * np.pi * recovery_freq * (t - recovery_start))
                else:
                    swing = alt_amp * np.sin(2.0 * np.pi * alt_freq * t)
                for j, ai in zip(rockers, r_act):
                    target = alt_bias + (swing if jn[j].startswith("left") else -swing)
                    q = d.qpos[m.jnt_qposadr[j]]
                    d.ctrl[ai] = 0.0 if rocker_free else float(np.clip(
                        rocker_gain * (np.deg2rad(target) - q) - rocker_damp * d.qvel[m.jnt_dofadr[j]],
                        -100, 100))
            else:
                wanted = schedule(x)
                rocker_deg += float(np.clip(wanted - rocker_deg, -slew * CTRL_DT, slew * CTRL_DT))
                for j, ai in zip(rockers, r_act):
                    q = d.qpos[m.jnt_qposadr[j]]
                    d.ctrl[ai] = 0.0 if rocker_free else float(np.clip(
                        rocker_gain * (np.deg2rad(rocker_deg) - q) - rocker_damp * d.qvel[m.jnt_dofadr[j]], -100, 100))
            next_ctrl += CTRL_DT

        mujoco.mj_step(m, d)
        t += SIM_DT
        if on_step is not None:
            on_step(m, d, t)
        max_z = max(max_z, float(d.qpos[2]))
        if on_frame is not None and n % 4 == 0:
            on_frame(m, d, t)
        if stop_when_done and stopped_at is None and float(d.qpos[0]) > course_end:
            wheel_lift = max(float(d.xpos[b][2]) - 0.163 for b in wb)
            if wheel_lift < 0.02 and abs(float(d.qpos[2]) - h0) < 0.03:
                stopped_at = round(t, 2)
                break
        w, xx, yy, zz = d.qpos[3:7]
        max_pitch = max(max_pitch, abs(float(np.degrees(np.arcsin(np.clip(2 * (w * yy - zz * xx), -1, 1))))))

    x, z = float(d.qpos[0]), float(d.qpos[2])
    reached_top = max_z > h0 + plat_top - 0.06
    back_down = z < h0 + 0.06
    past = x > course_end + 0.3
    return {
        "height": height, "stairs": stairs, "depth": depth, "friction": friction,
        "deg": deg, "assist": assist,
        "x": round(x, 2), "z": round(z, 3), "max_z": round(max_z, 3),
        "plat_z": round(h0 + plat_top, 3), "max_pitch_deg": round(max_pitch, 1),
        "reached_top": bool(reached_top), "back_down": bool(back_down), "past_end": bool(past),
        "stopped_at_s": stopped_at, "recovery_count": recovery_count,
        "success": bool(reached_top and back_down and past),
    }


def run(*args, **kwargs) -> dict:
    return simulate(*args, **kwargs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--stairs", type=int, default=3)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--friction", type=float, default=1.4)
    p.add_argument("--deg", type=float, default=12.0)
    p.add_argument("--front-lead", type=float, default=0.95)
    p.add_argument("--mid-lead", type=float, default=0.10)
    p.add_argument("--mid-trail", type=float, default=0.70)
    p.add_argument("--no-assist", action="store_true")
    p.add_argument("--const-deg", type=float, default=None)
    p.add_argument("--init-yaw", type=float, default=0.0)
    p.add_argument("--init-x", type=float, default=0.0)
    p.add_argument("--no-stop", action="store_true")
    p.add_argument("--mid-deg", type=float, default=None, help="rocker angle during the middle-wheel window")
    p.add_argument("--mode", choices=("roll", "lift"), default="roll")
    p.add_argument("--lift-deg", type=float, default=35.0)
    p.add_argument("--lift-hold", type=float, default=0.20)
    p.add_argument("--lift-lead", type=float, default=0.90)
    p.add_argument("--nosing", type=float, default=0.0, help="round the riser edge with this radius (m)")
    p.add_argument("--tread-from-height", action="store_true", help="45-degree stairs: tread depth equals step height")
    p.add_argument("--stair-angle", type=float, default=None, help="overall rise angle in degrees; sets tread = height/tan(angle)")
    p.add_argument("--wheel-radius", type=float, default=0.163)
    a = p.parse_args()
    if a.stair_angle:
        depth = a.height / math.tan(math.radians(a.stair_angle))
    elif a.tread_from_height:
        depth = a.height
    else:
        depth = a.depth
    print(json.dumps(run(a.height, a.stairs, depth, a.seconds, a.speed,
                         a.friction, deg=a.deg, const_deg=a.const_deg, init_yaw_deg=a.init_yaw,
                         init_x=a.init_x, stop_when_done=not a.no_stop, mid_deg=a.mid_deg,
                         mode=a.mode, lift_deg=a.lift_deg, lift_hold=a.lift_hold,
                         lift_lead=a.lift_lead, nosing=a.nosing, wheel_radius=a.wheel_radius,
                         front_lead=a.front_lead,
                         mid_lead=a.mid_lead, mid_trail=a.mid_trail,
                         assist=not a.no_assist)))
