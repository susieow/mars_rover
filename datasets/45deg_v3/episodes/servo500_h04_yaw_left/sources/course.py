"""Symmetric stair course with contact-driven rocker control.

A strict success requires completing the climb and descent, then stopping in a
level, aligned, settled pose on the far ground.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import os
from pathlib import Path

import mujoco
import numpy as np

from climb import ROBOT_XML, ROOT, build_robot, standing_height

CTRL_DT = 0.002


def build_course(height: float, stairs: int, depth: float, friction: float,
                 start_x: float = 1.6, top_len: float = 2.0, nosing: float = 0.0,
                 wheel_radius: float = 0.163, middle_collision: bool = True,
                 contact_timeconst: float = 0.02) -> Path:
    xml = ROBOT_XML.read_text()
    if not 0.004 <= contact_timeconst <= 0.02:
        raise ValueError("contact_timeconst must be within [0.004, 0.02] seconds")
    xml = re.sub(r"<geom ", f'<geom solref="{contact_timeconst} 1" ', xml)
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
    mu = f'friction="{friction} 0.02 0.001" solref="{contact_timeconst} 1"'
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
    scene = ROOT / f"scene_course_{os.getpid()}.xml"
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


from closed_loop_controller import ClosedLoopController

def simulate(height: float, stairs: int, depth: float, seconds: float = 90.0, speed: float = 3.0,
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
             dwell_amp: float = 0.0, dwell_hold: float = 1.5,
             controller_mode: str = "closed_loop", settle_time: float = 1.2,
             quality_pitch_limit: float = 60.0, front_lift_deg: float = 14.0,
             middle_lift_deg: float = 17.0, quality_roll_limit: float = 15.0,
             init_y: float = 0.0, on_control=None, wheel_gain: float = 10.0,
             descent_speed: float = 1.0, contact_timeconst: float = 0.02) -> dict:
    """Run one course with the contact-driven controller and strict final validation."""
    from closed_loop_controller import ClosedLoopController

    build_robot()
    scene = build_course(height, stairs, depth, friction, nosing=nosing,
                         wheel_radius=wheel_radius, middle_collision=middle_collision,
                         contact_timeconst=contact_timeconst)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(m.njnt)]
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
    wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
    steer = [i for i, n in enumerate(jn) if n.endswith("_steer_joint")]
    rockers = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
    wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]

    def act(name: str) -> int:
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_act")

    w_act = [act(jn[j]) for j in wheels]
    s_act = [act(jn[j]) for j in steer]
    r_act = [act(jn[j]) for j in rockers]

    h0 = standing_height(m, wb) + (wheel_radius - 0.163)
    d.qpos[:] = m.qpos0
    d.qpos[2] = h0
    d.qpos[0] = init_x
    d.qpos[1] = init_y
    yaw0 = np.deg2rad(init_yaw_deg)
    d.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
    mujoco.mj_forward(m, d)

    _, course_end = edges(height, stairs, depth)
    plat_top = height * stairs
    controller = ClosedLoopController(
        m, d,
        wheels=wheels,
        wheel_act=w_act,
        steer=steer,
        steer_act=s_act,
        rockers=rockers,
        rocker_act=r_act,
        wheel_radius=wheel_radius,
        speed=speed if controller_mode == "closed_loop" else 6.0,
        tau_max=tau_max,
        yaw0=0.0,
        front_lift_deg=(front_lift_deg if assist else 0.0),
        middle_lift_deg=(middle_lift_deg if assist else 0.0),
        rocker_gain=rocker_gain,
        rocker_damp=rocker_damp,
        rocker_slew_deg_s=min(float(slew), 30.0),
        dt=CTRL_DT,
        settle_time=settle_time,
        wheel_gain=wheel_gain,
        descent_speed=descent_speed,
    )

    if controller_mode != "closed_loop":
        raise ValueError("controller_mode must be 'closed_loop'; the legacy scheduler is disabled")

    t = 0.0
    next_ctrl = 0.0
    next_frame = 0.0
    max_z = -1.0
    max_pitch = 0.0
    max_roll = 0.0
    stopped_at = None
    last_state = {}

    for n in range(int(seconds / m.opt.timestep)):
        if t + 1e-9 >= next_ctrl:
            last_state = controller.update(t)
            if rocker_free:
                for actuator in r_act:
                    d.ctrl[actuator] = 0.0
            if on_control is not None:
                on_control(m, d, t, controller)
            next_ctrl += CTRL_DT

        mujoco.mj_step(m, d)
        t = float(d.time)
        if on_step is not None:
            on_step(m, d, t)
        max_z = max(max_z, float(d.qpos[2]))
        w, x, y, z = [float(v) for v in d.qpos[3:7]]
        pitch = float(np.degrees(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))))
        roll = float(np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))))
        max_pitch = max(max_pitch, abs(pitch))
        max_roll = max(max_roll, abs(roll))
        if on_frame is not None and t + 1e-9 >= next_frame:
            on_frame(m, d, t)
            next_frame += 0.02

        if controller.abort_reason:
            break
        if stop_when_done and controller.settle_ok:
            stopped_at = round(t, 2)
            break

    x = float(d.qpos[0])
    y = float(d.qpos[1])
    z = float(d.qpos[2])
    w, qx, qy, qz = [float(v) for v in d.qpos[3:7]]
    roll = float(np.degrees(np.arctan2(2.0 * (w * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))))
    pitch = float(np.degrees(np.arcsin(np.clip(2.0 * (w * qy - qz * qx), -1.0, 1.0))))
    yaw = float(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    yaw_error = float((-yaw + np.pi) % (2.0 * np.pi) - np.pi)
    wheel_speed_max = max((abs(float(d.qvel[m.jnt_dofadr[j]])) for j in wheels), default=0.0)
    linear_speed = float(np.linalg.norm(d.qvel[:3]))
    angular_speed = float(np.linalg.norm(d.qvel[3:6]))

    reached_top = bool(controller.platform_seen and max_z > h0 + plat_top - 0.06)
    back_down = bool(controller.down_seen and z < h0 + 0.06)
    past = bool(x > course_end + 0.3)
    final_shape_ok = bool(
        abs(y) < 0.25
        and abs(np.degrees(yaw_error)) < 10.0
        and abs(roll) < 5.0
        and abs(pitch) < 5.0
        and linear_speed < 0.05
        and angular_speed < 0.10
        and wheel_speed_max < 0.20
        and controller.settle_ok
    )
    quality_ok = bool(max_pitch <= quality_pitch_limit and max_roll <= quality_roll_limit)
    success = bool(reached_top and back_down and past and final_shape_ok and quality_ok and controller.abort_reason is None)
    return {
        "height": height,
        "stairs": stairs,
        "depth": depth,
        "friction": friction,
        "contact_timeconst": contact_timeconst,
        "speed": speed,
        "wheel_gain": wheel_gain,
        "descent_speed": descent_speed,
        "rocker_gain": rocker_gain,
        "rocker_damp": rocker_damp,
        "nosing": round(float(nosing), 4),
        "tau_max_requested": round(float(tau_max), 1),
        "tau_max_actual": round(min(float(tau_max), 50.0), 1),
        "front_lift_deg": round(float(front_lift_deg), 2),
        "middle_lift_deg": round(float(middle_lift_deg), 2),
        "quality_pitch_limit": round(float(quality_pitch_limit), 2),
        "deg": deg,
        "assist": assist,
        "controller": "closed_loop_v4",
        "simulation_dt": float(m.opt.timestep),
        "control_dt": CTRL_DT,
        "duration_s": float(d.time),
        "quality_roll_limit": quality_roll_limit,
        "ground_wheels": controller.last_contacts.get("ground_wheels", []),
        "stable_duration_s": (float(d.time) - controller.settle_since) if controller.settle_since is not None else 0.0,
        "x": round(x, 2),
        "y": round(y, 3),
        "z": round(z, 3),
        "max_z": round(max_z, 3),
        "plat_z": round(h0 + plat_top, 3),
        "max_pitch_deg": round(max_pitch, 1),
        "max_roll_deg": round(max_roll, 1),
        "final_pitch_deg": round(pitch, 2),
        "final_roll_deg": round(roll, 2),
        "final_yaw_deg": round(np.degrees(yaw), 2),
        "final_yaw_error_deg": round(np.degrees(yaw_error), 2),
        "linear_speed": round(linear_speed, 4),
        "angular_speed": round(angular_speed, 4),
        "wheel_speed_max": round(wheel_speed_max, 4),
        "reached_top": reached_top,
        "back_down": back_down,
        "past_end": past,
        "platform_seen": controller.platform_seen,
        "down_seen": controller.down_seen,
        "settled": controller.settle_ok,
        "final_shape_ok": final_shape_ok,
        "quality_ok": quality_ok,
        "stopped_at_s": stopped_at,
        "recovery_count": 0,
        "max_slip": round(controller.max_slip, 3),
        "max_abs_y": round(controller.max_abs_y, 3),
        "max_yaw_error_deg": round(np.degrees(controller.max_yaw_error), 2),
        "phase": controller.phase,
        "events": controller.event_log,
        "abort_reason": controller.abort_reason,
        "success": success,
    }


def run(*args, **kwargs) -> dict:
    return simulate(*args, **kwargs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--stairs", type=int, default=3)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--speed", type=float, default=3.0)
    p.add_argument("--friction", type=float, default=1.4)
    p.add_argument("--tau-max", type=float, default=50.0)
    p.add_argument("--deg", type=float, default=12.0)
    p.add_argument("--tread-from-height", action="store_true")
    p.add_argument("--stair-angle", type=float, default=None)
    p.add_argument("--nosing", type=float, default=0.0)
    p.add_argument("--wheel-radius", type=float, default=0.163)
    p.add_argument("--no-assist", action="store_true")
    p.add_argument("--no-stop", action="store_true")
    p.add_argument("--front-lift-deg", type=float, default=14.0)
    p.add_argument("--middle-lift-deg", type=float, default=17.0)
    p.add_argument("--quality-pitch-limit", type=float, default=38.0)
    a = p.parse_args()
    if a.stair_angle:
        depth = a.height / math.tan(math.radians(a.stair_angle))
    elif a.tread_from_height:
        depth = a.height
    else:
        depth = a.depth
    print(json.dumps(run(
        a.height, a.stairs, depth, a.seconds, a.speed, a.friction, a.tau_max,
        deg=a.deg, nosing=a.nosing, wheel_radius=a.wheel_radius,
        stop_when_done=not a.no_stop, assist=not a.no_assist,
        front_lift_deg=a.front_lift_deg,
        middle_lift_deg=a.middle_lift_deg,
        quality_pitch_limit=a.quality_pitch_limit,
    )) )
