"""Position-scheduled rocker assist.

Terrain is known, so schedule the suspension by base position instead of contact
events (contact triggering chattered). For each riser at x = start + i*depth:
  front window  x in [r - front_lead, r - mid_lead]  -> rocker = -deg (raise front)
  middle window x in [r - mid_lead, r + mid_trail]   -> rocker = +deg (raise middle)
Slew-rate limited so the suspension does not slam.
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

SIM_DT = 0.005
CTRL_DT = 0.02


def run(height: float, stairs: int, depth: float, seconds: float = 60.0, speed: float = 6.0,
        friction: float = 1.0, tau_max: float = 50.0, deg: float = 15.0,
        front_lead: float = 0.95, mid_lead: float = 0.10, mid_trail: float = 0.70,
        slew: float = 90.0, assist: bool = True, start_x: float = 1.6) -> dict:
    build_robot()
    scene = build_scene(height, stairs, depth, friction)
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

    h0 = standing_height(m, wb)
    d.qpos[:] = m.qpos0
    d.qpos[2] = h0
    mujoco.mj_forward(m, d)

    def wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def schedule(x: float) -> float:
        """Rocker target in degrees for the current base position."""
        if not assist:
            return 0.0
        for i in range(stairs):
            r = start_x + i * depth
            if r - front_lead < x < r - mid_lead:
                return -deg
            if r - mid_lead <= x < r + mid_trail:
                return deg
        return 0.0

    rocker_deg = 0.0
    t, next_ctrl = 0.0, 0.0
    max_pitch, min_dist = 0.0, 0.0
    phases = {"front": 0, "mid": 0}
    last_phase = ""

    for _ in range(int(seconds / SIM_DT)):
        if t >= next_ctrl:
            x = float(d.qpos[0])
            qw, qx, qy, qz = d.qpos[3:7]
            yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
            yaw_rate = float(d.qvel[5])
            y_ref = float(np.clip(-0.4 * d.qpos[1], -0.25, 0.25))
            corr = 4.0 * wrap(y_ref - yaw) + 0.6 * yaw_rate
            for k, (j, ai) in enumerate(zip(wheels, w_act)):
                target = speed - corr if k in left else speed + corr
                d.ctrl[ai] = float(np.clip(25.0 * (target - d.qvel[m.jnt_dofadr[j]]), -tau_max, tau_max))
            for j, ai in zip(steer, s_act):
                q = d.qpos[m.jnt_qposadr[j]]
                d.ctrl[ai] = float(np.clip(-20.0 * q - d.qvel[m.jnt_dofadr[j]], -40, 40))

            wanted = schedule(x)
            step_lim = slew * CTRL_DT
            rocker_deg += float(np.clip(wanted - rocker_deg, -step_lim, step_lim))
            phase = "front" if wanted < 0 else ("mid" if wanted > 0 else "")
            if phase != last_phase and phase:
                phases[phase] += 1
            last_phase = phase

            target = np.deg2rad(rocker_deg)
            for j, ai in zip(rockers, r_act):
                q = d.qpos[m.jnt_qposadr[j]]
                d.ctrl[ai] = float(np.clip(80.0 * (target - q) - 3.0 * d.qvel[m.jnt_dofadr[j]], -100, 100))
            next_ctrl += CTRL_DT

        mujoco.mj_step(m, d)
        t += SIM_DT
        w, xx, yy, zz = d.qpos[3:7]
        max_pitch = max(max_pitch, abs(float(np.degrees(np.arcsin(np.clip(2 * (w * yy - zz * xx), -1, 1))))))
        for i in range(d.ncon):
            min_dist = min(min_dist, float(d.contact[i].dist))

    x, z = float(d.qpos[0]), float(d.qpos[2])
    top_x = start_x + stairs * depth
    target_z = h0 + height * stairs
    return {
        "height": height, "stairs": stairs, "deg": deg, "assist": assist,
        "x": round(x, 2), "y": round(float(d.qpos[1]), 2), "z": round(z, 3),
        "target_z": round(target_z, 3), "max_pitch_deg": round(max_pitch, 1),
        "penetration_mm": round(-min_dist * 1000, 1), "phases": phases,
        "success": bool(abs(z - target_z) < 0.06 and x > top_x + 0.3),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--stairs", type=int, default=3)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--friction", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=50.0)
    p.add_argument("--deg", type=float, default=15.0)
    p.add_argument("--no-assist", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.height, a.stairs, a.depth, a.seconds, a.speed, a.friction,
                         a.tau_max, a.deg, assist=not a.no_assist)))
