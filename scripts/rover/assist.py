"""Rocker-assist controller: pulse the active suspension when a wheel meets a riser.

Sign convention measured on this model (base suspended):
  rocker -20 deg raises the front wheel 13 cm and lowers the middle wheel;
  rocker +20 deg raises the middle wheel 13 cm and lowers the front.

State machine per riser: front wheel touches -> raise front (-deg), hold, return;
then middle wheel touches -> raise middle (+deg), hold, return.
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
        friction: float = 1.0, tau_max: float = 50.0, deg: float = 15.0, hold_s: float = 1.6,
        assist: bool = True, start_x: float = 1.6) -> dict:
    build_robot()
    scene = build_scene(height, stairs, depth, friction)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
    steer = [i for i, n in enumerate(jn) if n.endswith("_steer_joint")]
    rockers = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
    wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]

    front_geoms = {g for g, n in enumerate(gn) if n and "front_wheel" in n}
    middle_geoms = {g for g, n in enumerate(gn) if n and "middle_wheel" in n}
    step_geoms = {g for g, n in enumerate(gn) if n and n.startswith("step")}

    act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")  # noqa: E731
    w_act = [act(jn[j]) for j in wheels]
    s_act = [act(jn[j]) for j in steer]
    r_act = [act(jn[j]) for j in rockers]
    left = [k for k, j in enumerate(wheels) if "left" in jn[j]]

    h0 = standing_height(m, wb)
    d.qpos[:] = m.qpos0
    d.qpos[2] = h0
    mujoco.mj_forward(m, d)

    def touches(a: set[int], b: set[int]) -> bool:
        for c in range(d.ncon):
            g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
            if (g1 in a and g2 in b) or (g1 in b and g2 in a):
                return True
        return False

    def wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    state = "idle"
    state_t = 0.0
    rocker_cmd = 0.0
    t, next_ctrl = 0.0, 0.0
    max_pitch, min_dist = 0.0, 0.0
    pulses = {"front": 0, "mid": 0}
    cooldown = 0.0

    for _ in range(int(seconds / SIM_DT)):
        if t >= next_ctrl:
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

            if assist:
                state_t += CTRL_DT
                cooldown = max(0.0, cooldown - CTRL_DT)
                if state == "idle" and cooldown == 0.0:
                    if touches(front_geoms, step_geoms):
                        state, state_t, rocker_cmd = "front", 0.0, -deg
                        pulses["front"] += 1
                    elif touches(middle_geoms, step_geoms):
                        state, state_t, rocker_cmd = "mid", 0.0, deg
                        pulses["mid"] += 1
                elif state == "idle" and cooldown > 0:
                    pass
                if state != "idle" and state_t > hold_s:
                    state, state_t, rocker_cmd = "idle", 0.0, 0.0
                    cooldown = 0.8

            target = np.deg2rad(rocker_cmd)
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
        "height": height, "stairs": stairs, "depth": depth, "friction": friction,
        "tau_max": tau_max, "assist": assist, "deg": deg, "hold_s": hold_s,
        "x": round(x, 2), "y": round(float(d.qpos[1]), 2), "z": round(z, 3),
        "target_z": round(target_z, 3), "max_pitch_deg": round(max_pitch, 1),
        "penetration_mm": round(-min_dist * 1000, 1), "pulses": pulses,
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
    p.add_argument("--hold", type=float, default=1.6)
    p.add_argument("--no-assist", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.height, a.stairs, a.depth, a.seconds, a.speed, a.friction,
                         a.tau_max, a.deg, a.hold, not a.no_assist)))
