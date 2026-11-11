"""Trace the first 30 s at a large step height to see where it loses the climb."""
import argparse
import json

import mujoco
import numpy as np

from course import simulate

state = {"next": 0.0, "rows": []}


def make_logger(model, every=1.0):
    jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    rockers = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
    groups = {}
    for key in ("front", "middle", "rear"):
        groups[key] = [i for i, n in enumerate(jn) if n.endswith(f"{key}_wheel_joint")]
    state.update(rockers=rockers, groups=groups, every=every)

    def on_step(m, d, t):
        if t < state["next"]:
            return
        state["next"] = t + state["every"]
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = float(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))))
        row = {"t": round(t, 1), "x": round(float(d.qpos[0]), 2), "z": round(float(d.qpos[2]), 2),
               "pitch": round(pitch, 1),
               "rocker": round(float(np.mean([np.degrees(d.qpos[m.jnt_qposadr[j]]) for j in rockers])), 1)}
        for key, js in groups.items():
            w = float(np.mean([d.qvel[m.jnt_dofadr[j]] for j in js]))
            row[key] = round(w, 2)
        state["rows"].append(row)

    return on_step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--mu", type=float, default=4.0)
    p.add_argument("--deg", type=float, default=30.0)
    p.add_argument("--seconds", type=float, default=30.0)
    a = p.parse_args()
    holder = {}

    def hook(m, d, t):
        if "fn" not in holder:
            holder["fn"] = make_logger(m)
        holder["fn"](m, d, t)

    r = simulate(a.height, 12, a.height, a.seconds, 6.0, a.mu, 120.0, a.deg,
                 pitch_limit=0.0, front_lead=9.5 * a.height, mid_lead=a.height,
                 mid_trail=7.0 * a.height, on_step=hook)
    print(f"{'t':>5} {'x':>6} {'z':>6} {'pitch':>6} {'rocker':>7} {'front':>6} {'mid':>6} {'rear':>6}")
    for row in state["rows"]:
        print(f"{row['t']:>5} {row['x']:>6} {row['z']:>6} {row['pitch']:>6} {row['rocker']:>7} "
              f"{row['front']:>6} {row['middle']:>6} {row['rear']:>6}")
    print(json.dumps({k: r[k] for k in ("x", "max_z", "plat_z", "reached_top", "back_down", "success")}))
