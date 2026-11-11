"""Diagnose what blocks the 0.20 m climb: rocker angle, middle-wheel height, contacts."""

import argparse
import json

import mujoco
import numpy as np

from course import simulate

state = {}


def make_logger(model, every=0.5, limit=40.0):
    jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    bn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    gn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    state.update(
        rockers=[i for i, n in enumerate(jn) if n.endswith("_rocker_joint")],
        mid_bodies=[i for i, n in enumerate(bn) if n.endswith("middle_wheel")],
        front_bodies=[i for i, n in enumerate(bn) if n.endswith("front_wheel")],
        gn=gn, next=0.0, rows=[], every=every, limit=limit,
    )

    def on_step(m, d, t):
        if t < state["next"] or t > state["limit"]:
            return
        state["next"] = t + state["every"]
        force = np.zeros(6)
        mid_fn = front_fn = 0.0
        mid_src = ""
        for c in range(d.ncon):
            con = d.contact[c]
            n1, n2 = gn[int(con.geom1)] or "", gn[int(con.geom2)] or ""
            terrain = (n1, n2) if n1.startswith(("up", "down", "platform", "ground")) else (n2, n1)
            if not terrain[0].startswith(("up", "down", "platform", "ground")):
                continue
            mujoco.mj_contactForce(m, d, c, force)
            fn = abs(float(force[0]))
            wheel = n1 if "_wheel" in n1 else (n2 if "_wheel" in n2 else "")
            if not wheel:
                continue
            if "middle_wheel" in wheel:
                mid_fn += fn
                mid_src = terrain[0]
            if "front_wheel" in wheel:
                front_fn += fn
        rocker_q = [np.degrees(d.qpos[m.jnt_qposadr[j]]) for j in state["rockers"]]
        mid_lift = max(float(d.xpos[b][2]) for b in state["mid_bodies"]) - 0.163
        front_lift = max(float(d.xpos[b][2]) for b in state["front_bodies"]) - 0.163
        state["rows"].append({
            "t": round(t, 1),
            "x": round(float(d.qpos[0]), 2),
            "z": round(float(d.qpos[2]), 3),
            "rocker": round(float(np.mean(rocker_q)), 1),
            "mid_lift": round(mid_lift * 1000, 1),
            "front_lift": round(front_lift * 1000, 1),
            "mid_fn": round(mid_fn),
            "front_fn": round(front_fn),
            "mid_on": mid_src,
        })

    return on_step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.20)
    p.add_argument("--friction", type=float, default=1.4)
    p.add_argument("--deg", type=float, default=12.0)
    p.add_argument("--seconds", type=float, default=40.0)
    a = p.parse_args()

    holder = {}

    def hook(m, d, t):
        if "fn" not in holder:
            holder["fn"] = make_logger(m)
        holder["fn"](m, d, t)

    result = simulate(a.height, 3, 0.5, a.seconds, 6.0, a.friction, 50.0, a.deg, on_step=hook)

    print(f"{'t':>5} {'x':>6} {'z':>6} {'rocker':>7} {'mid_lift':>9} {'front_lift':>10} "
          f"{'mid_fn':>7} {'front_fn':>8} mid_on")
    for r in state.get("rows", [])[:26]:
        print(f"{r['t']:>5} {r['x']:>6} {r['z']:>6} {r['rocker']:>7} {r['mid_lift']:>9} "
              f"{r['front_lift']:>10} {r['mid_fn']:>7} {r['front_fn']:>8} {r['mid_on']}")
    print(json.dumps(result))
