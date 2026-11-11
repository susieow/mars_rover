"""Log wheel spin vs body speed to separate slip from blockage at the 0.20 m stall."""

import argparse
import json

import mujoco
import numpy as np

from course import simulate

state = {}
R = 0.163


def make_logger(model, every=0.5, limit=30.0):
    jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    bn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    gn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    state.update(
        rockers=[i for i, n in enumerate(jn) if n.endswith("_rocker_joint")],
        mid_j=[i for i, n in enumerate(jn) if n.endswith("_middle_wheel_joint")],
        rear_j=[i for i, n in enumerate(jn) if n.endswith("_rear_wheel_joint")],
        mid_bodies=[i for i, n in enumerate(bn) if n.endswith("middle_wheel")],
        gn=gn, next=0.0, rows=[], every=every, limit=limit,
    )

    def forces(d, m):
        force = np.zeros(6)
        out = {"mid": 0.0, "front": 0.0, "rear": 0.0}
        for c in range(d.ncon):
            con = d.contact[c]
            n1, n2 = gn[int(con.geom1)] or "", gn[int(con.geom2)] or ""
            terrain = (n1, n2) if n1.startswith(("up", "down", "platform", "ground")) else (n2, n1)
            if not terrain[0].startswith(("up", "down", "platform", "ground")):
                continue
            wheel = n1 if "_wheel" in n1 else (n2 if "_wheel" in n2 else "")
            if not wheel:
                continue
            mujoco.mj_contactForce(m, d, c, force)
            fn = abs(float(force[0]))
            for key, short in (("middle_wheel", "mid"), ("front_wheel", "front"), ("rear_wheel", "rear")):
                if key in wheel:
                    out[short] += fn
        return out

    def on_step(m, d, t):
        if t < state["next"] or t > state["limit"]:
            return
        state["next"] = t + state["every"]
        fn = forces(d, m)
        w = np.mean([d.qvel[m.jnt_dofadr[j]] for j in state["mid_j"]])
        vx = float(d.qvel[0])
        surf = w * R
        slip = (surf - vx) / abs(surf) * 100 if abs(surf) > 0.05 else float("nan")
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = float(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))))
        rocker = np.mean([np.degrees(d.qpos[m.jnt_qposadr[j]]) for j in state["rockers"]])
        state["rows"].append({
            "t": round(t, 1), "x": round(float(d.qpos[0]), 2), "vx": round(vx, 2),
            "pitch": round(pitch, 1), "rocker": round(float(rocker), 1),
            "mid_lift": round((max(float(d.xpos[b][2]) for b in state["mid_bodies"]) - R) * 1000, 0),
            "mid_fn": round(fn["mid"]), "front_fn": round(fn["front"]), "rear_fn": round(fn["rear"]),
            "mid_w": round(float(w), 1), "mid_v": round(surf, 2),
            "slip_pct": None if np.isnan(slip) else round(float(slip), 0),
        })

    return on_step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.20)
    p.add_argument("--friction", type=float, default=1.4)
    p.add_argument("--deg", type=float, default=12.0)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--mid-deg", type=float, default=None)
    p.add_argument("--mode", default="roll")
    p.add_argument("--stairs", type=int, default=3)
    p.add_argument("--stair-angle", type=float, default=None)
    p.add_argument("--depth", type=float, default=0.5)
    a = p.parse_args()

    holder = {}

    def hook(m, d, t):
        if "fn" not in holder:
            holder["fn"] = make_logger(m, limit=a.seconds)
        holder["fn"](m, d, t)

    import math
    depth = a.height / math.tan(math.radians(a.stair_angle)) if a.stair_angle else a.depth
    result = simulate(a.height, a.stairs, depth, a.seconds, 6.0, a.friction, 50.0, a.deg,
                      on_step=hook, mid_deg=a.mid_deg, mode=a.mode)

    print(f"{'t':>5} {'x':>6} {'vx':>6} {'pitch':>6} {'rocker':>7} {'mid_lift':>8} "
          f"{'mid_fn':>7} {'mid_w':>6} {'mid_v':>6} {'slip%':>6} {'rear_fn':>8}")
    for r in state.get("rows", [])[:30]:
        print(f"{r['t']:>5} {r['x']:>6} {r['vx']:>6} {r['pitch']:>6} {r['rocker']:>7} "
              f"{r['mid_lift']:>8} {r['mid_fn']:>7} {r['mid_w']:>6} {r['mid_v']:>6} "
              f"{str(r['slip_pct']):>6} {r['rear_fn']:>8}")
    print(json.dumps(result))
