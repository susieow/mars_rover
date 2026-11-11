"""Trace base motion and wheel spin on 45-degree stairs for two friction levels."""
import mujoco
import numpy as np

from course import simulate


def run(mu):
    state = {"rows": [], "next": 0.0, "gn": None}

    def on_step(m, d, t):
        if t < state["next"]:
            return
        if state["gn"] is None:
            jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
            state["jn"] = jn
            state["wheels"] = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
            state["gn"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
        state["next"] = t + 1.0
        sumw = {}
        for i in state["wheels"]:
            key = "front" if "front" in state["jn"][i] else "middle" if "middle" in state["jn"][i] else "rear"
            sumw.setdefault(key, []).append(float(d.qvel[m.jnt_dofadr[i]]))
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = float(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))))
        state["rows"].append((round(t, 1), round(float(d.qpos[0]), 2), round(float(d.qpos[2]), 2),
                              round(pitch, 1), {k: round(float(np.mean(v)), 1) for k, v in sumw.items()}))

    r = simulate(0.10, 12, 0.10, seconds=12.0, speed=6.0, friction=mu,
                 tau_max=120.0, pitch_limit=0.0, on_step=on_step)
    print(f"mu={mu} final x={r['x']} z={r['z']} pitch={r['max_pitch_deg']} top={r['reached_top']}")
    for row in state["rows"]:
        print(row)
    print()


for mu in (2.5, 10.0):
    run(mu)
