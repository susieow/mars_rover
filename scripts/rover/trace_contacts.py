"""List every robot-terrain contact at a few instants, with penetration and normal force."""
import mujoco
import numpy as np

from course import simulate


def run(mu, times):
    state = {"gn": None, "next": 0.0, "times": list(times), "done": set()}

    def on_step(m, d, t):
        if not state["times"] or t < state["next"]:
            return
        if state["gn"] is None:
            state["gn"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
            state["bn"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
            state["jn"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        state["next"] = t + 0.05
        want = state["times"][0]
        if abs(t - want) > 0.02:
            return
        state["times"].pop(0)
        gn, force = state["gn"], np.zeros(6)
        print(f"--- mu={mu} t={t:.2f} x={d.qpos[0]:.2f} z={d.qpos[2]:.3f}")
        for c in range(d.ncon):
            con = d.contact[c]
            n1, n2 = gn[con.geom1] or "?", gn[con.geom2] or "?"
            if not (n1.startswith(("ground", "up", "down", "platform"))
                    or n2.startswith(("ground", "up", "down", "platform"))):
                continue
            mujoco.mj_contactForce(m, d, c, force)
            print(f"  {n1:28s} {n2:28s} dist={con.dist:+.4f} fn={force[0]:8.1f}")

    simulate(0.10, 12, 0.10, seconds=max(times) + 0.1, speed=6.0, friction=mu,
             tau_max=120.0, pitch_limit=0.0, on_step=on_step)


run(2.5, [1.5])
run(10.0, [0.5, 1.5, 3.0])
