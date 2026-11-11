"""At the stall, print drive torque per wheel and every contact normal and position."""
import mujoco
import numpy as np

from course import simulate

st = {"gn": None, "jn": None, "done": False, "an": None}


def on_step(m, d, t):
    if st["done"] or t < 12.0:
        return
    st["done"] = True
    gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    print(f"t={t:.2f} base x={d.qpos[0]:.2f} z={d.qpos[2]:.2f} qvel={np.round(d.qvel[:6], 2)}")
    for i, n in enumerate(jn):
        if n.endswith("_wheel_joint"):
            ai = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")
            print(f"  {n:28s} qvel={d.qvel[m.jnt_dofadr[i]]:6.2f} "
                  f"ctrl={d.ctrl[ai]:7.1f} force={d.actuator_force[ai]:7.1f} "
                  f"frcrange={m.jnt_actfrcrange[i] if m.jnt_limited[i] else None}")
    force = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        n1, n2 = gn[con.geom1] or "", gn[con.geom2] or ""
        if not (n1.startswith(("ground", "up", "down", "platform"))
                or n2.startswith(("ground", "up", "down", "platform"))):
            continue
        mujoco.mj_contactForce(m, d, c, force)
        robot = n1 if n1.startswith(("left", "right")) else n2
        o = c * 0
        print(f"  {robot:28s} dist={con.dist:+.4f} f=({force[0]:7.1f},{force[1]:7.1f},{force[2]:7.1f}) "
              f"pos=({con.pos[0]:5.2f},{con.pos[1]:5.2f},{con.pos[2]:5.2f}) "
              f"n=({con.frame[0]:+.2f},{con.frame[1]:+.2f},{con.frame[2]:+.2f})")


simulate(0.10, 12, 0.10, seconds=12.2, speed=6.0, friction=2.5,
         tau_max=120.0, pitch_limit=0.0, rocker_free=True, on_step=on_step)
