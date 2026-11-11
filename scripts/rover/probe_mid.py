"""Where is the middle wheel, and what is it touching, at the stall?"""
import mujoco
import numpy as np

from course import simulate

st = {"done": False}


def on_step(m, d, t):
    if st["done"] or t < 12.0:
        return
    st["done"] = True
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    for i, n in enumerate(bn):
        if n and "middle_wheel" in n:
            p = d.xpos[i]
            print(f"body {n}: pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
    for i, n in enumerate(gn):
        if n and "middle_wheel" in n and n.endswith("col_1"):
            g = d.geom_xpos[i]
            print(f"geom {n}: pos=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f}) size={m.geom_size[i]}")
    force = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        n1, n2 = gn[con.geom1] or "", gn[con.geom2] or ""
        if "middle_wheel" not in n1 + n2:
            continue
        mujoco.mj_contactForce(m, d, c, force)
        print(f"contact {n1} <-> {n2} pos=({con.pos[0]:.3f},{con.pos[1]:.3f},{con.pos[2]:.3f}) "
              f"n=({con.frame[0]:+.2f},{con.frame[1]:+.2f},{con.frame[2]:+.2f}) fn={force[0]:.1f}")


simulate(0.10, 12, 0.10, seconds=12.2, speed=6.0, friction=2.5,
         tau_max=120.0, pitch_limit=0.0, on_step=on_step)
