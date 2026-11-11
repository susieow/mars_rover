import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

build_robot()
scene = build_scene(0.001)
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]
d.qpos[:] = m.qpos0
d.qpos[2] = standing_height(m, wb)
mujoco.mj_forward(m, d)
print("h", round(float(d.qpos[2]), 3), "ncon", d.ncon, "ctrl size", m.nu)

for t in range(2400):
    for k, j in enumerate(wheels):
        d.ctrl[k] = float(np.clip(40 * (3.0 - d.qvel[m.jnt_dofadr[j]]), -50, 50))
    mujoco.mj_step(m, d)
    if t % 200 == 0:
        wq = [round(float(d.qvel[m.jnt_dofadr[j]]), 2) for j in wheels]
        print(
            f"t={t * 0.005:4.1f} x={d.qpos[0]:6.3f} z={d.qpos[2]:6.3f} "
            f"vx={d.qvel[0]:6.3f} vz={d.qvel[2]:6.3f} wq={wq} ncon={d.ncon} "
            f"ctrl0={d.ctrl[0]:5.1f}"
        )

print("== 接触对 ==")
for i in range(min(d.ncon, 12)):
    c = d.contact[i]
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
    print(f"  {g1} <-> {g2}  dist={c.dist:.4f}")
