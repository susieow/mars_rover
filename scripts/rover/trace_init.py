import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

build_robot()
scene = build_scene(0.10, 3, 0.5, 1.0)
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]
names = [bn[i] for i in wb]

h0 = standing_height(m, wb)
d.qpos[:] = m.qpos0
d.qpos[2] = h0
mujoco.mj_kinematics(m, d)
print("initial base z:", round(h0, 3))
for i, n in zip(wb, names):
    print(f"  {n:20s} z={d.xpos[i][2]:7.4f}")

print("== 静置 2 秒（无控制）==")
mujoco.mj_forward(m, d)
for _ in range(400):
    mujoco.mj_step(m, d)
print("base z:", round(float(d.qpos[2]), 3), "roll/pitch:",
      round(float(np.degrees(2 * np.arcsin(np.clip(d.qpos[4], -1, 1)))), 1),
      round(float(np.degrees(2 * np.arcsin(np.clip(d.qpos[5], -1, 1)))), 1))
for i, n in zip(wb, names):
    print(f"  {n:20s} z={d.xpos[i][2]:7.4f}")
print("contacts:", d.ncon)
