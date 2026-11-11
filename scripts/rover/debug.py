import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

build_robot()
scene = build_scene(0.10)
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)

jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]
print("actuators:", m.nu, "actuator names:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])

h0 = standing_height(m, wb)
d.qpos[:] = m.qpos0
d.qpos[2] = h0
mujoco.mj_forward(m, d)
print("standing h", round(h0, 3), "ncon", d.ncon)

for t in range(3000):
    for k, j in enumerate(wheels):
        d.ctrl[k] = float(np.clip(5.0 * (3.0 - d.qvel[m.jnt_dofadr[j]]), -50, 50))
    mujoco.mj_step(m, d)
    if t % 500 == 0:
        print(
            f"t={t * 0.005:4.1f} x={d.qpos[0]:6.3f} z={d.qpos[2]:5.3f} "
            f"vx={d.qvel[0]:6.3f} wq={d.qvel[m.jnt_dofadr[wheels[0]]]:6.2f} ncon={d.ncon}"
        )
