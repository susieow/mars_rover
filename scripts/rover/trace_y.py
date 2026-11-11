import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

build_robot()
scene = build_scene(0.10, 3, 0.5, 1.0)
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
steer = [i for i, n in enumerate(jn) if n.endswith("_steer_joint")]
rockers = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]
act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")  # noqa: E731
w_act = [act(jn[j]) for j in wheels]
s_act = [act(jn[j]) for j in steer]
r_act = [act(jn[j]) for j in rockers]

d.qpos[:] = m.qpos0
d.qpos[2] = standing_height(m, wb)
mujoco.mj_forward(m, d)

for t in range(int(30 / 0.005)):
    for j, ai in zip(wheels, w_act):
        d.ctrl[ai] = float(np.clip(25 * (6.0 - d.qvel[m.jnt_dofadr[j]]), -50, 50))
    for j, ai in zip(steer, s_act):
        d.ctrl[ai] = float(np.clip(-20 * d.qpos[m.jnt_qposadr[j]] - d.qvel[m.jnt_dofadr[j]], -40, 40))
    for ai in r_act:
        d.ctrl[ai] = 0.0
    mujoco.mj_step(m, d)
    if t % 600 == 0:
        print(f"t={t*0.005:5.1f} x={d.qpos[0]:6.2f} y={d.qpos[1]:6.2f} z={d.qpos[2]:5.3f} "
              f"ncon={d.ncon}")
