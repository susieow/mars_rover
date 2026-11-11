import json

import mujoco
import numpy as np

from assist2 import run  # noqa: F401  (ensures module import works)
from climb import build_robot, build_scene, standing_height

build_robot()
scene = build_scene(0.15, 3, 0.5, 1.0)
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
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

worst = (0.0, None, 0.0)
deg = 12.0
rocker_deg = 0.0
t, next_ctrl = 0.0, 0.0
for _ in range(int(60 / 0.005)):
    if t >= next_ctrl:
        x = float(d.qpos[0])
        wanted = 0.0
        for i in range(3):
            r = 1.6 + i * 0.5
            if r - 0.95 < x < r - 0.10:
                wanted = -deg
            elif r - 0.10 <= x < r + 0.70:
                wanted = deg
        rocker_deg += float(np.clip(wanted - rocker_deg, -1.8, 1.8))
        for j, ai in zip(wheels, w_act):
            d.ctrl[ai] = float(np.clip(25 * (6.0 - d.qvel[m.jnt_dofadr[j]]), -50, 50))
        for j, ai in zip(steer, s_act):
            d.ctrl[ai] = float(np.clip(-20 * d.qpos[m.jnt_qposadr[j]] - d.qvel[m.jnt_dofadr[j]], -40, 40))
        for j, ai in zip(rockers, r_act):
            d.ctrl[ai] = float(np.clip(80 * (np.deg2rad(rocker_deg) - d.qpos[m.jnt_qposadr[j]])
                                       - 3 * d.qvel[m.jnt_dofadr[j]], -100, 100))
        next_ctrl += 0.02
    mujoco.mj_step(m, d)
    t += 0.005
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < worst[0]:
            worst = (float(c.dist), (gn[c.geom1], gn[c.geom2]), round(t, 2))

print(json.dumps({"worst_penetration_mm": round(-worst[0] * 1000, 1),
                  "pair": worst[1], "at_s": worst[2],
                  "final": [round(float(d.qpos[0]), 2), round(float(d.qpos[2]), 3)]}))
