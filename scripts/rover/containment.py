"""Direct geometric check: does any wheel centre ever enter a step box?

MuJoCo's contact.dist for a cylinder pressed on a box edge reports large overlap that
does not correspond to the visible mesh. Testing containment of the wheel centre inside
the box volume is unambiguous.
"""

import json

import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

build_robot()
scene = build_scene(0.15, 3, 0.5, 1.0)
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

start, depth, n_steps, h = 1.6, 0.5, 3, 0.15
boxes = [(start + i * depth, start + (i + 1) * depth, h * (i + 1)) for i in range(n_steps)]

d.qpos[:] = m.qpos0
d.qpos[2] = standing_height(m, wb)
mujoco.mj_forward(m, d)

worst_depth = 0.0
worst_info = None
rocker_deg, t, next_ctrl = 0.0, 0.0, 0.0

for _ in range(int(60 / 0.005)):
    if t >= next_ctrl:
        x = float(d.qpos[0])
        wanted = 0.0
        for i in range(n_steps):
            r = start + i * depth
            if r - 0.95 < x < r - 0.10:
                wanted = -12.0
            elif r - 0.10 <= x < r + 0.70:
                wanted = 12.0
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

    for b in wb:
        px, py, pz = d.xpos[b]
        for x0, x1, top in boxes:
            if x0 < px < x1 and abs(py) < 2.5 and 0 < pz < top:
                depth_in = min(px - x0, x1 - px, top - pz)
                if depth_in > worst_depth:
                    worst_depth = depth_in
                    worst_info = (bn[b], round(t, 2), [round(px, 3), round(pz, 3)])

print(json.dumps({
    "wheel_centre_inside_box_mm": round(worst_depth * 1000, 1),
    "detail": worst_info,
    "final_x": round(float(d.qpos[0]), 2),
    "final_z": round(float(d.qpos[2]), 3),
}))
