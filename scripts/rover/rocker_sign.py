import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

build_robot()
scene = build_scene(0.001, 1, 0.5, 1.0)
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
rockers = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
wheels = {n: i for i, n in enumerate(bn) if n.endswith("_wheel")}
wb = list(wheels.values())
act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")  # noqa: E731
r_act = [act(jn[j]) for j in rockers]

print(f"{'rocker':>8} {'L_front':>8} {'L_mid':>8} {'L_rear':>8} {'pitch':>7}")
for deg in (-20, -10, 0, 10, 20):
    d.qpos[:] = m.qpos0
    d.qpos[2] = standing_height(m, wb)
    mujoco.mj_forward(m, d)
    target = np.deg2rad(deg)
    for _ in range(2000):
        for j, ai in zip(rockers, r_act):
            q = d.qpos[m.jnt_qposadr[j]]
            d.ctrl[ai] = float(np.clip(80.0 * (target - q) - 3.0 * d.qvel[m.jnt_dofadr[j]], -100, 100))
        mujoco.mj_step(m, d)
    pitch = float(np.degrees(np.arcsin(np.clip(2 * (d.qpos[3] * d.qpos[5] - d.qpos[6] * d.qpos[4]), -1, 1))))
    print(f"{deg:>8} {d.xpos[wheels['left_front_wheel']][2]:>8.3f} "
          f"{d.xpos[wheels['left_middle_wheel']][2]:>8.3f} "
          f"{d.xpos[wheels['left_rear_wheel']][2]:>8.3f} {pitch:>7.1f}")
