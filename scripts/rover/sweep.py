import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height


def trial(kp, target, seconds=6.0):
    build_robot()
    scene = build_scene(0.001)  # flat: step height 0
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
    wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]
    d.qpos[:] = m.qpos0
    d.qpos[2] = standing_height(m, wb)
    mujoco.mj_forward(m, d)

    for t in range(int(seconds / 0.005)):
        for k, j in enumerate(wheels):
            d.ctrl[k] = float(np.clip(kp * (target - d.qvel[m.jnt_dofadr[j]]), -50, 50))
        mujoco.mj_step(m, d)
    w_mean = float(np.mean([d.qvel[m.jnt_dofadr[j]] for j in wheels]))
    return round(float(d.qpos[0]), 3), round(float(d.qvel[0]), 3), round(w_mean, 2), round(w_mean * 0.163, 2)


print(f"{'kp':>4} {'target':>7} {'x':>7} {'vx':>7} {'wheel_w':>8} {'wheel_v':>8}")
for kp in (10, 20, 40):
    for target in (1.5, 2.0, 3.0):
        x, vx, w, wv = trial(kp, target)
        print(f"{kp:>4} {target:>7} {x:>7} {vx:>7} {w:>8} {wv:>8}")
