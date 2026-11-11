import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("/home/agilex/wzy/rover/assets/mars_rover/rover_actuated.xml")
d = mujoco.MjData(m)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
rockers = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
base = bn.index("base_link")
front = bn.index("left_front_wheel")
mid = bn.index("left_middle_wheel")
rear = bn.index("left_rear_wheel")

print(f"{'rocker':>8} {'front-base':>11} {'mid-base':>10} {'rear-base':>10}")
for deg in (-20, -10, 0, 10, 20):
    d.qpos[:] = m.qpos0
    d.qpos[2] = 2.0
    for j in rockers:
        d.qpos[m.jnt_qposadr[j]] = np.deg2rad(deg)
    mujoco.mj_kinematics(m, d)
    print(f"{deg:>8} {d.xpos[front][2] - d.xpos[base][2]:>11.3f} "
          f"{d.xpos[mid][2] - d.xpos[base][2]:>10.3f} "
          f"{d.xpos[rear][2] - d.xpos[base][2]:>10.3f}")
