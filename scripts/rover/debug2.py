import mujoco
import numpy as np

from climb import build_robot, build_scene, standing_height

build_robot()
scene = build_scene(0.10)
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)

jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]

print("== wheel joint limits ==")
for j in wheels:
    print(f"  {jn[j]:28s} limited={m.jnt_limited[j]} range={m.jnt_range[j] if m.jnt_limited[j] else 'none'}")

print("== wheel geoms ==")
for g in range(m.ngeom):
    b = m.geom_bodyid[g]
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if bn and bn.endswith("_wheel"):
        print(f"  {bn:20s} type={m.geom_type[g]} size={np.round(m.geom_size[g], 3)} "
              f"quat={np.round(m.geom_quat[g], 3)} friction={np.round(m.geom_friction[g], 3)}")

print("== 离地空转测试 ==")
d.qpos[:] = m.qpos0
d.qpos[2] = 2.0
mujoco.mj_forward(m, d)
for k, j in enumerate(wheels):
    d.ctrl[k] = 10.0
for t in range(400):
    mujoco.mj_step(m, d)
print("  空转 2 秒后 qvel:", np.round([d.qvel[m.jnt_dofadr[j]] for j in wheels], 2))
print("  ncon:", d.ncon)

print("== 着地静摩擦测试 ==")
d2 = mujoco.MjData(m)
d2.qpos[:] = m.qpos0
d2.qpos[2] = standing_height(m, [i for i in range(m.nbody)
                                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("_wheel")])
mujoco.mj_forward(m, d2)
for k in range(6):
    d2.ctrl[k] = 50.0
for t in range(400):
    mujoco.mj_step(m, d2)
print("  满扭矩 2 秒后 x:", round(float(d2.qpos[0]), 3), "qvel:", np.round([d2.qvel[m.jnt_dofadr[j]] for j in wheels], 2))
