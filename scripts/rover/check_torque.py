"""Measure the force actually applied at the wheel joints under tau_max=120."""
import mujoco
from course import build_course
from climb import build_robot

build_robot()
m = mujoco.MjModel.from_xml_path(str(build_course(0.10, 12, 0.10, 4.0)))
d = mujoco.MjData(m)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
wj = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
print("jnt_actfrcrange:")
for j in wj:
    print(" ", jn[j], m.jnt_actfrcrange[j])
wa = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn[j] + "_act") for j in wj]
print("ctrlrange:", m.actuator_ctrlrange[wa[0]])
mujoco.mj_forward(m, d)
for j, ai in zip(wj, wa):
    d.ctrl[ai] = 120.0
for _ in range(50):
    mujoco.mj_step(m, d)
print("applied actuator_force after 50 steps:")
for j, ai in zip(wj, wa):
    print(" ", jn[j], "ctrl=%.0f force=%.1f qfrc=%.1f" % (
        d.ctrl[ai], d.actuator_force[ai], d.qfrc_actuator[m.jnt_dofadr[j]]))
