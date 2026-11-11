"""Log wheel-terrain contact forces during the rocker-assist climb."""

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
gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
wheels = [i for i, n in enumerate(jn) if n.endswith("_wheel_joint")]
steer = [i for i, n in enumerate(jn) if n.endswith("_steer_joint")]
rockers = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
wb = [i for i, n in enumerate(bn) if n.endswith("_wheel")]
act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")  # noqa: E731
w_act = [act(jn[j]) for j in wheels]
s_act = [act(jn[j]) for j in steer]
r_act = [act(jn[j]) for j in rockers]
wheel_g = {g for g, n in enumerate(gn) if n and "_wheel" in n and "step" not in n}
step_g = {g for g, n in enumerate(gn) if n and (n.startswith("step") or n == "platform")}

d.qpos[:] = m.qpos0
d.qpos[2] = standing_height(m, wb)
mujoco.mj_forward(m, d)

start, depth, n_steps = 1.6, 0.5, 3
rocker_deg, t, next_ctrl = 0.0, 0.0, 0.0
events = []
worst = {"normal": 0.0, "ratio": 0.0, "dist_mm": 0.0}
force6 = np.zeros(6)

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

    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ((g1 in wheel_g and g2 in step_g) or (g2 in wheel_g and g1 in step_g)):
            continue
        mujoco.mj_contactForce(m, d, i, force6)
        normal = abs(float(force6[0]))
        tangent = float(np.linalg.norm(force6[1:3]))
        ratio = tangent / normal if normal > 1e-9 else 0.0
        worst["normal"] = max(worst["normal"], normal)
        worst["ratio"] = max(worst["ratio"], ratio)
        worst["dist_mm"] = min(worst["dist_mm"], float(c.dist) * 1000)
        if len(events) < 4000:
            events.append({
                "t": round(t, 2),
                "geom": gn[g1 if g1 in wheel_g else g2],
                "step": gn[g2 if g1 in wheel_g else g1],
                "fn": round(normal, 1),
                "ft": round(tangent, 1),
                "ratio": round(ratio, 2),
                "dist_mm": round(float(c.dist) * 1000, 2),
            })

peak = sorted(events, key=lambda e: -e["fn"])[:5]
slip = [e for e in events if e["ratio"] > 0.8][:5]
print(json.dumps({
    "samples": len(events),
    "max_normal_N": round(worst["normal"], 1),
    "max_tangent_ratio": round(worst["ratio"], 2),
    "min_dist_mm": round(worst["dist_mm"], 2),
    "peak_normal_events": peak,
    "slip_examples": slip,
}, indent=1))
with open("/home/agilex/wzy/videos/contact_forces.json", "w") as fh:
    json.dump(events, fh)
