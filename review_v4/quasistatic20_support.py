"""Isolated kinematic study; never steps or edits the physical model."""
from pathlib import Path
import json
import hashlib
import math
import mujoco
import numpy as np

ROOT = Path("/home/agilex/wzy/rover/review_v4")
SOURCE = ROOT / "high20_slow_cycle"
m = mujoco.MjModel.from_xml_path(str(SOURCE / "scene.xml"))
d = mujoco.MjData(m)
a = np.load(SOURCE / "trajectory.npz")
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(m.njnt)]
gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
rj = [i for i, n in enumerate(jn) if n.endswith("_rocker_joint")]
sj = [i for i, n in enumerate(jn) if n.endswith("_steer_joint")]
wh = [i for i, n in enumerate(gn) if "_wheel_col_1" in n]
roles = {role: [i for i, g in enumerate(wh) if role in gn[g]] for role in ["front", "middle", "rear"]}
front, mid, rear = [roles[role] for role in ["front", "middle", "rear"]]
indices = np.where(a["time"] > 12.0)[0]
ix = indices[np.argmin(a["qpos"][indices][:, m.jnt_qposadr[rj]].mean(axis=1))]
template = a["qpos"][ix].copy()
q0 = float(np.rad2deg(template[m.jnt_qposadr[rj]]).mean())
p0 = float(np.rad2deg(np.arcsin(2*(template[3]*template[5]-template[4]*template[6]))))
d.qpos[:] = template
mujoco.mj_kinematics(m, d)
edge = np.array([2.0, 0.6])
old_center = d.geom_xpos[wh][mid][:, [0, 2]].mean(axis=0)
alpha = np.arctan2(*(old_center-edge)[::-1])
radius = float(m.geom_size[np.array(wh)[mid], 0].mean())
boxes = [g for g, n in enumerate(gn) if n.startswith("up") or n == "platform"]
box_centers = m.geom_pos[boxes][:, [0, 2]]
box_sizes = m.geom_size[boxes][:, [0, 2]]
radii = m.geom_size[wh, 0]


def pose(x, z, pitch, q):
    d.qpos[:] = template
    d.qpos[:3] = [x, 0, z]
    d.qpos[3:7] = [np.cos(np.deg2rad(pitch)/2), 0, np.sin(np.deg2rad(pitch)/2), 0]
    d.qpos[m.jnt_qposadr[rj]] = np.deg2rad(q)
    d.qpos[m.jnt_qposadr[sj]] = 0
    mujoco.mj_kinematics(m, d)
    centers = d.geom_xpos[wh][:, [0, 2]].copy()
    delta = abs(centers[:, None, :] - box_centers[None, :, :]) - box_sizes[None, :, :]
    signed = np.linalg.norm(np.maximum(delta, 0), axis=2) + np.minimum(np.max(delta, axis=2), 0)
    distances = np.column_stack([centers[:, 1]-radii, signed-radii[:, None]])
    nearest = np.argmin(distances, axis=1)
    gaps = np.min(distances, axis=1)
    names = ["ground"] + [gn[g] for g in boxes]
    return centers, gaps, [names[int(i)] for i in nearest]


def validate(x, z, pitch, q):
    centers, gaps, nearest = pose(x, z, pitch, q)
    mujoco.mj_forward(m, d)
    nonwheel = []
    wheel_penetration = 0.0
    for contact in d.contact:
        names = [gn[int(contact.geom1)], gn[int(contact.geom2)]]
        if any("_wheel_col_" in name for name in names):
            wheel_penetration = max(wheel_penetration, -float(contact.dist))
        elif float(contact.dist) < 0:
            nonwheel.append({"pair": names, "penetration_m": -float(contact.dist)})
    return {
        "x": float(x), "z": float(z), "pitch_deg": float(pitch), "rocker_deg": float(q),
        "wheel_centers_xz": {gn[g]: centers[i].tolist() for i, g in enumerate(wh)},
        "wheel_gap_m": gaps.tolist(), "nearest_terrain": nearest,
        "nonwheel_contacts": nonwheel, "wheel_penetration_mujoco_m": wheel_penetration,
        "com_xyz": d.subtree_com[1].tolist(),
        "qpos": d.qpos.tolist(),
    }


def anchored(q, pitch):
    centers, _, _ = pose(0, 0, pitch, q)
    angle = alpha - np.deg2rad((pitch+q)-(p0+q0))
    anchor = edge + radius*np.array([np.cos(angle), np.sin(angle)])
    translation = anchor-centers[mid].mean(axis=0)
    centers, gaps, nearest = pose(float(translation[0]), float(translation[1]), pitch, q)
    return translation, centers, gaps


curve = []
for q in np.arange(-15.0, -4.9, 0.5):
    candidates = []
    for pitch in np.arange(-55.0, -29.99, 0.05):
        base, centers, gaps = anchored(float(q), float(pitch))
        if gaps.min() >= -0.001 and 1.3 < base[0] < 2.3:
            candidates.append((float(centers[rear, 0].mean()), float(pitch), base))
    found = None
    for _, pitch, base in sorted(candidates, reverse=True, key=lambda row: row[0]):
        trial = validate(float(base[0]), float(base[1]), pitch, float(q))
        if not trial["nonwheel_contacts"] and trial["wheel_penetration_mujoco_m"] <= 0.002:
            found = trial
            break
    if found is not None:
        curve.append(found)
        c = found["wheel_centers_xz"]
        print("ANCHOR", round(q, 2), round(found["pitch_deg"], 2), round(found["x"], 4),
              round(found["z"], 4), "rear", np.round(np.mean([c[gn[wh[i]]] for i in rear], axis=0), 4).tolist(),
              "frontgap", round(max(found["wheel_gap_m"][i] for i in front), 4), flush=True)
    else:
        print("ANCHOR", round(q, 2), "NONE", flush=True)

free = []
q_grid = np.arange(-89.0, 20.001, 0.5)
pitch_grid = np.arange(-60.0, 0.001, 0.5)


def grid_candidates(x, qs, pitches):
    local = np.array([pose(0.0, 0.0, 0.0, float(q))[0] for q in qs])
    ca = np.cos(np.deg2rad(pitches))[None, :, None]
    sa = np.sin(np.deg2rad(pitches))[None, :, None]
    xs = float(x) + ca*local[:, None, :, 0] + sa*local[:, None, :, 1]
    zs = -sa*local[:, None, :, 0] + ca*local[:, None, :, 1]
    minimum_z = np.broadcast_to(radii, xs.shape).copy()
    for center, size in zip(box_centers, box_sizes):
        dx = np.maximum(abs(xs-center[0])-size[0], 0)
        raised = center[1]+size[1]+np.sqrt(np.maximum(radii**2-dx**2, 0))
        minimum_z = np.maximum(minimum_z, np.where(dx <= radii, raised, -np.inf))
    lift = np.max(minimum_z-zs, axis=2)
    gaps = zs+lift[:, :, None]-minimum_z
    score = np.max(gaps[:, :, front+mid], axis=2)
    order = np.argsort(score, axis=None)[:600]
    valid = []
    for flat in order:
        qi, pi = np.unravel_index(flat, score.shape)
        z, pitch, q = float(lift[qi, pi]), float(pitches[pi]), float(qs[qi])
        if not .5 <= z <= 2.0:
            continue
        candidate = validate(float(x), z, pitch, q)
        if not candidate["nonwheel_contacts"] and candidate["wheel_penetration_mujoco_m"] <= .002:
            candidate["support_gap_max_m"] = float(max(candidate["wheel_gap_m"][i] for i in front+mid))
            candidate["vertical_support_gap_max_m"] = float(score[qi, pi])
            valid.append(candidate)
            if len(valid) >= 12:
                break
    return valid


for x in np.arange(1.4, 2.001, 0.05):
    candidates = grid_candidates(float(x), q_grid, pitch_grid)
    if candidates:
        coarse = min(candidates, key=lambda r: r["support_gap_max_m"])
        fine_q = np.arange(max(-89, coarse["rocker_deg"]-.6), min(20, coarse["rocker_deg"]+.6)+.001, .05)
        fine_p = np.arange(max(-60, coarse["pitch_deg"]-.6), min(0, coarse["pitch_deg"]+.6)+.001, .05)
        candidates += grid_candidates(float(x), fine_q, fine_p)
        best = min(candidates, key=lambda r: r["support_gap_max_m"])
        free.append(best)
        c = best["wheel_centers_xz"]
        print("FREE", round(float(x), 3), round(best["pitch_deg"], 2), round(best["rocker_deg"], 2),
              "supportgap", round(best["support_gap_max_m"], 4),
              "rear", np.round(np.mean([c[gn[wh[i]]] for i in rear], axis=0), 4).tolist(), flush=True)
    else:
        print("FREE", round(float(x), 3), "NONE", flush=True)

report = {
    "analysis_only": True, "dynamics_simulated": False, "source_scene": str(SOURCE/"scene.xml"),
    "source_scene_sha256": hashlib.sha256((SOURCE/"scene.xml").read_bytes()).hexdigest(),
    "source_trajectory": str(SOURCE/"trajectory.npz"), "source_time": float(a["time"][ix]),
    "source_pitch_deg": p0, "source_rocker_mean_deg": q0,
    "assumptions": ["Actual MuJoCo forward kinematics; no model edit and no mj_step.",
                    "Yaw/roll/steer fixed zero; symmetric rocker command; original wheel phases retained.",
                    "Wheel search uses circle-box cross-section distances; retained poses checked with MuJoCo collision.",
                    "Anchored middle tire rotates about up2 edge with fixed relative wheel joint angle.",
                    "No force, actuator limit or dynamic stability proof."],
    "anchored_curve": curve, "front_middle_support_search": free,
}
out = ROOT/"quasistatic20_support.json"
out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print("REPORT", out, flush=True)
