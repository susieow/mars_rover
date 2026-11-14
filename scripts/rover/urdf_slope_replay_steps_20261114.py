"""Recompute the recorded step level from saved trajectories, trailing axle.

The old rule took the highest loaded wheel, which is the front axle. The table
now reports the rear axle so a failed climb reads how much of the staircase the
whole rover cleared. Dry run unless --write is given.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("/home/agilex/wzy/rover")
sys.path.insert(0, str(ROOT / "scripts/rover"))
import speed20_experiment as common  # noqa: E402
import urdf_slope_runtime_20261113 as pb  # noqa: E402

INDEX = ROOT / "docs/urdf_slope_index.jsonl"
WRITE = "--write" in sys.argv


def levels(m, d, c, terrain, rise):
    forces = pb.wheel_forces(m, d, c, terrain)
    rear, front = 0, 0
    for i, row in enumerate(c.wheels):
        if forces[i] <= pb.SUPPORT_N:
            continue
        geom = row[6]
        height = float(d.geom_xpos[geom, 2]) - float(m.geom_size[geom, 0])
        level = int(round(height / rise))
        if row[4] == 2:
            rear = max(rear, level)
        elif row[4] == 0:
            front = max(front, level)
    return rear, front


def resolve(row):
    """Phase D metadata JSONs live apart from the scene they were replayed from."""
    result_path = ROOT / row["result_path"]
    direct = result_path.with_name(result_path.stem)
    control = json.loads(result_path.read_text(encoding="utf-8")).get("control_id", {})
    base = (f'{row["friction"]}_{row["angle_deg"]:g}_{row["order"]}'
            f'_{control.get("rocker_amp_deg", 18.0):g}')
    suffix = f'_v{row["speed_rad_s"]:g}_{row["trigger"]}'
    folders = [direct.parent, ROOT / "runs/urdf_slope_20261112/pd/screen_replay"]
    stems = [direct, *(folder / (base + suffix) for folder in folders)]
    for stem in stems:
        scene = stem.with_name(stem.name + ".xml")
        traj = stem.with_name(stem.name + "_trajectory.npz")
        if scene.exists() and traj.exists():
            return scene, traj
    return None, None


rows = [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]
stats = Counter()
for row in rows:
    result_path = ROOT / row["result_path"]
    scene, traj = resolve(row)
    if scene is None:
        stats["missing"] += 1
        print("missing", row["result_path"])
        continue
    m = mujoco.MjModel.from_xml_path(str(scene))
    m.opt.timestep = 0.001
    d = mujoco.MjData(m)
    c = common.Controller(m, d)
    platform = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "platform")
    rise = 2.0 * float(m.geom_pos[platform, 2]) / row["steps_total"]
    terrain = {g for g in range(m.ngeom)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("up")
               or (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == "platform"}
    qpos = np.load(traj)["qpos"]
    d.qvel[:] = 0.0
    rear, front = 0, 0
    for k in range(len(qpos)):
        d.qpos[:] = qpos[k]
        mujoco.mj_forward(m, d)
        r, f = levels(m, d, c, terrain, rise)
        rear, front = max(rear, r), max(front, f)
    old = row["steps_climbed"]
    stats["matched" if front == old else "differs"] += 1
    if front != old:
        print(f"old {old:2d} replay front {front:2d} rear {rear:2d}  {scene.parent.name}/{scene.stem}")
    stats[f"rear-{rear}"] += 1
    row["steps_climbed"] = rear
    row["steps_climbed_front"] = front
    if WRITE:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["result"]["steps_climbed"] = rear
        data["result"]["steps_climbed_front"] = front
        data["result"]["steps_climbed_rule"] = "trailing_axle"
        result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(dict(stats))
if WRITE:
    INDEX.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                     encoding="utf-8")
    print("index rewritten")
