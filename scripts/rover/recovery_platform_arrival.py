"""Derive diagnostic arrival from saved states without changing raw results."""
from pathlib import Path
import hashlib

import mujoco
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assess_arrival(folder, result):
    # Original settled success is stronger than this display-only arrival test.
    if result['ascent_completed']:
        return {'platform_reached': True, 'platform_arrival': {
            'method': 'original_platform_settled_success', 'first_time_s': None}}
    folder = Path(folder)
    scene, trajectory = folder / 'scene.xml', folder / 'trajectory.npz'
    for path, field in [(scene, 'scene_sha256'), (trajectory, 'trajectory_sha256')]:
        if sha256(path) != result[field]:
            raise ValueError(f'Recorded evidence hash mismatch: {path}')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    platform = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'platform')
    wheels = [g for g in range(model.ngeom)
              if 'wheel_col' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
    if platform < 0 or len(wheels) != 6:
        raise ValueError(f'Expected one platform and six wheel collision geoms: {folder}')
    with np.load(trajectory, allow_pickle=False) as saved:
        times = saved['time'].reshape(-1)
        poses = saved['qpos']
        support = saved['platform_support']
    if len(poses) != len(times) or len(support) != len(times) - 1:
        raise ValueError(f'Unexpected saved-state alignment: {folder}')
    dt = float(np.median(np.diff(times)))
    stride = max(1, round(0.01 / dt))
    position, size = model.geom_pos[platform], model.geom_size[platform]
    top = float(position[2] + size[2])
    first = since = None
    last = None
    longest = 0.0
    touched = False
    for i in range(1, len(times), stride):
        data.qpos[:] = poses[i]
        mujoco.mj_kinematics(model, data)
        centers = data.geom_xpos[wheels]
        # Cylinder extent accounts for wheel/rocker/chassis orientation.
        axis_z = data.geom_xmat[wheels].reshape(-1, 3, 3)[:, 2, 2]
        extent_z = (model.geom_size[wheels, 0] * np.sqrt(np.maximum(0, 1-axis_z**2))
                    + model.geom_size[wheels, 1] * np.abs(axis_z))
        inside = bool(np.all(np.abs(centers[:, :2] - position[:2]) <= size[:2])
                      and np.all(centers[:, 2] - extent_z >= top - 0.02))
        if not inside:
            since, touched = None, False
            continue
        if since is None:
            since = float(times[i])
        touched = touched or bool(np.any(support[max(0, i-stride):i]))
        duration = float(times[i]) - since
        longest = max(longest, duration)
        if duration >= 0.1 - 1e-8 and touched:
            if first is None:
                first = since
            last = float(times[i])
    return {'platform_reached': first is not None, 'platform_arrival': {
        'method': 'saved_six_wheel_geometry_and_platform_contact_v1',
        'scope': 'display_progress_only; not settled_success_or_quality_certification',
        'first_time_s': first, 'last_confirmed_time_s': last,
        'longest_geometry_interval_s': round(longest, 6),
        'sample_interval_s': stride * dt, 'minimum_duration_s': 0.1,
        'wheel_bottom_tolerance_m': 0.02,
        'requires_platform_contact_in_interval': True}}
