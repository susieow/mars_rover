"""Publish recorded-state video and extrema-preserving curves for the recovery batch."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np

import recovery20_experiment as exp
import render_dataset

MEDIA = Path('/home/agilex/wzy/videos/review_high_stairs/recovery20_t22_20261111')
WIDTH, HEIGHT, FPS = 480, 360, 20


def render(tag):
    folder = exp.RUN / 'cases' / tag
    result = json.loads((folder / 'result.json').read_text())
    existing = folder / 'media.json'
    if existing.exists():
        saved = json.loads(existing.read_text())
        if exp.common.digest(saved['video_path']) != saved['video_sha256']:
            raise RuntimeError('existing video differs from manifest')
        return
    if exp.common.digest(folder / 'trajectory.npz') != result['trajectory_sha256']:
        raise RuntimeError('continuation trajectory hash mismatch')
    parent_path = exp.PARENTS / result['parent_id'] / 'trajectory.npz'
    if exp.common.digest(parent_path) != result['parent_trajectory_sha256']:
        raise RuntimeError('parent trajectory hash mismatch')
    with np.load(parent_path, allow_pickle=False) as z:
        start, end = round(17 / result['physics_step_s']), result['parent_step']
        prefix_t, prefix_q = z['time'][start:end], z['qpos'][start:end]
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as z:
        times, poses = np.r_[prefix_t, z['time']], np.concatenate([prefix_q, z['qpos']])
    meta = {'audit': {'velocity_limit_source': {'path': str(exp.common.URDF)}}, 'urdf_sha256': exp.common.digest(exp.common.URDF)}
    model, option, visual = render_dataset._visual_render_model(exp.SCENE, folder, meta)
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    data = mujoco.MjData(model)
    cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
    for camera, azimuth, elevation in zip(cameras, (90., 45.), (-17., -22.)):
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance, camera.azimuth, camera.elevation = 4.8, azimuth, elevation
    duration = float(times[-1] - times[0])
    requested = times[0] + np.arange(math.floor(duration * FPS + 1e-8) + 1) / FPS
    if abs(requested[-1] - times[-1]) > 1e-9:
        requested = np.r_[requested, times[-1]]
    selected = render_dataset._nearest(times, requested)
    error = float(np.max(abs(times[selected] - requested)))
    if selected[0] != 0 or selected[-1] != len(times) - 1 or error > result['physics_step_s'] / 2 + 1e-8:
        raise RuntimeError('render sampling differs from saved state timeline')
    MEDIA.mkdir(parents=True, exist_ok=True)
    path = MEDIA / f'{tag}.mp4'; temporary = MEDIA / f'.{tag}.tmp.mp4'
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    process = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{WIDTH * 2}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'ultrafast', '-threads', '1', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in selected:
            data.time = times[index]; data.qpos[:] = poses[index]; data.qvel[:] = 0.
            mujoco.mj_forward(model, data)
            frames = []
            for camera in cameras:
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera, scene_option=option)
                frames.append(renderer.render().copy())
            process.stdin.write(np.concatenate(frames, axis=1).tobytes())
    finally:
        process.stdin.close(); errors = process.stderr.read().decode('utf-8', errors='replace')
        code = process.wait(); renderer.close()
    if code:
        raise RuntimeError(f'ffmpeg failed: {errors}')
    info = render_dataset._probe(temporary)
    if info['frames'] != len(selected) or info['width'] != 960 or info['height'] != 360:
        raise RuntimeError('media dimensions or frame count mismatch')
    temporary.replace(path)
    report = {'id': tag, 'render_status': 'complete', 'video_path': str(path), 'video_sha256': exp.common.digest(path),
        'render_source': 'original_recorded_parent_and_continuation_states',
        'first_time_s': float(times[0]), 'last_time_s': float(times[-1]), 'duration_s': duration,
        'intervention_video_time_s': float(22 - times[0]), 'fps': FPS, 'frames': len(selected),
        'max_sample_error_s': error, 'parent_trajectory_sha256': result['parent_trajectory_sha256'],
        'trajectory_sha256': result['trajectory_sha256'], 'scene_sha256': result['scene_sha256'],
        'visual_mode': visual['mode'], 'render_scene_sha256': visual['render_scene_sha256']}
    exp.write(existing, report)


def curves(tag):
    folder = exp.RUN / 'cases' / tag
    if (folder / 'curves.json').exists():
        return
    r = json.loads((folder / 'result.json').read_text())
    m, _, c, _, _, _ = exp.restore(r['physics_step_s'])
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as z:
        times, pos, vel, tau = z['time'], z['qpos'], z['qvel'], z['qfrc_actuator']
        wheels, rockers, contact = z['wheel_targets'], z['rocker_targets_rad'], z['contact_force_n']
        n, dt = len(tau), r['physics_step_s']; sub = round(.001 / dt)
        control_i = np.arange(n) // sub * sub
        ticks = np.arange(0, n, sub)
        qw, qx, qy, qz = pos[:, 3:7].T
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
        steer = np.empty(len(ticks)); previous, rate = c.yaw_previous, c.yaw_rate
        for i, j in enumerate(ticks):
            if previous is not None:
                rate = .65 * rate + .35 * exp.common.wrap(yaw[j] - previous) / .001
            previous = yaw[j]
            heading = np.clip(-math.atan(1.5 * pos[j, 1] + .3 * vel[j, 1]), -.35, .35)
            error = exp.common.wrap(heading - yaw[j])
            steer[i] = np.clip(.45 * error - .1 * rate, -.15, .15)
        out = {'id': tag, 'first_time_s': float(times[0]), 'last_time_s': float(times[-1]), 'sample_bin_s': .05,
            'note': 'Actual states at step boundaries; actuator force and targets refer to the preceding step. Per-bin velocity, torque and angle extrema are retained.',
            'joints': {}, 'body': {}}
        bins = range(0, n, max(1, round(.05 / dt)))
        columns = ['time_s', 'actual_rad_s', 'target_rad_s', 'torque_nm', 'angle_deg', 'target_deg', 'contact_force_n']
        for name, limit in r['audit']['joint_velocity_limit_rad_s'].items():
            joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            qp, vp = int(m.jnt_qposadr[joint]), int(m.jnt_dofadr[joint])
            speed, torque, angle = vel[1:, vp], tau[:, vp], np.degrees(pos[1:, qp])
            selected = [0, n - 1]
            for lo in bins:
                hi = min(n, lo + max(1, round(.05 / dt)))
                selected.extend([lo, hi - 1])
                for values in (speed, torque, angle):
                    selected.extend([lo + int(np.argmin(values[lo:hi])), lo + int(np.argmax(values[lo:hi]))])
            ix = np.unique(selected)
            angle_target = np.full(n, np.nan); force = np.zeros(n)
            if name.endswith('_wheel_joint'):
                wi = r['wheel_names'].index(name); target = wheels[:, wi]; force = contact[:, wi]
                force_limit = 50.
            elif name.endswith('_rocker_joint'):
                ri = next(i for i, row in enumerate(c.rockers) if row[0] == joint)
                angle_target = np.degrees(rockers[:, ri])
                target = np.clip(4 * (rockers[:, ri] - pos[control_i, qp]), -.7, .7)
                force_limit = 100.
            else:
                reference = steer[np.arange(n) // sub] * (1 if 'front' in name else -1)
                angle_target = np.degrees(reference)
                target = np.clip(8 * (reference - pos[control_i, qp]), -1.2, 1.2)
                force_limit = 40.
            rows = np.column_stack((times[ix + 1], speed[ix], target[ix], torque[ix], angle[ix], angle_target[ix], force[ix]))
            clean = [[None if not math.isfinite(v) else round(float(v), 7) for v in row] for row in rows]
            out['joints'][name] = {'columns': columns, 'rows': clean, 'speed_limit': limit, 'audit_speed_threshold': limit + .01,
                'torque_limit': force_limit, 'max_abs_velocity_rad_s': float(np.max(abs(speed))),
                'max_abs_torque_nm': float(np.max(abs(torque))), 'mean_velocity_rad_s': float(np.mean(speed)),
                'torque_saturation_fraction': float(np.mean(abs(torque) >= force_limit - .01))}
        ix = np.unique(np.r_[np.arange(0, n + 1, max(1, round(.01 / dt))), n])
        out['body'] = {'columns': ['time_s', 'progress_x_m', 'roll_deg', 'pitch_deg', 'yaw_deg'],
            'rows': np.round(np.column_stack((times[ix], pos[ix, 0] - pos[0, 0], roll[ix], pitch[ix], np.degrees(yaw[ix]))), 7).tolist()}
    exp.write(folder / 'curves.json', out)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--case'); parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        files = [exp.RUN / 'cases' / args.case / 'result.json'] if args.case else sorted((exp.RUN / 'cases').glob('*/result.json'))
        for f in files:
            tag = f.parent.name
            if (f.parent / 'media.json').exists() and (f.parent / 'curves.json').exists():
                continue
            curves(tag); render(tag)
            print(json.dumps({'id': tag, 'media': 'complete'}), flush=True)
        if not args.watch:
            break
        done = exp.RUN / 'experiment_complete.json'
        if done.exists():
            count = json.loads(done.read_text())['completed']
            if len(list((exp.RUN / 'cases').glob('*/media.json'))) == count:
                break
        time.sleep(3)


if __name__ == '__main__':
    main()
