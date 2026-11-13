"""Render recorded experiment-two states to mp4 so every row can be judged by eye.

One process per shard keeps each EGL context in its own process; forking a
driver context across workers is fragile on this host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

import render_dataset

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/urdf_slope_20261112"
MEDIA = RUNS / "media"
URDF = ROOT / "assets/mars_rover/urdf/mars_rover.urdf"
WIDTH, HEIGHT, FPS = 480, 360, 20
FOLDERS = ("pb", "pb_v2", "pd/screen_replay", "pd/confirm")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    out = []
    for folder in FOLDERS:
        for path in sorted((RUNS / folder).glob("*_trajectory.npz")):
            stem = path.name[:-len("_trajectory.npz")]
            if (path.parent / f"{stem}.xml").is_file():
                out.append((folder, stem))
    return out


def media_key(folder, stem):
    return f"{folder}/{stem}".replace("/", "__")


def render(folder, stem):
    trajectory = RUNS / folder / f"{stem}_trajectory.npz"
    scene = RUNS / folder / f"{stem}.xml"
    with np.load(trajectory, allow_pickle=False) as archive:
        times = archive["time"].reshape(-1).copy()
        poses = archive["qpos"].copy()
        dt = float(archive["simulation_dt"])
    metadata = {
        "audit": {"velocity_limit_source": {"path": str(URDF)}},
        "urdf_sha256": digest(URDF),
    }
    model, option, visual = render_dataset._visual_render_model(scene, scene.parent, metadata)
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
    for camera, azimuth, elevation in zip(cameras, (90.0, 30.0), (-14.0, -12.0)):
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance, camera.azimuth, camera.elevation = (4.2 if azimuth == 90.0 else 3.6), azimuth, elevation
    duration = float(times[-1] - times[0])
    requested = times[0] + np.arange(math.floor(duration * FPS + 1e-8) + 1) / FPS
    if times[-1] - requested[-1] > 1e-9:
        requested = np.r_[requested, times[-1]]
    selected = render_dataset._nearest(times, requested)
    sampling_error = float(np.max(np.abs(times[selected] - requested)))
    if selected[0] != 0 or selected[-1] != len(times) - 1 or sampling_error > max(dt, float(np.median(np.diff(times)))) / 2 + 1e-9:
        raise RuntimeError("render sampling differs from saved timeline")

    MEDIA.mkdir(parents=True, exist_ok=True)
    key = media_key(folder, stem)
    out = MEDIA / f"{key}.mp4"
    temporary = MEDIA / f".{key}.tmp.mp4"
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH * 2}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
         "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", "-crf", "26",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in selected:
            data.time = float(times[index])
            data.qpos[:] = poses[index]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            frames = []
            for camera in cameras:
                camera.lookat[:] = data.qpos[:3] + np.array([0.8, 0.0, 0.2])
                renderer.update_scene(data, camera=camera, scene_option=option)
                frames.append(renderer.render().copy())
            process.stdin.write(np.concatenate(frames, axis=1).tobytes())
    finally:
        process.stdin.close()
        errors = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
        renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg failed: {errors}")
    info = render_dataset._probe(temporary)
    if info["frames"] != len(selected) or info["width"] != WIDTH * 2 or info["height"] != HEIGHT:
        raise RuntimeError("media dimensions or frame count mismatch")
    temporary.replace(out)
    report = {
        "media_key": key, "phase_folder": folder, "stem": stem,
        "video_path": str(out.relative_to(ROOT)),
        "video_sha256": digest(out),
        "fps": FPS, "frames": len(selected), "duration_s": duration,
        "max_sample_error_s": sampling_error,
        "trajectory_sha256": digest(trajectory),
        "render_scene_sha256": visual["render_scene_sha256"],
    }
    (MEDIA / f"{key}.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pending = []
    for index, (folder, stem) in enumerate(cases()):
        if index % args.shards != args.shard:
            continue
        if not args.force and (MEDIA / f"{media_key(folder, stem)}.mp4").exists():
            continue
        pending.append((folder, stem))
    print(json.dumps({"shard": args.shard, "shards": args.shards, "pending": len(pending)}), flush=True)
    for folder, stem in pending:
        try:
            report = render(folder, stem)
            print(json.dumps({"ok": report["media_key"], "frames": report["frames"]}), flush=True)
        except Exception as error:  # keep the shard alive so one bad run does not cost the rest
            print(json.dumps({"fail": f"{folder}/{stem}", "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
