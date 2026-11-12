"""Render original recorded states from the URDF slope control experiment."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import argparse

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

import urdf_slope_control_20261112 as exp
import render_dataset


WIDTH = 480
HEIGHT = 360
FPS = 20


def render(case_id):
    folder = exp.RUN_ROOT / "cases" / case_id
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if exp.digest(folder / "trajectory.npz") != result["trajectory_sha256"]:
        raise RuntimeError("trajectory hash mismatch")
    with np.load(folder / "trajectory.npz", allow_pickle=False) as archive:
        times = archive["time"].reshape(-1).copy()
        poses = archive["qpos"].copy()
    metadata = {
        "audit": {"velocity_limit_source": {"path": str(exp.URDF)}},
        "urdf_sha256": exp.digest(exp.URDF),
    }
    model, option, visual = render_dataset._visual_render_model(
        folder / "scene.xml", folder, metadata
    )
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
    for camera, azimuth, elevation in zip(cameras, (90.0, 45.0), (-16.0, -22.0)):
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance, camera.azimuth, camera.elevation = 4.4, azimuth, elevation
    duration = float(times[-1] - times[0])
    requested = times[0] + np.arange(math.floor(duration * FPS + 1e-8) + 1) / FPS
    if abs(requested[-1] - times[-1]) > 1e-9:
        requested = np.r_[requested, times[-1]]
    selected = render_dataset._nearest(times, requested)
    sampling_error = float(np.max(np.abs(times[selected] - requested)))
    if selected[0] != 0 or selected[-1] != len(times) - 1 or sampling_error > exp.DT / 2 + 1e-8:
        raise RuntimeError("render sampling differs from saved timeline")
    exp.VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    path = exp.VIDEO_ROOT / f"{case_id}.mp4"
    temporary = exp.VIDEO_ROOT / f".{case_id}.tmp.mp4"
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH * 2}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "1",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for index in selected:
            data.time = float(times[index])
            data.qpos[:] = poses[index]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            frames = []
            for camera in cameras:
                camera.lookat[:] = data.qpos[:3]
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
    temporary.replace(path)
    report = {
        "id": case_id,
        "render_status": "complete",
        "video_path": str(path),
        "video_sha256": exp.digest(path),
        "render_source": "original_recorded_urdf_slope_control_states",
        "first_time_s": float(times[0]),
        "last_time_s": float(times[-1]),
        "duration_s": duration,
        "intervention_video_time_s": result.get("intervention_s"),
        "fps": FPS,
        "frames": len(selected),
        "max_sample_error_s": sampling_error,
        "trajectory_sha256": result["trajectory_sha256"],
        "scene_sha256": result["scene_sha256"],
        "visual_mode": visual["mode"],
        "render_scene_sha256": visual["render_scene_sha256"],
    }
    exp.write_json(folder / "media.json", report)
    print(json.dumps({"id": case_id, "video": str(path), "frames": len(selected)}))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    args = parser.parse_args()
    cases = [args.case] if args.case else sorted(
        path.parent.name for path in (exp.RUN_ROOT / "cases").glob("*/result.json")
    )
    for case_id in cases:
        if not (exp.RUN_ROOT / "cases" / case_id / "media.json").exists():
            render(case_id)


if __name__ == "__main__":
    main()
