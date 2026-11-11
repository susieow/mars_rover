"""Render recorded rover states without running a controller or simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

WIDTH, HEIGHT, FPS = 1280, 720, 50
SHEET_CROP_TOP = 220
VIDEOS = Path("/home/agilex/wzy/videos")


def _resolve(value, folder):
    path = Path(value)
    return (path if path.is_absolute() else folder / path).resolve(strict=True)


def _nearest(times, requested):
    right = np.searchsorted(times, requested, side="left").clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(times[left] - requested) <= abs(times[right] - requested), left, right)


def _frame_times(start, end):
    result = start + np.arange(int(math.floor((end - start) * FPS + 1e-9)) + 1) / FPS
    if abs(float(result[-1]) - end) <= 1e-9:
        result[-1] = end
    else:
        result = np.append(result, end)
    return result


def _load_trajectory(path):
    with np.load(path, allow_pickle=False) as archive:
        control_time = archive["time"].copy()
        control_qpos = archive["qpos"].copy()
        control_qvel = archive["qvel"].copy()
        phase = archive["phase"].copy()
        simulation_dt = float(archive["simulation_dt"])
        scene_hash = str(archive["scene_sha256"].item())
        if bool(archive["physics_recorded"]):
            times = np.concatenate((control_time[:1], archive["physics_time"]))
            poses = np.concatenate((control_qpos[:1], archive["physics_qpos"]), axis=0)
            velocities = np.concatenate((control_qvel[:1], archive["physics_qvel"]), axis=0)
            source = "physics_qpos"
        else:
            times, poses, velocities, source = control_time, control_qpos, control_qvel, "qpos"
    if len(control_time) != len(phase) + 1 or len(times) < 2 or len(times) != len(poses):
        raise ValueError("trajectory state/action lengths are inconsistent")
    if not np.isfinite(simulation_dt) or simulation_dt <= 0:
        raise ValueError("simulation_dt must be positive and finite")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(poses)) or not np.all(np.isfinite(velocities)) or len(velocities) != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("trajectory has invalid timestamps or poses")
    if abs(float(times[-1]) - float(control_time[-1])) > 1e-10 or not np.array_equal(poses[-1], control_qpos[-1]):
        raise ValueError("physical samples do not contain the saved terminal state")
    return times, poses, velocities, control_time, phase, simulation_dt, scene_hash, source


def _orientation(poses, model):
    free_joints = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if not len(free_joints):
        raise ValueError("rover rendering requires a free base joint")
    offset = int(model.jnt_qposadr[free_joints[0]]) + 3
    w, x, y, z = poses[:, offset:offset + 4].T
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return pitch, roll


def _probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,duration:format=duration", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    if len(data.get("streams", [])) != 1:
        raise RuntimeError("encoded video must contain one video stream")
    stream = data["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "frames": int(stream["nb_frames"]), "fps": float(numerator) / float(denominator),
            "video_duration_s": float(stream.get("duration", data["format"]["duration"]))}


def _ppm(path, pixels):
    with path.open("wb") as stream:
        stream.write(f"P6\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode())
        stream.write(pixels.tobytes())


def _sheet(frames, captions, destination, title, temporary, columns=4, cell_width=320):
    image_height, heading = cell_width * 9 // 16, 48
    cell_height = image_height + 42
    font_size = 18 if cell_width >= 640 else 13
    sheet = np.full((heading + math.ceil(len(frames) / columns) * cell_height, columns * cell_width, 3), [17, 24, 32], dtype=np.uint8)
    labels = [(12, 12, title)]
    for index, (frame, caption) in enumerate(zip(frames, captions)):
        x, y = (index % columns) * cell_width, heading + (index // columns) * cell_height
        rows = np.linspace(0, frame.shape[0] - 1, image_height).astype(int)
        cols = np.linspace(0, frame.shape[1] - 1, cell_width).astype(int)
        sheet[y:y + image_height, x:x + cell_width] = frame[rows[:, None], cols]
        labels.extend([(x + 7, y + image_height + 3, caption[0]), (x + 7, y + image_height + 21, caption[1])])
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        source = temporary / "sheet.ppm"
        _ppm(source, sheet)
        filters = []
        for index, (x, y, label) in enumerate(labels):
            text_path = temporary / f"label_{index:02d}.txt"
            text_path.write_text(label, encoding="utf-8")
            font = f"fontfile={font_path}:" if font_path.is_file() else ""
            filters.append(f"drawtext={font}textfile={text_path}:expansion=none:fontcolor=white:fontsize={font_size}:x={x}:y={y}")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", ",".join(filters),
                        "-frames:v", "1", "-q:v", "2", "-update", "1", str(destination)],
                       capture_output=True, check=True)
        return "ffmpeg"
    image = Image.fromarray(sheet)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), font_size) if font_path.is_file() else ImageFont.load_default()
    for x, y, label in labels:
        draw.text((x, y), label, font=font, fill=(240, 244, 248))
    image.save(destination, quality=94)
    return "PIL"


def _visual_render_model(scene_path, folder, metadata):
    urdf_path = Path(metadata["audit"]["velocity_limit_source"]["path"])
    if hashlib.sha256(urdf_path.read_bytes()).hexdigest() != metadata["urdf_sha256"]:
        raise ValueError("URDF changed since trajectory recording")
    urdf = ET.parse(urdf_path).getroot()
    scene = ET.parse(scene_path)
    asset_nodes = {node.get("name"): node for node in scene.getroot().findall("./asset/mesh")}
    geometry = {}
    for body in scene.getroot().findall("./worldbody/body"):
        geometry.update({node.get("name"): node for node in body.iter("geom")})
    provenance_path = Path("/home/agilex/wzy/rover/render_assets/provenance.json")
    provenance = json.loads(provenance_path.read_text())
    exact_base = Path(provenance["output_path"])
    if hashlib.sha256(exact_base.read_bytes()).hexdigest() != provenance["output_sha256"]:
        raise ValueError("exact visual OBJ differs from conversion provenance")
    assets, restored = {}, []
    for link in urdf.findall("link"):
        name = "vis_" + link.get("name")
        visuals = link.findall("visual")
        if name not in geometry:
            if visuals:
                raise ValueError("missing URDF visual geometry: " + name)
            continue
        if len(visuals) != 1:
            raise ValueError("expected one URDF visual per rover link: " + name)
        visual = visuals[0]
        source = (urdf_path.parent / visual.find("geometry/mesh").get("filename")).resolve(strict=True)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        node = asset_nodes[geometry[name].get("mesh")]
        render_asset = source
        if name == "vis_base_link":
            if str(source) != provenance["source_path"] or source_hash != provenance["source_sha256"]:
                raise ValueError("exact visual OBJ is not derived from the original URDF mesh")
            render_asset = exact_base
            restored.append(name)
        node.set("file", str(render_asset))
        assets[str(render_asset)] = hashlib.sha256(render_asset.read_bytes()).hexdigest()
    collision_names = []
    for name, node in geometry.items():
        if name.startswith("vis_"):
            node.set("group", "1")
        else:
            node.set("group", "5")
            collision_names.append(name)
    render_xml = ET.tostring(scene.getroot(), encoding="unicode")
    render_scene = folder / "render_scene.xml"
    render_scene.write_text(render_xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(render_scene))
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    info = {
        "mode": "original_urdf_visual_meshes_only",
        "physics_scene_path": str(scene_path),
        "render_scene_path": str(render_scene),
        "render_scene_sha256": hashlib.sha256(render_scene.read_bytes()).hexdigest(),
        "urdf_path": str(urdf_path),
        "urdf_sha256": metadata["urdf_sha256"],
        "render_mesh_sha256": assets,
        "hidden_robot_collision_geometries": collision_names,
        "hidden_robot_collision_count": len(collision_names),
        "visual_geom_count": sum(name.startswith("vis_") for name in geometry),
        "lossless_restored_visuals": restored,
        "conversion_provenance": provenance,
        "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physics_trajectory_modified": False,
        "render_headlight": {"ambient": [0.35, 0.35, 0.35], "diffuse": [0.65, 0.65, 0.65]},
    }
    return model, option, info


def render_metadata(metadata_path, *, video_dir=VIDEOS, contact_sheet_only=False, camera_preview_only=False, review_sheets=False):
    if mujoco.get_mjcb_control() is not None:
        raise ValueError("offline rendering requires no global MuJoCo control callback")
    metadata_path = Path(metadata_path).resolve(strict=True)
    folder = metadata_path.parent
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("automated_checks_passed") is not True and not (contact_sheet_only or camera_preview_only):
        raise ValueError("refusing to render: automated_checks_passed must be true")
    case = str(metadata.get("case", folder.name))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", case):
        raise ValueError("case must be a safe ASCII filename component")
    scene_path = _resolve(metadata["scene_path"], folder)
    trajectory_path = _resolve(metadata["trajectory_path"], folder)
    times, poses, velocities, control_time, phases, simulation_dt, scene_hash, source = _load_trajectory(trajectory_path)
    if hashlib.sha256(scene_path.read_bytes()).hexdigest() != scene_hash:
        raise ValueError("scene XML differs from the scene associated with the recorded trajectory")
    requested = _frame_times(float(times[0]), float(times[-1]))
    selected = _nearest(times, requested)
    max_sampling_error = float(np.max(abs(times[selected] - requested)))
    if max_sampling_error > simulation_dt / 2 + 1e-9:
        raise ValueError("saved trajectory samples are too sparse for faithful 50 fps rendering")
    if selected[0] != 0 or selected[-1] != len(times) - 1:
        raise ValueError("video sampling must include the exact initial and terminal poses")
    model, scene_option, visual_info = _visual_render_model(scene_path, folder, metadata)
    if poses.shape[1] != model.nq or velocities.shape[1] != model.nv:
        raise ValueError("saved qpos/qvel dimensions differ from the scene")
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    pitch, roll = _orientation(poses, model)
    worst = metadata.get("audit", {}).get("worst_contact") or {}
    worst_time = worst.get("time_s")
    if worst_time is not None and (not math.isfinite(float(worst_time)) or not times[0] <= float(worst_time) <= times[-1] + 1e-9):
        raise ValueError("worst_contact time is outside the saved trajectory")
    sheet_indices = list(_nearest(times, np.linspace(times[0], times[-1], 12)))
    sheet_indices += [int(np.argmax(abs(pitch))), int(np.argmax(abs(roll))),
                      int(_nearest(times, float(worst_time))) if worst_time is not None else len(times) - 1,
                      len(times) - 1]
    roles = [f"uniform {i + 1}/12" for i in range(12)] + [
        f"max |pitch| {np.max(abs(pitch)):.2f} deg", f"max |roll| {np.max(abs(roll)):.2f} deg",
        f"worst contact {float(worst.get('penetration_m', 0)) * 1000:.2f} mm" if worst_time is not None else "worst contact: not recorded",
        "exact terminal state",
    ]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)]
    for suffix, values, addresses, label, unit in (
        ("_steer_joint", poses, model.jnt_qposadr, "max |steer angle|", "deg"),
        ("_steer_joint", velocities, model.jnt_dofadr, "max |steer speed|", "rad/s"),
        ("_rocker_joint", velocities, model.jnt_dofadr, "max |rocker speed|", "rad/s"),
    ):
        joints = [i for i, name in enumerate(joint_names) if name.endswith(suffix)]
        if not joints:
            continue
        samples = np.abs(values[:, [int(addresses[i]) for i in joints]])
        sample_index, _ = np.unravel_index(int(np.argmax(samples)), samples.shape)
        maximum = float(np.max(samples))
        if unit == "deg":
            maximum = math.degrees(maximum)
        sheet_indices.append(int(sample_index))
        roles.append(f"{label} {maximum:.2f} {unit}")
    robot_geoms = np.flatnonzero((model.geom_bodyid != 0) & (model.geom_group == 1))
    if not len(robot_geoms):
        raise ValueError("scene has no robot geometry to frame")
    side_camera, quarter_camera = mujoco.MjvCamera(), mujoco.MjvCamera()
    for camera in (side_camera, quarter_camera):
        mujoco.mjv_defaultFreeCamera(model, camera)
    side_camera.azimuth, side_camera.elevation = 90.0, -18.0
    quarter_camera.azimuth, quarter_camera.elevation = 135.0, -24.0
    video_dir = Path(video_dir).resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"v3_{case}.mp4"
    preview_path = folder / "contact_sheet.jpg"
    rendered = {}
    sheet_needed = set(sheet_indices)
    review_indices = []
    review_pages = []
    if review_sheets:
        review_times = times[0] + np.arange(int(math.floor((times[-1] - times[0]) / 0.5 + 1e-9)) + 1) * 0.5
        if abs(float(review_times[-1]) - float(times[-1])) <= 1e-9:
            review_times[-1] = times[-1]
        else:
            review_times = np.append(review_times, times[-1])
        review_indices = [int(index) for index in _nearest(times, review_times)]
        sheet_needed.update(review_indices)
    frames = len(selected) if contact_sheet_only else 0
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH // 2)

    def render_index(index):
        data.time = float(times[index])
        data.qpos[:] = poses[index]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        centers = data.geom_xpos[robot_geoms]
        radii = model.geom_rbound[robot_geoms]
        lower = np.min(centers - radii[:, None], axis=0)
        upper = np.max(centers + radii[:, None], axis=0)
        focus = (lower + upper) / 2
        side_camera.lookat[:] = focus
        quarter_camera.lookat[:] = focus
        side_camera.distance = max(4.0, float(upper[0] - lower[0]) * 1.35)
        quarter_camera.distance = max(4.2, float(upper[0] - lower[0]) * 1.45)
        renderer.update_scene(data, camera=side_camera, scene_option=scene_option)
        visible = {int(g.objid) for g in renderer.scene.geoms[:renderer.scene.ngeom]
                   if int(g.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)}
        if any(model.geom_bodyid[g] != 0 and model.geom_group[g] != 1 for g in visible):
            raise RuntimeError("robot collision geometry leaked into visual rendering")
        if not set(map(int, robot_geoms)).issubset(visible):
            raise RuntimeError("URDF visual geometry missing from rendered scene")
        side = renderer.render().copy()
        renderer.update_scene(data, camera=quarter_camera, scene_option=scene_option)
        quarter = renderer.render().copy()
        image = np.concatenate((side, quarter), axis=1)
        if index in sheet_needed:
            rendered[index] = side[SHEET_CROP_TOP:SHEET_CROP_TOP + 360].copy()
        return image

    if camera_preview_only:
        try:
            preview_indices = [int(np.argmax(abs(pitch))), sheet_indices[14], len(times) - 1]
            preview_frames, preview_captions = [], []
            for index in preview_indices:
                image = render_index(index)
                phase_index = int(np.searchsorted(control_time, times[index], side="right") - 1)
                phase = str(phases[min(max(phase_index, 0), len(phases) - 1)])
                for start, view in ((0, "SIDE azimuth 90"), (WIDTH // 2, "THREE-QUARTER azimuth 135")):
                    preview_frames.append(image[SHEET_CROP_TOP:SHEET_CROP_TOP + 360, start:start + WIDTH // 2])
                    preview_captions.append((f"t={times[index]:.3f}s | {phase}", view))
            with tempfile.TemporaryDirectory(prefix="render_dataset_preview_") as temporary_name:
                temporary = Path(temporary_name)
                temporary_preview = temporary / "camera_preview.jpg"
                _sheet(preview_frames, preview_captions, temporary_preview,
                       f"{case} | camera comparison | diagnostic only", temporary, columns=2, cell_width=640)
                preview = folder / "camera_preview.jpg"
                preview.write_bytes(temporary_preview.read_bytes())
            return {"camera_preview_path": str(preview), "sample_times_s": times[preview_indices].tolist(),
                    "video_reencoded": False, "metadata_changed": False, "visual_geometry": visual_info}
        finally:
            renderer.close()

    try:
        with tempfile.TemporaryDirectory(prefix="render_dataset_") as temporary_name:
            temporary = Path(temporary_name)
            partial_video = video_dir / f".{video_path.stem}.{os.getpid()}.tmp.mp4"
            if contact_sheet_only:
                partial_video = video_path
            else:
                with tempfile.TemporaryFile() as errors:
                    process = subprocess.Popen([
                        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(partial_video),
                    ], stdin=subprocess.PIPE, stderr=errors, stdout=subprocess.DEVNULL)
                    writing_error = None
                    try:
                        for index in selected:
                            image = render_index(int(index))
                            process.stdin.write(image.tobytes())
                            frames += 1
                    except Exception as error:
                        writing_error = error
                    finally:
                        try:
                            process.stdin.close()
                        except BrokenPipeError:
                            pass
                        return_code = process.wait()
                    errors.seek(0)
                    encoder_error = errors.read().decode("utf-8", errors="replace")
                    if return_code != 0:
                        raise RuntimeError(f"ffmpeg failed ({return_code}): {encoder_error.strip()}")
                    if writing_error is not None:
                        raise writing_error
            encoding = _probe(partial_video)
            if encoding["frames"] != frames or encoding["width"] != WIDTH or encoding["height"] != HEIGHT or abs(encoding["fps"] - FPS) > 1e-9:
                raise RuntimeError(f"encoded video properties do not match the rendered frames: {encoding}")
            if abs(encoding["video_duration_s"] - frames / FPS) > 1 / FPS:
                raise RuntimeError("encoded duration does not match the 50 fps frame count")
            for index in sheet_needed:
                if index not in rendered:
                    render_index(index)
            captions = []
            keyframes = []
            for index, role in zip(sheet_indices, roles):
                phase_index = int(np.searchsorted(control_time, times[index], side="right") - 1)
                phase = str(phases[min(max(phase_index, 0), len(phases) - 1)])
                captions.append((f"t={times[index]:.3f}s | {phase}", role))
                keyframes.append({"source_index": int(index), "time_s": float(times[index]), "phase": phase, "role": role})
            partial_preview = temporary / "contact_sheet.jpg"
            sheet_backend = _sheet([rendered[index] for index in sheet_indices], captions,
                                   partial_preview, f"{case} | recorded states | physical time", temporary)
            for page_index, start in enumerate(range(0, len(review_indices), 20), start=1):
                indices = review_indices[start:start + 20]
                page_captions = []
                for ordinal, index in enumerate(indices, start=start + 1):
                    phase_index = int(np.searchsorted(control_time, times[index], side="right") - 1)
                    phase = str(phases[min(max(phase_index, 0), len(phases) - 1)])
                    page_captions.append((f"t={times[index]:.3f}s | {phase}", f"sequence {ordinal}/{len(review_indices)} | 0.5s sampling"))
                partial_page = temporary / f"review_{page_index:02d}.jpg"
                _sheet([rendered[index] for index in indices], page_captions, partial_page,
                       f"{case} | continuous sampling | page {page_index} | side view", temporary)
                page_path = folder / partial_page.name
                review_pages.append({"path": str(page_path), "frames": len(indices),
                                     "source_times_s": times[indices].tolist(), "temporary": partial_page})
            latest = json.loads(metadata_path.read_text(encoding="utf-8"))
            if latest.get("automated_checks_passed") is not True and not contact_sheet_only:
                raise ValueError("automated checks were revoked during rendering; metadata was not updated")
            if _resolve(latest["scene_path"], folder) != scene_path or _resolve(latest["trajectory_path"], folder) != trajectory_path:
                raise ValueError("metadata scene or trajectory changed during rendering")
            if not contact_sheet_only:
                os.replace(partial_video, video_path)
            preview_bytes = partial_preview.read_bytes()
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".jpg", delete=False) as stream:
                preview_temporary = Path(stream.name)
                stream.write(preview_bytes)
            os.replace(preview_temporary, preview_path)
            for page in review_pages:
                with tempfile.NamedTemporaryFile(dir=folder, suffix=".jpg", delete=False) as stream:
                    page_temporary = Path(stream.name)
                    stream.write(page.pop("temporary").read_bytes())
                os.replace(page_temporary, page["path"])
            render_info = {
                **encoding, "fps": FPS, "visual_geometry": visual_info, "source_duration_s": float(times[-1] - times[0]),
                "source_start_time_s": float(times[0]), "source_end_time_s": float(times[-1]),
                "sampling": "nearest recorded physical state, no interpolation; exact initial and terminal poses",
                "source_field": source, "max_sampling_error_s": max_sampling_error,
                "frame_source_times_s": times[selected].tolist(),
                "initial_frame_included": True, "terminal_frame_included": True,
                "camera": {"layout": "side and three-quarter, 640x720 each",
                           "side_azimuth_deg": 90.0, "side_elevation_deg": -18.0,
                           "quarter_azimuth_deg": 135.0, "quarter_elevation_deg": -24.0,
                           "tracking": "robot-centered near view", "contact_sheet": "side view center crop"},
                "contact_sheet_backend": sheet_backend, "contact_sheet_keyframes": keyframes,
                "contact_sheet_diagnostic_only": latest.get("automated_checks_passed") is not True,
                "ffmpeg_returncode": 0, "controller_executed": False, "simulation_stepped": False,
            }
            if review_sheets:
                render_info.update(review_sample_interval_s=0.5, review_sheets=review_pages,
                                   review_sampling_note="Half-second side-view snapshots plus exact terminal state; not a full-frame video audit.")
            if contact_sheet_only:
                existing_render = dict(latest.get("render", {}))
                for key in ("contact_sheet_backend", "contact_sheet_keyframes", "contact_sheet_diagnostic_only"):
                    existing_render[key] = render_info[key]
                if review_sheets:
                    for key in ("review_sample_interval_s", "review_sheets", "review_sampling_note"):
                        existing_render[key] = render_info[key]
                existing_render["contact_sheet_camera"] = render_info["camera"]
                render_info = existing_render
            latest.update(out=str(video_path), preview_path=str(preview_path), render=render_info)
            with tempfile.NamedTemporaryFile(dir=folder, mode="w", suffix=".json", encoding="utf-8", delete=False) as stream:
                metadata_temporary = Path(stream.name)
                json.dump(latest, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(metadata_temporary, metadata_path)
    finally:
        renderer.close()
    return {"metadata_path": str(metadata_path), "out": str(video_path), "preview_path": str(preview_path),
            "frames": frames, "fps": FPS, "source_duration_s": float(times[-1] - times[0]),
            "video_duration_s": encoding["video_duration_s"], "keyframes": len(sheet_indices),
            "review_paths": [page["path"] for page in review_pages]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path, help="episode metadata.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--contact-sheet-only", action="store_true", help="refresh diagnostic keyframes without encoding the existing video")
    mode.add_argument("--camera-preview-only", action="store_true", help="render a small camera comparison without changing video or metadata")
    parser.add_argument("--review-sheets", action="store_true", help="save side-view snapshots every 0.5 seconds in pages of at most 20 frames")
    arguments = parser.parse_args()
    print(json.dumps(render_metadata(arguments.metadata, contact_sheet_only=arguments.contact_sheet_only,
                                     camera_preview_only=arguments.camera_preview_only,
                                     review_sheets=arguments.review_sheets), ensure_ascii=False), flush=True)
