"""Recompute physical quality and course completion from saved controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dataset_io import replay_check
from episode_audit import EpisodeAudit


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SOURCES = {"course.py", "closed_loop_controller.py", "episode_audit.py",
                    "dataset_io.py", "generate_dataset.py", "verify_dataset.py", "climb.py"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mesh_hashes(scene_path):
    scene_path = Path(scene_path).resolve(strict=True)
    xml = ET.parse(scene_path).getroot()
    if xml.findall(".//include"):
        raise ValueError("mesh provenance requires a self-contained scene XML without includes")
    compiler = xml.find("compiler")
    options = {} if compiler is None else compiler.attrib
    meshdir = Path(options.get("meshdir", options.get("assetdir", ".")))
    if not meshdir.is_absolute():
        meshdir = scene_path.parent / meshdir
    paths = set()
    for mesh in xml.findall("./asset/mesh"):
        filename = mesh.get("file")
        if filename is None:
            continue
        path = Path(filename)
        if options.get("strippath", "false").lower() == "true":
            path = Path(path.name)
        if not path.is_absolute():
            path = meshdir / path
        paths.add(path.resolve(strict=True))
    return {str(path): sha256(path) for path in sorted(paths)}


def check_meshes(metadata, scene_path):
    expected = metadata.get("mesh_sha256")
    if not isinstance(expected, dict):
        raise ValueError("mesh_sha256 provenance is missing")
    actual = mesh_hashes(scene_path)
    if set(expected) != set(actual):
        raise ValueError("scene mesh files differ from mesh_sha256 provenance")
    for path, digest in actual.items():
        if expected[path] != digest:
            raise ValueError(f"mesh content hash mismatch: {path}")
    return actual


def check_sources(metadata, episode):
    manifest = metadata.get("source_sha256")
    if not isinstance(manifest, dict) or not REQUIRED_SOURCES.issubset(manifest):
        raise ValueError("source_sha256 is missing required source snapshots")
    if metadata.get("source_sha256_end") != manifest:
        raise ValueError("source files changed during generation or end hashes are missing")
    if metadata.get("source_loaded_sha256") != manifest:
        raise ValueError("saved source snapshots differ from the generator process imports")
    checked = {}
    for name, expected in manifest.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("source_sha256 contains an invalid filename")
        actual = sha256(episode / "sources" / name)
        if actual != expected:
            raise ValueError(f"saved source hash mismatch: {name}")
        checked[name] = actual
    return checked


def inspect_course(model):
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g): g
             for g in range(model.ngeom) if int(model.geom_bodyid[g]) == 0}
    expected = {"ground", "platform", *(f"up{i}" for i in range(12)),
                *(f"down{i}" for i in range(11))}
    if set(names) != expected:
        raise ValueError("course must contain only ground, 12 ascending boxes, platform and 11 descending boxes")
    boxes = {}
    for name, g in names.items():
        expected_type = mujoco.mjtGeom.mjGEOM_PLANE if name == "ground" else mujoco.mjtGeom.mjGEOM_BOX
        if int(model.geom_type[g]) != int(expected_type):
            raise ValueError(f"course geometry type is invalid: {name}")
        if not np.allclose(model.geom_quat[g], [1, 0, 0, 0], atol=1e-10, rtol=0):
            raise ValueError(f"course geometry is rotated or chamfered: {name}")
        if int(model.geom_contype[g]) != 2 or int(model.geom_conaffinity[g]) != 1:
            raise ValueError(f"course collision mask is invalid: {name}")
        if float(model.geom_margin[g]) != 0 or float(model.geom_gap[g]) != 0:
            raise ValueError(f"course contact margin or gap is nonzero: {name}")
        if name == "ground":
            if not np.allclose(model.geom_pos[g], 0, atol=1e-10, rtol=0):
                raise ValueError("ground must be a horizontal plane at z=0")
            continue
        p, s = model.geom_pos[g], model.geom_size[g]
        if np.any(s <= 0) or abs(float(p[1])) > 1e-10 or abs(float(p[2] - s[2])) > 1e-10:
            raise ValueError(f"course box is not centered on the corridor or grounded: {name}")
        boxes[name] = {"id": g, "left": float(p[0] - s[0]), "right": float(p[0] + s[0]),
                       "top": float(p[2] + s[2]), "half_width": float(s[1])}
    first, platform = boxes["up0"], boxes["platform"]
    height, depth = first["top"], first["right"] - first["left"]
    if not math.isclose(height, depth, abs_tol=1e-9, rel_tol=0):
        raise ValueError("stair slope is not 45 degrees (rise differs from run)")
    previous_right = first["left"]
    for name in [*(f"up{i}" for i in range(12)), "platform", *(f"down{i}" for i in range(11))]:
        box = boxes[name]
        top = (int(name[2:]) + 1) * height if name.startswith("up") else (
            (11 - int(name[4:])) * height if name.startswith("down") else 12 * height)
        if not math.isclose(box["left"], previous_right, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"stair course has an overlap or gap: {name}")
        if not math.isclose(box["top"], top, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"stair course has an incorrect riser height: {name}")
        if name != "platform" and not math.isclose(box["right"] - box["left"], depth, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"stair course has an incorrect tread depth: {name}")
        if not math.isclose(box["half_width"], first["half_width"], abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"stair course has an inconsistent corridor width: {name}")
        previous_right = box["right"]
    return {"slope_deg": math.degrees(math.atan2(height, depth)), "ascending_risers": 12,
            "descending_risers_including_ground": 12, "height_m": height, "depth_m": depth,
            "chamfer_or_nosing": False, "start_x": first["left"], "end_x": previous_right,
            "platform_start_x": platform["left"], "platform_end_x": platform["right"],
            "platform_top_z": platform["top"], "half_width_m": first["half_width"],
            "terrain_ids": names}


class RouteAudit:
    MAX_ABS_Y_M = 0.5
    MAX_ABS_YAW_DEG = 30.0
    MAX_ABS_ROLL_DEG = 15.0
    MAX_ABS_PITCH_DEG = 60.0

    def __init__(self, model, audit):
        self.course = inspect_course(model)
        self.audit = audit
        self.events = {}
        self.max_abs_y = self.max_abs_yaw = self.max_abs_roll = self.max_abs_pitch = 0.0
        self.wheel_outside_course = False
        self.platform_wheels_seen = set()
        self.up_geoms_seen, self.down_geoms_seen = set(), set()
        self.final_base_x = None
        self.final_wheel_min_x = None
        self.force = np.zeros(6)
        self.steps = 0

    def step(self, model, data, t):
        self.steps += 1
        course = self.course
        x, y = map(float, data.qpos[:2])
        self.final_base_x = x
        wheel_geoms = sorted(self.audit.wheel_by_geom)
        wheel_centers = data.geom_xpos[wheel_geoms]
        self.final_wheel_min_x = float(np.min(wheel_centers[:, 0]))
        # Each wheel's collision bounding sphere must remain within the staircase width.
        if np.any(np.abs(wheel_centers[:, 1]) + model.geom_rbound[wheel_geoms] > course["half_width_m"]):
            self.wheel_outside_course = True
        w, qx, qy, qz = map(float, data.qpos[3:7])
        roll = math.degrees(math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
        pitch = math.degrees(math.asin(float(np.clip(2 * (w * qy - qz * qx), -1, 1))))
        yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        self.max_abs_y = max(self.max_abs_y, abs(y))
        self.max_abs_yaw = max(self.max_abs_yaw, abs(yaw))
        self.max_abs_roll = max(self.max_abs_roll, abs(roll))
        self.max_abs_pitch = max(self.max_abs_pitch, abs(pitch))
        if self.steps == 1 and np.max(wheel_centers[:, 0]) < course["start_x"]:
            self.events["start_before_stairs_s"] = float(t)
        support = {name: {} for name in ("ground", "platform")}
        up, down = set(), set()
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if int(contact.efc_address) < 0 or abs(float(contact.frame[2])) <= 0.7:
                continue
            if a in self.audit.terrain_geoms:
                terrain, wheel_geom = a, b
            elif b in self.audit.terrain_geoms:
                terrain, wheel_geom = b, a
            else:
                continue
            wheel = self.audit.wheel_by_geom.get(wheel_geom)
            if wheel is None:
                continue
            mujoco.mj_contactForce(model, data, index, self.force)
            force = max(0.0, float(self.force[0]))
            name = self.audit.geom_names[terrain]
            if name in support:
                support[name][wheel] = support[name].get(wheel, 0.0) + force
            elif force > self.audit.MIN_CONTACT_FORCE_N:
                if name.startswith("up"):
                    up.add(name)
                elif name.startswith("down"):
                    down.add(name)
        self.up_geoms_seen.update(up)
        self.down_geoms_seen.update(down)
        platform_wheels = {name for name, force in support["platform"].items()
                           if force > self.audit.MIN_CONTACT_FORCE_N}
        self.platform_wheels_seen.update(platform_wheels)
        grounded = len([force for force in support["ground"].values()
                        if force > self.audit.MIN_CONTACT_FORCE_N]) == 6
        if grounded and x < course["start_x"] and "ascent_s" not in self.events:
            self.events.setdefault("initial_six_wheels_ground_s", float(t))
        if up and "initial_six_wheels_ground_s" in self.events:
            self.events.setdefault("ascent_s", float(t))
        if len(platform_wheels) == 6 and "ascent_s" in self.events and course["platform_start_x"] < x < course["platform_end_x"]:
            self.events.setdefault("all_six_supported_on_platform_s", float(t))
        if down and "all_six_supported_on_platform_s" in self.events:
            self.events.setdefault("descent_s", float(t))
        if grounded and self.final_wheel_min_x > course["end_x"] and "descent_s" in self.events:
            self.events.setdefault("all_six_ground_after_stairs_s", float(t))

    def result(self):
        required = ("start_before_stairs_s", "initial_six_wheels_ground_s", "ascent_s",
                    "all_six_supported_on_platform_s", "descent_s", "all_six_ground_after_stairs_s")
        reasons = [f"route_missing:{key}" for key in required if key not in self.events]
        if self.max_abs_y > self.MAX_ABS_Y_M or self.wheel_outside_course:
            reasons.append("route_outside_corridor")
        if self.max_abs_yaw > self.MAX_ABS_YAW_DEG:
            reasons.append("route_yaw_over_30deg")
        if self.max_abs_roll > self.MAX_ABS_ROLL_DEG:
            reasons.append("route_roll_over_15deg")
        if self.max_abs_pitch > self.MAX_ABS_PITCH_DEG:
            reasons.append("route_pitch_over_60deg")
        if self.final_base_x is None or self.final_base_x <= self.course["end_x"] + 0.3:
            reasons.append("route_final_base_not_past_end")
        if self.final_wheel_min_x is None or self.final_wheel_min_x <= self.course["end_x"]:
            reasons.append("route_final_wheels_not_past_end")
        if self.audit.continuous_final_stability_s + 1e-9 < self.audit.MIN_FINAL_STABILITY_S:
            reasons.append("route_final_ground_stability_under_1s")
        geometry = {key: value for key, value in self.course.items() if key != "terrain_ids"}
        return {"route_ok": not reasons, "rejection_reasons": reasons, "geometry": geometry,
                "events": self.events, "platform_wheels_seen": sorted(self.platform_wheels_seen),
                "ascending_treads_contacted": sorted(self.up_geoms_seen),
                "descending_treads_contacted": sorted(self.down_geoms_seen),
                "max_abs_y_m": self.max_abs_y, "max_abs_yaw_deg": self.max_abs_yaw,
                "max_abs_roll_deg": self.max_abs_roll, "max_abs_pitch_deg": self.max_abs_pitch,
                "wheel_outside_course": self.wheel_outside_course,
                "final_base_x_m": self.final_base_x, "final_wheel_min_x_m": self.final_wheel_min_x}


def verify_episode(metadata_path):
    path = Path(metadata_path).resolve(strict=True)
    metadata = json.loads(path.read_text())
    episode = path.parent
    sources = check_sources(metadata, episode)
    meshes = check_meshes(metadata, episode / "scene.xml")
    state = {}

    def observe(model, data, t):
        if not state:
            state["audit"] = EpisodeAudit(model)
            state["route"] = RouteAudit(model, state["audit"])
        state["audit"].step(model, data, t)
        state["route"].step(model, data, t)

    replay = replay_check(episode / "scene.xml", episode / "trajectory.npz", on_step=observe)
    check_meshes(metadata, episode / "scene.xml")
    check_sources(metadata, episode)
    audit, route = state["audit"].result(), state["route"].result()
    reasons = [*audit["quality_rejection_reasons"], *route["rejection_reasons"]]
    if not replay["replay_ok"]:
        reasons.append("independent_action_replay_mismatch")
    if not replay["physics_checked"]:
        reasons.append("missing_per_physics_step_recording")
    return {"metadata_path": str(path), "verify_ok": not reasons, "rejection_reasons": reasons,
            "source_snapshot_hashes_verified": sources,
            "mesh_hashes_verified": meshes,
            "current_verifier_sha256": sha256(__file__),
            "current_audit_sha256": sha256(Path(__file__).with_name("episode_audit.py")),
            "replay": replay, "audit": audit, "route": route}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", nargs="*", type=Path)
    parser.add_argument("--accepted", action="store_true", help="verify every accepted episode in dataset-root")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/45deg_v3")
    args = parser.parse_args()
    paths = list(args.metadata)
    selection_errors = []
    if args.accepted:
        for path in sorted((args.dataset_root / "episodes").glob("*/metadata.json")):
            try:
                if json.loads(path.read_text()).get("dataset_accepted") is True:
                    paths.append(path)
            except (OSError, ValueError) as error:
                selection_errors.append({"metadata_path": str(path), "error": str(error)})
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    results = []
    for path in paths:
        try:
            results.append(verify_episode(path))
        except Exception as error:
            results.append({"metadata_path": str(path), "verify_ok": False,
                            "rejection_reasons": [f"verification_error:{type(error).__name__}:{error}"]})
    ok = bool(results) and not selection_errors and all(result["verify_ok"] for result in results)
    print(json.dumps({"verify_ok": ok, "verification_scope": "independent_physics_and_full_route",
                      "episodes_checked": len(results), "selection_errors": selection_errors,
                      "selection_rejection_reason": "no_selected_episodes" if not results else None,
                      "episodes": results}, indent=2, allow_nan=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
