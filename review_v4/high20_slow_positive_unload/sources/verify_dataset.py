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
ROBOT_XML = ROOT / "assets/mars_rover/rover_actuated.xml"
URDF = ROOT / "assets/mars_rover/urdf/mars_rover.urdf"
MODEL_HASH_RECORD = ROOT / "review_v3/model_sha256.txt"
LEGACY_COURSE = ROOT / "scripts/rover/course.py.before_closed_loop_20261110.py"
LEGACY_COURSE_SHA256 = "946c632d41f9bf99fe90c6ea93e081279787d0c9823d935c7002c47611f53a39"
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


def inherited_collision_policy():
    if sha256(LEGACY_COURSE) != LEGACY_COURSE_SHA256:
        raise ValueError("legacy course collision-policy source differs from the pinned SHA256")
    return {"source_path": str(LEGACY_COURSE), "source_sha256": LEGACY_COURSE_SHA256,
            "source_line": 28, "original_xml_collision_geom_mask": {"contype": 1, "conaffinity": 1},
            "inherited_course_collision_geom_mask": {"contype": 1, "conaffinity": 2},
            "terrain_mask": {"contype": 2, "conaffinity": 1},
            "visual_geom_mask": {"contype": 0, "conaffinity": 0},
            "self_collision_enabled": False}


def check_reference_provenance(metadata):
    pinned = MODEL_HASH_RECORD.read_text().strip()
    if len(pinned) != 64 or any(character not in "0123456789abcdef" for character in pinned):
        raise ValueError("original model SHA256 record is invalid")
    model_hash = sha256(ROBOT_XML)
    if model_hash != pinned or model_hash != metadata.get("model_sha256"):
        raise ValueError("original robot XML differs from the pinned or episode model SHA256")
    urdf_hash = sha256(URDF)
    if urdf_hash != metadata.get("urdf_sha256"):
        raise ValueError("URDF velocity-limit source SHA256 is missing or differs from the episode")
    recorded_audit_hash = metadata.get("audit", {}).get("velocity_limit_source", {}).get("sha256")
    if recorded_audit_hash != urdf_hash:
        raise ValueError("recorded velocity audit used a different URDF source")
    policy = inherited_collision_policy()
    if metadata.get("model_identity", {}).get("inherited_collision_policy") != policy:
        raise ValueError("inherited course collision policy provenance is missing or inconsistent")
    return {"model_path": str(ROBOT_XML), "model_sha256": model_hash,
            "model_hash_record": str(MODEL_HASH_RECORD), "urdf_path": str(URDF), "urdf_sha256": urdf_hash,
            "inherited_collision_policy": policy}


def compare_robot_models(original, scene, metadata):
    """Compare robot mechanics by name; world terrain and declared contact settings are separate."""
    mismatches = []
    counts = {}

    def same(first, second):
        first, second = np.asarray(first), np.asarray(second)
        return first.shape == second.shape and bool(np.allclose(first, second, atol=1e-12, rtol=0))

    def require(label, first, second):
        if not same(first, second):
            mismatches.append(f"robot_model_mismatch:{label}")

    def named(model, kind, count, robot_geoms=False):
        result = {}
        for index in range(count):
            if robot_geoms and int(model.geom_bodyid[index]) == 0:
                continue
            name = mujoco.mj_id2name(model, kind, index)
            if not name or name in result:
                raise ValueError("model invariance requires unique names for all robot objects")
            result[name] = index
        return result

    def objects(label, kind, size, fields, robot_geoms=False):
        before = named(original, kind, getattr(original, size), robot_geoms)
        after = named(scene, kind, getattr(scene, size), robot_geoms)
        if set(before) != set(after):
            mismatches.append(f"robot_model_mismatch:{label}:names")
        counts[label] = len(before)
        for name in sorted(set(before) & set(after)):
            for field in fields:
                require(f"{label}:{name}:{field}", getattr(original, field)[before[name]],
                        getattr(scene, field)[after[name]])
        return before, after

    def object_name(model, kind, index):
        return mujoco.mj_id2name(model, kind, int(index))

    def same_reference(label, first, second):
        if first != second:
            mismatches.append(f"robot_model_mismatch:{label}")

    for dimension in ("nq", "nv", "na", "neq", "ntendon", "nwrap", "nsensor", "nplugin",
                      "nflex", "nskin", "npair", "nexclude"):
        require(dimension, getattr(original, dimension), getattr(scene, dimension))
    # New global forces or constraints could otherwise preserve the listed robot arrays.
    for name in sorted(name for name in dir(original.opt) if not name.startswith("_") and name != "timestep"):
        if not callable(getattr(original.opt, name)):
            require(f"option:{name}", getattr(original.opt, name), getattr(scene.opt, name))
    recorded_dt = metadata.get("simulation_dt")
    if not isinstance(recorded_dt, (int, float)) or not math.isfinite(recorded_dt) or recorded_dt <= 0:
        mismatches.append("robot_model_mismatch:missing_explicit_simulation_dt")
    else:
        require("option:timestep_vs_metadata", float(scene.opt.timestep), recorded_dt)

    bodies = objects("body", mujoco.mjtObj.mjOBJ_BODY, "nbody", (
        "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
        "body_gravcomp", "body_dofnum", "body_mocapid"))
    for name in sorted(set(bodies[0]) & set(bodies[1])):
        b, s = bodies[0][name], bodies[1][name]
        same_reference(f"body:{name}:parent", object_name(original, mujoco.mjtObj.mjOBJ_BODY, original.body_parentid[b]),
                       object_name(scene, mujoco.mjtObj.mjOBJ_BODY, scene.body_parentid[s]))

    joints = objects("joint", mujoco.mjtObj.mjOBJ_JOINT, "njnt", (
        "jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited", "jnt_stiffness",
        "jnt_stiffnesspoly", "jnt_margin", "jnt_solref", "jnt_solimp", "jnt_actfrclimited",
        "jnt_actfrcrange", "jnt_actgravcomp"))
    for name in sorted(set(joints[0]) & set(joints[1])):
        b, s = joints[0][name], joints[1][name]
        same_reference(f"joint:{name}:body", object_name(original, mujoco.mjtObj.mjOBJ_BODY, original.jnt_bodyid[b]),
                       object_name(scene, mujoco.mjtObj.mjOBJ_BODY, scene.jnt_bodyid[s]))
        kind = int(original.jnt_type[b])
        nq, nv = (7, 6) if kind == 0 else ((4, 3) if kind == 1 else (1, 1))
        qb, qs = int(original.jnt_qposadr[b]), int(scene.jnt_qposadr[s])
        vb, vs = int(original.jnt_dofadr[b]), int(scene.jnt_dofadr[s])
        for field in ("qpos0", "qpos_spring"):
            require(f"joint:{name}:{field}", getattr(original, field)[qb:qb + nq], getattr(scene, field)[qs:qs + nq])
        for field in ("dof_damping", "dof_dampingpoly", "dof_armature", "dof_frictionloss", "dof_solref", "dof_solimp"):
            require(f"joint:{name}:{field}", getattr(original, field)[vb:vb + nv], getattr(scene, field)[vs:vs + nv])

    geoms = objects("robot_geom", mujoco.mjtObj.mjOBJ_GEOM, "ngeom", (
        "geom_type", "geom_size", "geom_pos", "geom_quat",
        "geom_condim", "geom_priority", "geom_friction", "geom_solimp", "geom_solmix", "geom_margin",
        "geom_gap", "geom_fluid", "geom_adhesion", "geom_surfacevel", "geom_rbound", "geom_rgba", "geom_group"), True)
    contact_timeconst = metadata.get("contact_timeconst")
    if not isinstance(contact_timeconst, (int, float)) or not 0.004 <= contact_timeconst <= 0.02:
        mismatches.append("robot_model_mismatch:missing_or_invalid_contact_timeconst")
    solref_differences = 0
    collision_mask_differences = 0
    for name in sorted(set(geoms[0]) & set(geoms[1])):
        b, s = geoms[0][name], geoms[1][name]
        original_mask = (int(original.geom_contype[b]), int(original.geom_conaffinity[b]))
        expected_mask = (1, 2) if original_mask == (1, 1) else original_mask
        actual_mask = (int(scene.geom_contype[s]), int(scene.geom_conaffinity[s]))
        require(f"robot_geom:{name}:inherited_collision_mask", actual_mask, expected_mask)
        collision_mask_differences += int(actual_mask != original_mask)
        same_reference(f"robot_geom:{name}:body", object_name(original, mujoco.mjtObj.mjOBJ_BODY, original.geom_bodyid[b]),
                       object_name(scene, mujoco.mjtObj.mjOBJ_BODY, scene.geom_bodyid[s]))
        if int(original.geom_type[b]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            same_reference(f"robot_geom:{name}:mesh", object_name(original, mujoco.mjtObj.mjOBJ_MESH, original.geom_dataid[b]),
                           object_name(scene, mujoco.mjtObj.mjOBJ_MESH, scene.geom_dataid[s]))
        if isinstance(contact_timeconst, (int, float)):
            require(f"robot_geom:{name}:solref_vs_metadata", scene.geom_solref[s], [contact_timeconst, 1.0])
        solref_differences += int(not same(original.geom_solref[b], scene.geom_solref[s]))

    meshes = objects("mesh", mujoco.mjtObj.mjOBJ_MESH, "nmesh", (
        "mesh_pos", "mesh_quat", "mesh_scale", "mesh_vertnum", "mesh_facenum"))
    for name in sorted(set(meshes[0]) & set(meshes[1])):
        b, s = meshes[0][name], meshes[1][name]
        for field, address, count in (("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
                                      ("mesh_face", "mesh_faceadr", "mesh_facenum")):
            ib, js = int(getattr(original, address)[b]), int(getattr(scene, address)[s])
            nb, ns = int(getattr(original, count)[b]), int(getattr(scene, count)[s])
            require(f"mesh:{name}:{field}", getattr(original, field)[ib:ib + nb], getattr(scene, field)[js:js + ns])

    actuators = objects("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, "nu", (
        "actuator_trntype", "actuator_gear", "actuator_ctrllimited", "actuator_ctrlrange",
        "actuator_forcelimited", "actuator_forcerange", "actuator_actlimited", "actuator_actrange",
        "actuator_gaintype", "actuator_gainprm", "actuator_biastype", "actuator_biasprm",
        "actuator_dyntype", "actuator_dynprm", "actuator_actearly", "actuator_actnum",
        "actuator_cranklength", "actuator_lengthrange", "actuator_armature", "actuator_damping",
        "actuator_dampingpoly", "actuator_delay", "actuator_history", "actuator_ctrlspec"))
    for name in sorted(set(actuators[0]) & set(actuators[1])):
        b, s = actuators[0][name], actuators[1][name]
        same_reference(f"actuator:{name}:joint", object_name(original, mujoco.mjtObj.mjOBJ_JOINT, original.actuator_trnid[b, 0]),
                       object_name(scene, mujoco.mjtObj.mjOBJ_JOINT, scene.actuator_trnid[s, 0]))
        require(f"actuator:{name}:secondary_transmission", original.actuator_trnid[b, 1], scene.actuator_trnid[s, 1])
    return {"model_identity_ok": not mismatches, "structure_mass_joints_actuators_match": not mismatches,
            "rejection_reasons": mismatches, "compared_objects": counts,
            "reference_model_path": str(ROBOT_XML), "reference_model_sha256": metadata.get("model_sha256"),
            "numeric_absolute_tolerance": 1e-12, "original_simulation_dt": float(original.opt.timestep),
            "scene_simulation_dt": float(scene.opt.timestep), "declared_contact_timeconst": contact_timeconst,
            "robot_geom_solref_differences": solref_differences,
            "inherited_robot_collision_mask_differences": collision_mask_differences,
            "inherited_collision_policy": metadata.get("model_identity", {}).get("inherited_collision_policy"),
            "literal_full_physics_model_identical": False,
            "allowed_differences": ["world terrain", "declared geom contact solref", "explicitly recorded simulation_dt",
                                    "legacy course collision mask 1/1 to 1/2; self-collision disabled"]}


def inspect_course(model, expected_parameters=None):
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g): g
             for g in range(model.ngeom) if int(model.geom_bodyid[g]) == 0}
    stairs = sum(isinstance(name, str) and name.startswith("up") for name in names)
    if stairs < 1:
        raise ValueError("course must have at least one ascending riser")
    expected = {"ground", "platform", *(f"up{i}" for i in range(stairs)),
                *(f"down{i}" for i in range(stairs - 1))}
    if set(names) != expected:
        raise ValueError("course must contain consecutive ascending boxes, platform, matching descending boxes and ground only")
    if expected_parameters is not None:
        declared_stairs = expected_parameters.get("stairs")
        if type(declared_stairs) is not int or declared_stairs != stairs:
            raise ValueError("scene stair count differs from declared parameters")
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
    if expected_parameters is not None:
        for key, actual in (("height", height), ("depth", depth)):
            declared = expected_parameters.get(key)
            if (type(declared) not in (int, float) or not math.isfinite(declared)
                    or not math.isclose(actual, declared, abs_tol=1e-9, rel_tol=0)):
                raise ValueError(f"scene {key} differs from declared parameters")
    if not math.isclose(height, depth, abs_tol=1e-9, rel_tol=0):
        raise ValueError("stair slope is not 45 degrees (rise differs from run)")
    previous_right = first["left"]
    for name in [*(f"up{i}" for i in range(stairs)), "platform", *(f"down{i}" for i in range(stairs - 1))]:
        box = boxes[name]
        top = (int(name[2:]) + 1) * height if name.startswith("up") else (
            (stairs - 1 - int(name[4:])) * height if name.startswith("down") else stairs * height)
        if not math.isclose(box["left"], previous_right, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"stair course has an overlap or gap: {name}")
        if not math.isclose(box["top"], top, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"stair course has an incorrect riser height: {name}")
        if name != "platform" and not math.isclose(box["right"] - box["left"], depth, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"stair course has an incorrect tread depth: {name}")
        if not math.isclose(box["half_width"], first["half_width"], abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"stair course has an inconsistent corridor width: {name}")
        previous_right = box["right"]
    return {"slope_deg": math.degrees(math.atan2(height, depth)), "ascending_risers": stairs,
            "descending_risers_including_ground": stairs, "height_m": height, "depth_m": depth,
            "chamfer_or_nosing": False, "start_x": first["left"], "end_x": previous_right,
            "platform_start_x": platform["left"], "platform_end_x": platform["right"],
            "platform_top_z": platform["top"], "half_width_m": first["half_width"],
            "terrain_ids": names}


class RouteAudit:
    MAX_ABS_Y_M = 0.5
    MAX_ABS_YAW_DEG = 30.0
    MAX_ABS_ROLL_DEG = 15.0
    MAX_ABS_PITCH_DEG = 60.0
    MAX_ABS_STEER_DEG = 30.0

    def __init__(self, model, audit, expected_parameters=None):
        self.course = inspect_course(model, expected_parameters)
        self.audit = audit
        self.events = {}
        self.max_abs_y = self.max_abs_yaw = self.max_abs_roll = self.max_abs_pitch = 0.0
        self.steer_joints = [(name, q) for name, kind, q, _ in audit.joints if kind == "steer"]
        self.steer_peak_by_joint = {name: {"abs_angle_deg": 0.0, "signed_angle_deg": 0.0, "time_s": None}
                                    for name, _ in self.steer_joints}
        self.max_abs_steer_rad = 0.0
        self.steer_peak = {"joint": None, "abs_angle_deg": 0.0, "signed_angle_deg": 0.0, "time_s": None}
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
        for name, q in self.steer_joints:
            angle = float(data.qpos[q])
            absolute_deg = math.degrees(abs(angle))
            peak = self.steer_peak_by_joint[name]
            if peak["time_s"] is None or absolute_deg > peak["abs_angle_deg"]:
                peak.update(abs_angle_deg=absolute_deg, signed_angle_deg=math.degrees(angle), time_s=float(t))
            if self.steer_peak["time_s"] is None or abs(angle) > self.max_abs_steer_rad:
                self.max_abs_steer_rad = abs(angle)
                self.steer_peak = {"joint": name, "abs_angle_deg": absolute_deg,
                                   "signed_angle_deg": math.degrees(angle), "time_s": float(t)}
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
        if self.max_abs_steer_rad > math.radians(self.MAX_ABS_STEER_DEG):
            reasons.append("route_steer_over_30deg")
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
                "max_abs_steer_deg": math.degrees(self.max_abs_steer_rad),
                "steer_peak": dict(self.steer_peak),
                "steer_peak_by_joint": {name: dict(peak) for name, peak in self.steer_peak_by_joint.items()},
                "task_steer_limit_deg": self.MAX_ABS_STEER_DEG,
                "wheel_outside_course": self.wheel_outside_course,
                "final_base_x_m": self.final_base_x, "final_wheel_min_x_m": self.final_wheel_min_x}


def verify_episode(metadata_path):
    path = Path(metadata_path).resolve(strict=True)
    metadata = json.loads(path.read_text())
    episode = path.parent
    sources = check_sources(metadata, episode)
    meshes = check_meshes(metadata, episode / "scene.xml")
    reference = check_reference_provenance(metadata)
    original = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    state = {}

    def observe(model, data, t):
        if not state:
            state["model_invariance"] = compare_robot_models(original, model, metadata)
            state["audit"] = EpisodeAudit(model)
            state["route"] = RouteAudit(model, state["audit"], metadata["parameters"])
        state["audit"].step(model, data, t)
        state["route"].step(model, data, t)

    replay = replay_check(episode / "scene.xml", episode / "trajectory.npz", on_step=observe)
    check_meshes(metadata, episode / "scene.xml")
    check_sources(metadata, episode)
    check_reference_provenance(metadata)
    audit, route = state["audit"].result(), state["route"].result()
    reasons = [*audit["quality_rejection_reasons"], *route["rejection_reasons"],
               *state["model_invariance"]["rejection_reasons"]]
    if not replay["replay_ok"]:
        reasons.append("independent_action_replay_mismatch")
    if not replay["physics_checked"]:
        reasons.append("missing_per_physics_step_recording")
    return {"metadata_path": str(path), "verify_ok": not reasons, "rejection_reasons": reasons,
            "source_snapshot_hashes_verified": sources,
            "mesh_hashes_verified": meshes,
            "reference_provenance": reference,
            "model_identity": state["model_invariance"],
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
