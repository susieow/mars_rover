#!/usr/bin/env python3
"""P0 model compliance for experiment two.

The URDF importer drops <contact> friction and cannot express joint velocity
limits, so this module writes every declared value into the MJCF explicitly,
recompiles it, and compares the compiled model field by field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from climb import visual_geoms

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "assets/mars_rover/urdf/mars_rover.urdf"
EXPECTED_URDF_SHA = "c12222dfd7b770b21f5620be33d55ac2ab974fb499f32b7d5c16045b1f5ca061"
RUN_DIR = ROOT / "runs/urdf_slope_20261112"
MODEL_DIR = RUN_DIR / "model"
MJCF = MODEL_DIR / "rover_p0.xml"
SPEC = MODEL_DIR / "model_spec.json"

WHEEL_RADIUS = 0.162697
ROBOT_FRICTION = (0.6, 0.005, 0.0001)
ROBOT_CONTYPE = 1
ROBOT_CONAFFINITY = 3
TERRAIN_FRICTION = (4.0, 0.02, 0.001)
CTRL_LIMIT = {"wheel": 50.0, "steer": 40.0, "rocker": 100.0}
TIMESTEP = 0.001
SETTLE_SECONDS = 3.0
SETTLE_PENETRATION_LIMIT_M = 0.05
SETTLE_SELF_FORCE_LIMIT_N = 1.0
GEOM_TYPE = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vec(text: str | None, n: int) -> list[float]:
    if text is None:
        return [0.0] * n
    values = [float(v) for v in text.split()]
    if len(values) != n:
        raise ValueError(f"expected {n} values, got {text!r}")
    return values


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def joint_class(name: str) -> str:
    for kind in ("wheel", "steer", "rocker"):
        if name.endswith(f"_{kind}_joint"):
            return kind
    raise ValueError(f"unclassified joint {name}")


def parse_urdf() -> dict:
    root = ET.parse(URDF).getroot()
    links = []
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        entry = {"name": name, "has_inertial": inertial is not None,
                 "mass": 0.0, "com": [0.0, 0.0, 0.0],
                 "inertia": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        if inertial is not None:
            entry["mass"] = float(inertial.find("mass").get("value"))
            origin = inertial.find("origin")
            entry["com"] = vec(origin.get("xyz") if origin is not None else None, 3)
            tensor = inertial.find("inertia")
            entry["inertia"] = [float(tensor.get(k)) for k in
                                ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
        links.append(entry)

    joints = []
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        joints.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else None,
            "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else None,
            "velocity": float(limit.get("velocity")) if limit is not None else None,
            "effort": float(limit.get("effort")) if limit is not None else None,
            "damping": float(dynamics.get("damping", 0.0)) if dynamics is not None else 0.0,
            "frictionloss": float(dynamics.get("friction", 0.0)) if dynamics is not None else 0.0,
        })

    collisions = []
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            shape = list(geometry)[0]
            origin = collision.find("origin")
            contact = collision.find("contact")
            entry = {
                "name": collision.get("name"),
                "link": link.get("name"),
                "shape": shape.tag,
                "xyz": vec(origin.get("xyz") if origin is not None else None, 3),
                "rpy": vec(origin.get("rpy") if origin is not None else None, 3),
                "friction_declared": None,
                "priority": 0,
                "condim": 3,
                "size": None,
            }
            if shape.tag == "box":
                entry["size"] = [v / 2 for v in vec(shape.get("size"), 3)]
            elif shape.tag in ("cylinder", "capsule"):
                entry["size"] = [float(shape.get("radius")), float(shape.get("length")) / 2, 0.0]
            elif shape.tag == "sphere":
                entry["size"] = [float(shape.get("radius")), 0.0, 0.0]
            elif shape.tag == "mesh":
                entry["mesh"] = Path(shape.get("filename")).name
            if contact is not None:
                if contact.get("lateral_friction") is not None:
                    entry["friction_declared"] = float(contact.get("lateral_friction"))
                entry["priority"] = int(contact.get("priority", 0))
                entry["condim"] = int(contact.get("condim", 3))
            collisions.append(entry)

    frames = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        frames[joint.find("child").get("link")] = {
            "parent": joint.find("parent").get("link"),
            "xyz": vec(origin.get("xyz") if origin is not None else None, 3),
            "quat": rpy_to_quat(*vec(origin.get("rpy") if origin is not None else None, 3)),
        }
    return {"links": links, "joints": joints, "collisions": collisions, "frames": frames}


def _fmt(values) -> str:
    if np.isscalar(values):
        return f"{float(values):.17g}"
    return " ".join(f"{float(v):.17g}" for v in values)


def _set_attr(tag: str, attr: str, value: str) -> str:
    pattern = rf'\b{attr}="[^"]*"'
    if re.search(pattern, tag):
        return re.sub(pattern, f'{attr}="{value}"', tag, count=1)
    if tag.endswith("/>"):
        return tag[:-2].rstrip() + f' {attr}="{value}"/>'
    return tag[:-1].rstrip() + f' {attr}="{value}">'


def high_precision(xml: str, model) -> str:
    """mj_saveLastXML writes six significant digits; restore the compiled doubles."""

    def body_block(match: re.Match[str]) -> str:
        name = match.group(1)
        body = body_id(model, name)
        head = match.group(0).split("\n", 1)[0]
        head = _set_attr(head, "pos", _fmt(model.body_pos[body]))
        head = _set_attr(head, "quat", _fmt(model.body_quat[body]))
        inertial = (f'<inertial pos="{_fmt(model.body_ipos[body])}" '
                    f'quat="{_fmt(model.body_iquat[body])}" '
                    f'mass="{float(model.body_mass[body]):.17g}" '
                    f'diaginertia="{_fmt(model.body_inertia[body])}"/>')
        return head + "\n" + match.group(2) + inertial

    xml = re.sub(r'<body name="([^"]+)"[^>]*>\n(\s*)<inertial [^>]*/>', body_block, xml)

    def body_only(match: re.Match[str]) -> str:
        body = body_id(model, match.group(1))
        tag = _set_attr(match.group(0), "pos", _fmt(model.body_pos[body]))
        return _set_attr(tag, "quat", _fmt(model.body_quat[body]))

    xml = re.sub(r'<body name="([^"]+)"[^>]*>', body_only, xml)

    def joint_tag(match: re.Match[str]) -> str:
        joint = joint_id(model, match.group(1))
        tag = match.group(0)
        if bool(model.jnt_limited[joint]):
            tag = _set_attr(tag, "range", _fmt(model.jnt_range[joint]))
        if bool(model.jnt_actfrclimited[joint]):
            tag = _set_attr(tag, "actuatorfrcrange", _fmt(model.jnt_actfrcrange[joint]))
        dof = int(model.jnt_dofadr[joint])
        tag = _set_attr(tag, "damping", _fmt(model.dof_damping[dof]))
        tag = _set_attr(tag, "frictionloss", _fmt(model.dof_frictionloss[dof]))
        return _set_attr(tag, "axis", _fmt(model.jnt_axis[joint]))

    xml = re.sub(r'<joint name="([^"]+)"[^>]*/>', joint_tag, xml)

    def geom_tag(match: re.Match[str]) -> str:
        geom = geom_id(model, match.group(1))
        kind = mujoco.mjtGeom(int(model.geom_type[geom])).name.removeprefix("mjGEOM_").lower()
        count = {"box": 3, "cylinder": 2, "sphere": 1, "capsule": 2}.get(kind, 3)
        tag = _set_attr(match.group(0), "size", _fmt(model.geom_size[geom][:count]))
        tag = _set_attr(tag, "pos", _fmt(model.geom_pos[geom]))
        return _set_attr(tag, "quat", _fmt(model.geom_quat[geom]))

    return re.sub(r'<geom name="([^"]+)"[^>]*/>', geom_tag, xml)


def build(source: dict) -> None:
    actual = sha256(URDF)
    if actual != EXPECTED_URDF_SHA:
        raise SystemExit(f"URDF hash mismatch: {actual}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    text = URDF.read_text().replace(
        "</robot>", '  <mujoco><compiler fusestatic="false" angle="radian"/></mujoco>\n</robot>')
    pinned = MODEL_DIR / "urdf_pinned.urdf"
    pinned.write_text(text)
    imported = mujoco.MjModel.from_xml_path(str(pinned))
    raw = MODEL_DIR / "rover_p0_raw.xml"
    mujoco.mj_saveLastXML(str(raw), imported)
    xml = high_precision(raw.read_text(), imported)

    xml, count = re.subn(r'(<body name="base_link"[^>]*>)',
                         r'\1\n      <freejoint name="root"/>', xml, count=1)
    if count != 1:
        raise RuntimeError("base_link not found")

    collisions = {c["name"] for c in source["collisions"]}

    def tag_collision(match: re.Match[str]) -> str:
        if match.group(1) not in collisions:
            return match.group(0)
        attrs = (f'friction="{ROBOT_FRICTION[0]} {ROBOT_FRICTION[1]} {ROBOT_FRICTION[2]}" '
                 f'contype="{ROBOT_CONTYPE}" conaffinity="{ROBOT_CONAFFINITY}" '
                 f'priority="0" condim="3"')
        return match.group(0)[:-2].rstrip() + " " + attrs + "/>"

    xml, tagged = re.subn(r'<geom [^>]*name="([^"]+)"[^>]*/>', tag_collision, xml)
    if tagged != len(collisions):
        raise RuntimeError(f"tagged {tagged} geoms, expected {len(collisions)}")
    if 'contype="1" conaffinity="3"' not in xml:
        raise RuntimeError("collision bitmask injection failed")

    actuators = []
    for joint in source["joints"]:
        limit = CTRL_LIMIT[joint_class(joint["name"])]
        if abs(joint["effort"] - limit) > 1e-9:
            raise RuntimeError(f"URDF effort {joint['effort']} for {joint['name']} != {limit}")
        actuators.append(f'    <motor name="{joint["name"]}_act" joint="{joint["name"]}" '
                         f'gear="1" ctrlrange="-{limit:g} {limit:g}"/>')
    xml = xml.replace("</mujoco>", "  <actuator>\n" + "\n".join(actuators) +
                      "\n  </actuator>\n</mujoco>", 1)

    option = ('  <option timestep="0.001" gravity="0 0 -9.81" solver="Newton" cone="elliptic" '
              'impratio="10" iterations="200"/>\n')
    xml = xml.replace("<worldbody>", option + "  <worldbody>", 1)

    visual = visual_geoms()
    assets = "\n".join(f'    <mesh name="vis_{link}" file="{ROOT / "assets/mars_rover/meshes" / data["file"]}"/>'
                       for link, data in visual.items())
    xml = xml.replace("  <worldbody>", f"  <asset>\n{assets}\n  </asset>\n  <worldbody>", 1)
    missing = []
    for link, data in visual.items():
        pos = " ".join(f"{v:.9g}" for v in data["xyz"])
        quat = " ".join(f"{v:.9g}" for v in data["quat"])
        geom = (f'      <geom name="vis_{link}" type="mesh" mesh="vis_{link}" pos="{pos}" '
                f'quat="{quat}" rgba="{data["rgba"]}" contype="0" conaffinity="0" '
                f'group="1" density="0"/>')
        xml, found = re.subn(rf'(<body name="{re.escape(link)}"[^>]*>)',
                             lambda m: f"{m.group(0)}\n{geom}", xml, count=1)
        if found != 1:
            missing.append(link)
    if missing:
        raise RuntimeError(f"visual bodies missing: {missing}")

    MJCF.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    print(f"[build] {MJCF.relative_to(ROOT)}  bodies={model.nbody} geoms={model.ngeom} "
          f"joints={model.njnt} actuators={model.nu} mass={model.body_mass.sum():.6f}")


def row(section: str, name: str, field: str, declared, compiled, source: str,
        mapping: str, gate: bool = True, ok: bool = True, unit: str = "",
        note: str = "", detail=None) -> dict:
    out = {"section": section, "name": name, "field": field,
           "declared": declared, "compiled": compiled, "unit": unit,
           "source": source, "mapping": mapping, "gate": gate, "pass": bool(ok),
           "note": note}
    if detail is not None:
        out["detail"] = detail
    return out


def close(a, b, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol))


def quat_close(a, b) -> bool:
    return abs(float(np.dot(a, b))) >= 1.0 - 1e-6


def body_id(model, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise KeyError(f"body {name}")
    return value


def geom_id(model, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if value < 0:
        raise KeyError(f"geom {name}")
    return value


def joint_id(model, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if value < 0:
        raise KeyError(f"joint {name}")
    return value


def check_links(model, source) -> list[dict]:
    rows = []
    for link in source["links"]:
        body = body_id(model, link["name"])
        frame = source["frames"].get(link["name"],
                                     {"xyz": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]})
        rows.append(row("link", link["name"], "frame_pos", frame["xyz"],
                        model.body_pos[body].tolist(),
                        f"urdf:joint[child='{link['name']}']/origin/@xyz", "direct",
                        unit="m", ok=close(frame["xyz"], model.body_pos[body])))
        rows.append(row("link", link["name"], "frame_quat", frame["quat"],
                        model.body_quat[body].tolist(),
                        f"urdf:joint[child='{link['name']}']/origin/@rpy", "direct",
                        ok=quat_close(frame["quat"], model.body_quat[body])))
        if not link["has_inertial"]:
            rows.append(row("link", link["name"], "mass", 0.0, float(model.body_mass[body]),
                            f"urdf:link[@name='{link['name']}']", "absent_zero",
                            ok=abs(float(model.body_mass[body])) < 1e-12))
            continue
        rows.append(row("link", link["name"], "mass", link["mass"],
                        float(model.body_mass[body]),
                        f"urdf:link[@name='{link['name']}']/inertial/mass/@value", "direct",
                        unit="kg", ok=close(link["mass"], model.body_mass[body])))
        rows.append(row("link", link["name"], "com", link["com"],
                        model.body_ipos[body].tolist(),
                        f"urdf:link[@name='{link['name']}']/inertial/origin/@xyz", "direct",
                        unit="m", ok=close(link["com"], model.body_ipos[body])))
        ixx, ixy, ixz, iyy, iyz, izz = link["inertia"]
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        eigenvalues = np.linalg.eigvalsh(tensor)
        rows.append(row("link", link["name"], "inertia", link["inertia"],
                        model.body_inertia[body].tolist(),
                        f"urdf:link[@name='{link['name']}']/inertial/inertia", "direct",
                        unit="kg m^2", ok=close(np.sort(eigenvalues), np.sort(model.body_inertia[body])),
                        note="declared tensor, compiled principal diagonal",
                        detail={"urdf_eigenvalues": eigenvalues.tolist(),
                                "compiled_quat": model.body_iquat[body].tolist()}))
    return rows


def check_joints(model, source) -> list[dict]:
    rows = []
    for joint in source["joints"]:
        jid = joint_id(model, joint["name"])
        limited = bool(model.jnt_limited[jid])
        declared_range = None
        if joint["lower"] is not None and joint["upper"] is not None:
            declared_range = [joint["lower"], joint["upper"]]
        compiled_range = model.jnt_range[jid].tolist() if limited else None
        range_ok = (declared_range is None and compiled_range is None) or (
            declared_range is not None and compiled_range is not None and close(declared_range, compiled_range))
        rows.append(row("joint", joint["name"], "range", declared_range, compiled_range,
                        f"urdf:joint[@name='{joint['name']}']/limit", "direct",
                        unit="rad", ok=range_ok,
                        note="" if limited else "continuous joint has no position limit"))
        rows.append(row("joint", joint["name"], "velocity", joint["velocity"], None,
                        f"urdf:joint[@name='{joint['name']}']/limit/@velocity", "audit_only",
                        gate=False, unit="rad/s",
                        note="MuJoCo has no joint velocity limit; strict audit enforces it"))
        rows.append(row("joint", joint["name"], "actfrcrange",
                        [-joint["effort"], joint["effort"]],
                        model.jnt_actfrcrange[jid].tolist(),
                        f"urdf:joint[@name='{joint['name']}']/limit/@effort", "direct",
                        unit="N m", ok=close([-joint["effort"], joint["effort"]],
                                             model.jnt_actfrcrange[jid])))
        rows.append(row("joint", joint["name"], "actfrclimited", True,
                        bool(model.jnt_actfrclimited[jid]),
                        f"urdf:joint[@name='{joint['name']}']/limit/@effort", "direct",
                        ok=bool(model.jnt_actfrclimited[jid])))
        rows.append(row("joint", joint["name"], "damping", joint["damping"],
                        float(model.dof_damping[model.jnt_dofadr[jid]]),
                        f"urdf:joint[@name='{joint['name']}']/dynamics/@damping", "direct",
                        unit="N m s/rad", ok=close(joint["damping"],
                                                   model.dof_damping[model.jnt_dofadr[jid]])))
        rows.append(row("joint", joint["name"], "frictionloss", joint["frictionloss"],
                        float(model.dof_frictionloss[model.jnt_dofadr[jid]]),
                        f"urdf:joint[@name='{joint['name']}']/dynamics/@friction", "direct",
                        unit="N m", ok=close(joint["frictionloss"],
                                             model.dof_frictionloss[model.jnt_dofadr[jid]])))
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                     joint["name"] + "_act")
        if actuator < 0:
            raise KeyError(f"actuator for {joint['name']}")
        limit = CTRL_LIMIT[joint_class(joint["name"])]
        rows.append(row("actuator", joint["name"] + "_act", "ctrlrange",
                        [-limit, limit], model.actuator_ctrlrange[actuator].tolist(),
                        "experiment:control_limit", "direct", unit="N m",
                        ok=close([-limit, limit], model.actuator_ctrlrange[actuator]),
                        note="equal to the URDF rated effort"))
        rows.append(row("actuator", joint["name"] + "_act", "gear", 1.0,
                        float(model.actuator_gear[actuator, 0]), "experiment:gear",
                        "direct", ok=close(1.0, model.actuator_gear[actuator, 0])))
        rows.append(row("actuator", joint["name"] + "_act", "forcelimited", False,
                        bool(model.actuator_forcelimited[actuator]), "experiment:forcerange",
                        "none", gate=False,
                        note="force is bounded by ctrlrange and joint actuatorfrcrange"))
    return rows


def check_geoms(model, source) -> list[dict]:
    rows = []
    for collision in source["collisions"]:
        gid = geom_id(model, collision["name"])
        rows.append(row("geom", collision["name"], "body", collision["link"],
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
                        if int(model.geom_bodyid[gid]) else "world", "urdf:body", "direct",
                        ok=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                             int(model.geom_bodyid[gid])) == collision["link"]))
        compiled_type = mujoco.mjtGeom(int(model.geom_type[gid])).name.removeprefix("mjGEOM_").lower()
        rows.append(row("geom", collision["name"], "type", collision["shape"], compiled_type,
                        f"urdf:collision[@name='{collision['name']}']/geometry", "direct",
                        ok=collision["shape"] == compiled_type))
        if collision["size"] is not None:
            rows.append(row("geom", collision["name"], "size", collision["size"],
                            model.geom_size[gid].tolist(),
                            f"urdf:collision[@name='{collision['name']}']/geometry", "direct",
                            unit="m", ok=close(collision["size"], model.geom_size[gid])))
        compiled_quat = model.geom_quat[gid].tolist()
        declared_quat = rpy_to_quat(*collision["rpy"])
        rows.append(row("geom", collision["name"], "pos", collision["xyz"],
                        model.geom_pos[gid].tolist(),
                        f"urdf:collision[@name='{collision['name']}']/origin/@xyz", "direct",
                        unit="m", ok=close(collision["xyz"], model.geom_pos[gid])))
        rows.append(row("geom", collision["name"], "quat", declared_quat, compiled_quat,
                        f"urdf:collision[@name='{collision['name']}']/origin/@rpy", "direct",
                        ok=quat_close(declared_quat, compiled_quat)))
        declared_friction = [ROBOT_FRICTION[0] if collision["friction_declared"] is None
                             else collision["friction_declared"],
                             ROBOT_FRICTION[1], ROBOT_FRICTION[2]]
        provenance = "urdf_contact" if collision["friction_declared"] is not None else "user_required"
        rows.append(row("geom", collision["name"], "friction", declared_friction,
                        model.geom_friction[gid].tolist(),
                        f"urdf:collision[@name='{collision['name']}']/contact/@lateral_friction",
                        provenance, ok=close(declared_friction, model.geom_friction[gid]),
                        note=("declared 0.6 in URDF" if provenance == "urdf_contact"
                              else "wheel collision has no contact element; 0.6 is a user requirement")))
        rows.append(row("geom", collision["name"], "priority", collision["priority"],
                        int(model.geom_priority[gid]),
                        f"urdf:collision[@name='{collision['name']}']/contact/@priority",
                        "direct", ok=collision["priority"] == int(model.geom_priority[gid])))
        rows.append(row("geom", collision["name"], "condim", collision["condim"],
                        int(model.geom_condim[gid]),
                        f"urdf:collision[@name='{collision['name']}']/contact/@condim",
                        "direct", ok=collision["condim"] == int(model.geom_condim[gid])))
        rows.append(row("geom", collision["name"], "contype", ROBOT_CONTYPE,
                        int(model.geom_contype[gid]), "experiment:self_collision", "direct",
                        ok=int(model.geom_contype[gid]) == ROBOT_CONTYPE))
        rows.append(row("geom", collision["name"], "conaffinity", ROBOT_CONAFFINITY,
                        int(model.geom_conaffinity[gid]), "experiment:self_collision", "direct",
                        ok=int(model.geom_conaffinity[gid]) == ROBOT_CONAFFINITY))
    return rows


def check_options(model) -> list[dict]:
    options = [
        ("timestep", TIMESTEP, float(model.opt.timestep), "s"),
        ("gravity", [0.0, 0.0, -9.81], model.opt.gravity.tolist(), "m/s^2"),
        ("solver", "Newton", mujoco.mjtSolver(int(model.opt.solver)).name.removeprefix("mjSOL_").lower(), ""),
        ("cone", "elliptic", mujoco.mjtCone(int(model.opt.cone)).name.removeprefix("mjCONE_").lower(), ""),
        ("impratio", 10.0, float(model.opt.impratio), ""),
        ("iterations", 200, int(model.opt.iterations), ""),
        ("integrator", "Euler", mujoco.mjtIntegrator(int(model.opt.integrator)).name.removeprefix("mjINT_").lower(), ""),
        ("disableflags", 0, int(model.opt.disableflags), ""),
    ]
    rows = []
    for field, declared, compiled, unit in options:
        if isinstance(declared, str):
            ok = declared.lower() == str(compiled).lower()
        else:
            ok = close(declared, compiled)
        rows.append(row("option", "option", field, declared, compiled,
                        "experiment:simulation_assumption", "direct", unit=unit, ok=ok))
    return rows


def check_counts(model, source) -> list[dict]:
    collisions = sum(1 for g in range(model.ngeom)
                     if re.search(r"_col_\d+$",
                                  mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
    rows = [
        row("counts", "model", "bodies", 96, int(model.nbody), "urdf:link", "direct",
            ok=int(model.nbody) == 96),
        row("counts", "model", "links", len(source["links"]), int(model.nbody) - 1,
            "urdf:link", "direct", ok=int(model.nbody) - 1 == len(source["links"])),
        row("counts", "model", "collisions", len(source["collisions"]), collisions,
            "urdf:collision", "direct", ok=collisions == len(source["collisions"])),
        row("counts", "model", "movable_joints", 12, int(model.njnt) - 1,
            "urdf:joint", "direct", ok=int(model.njnt) - 1 == 12),
        row("counts", "model", "actuators", 12, int(model.nu), "experiment:actuator",
            "direct", ok=int(model.nu) == 12),
        row("counts", "model", "total_mass", 126.746, float(model.body_mass.sum()),
            "urdf:link/inertial/mass", "direct", unit="kg",
            ok=close(126.746, model.body_mass.sum(), rtol=1e-6)),
        row("counts", "model", "nq", 19, int(model.nq), "experiment:freejoint",
            "direct", ok=int(model.nq) == 19),
        row("counts", "model", "nv", 18, int(model.nv), "experiment:freejoint",
            "direct", ok=int(model.nv) == 18),
    ]
    return rows


def static_scan(model) -> dict:
    data = mujoco.MjData(model)
    movable = [j for j in range(model.njnt)
               if int(model.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE,
                                             mujoco.mjtJoint.mjJNT_SLIDE)]
    base = model.qpos0.copy()

    def pairs_at() -> list[tuple[str, str, float]]:
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        found = []
        for i in range(data.ncon):
            contact = data.contact[i]
            if float(contact.dist) >= 0.0:
                continue
            a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            found.append((min(a, b), max(a, b), float(contact.dist)))
        return found

    data.qpos[:] = base
    neutral_raw = pairs_at()
    neutral_by_pair = {}
    for a, b, dist in neutral_raw:
        key = (a, b)
        neutral_by_pair[key] = min(neutral_by_pair.get(key, 0.0), dist)
    neutral = [{"pair": [a, b], "dist_m": dist} for (a, b), dist in sorted(neutral_by_pair.items())]
    by_joint = {}
    for jid in movable:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        adr = int(model.jnt_qposadr[jid])
        if bool(model.jnt_limited[jid]):
            values = np.linspace(model.jnt_range[jid][0], model.jnt_range[jid][1], 41)
        else:
            values = np.linspace(-math.pi, math.pi, 41)
        found = {}
        for value in values:
            data.qpos[:] = base
            data.qpos[adr] = value
            for a, b, dist in pairs_at():
                key = (a, b)
                entry = found.setdefault(key, {"pair": list(key),
                                               "min_deg": math.degrees(value),
                                               "max_deg": math.degrees(value),
                                               "min_dist_m": dist})
                entry["min_deg"] = min(entry["min_deg"], math.degrees(value))
                entry["max_deg"] = max(entry["max_deg"], math.degrees(value))
                entry["min_dist_m"] = min(entry["min_dist_m"], dist)
        by_joint[name] = sorted(found.values(), key=lambda e: e["pair"])
    return {
        "pass": not neutral,
        "neutral": neutral,
        "by_joint": by_joint,
        "samples_per_joint": 41,
    }


def ground_scene() -> str:
    xml = MJCF.read_text()
    mu = " ".join(f"{v:g}" for v in TERRAIN_FRICTION)
    plane = (f'    <geom name="ground" type="plane" size="30 30 0.1" contype="2" '
             f'conaffinity="1" friction="{mu}" rgba=".72 .72 .72 1"/>\n')
    return xml.replace("<worldbody>", "<worldbody>\n" + plane, 1)


def settle(rocker_deg: float) -> dict:
    model = mujoco.MjModel.from_xml_string(ground_scene())
    data = mujoco.MjData(model)
    wheel_bodies = [b for b in range(model.nbody)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("_wheel")]
    rockers = [j for j in range(model.njnt)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("_rocker_joint")]
    for jid in rockers:
        data.qpos[int(model.jnt_qposadr[jid])] = math.radians(rocker_deg)
    lo, hi = 0.2, 1.5
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        data.qpos[:7] = model.qpos0[:7]
        data.qpos[2] = mid
        mujoco.mj_kinematics(model, data)
        if min(float(data.xpos[b][2]) for b in wheel_bodies) - WHEEL_RADIUS < 0.0:
            lo = mid
        else:
            hi = mid
    data.qpos[2] = 0.5 * (lo + hi) + 0.002
    mujoco.mj_forward(model, data)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    limits = model.actuator_ctrlrange.copy()
    limited = [(j, int(model.jnt_qposadr[j]), float(model.jnt_range[j][0]),
                float(model.jnt_range[j][1]))
               for j in range(model.njnt) if bool(model.jnt_limited[j])]
    warnings = {i: 0 for i in range(len(data.warning))}
    max_penetration = 0.0
    worst_pair = None
    max_overshoot = 0.0
    nonfinite = False
    self_pairs = set()
    self_force: dict[tuple[str, str], float] = {}
    ground_force: dict[str, float] = {}
    force = np.zeros(6)
    steps = int(round(SETTLE_SECONDS / model.opt.timestep))
    for _ in range(steps):
        for i in range(model.nu):
            jid = int(model.actuator_trnid[i, 0])
            adr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname.endswith("_wheel_joint"):
                ctrl = -0.5 * float(data.qvel[dof])
            else:
                target = math.radians(rocker_deg) if jname.endswith("_rocker_joint") else 0.0
                ctrl = 25.0 * (target - float(data.qpos[adr])) - 0.6 * float(data.qvel[dof])
            data.ctrl[i] = float(np.clip(ctrl, limits[i][0], limits[i][1]))
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                and np.isfinite(data.qfrc_actuator).all()):
            nonfinite = True
            break
        for i, warning in enumerate(data.warning):
            if int(warning.number) != warnings[i]:
                warnings[i] = int(warning.number)
        ground_force = {}
        for i in range(data.ncon):
            contact = data.contact[i]
            depth = max(0.0, -float(contact.dist))
            if depth > max_penetration:
                max_penetration = depth
                a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                worst_pair = [a, b]
            if int(contact.geom1) != 0 and int(contact.geom2) != 0:
                pair = tuple(sorted((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                       int(contact.geom1)),
                                     mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                       int(contact.geom2)))))
                self_pairs.add(pair)
                mujoco.mj_contactForce(model, data, i, force)
                self_force[pair] = max(self_force.get(pair, 0.0), abs(float(force[0])))
            elif int(contact.geom1) == 0 or int(contact.geom2) == 0:
                other = int(contact.geom2) if int(contact.geom1) == 0 else int(contact.geom1)
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
                if name and name.endswith("_col_1") and "wheel" in name:
                    mujoco.mj_contactForce(model, data, i, force)
                    ground_force[name] = ground_force.get(name, 0.0) + max(0.0, float(force[0]))
        for _, adr, low, high in limited:
            max_overshoot = max(max_overshoot, low - float(data.qpos[adr]),
                                float(data.qpos[adr]) - high, 0.0)

    warned = {int(k): v for k, v in warnings.items() if v}
    peak_self_force = max(self_force.values(), default=0.0)
    passed = (not nonfinite and not warned
              and max_penetration <= SETTLE_PENETRATION_LIMIT_M
              and peak_self_force <= SETTLE_SELF_FORCE_LIMIT_N
              and max_overshoot <= 1e-9)
    return {
        "pass": passed,
        "rocker_deg": rocker_deg,
        "seconds": SETTLE_SECONDS,
        "nonfinite": nonfinite,
        "warnings": warned,
        "max_penetration_m": max_penetration,
        "worst_pair": worst_pair,
        "max_joint_overshoot_rad": max_overshoot,
        "self_contact_pairs": sorted([list(p) for p in self_pairs]),
        "self_contact_force_n": {" | ".join(pair): value for pair, value in sorted(self_force.items())},
        "ground_normal_force_n": dict(sorted(ground_force.items())),
        "self_force_limit_n": SETTLE_SELF_FORCE_LIMIT_N,
        "final_rocker_deg": [float(np.degrees(data.qpos[int(model.jnt_qposadr[j])]))
                             for j in rockers],
        "penetration_limit_m": SETTLE_PENETRATION_LIMIT_M,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse the existing MJCF and re-run the audit")
    args = parser.parse_args(argv)

    source = parse_urdf()
    if not args.skip_build or not MJCF.exists():
        build(source)
    model = mujoco.MjModel.from_xml_path(str(MJCF))

    rows = (check_links(model, source) + check_joints(model, source)
            + check_geoms(model, source) + check_options(model)
            + check_counts(model, source))
    checks = {
        "static_scan": static_scan(model),
        "settle": [settle(0.0), settle(18.0)],
    }
    failed = [r for r in rows if r["gate"] and not r["pass"]]
    all_pass = (not failed and checks["static_scan"]["pass"]
                and all(s["pass"] for s in checks["settle"]))

    spec = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "urdf": str(URDF.relative_to(ROOT)),
        "urdf_sha256": sha256(URDF),
        "mjcf": str(MJCF.relative_to(ROOT)),
        "mjcf_sha256": sha256(MJCF),
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "assumptions": {
            "gravity": [0.0, 0.0, -9.81],
            "timestep_s": TIMESTEP,
            "integrator": "Euler",
            "solver": "Newton",
            "cone": "elliptic",
            "impratio": 10,
            "iterations": 200,
            "solref": "0.004 1 (MuJoCo default)",
            "robot_friction": list(ROBOT_FRICTION),
            "robot_friction_note": "76 structure collisions declare 0.6; 6 wheels have no contact element and take the user-required 0.6",
            "self_collision": "robot contype=1 conaffinity=3, terrain contype=2 conaffinity=1",
            "terrain_friction_ground": 4.0,
            "mixing": "same priority 0, element-wise max",
            "mesh_paths": "absolute",
        },
        "friction_provenance": {
            "structure_urdf": sum(1 for c in source["collisions"] if c["friction_declared"] is not None),
            "wheel_urdf": sum(1 for c in source["collisions"] if c["friction_declared"] is None),
        },
        "summary": {
            "rows": len(rows),
            "gate_rows": sum(1 for r in rows if r["gate"]),
            "gate_failed": len(failed),
            "all_pass": all_pass,
        },
        "checks": checks,
        "rows": rows,
    }
    SPEC.write_text(json.dumps(spec, indent=2) + "\n")

    print(f"[spec] {SPEC.relative_to(ROOT)}  rows={len(rows)} gate_failed={len(failed)}")
    for failure in failed[:20]:
        print(f"  FAIL {failure['section']}/{failure['name']}/{failure['field']}: "
              f"declared={failure['declared']} compiled={failure['compiled']}")
    scan = checks["static_scan"]
    print(f"[scan] neutral_pairs={len(scan['neutral'])} pass={scan['pass']}")
    for entry in scan["neutral"]:
        print(f"  neutral overlap {entry['pair'][0]} | {entry['pair'][1]} "
              f"dist={entry['dist_m'] * 1000:.2f}mm")
    for name, entries in scan["by_joint"].items():
        if entries:
            print(f"  {name}: {len(entries)} self-contact pair(s)")
    for run in checks["settle"]:
        print(f"[settle] rocker={run['rocker_deg']:g} pass={run['pass']} "
              f"penetration={run['max_penetration_m'] * 1000:.2f}mm "
              f"overshoot={run['max_joint_overshoot_rad']:.3g}rad "
              f"warnings={run['warnings']} "
              f"self_force={run['self_contact_force_n']}")
    print(f"[p0] {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
