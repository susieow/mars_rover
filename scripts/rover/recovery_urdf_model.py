"""Build recovery scenes from the pinned Phase B URDF, never the legacy MJCF."""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import urdf_slope_model as p0


def build_scene(folder, friction, angle_deg):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    source = p0.parse_urdf()
    old = p0.MODEL_DIR, p0.MJCF
    try:
        p0.MODEL_DIR = folder / "model"
        p0.MJCF = p0.MODEL_DIR / "rover_p0.xml"
        p0.build(source)
        tree = ET.parse(p0.MJCF)
    finally:
        p0.MODEL_DIR, p0.MJCF = old
    (folder / "source.urdf").write_bytes(p0.URDF.read_bytes())
    world = tree.getroot().find("worldbody")
    ET.SubElement(world, "light", name="sun", pos="0 0 14", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="80 80 0.1",
                  contype="2", conaffinity="1", friction="4 0.02 0.001", solref="0.004 1")
    depth = 0.2 / math.tan(math.radians(angle_deg))
    x = 1.6
    blocks = [(f"up{i}", depth, 0.2 * (i + 1), friction) for i in range(15)]
    blocks += [("platform", 2.0, 3.0, 4.0)]
    blocks += [(f"down{i}", depth, 0.2 * (14 - i), friction) for i in range(14)]
    for name, length, height, mu in blocks:
        ET.SubElement(world, "geom", name=name, type="box",
                      pos=f"{x + length / 2} 0 {height / 2}",
                      size=f"{length / 2} 2.5 {height / 2}",
                      contype="2", conaffinity="1", friction=f"{mu} 0.02 0.001",
                      solref="0.004 1", rgba=".58 .47 .36 1")
        x += length
    scene = folder / "scene.xml"
    tree.write(scene, encoding="unicode")
    return scene


def model_compliance(model, scene, friction):
    source = p0.parse_urdf()
    rows = (p0.check_links(model, source) + p0.check_joints(model, source)
            + p0.check_geoms(model, source) + p0.check_options(model))
    # The upstream link check compares eigenvalues; also verify tensor orientation.
    for link in source["links"]:
        if not link["has_inertial"]:
            continue
        body = p0.body_id(model, link["name"])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
        rotation = rotation.reshape(3, 3)
        compiled = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
        ixx, ixy, ixz, iyy, iyz, izz = link["inertia"]
        declared = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        rows.append(p0.row("link", link["name"], "inertia_tensor", declared.tolist(),
                           compiled.tolist(), "urdf:inertial/inertia", "direct",
                           ok=p0.close(declared, compiled)))
    xml = ET.parse(p0.URDF).getroot()
    for joint in source["joints"]:
        node = xml.find(f"joint[@name='{joint['name']}']")
        axis = node.find("axis")
        expected = np.array(p0.vec(axis.get("xyz") if axis is not None else "1 0 0", 3))
        expected /= np.linalg.norm(expected)
        jid = p0.joint_id(model, joint["name"])
        rows.append(p0.row("joint", joint["name"], "axis", expected.tolist(),
                           model.jnt_axis[jid].tolist(), "urdf:joint/axis", "direct",
                           ok=p0.close(expected, model.jnt_axis[jid])))
    actual_sha = p0.sha256(p0.URDF)
    rows.append(p0.row("source", "URDF", "sha256", p0.EXPECTED_URDF_SHA,
                       actual_sha, "pinned Phase B URDF", "direct",
                       ok=actual_sha == p0.EXPECTED_URDF_SHA))
    rows.append(p0.row("model", "collision", "exclusions", 0, int(model.nexclude),
                       "Phase B collision policy", "direct", ok=model.nexclude == 0))
    for geom in ET.parse(scene).getroot().findall("./worldbody/geom"):
        name = geom.get("name", "")
        expected = 4.0 if name in ("ground", "platform") else float(friction)
        gid = p0.geom_id(model, name)
        rows.append(p0.row("terrain", name, "friction_slide", expected,
                           float(model.geom_friction[gid, 0]), "experiment assumption", "direct",
                           ok=float(model.geom_friction[gid, 0]) == expected))
    failed = [row for row in rows if row["gate"] and not row["pass"]]
    return {"urdf_sha256": actual_sha, "passed": not failed,
            "scope": "compiled_model_fields_only", "checks": rows, "failed": failed,
            "assumptions": {"ground_and_platform_friction": 4.0,
                            "wheel_friction_source": "user-confirmed 0.6",
                            "self_collision": "Phase B bitmasks; no explicit exclusions"}}
