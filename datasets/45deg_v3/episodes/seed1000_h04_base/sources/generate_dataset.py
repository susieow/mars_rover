"""Record complete traversals and refuse invalid or unreproducible episodes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

SOURCE_DIR = Path(__file__).resolve().parent
SOURCE_FILES = ("course.py", "closed_loop_controller.py", "episode_audit.py", "dataset_io.py",
                "generate_dataset.py", "verify_dataset.py", "climb.py")
# A long-lived generator must not label already-imported code with newer file contents.
LOADED_SOURCE_SHA256 = {name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest()
                        for name in SOURCE_FILES}

import mujoco

import course
from climb import ROOT, ROBOT_XML, URDF
from dataset_io import EpisodeRecorder
from episode_audit import EpisodeAudit
from verify_dataset import inherited_collision_policy, mesh_hashes, sha256, verify_episode

BASE = Path("/home/agilex/wzy/rover/datasets/45deg_v3")

def digest(path):
    return sha256(path)


def current_source_hashes():
    return {name: digest(SOURCE_DIR / name) for name in SOURCE_FILES}


def source_change_reasons(start, end):
    return [f"source_changed_during_generation:{name}" for name in sorted(set(start) | set(end))
            if start.get(name) != end.get(name)]


def freeze_sources(folder):
    source_dir = folder / "sources"
    source_dir.mkdir()
    captured = {name: (SOURCE_DIR / name).read_bytes() for name in SOURCE_FILES}
    hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in captured.items()}
    if hashes != LOADED_SOURCE_SHA256:
        raise RuntimeError("source files changed since imports; start a fresh generator process")
    for name, contents in captured.items():
        (source_dir / name).write_bytes(contents)
    return hashes


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def generate(episode_id, height, params):
    folder = BASE / "episodes" / episode_id
    if folder.exists():
        raise FileExistsError(f"episode already exists: {folder}")
    folder.mkdir(parents=True)
    source_hashes = freeze_sources(folder)
    model_hash = digest(ROBOT_XML)
    urdf_hash = digest(URDF)
    collision_policy = inherited_collision_policy()
    meshes = mesh_hashes(ROBOT_XML)
    recorder = EpisodeRecorder(control_dt=course.CTRL_DT, record_physics=True)
    audit_holder = {}
    def on_control(m, d, t, controller):
        if "audit" not in audit_holder:
            audit_holder["audit"] = EpisodeAudit(m)
        recorder.on_control(m, d, t, controller)
    def on_step(m, d, t):
        recorder.on_step(m, d, t)
        audit_holder["audit"].step(m, d, t)
    result = course.simulate(height, 12, height, on_control=on_control, on_step=on_step, **params)
    scene = folder / "scene.xml"
    xml = (ROOT / f"scene_course_{os.getpid()}.xml").read_text()
    xml = xml.replace('file="meshes/', f'file="{ROOT}/meshes/')
    scene.write_text(xml)
    archive = folder / "trajectory.npz"
    record = recorder.save(archive, scene)
    audit = audit_holder["audit"].result()
    reasons = list(audit["quality_rejection_reasons"])
    if not result["success"]:
        reasons.append("full_route_or_final_pose_failed")
    if result["max_abs_y"] > 0.5:
        reasons.append("lateral_drift_over_0.5m")
    if result["max_yaw_error_deg"] > 30:
        reasons.append("heading_deviation_over_30deg")
    if audit.get("continuous_final_stability_s", 0) < 1.0:
        reasons.append("continuous_stability_less_than_1s")
    source_end = current_source_hashes()
    reasons.extend(source_change_reasons(source_hashes, source_end))
    if digest(ROBOT_XML) != model_hash:
        reasons.append("robot_model_changed_during_generation")
    if digest(URDF) != urdf_hash:
        reasons.append("urdf_changed_during_generation")
    if mesh_hashes(scene) != meshes:
        reasons.append("mesh_content_changed_during_generation")
    metadata_path = folder / "metadata.json"
    summary = {
        **result, **record,
        "case": episode_id,
        "dataset": "45deg_v3",
        "dataset_accepted": False,
        "simulation_success": result["success"],
        "physics_valid": not audit["quality_rejection_reasons"],
        "automated_checks_passed": False,
        "visual_review_status": "not_eligible",
        "quality_rejection_reasons": sorted(set([*reasons, "independent_verification_pending"])),
        "final_wheel_contacts": audit["final_ground_contact_by_wheel"],
        "max_contact_penetration_m": audit["max_contact_penetration_m"],
        "measured_actuator_peaks_nm": audit["actuator_force_peak_by_kind_nm"],
        "continuous_final_stability_s": audit.get("continuous_final_stability_s", 0),
        "parameters": dict(height=height, stairs=12, depth=height, **params),
        "metadata_path": str(metadata_path),
        "scene_path": str(scene),
        "model_sha256": model_hash,
        "urdf_sha256": urdf_hash,
        "model_identity": {"verification_pending": True, "reference_model_sha256": model_hash,
                           "inherited_collision_policy": collision_policy},
        "source_sha256": source_hashes,
        "source_sha256_end": source_end,
        "source_loaded_sha256": dict(LOADED_SOURCE_SHA256),
        "mesh_sha256": meshes,
        "mujoco_version": mujoco.__version__,
        "audit": audit,
        "scope": "full_ascent_platform_descent_ground_stop",
        "limitations": [
            "Simulation demonstrations; friction and contact response are not hardware calibrated.",
            "Original robot model retained, including disabled self-collision and collision/visual tire-width differences.",
            "Controller uses privileged simulator contacts and world pose; not a sensor-only deployment controller."
        ],
    }
    atomic_json(metadata_path, summary)
    try:
        verification = verify_episode(metadata_path)
    except Exception as error:
        verification = {"verify_ok": False,
                        "rejection_reasons": [f"independent_verification_error:{type(error).__name__}:{error}"]}
    reasons.extend(verification["rejection_reasons"])
    source_end = current_source_hashes()
    changes = source_change_reasons(source_hashes, source_end)
    reasons.extend(changes)
    if changes:
        verification["verify_ok"] = False
        verification["rejection_reasons"] = sorted(set([*verification["rejection_reasons"], *changes]))
    reasons = sorted(set(reasons))
    summary.update({
        "source_sha256_end": source_end,
        "independent_verification": verification,
        "model_identity": verification.get("model_identity", summary["model_identity"]),
        "replay": verification.get("replay", {"replay_ok": False, "reason": "independent_verification_incomplete"}),
        "physics_valid": verification.get("audit", audit)["audit_quality_ok"],
        "automated_checks_passed": bool(verification["verify_ok"]) and not reasons,
        "visual_review_status": "pending" if verification["verify_ok"] and not reasons else "not_eligible",
        "quality_rejection_reasons": reasons,
    })
    atomic_json(metadata_path, summary)
    compact = {k:v for k,v in summary.items() if k not in {
        "audit", "events", "source_sha256", "source_sha256_end", "source_loaded_sha256",
        "mesh_sha256", "independent_verification", "limitations"}}
    with (BASE / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(compact, ensure_ascii=False) + "\n")
    print(json.dumps({k:summary[k] for k in ["case","simulation_success","automated_checks_passed",
                     "quality_rejection_reasons","duration_s","max_roll_deg","max_contact_penetration_m"]}),
          flush=True)
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heights", default="4")
    p.add_argument("--prefix", default="seed")
    p.add_argument("--variants", default="base")
    p.add_argument("--contact-timeconst", type=float, default=0.004)
    p.add_argument("--speed", type=float, default=6.0)
    a = p.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    for cm in a.heights.split(","):
        for variant in a.variants.split(","):
            offsets = {
                "base": {"init_yaw_deg":0.0,"init_y":0.0,"init_x":0.0},
                "yaw_left": {"init_yaw_deg":1.0,"init_y":0.015,"init_x":-0.02},
                "yaw_right": {"init_yaw_deg":-1.0,"init_y":-0.015,"init_x":0.02},
            }
            if variant not in offsets:
                raise ValueError(f"unknown variant: {variant}")
            params = dict(seconds=90.0, speed=a.speed, friction=4.0, tau_max=50.0,
                          nosing=0.0, front_lift_deg=0.0, middle_lift_deg=0.0,
                          rocker_gain=600.0, rocker_damp=150.0, wheel_gain=10.0,
                          descent_wheel_gain=40.0, steer_damp=50.0,
                          descent_speed=1.0, settle_time=1.2,
                          quality_pitch_limit=60.0, quality_roll_limit=15.0,
                          contact_timeconst=a.contact_timeconst, **offsets[variant])
            generate(f"{a.prefix}_h{int(cm):02d}_{variant}", float(cm)/100.0, params)
