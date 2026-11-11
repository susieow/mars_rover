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
                "generate_dataset.py", "verify_dataset.py", "climb.py", "dataset_plan.py", "publish_dataset.py")
# A long-lived generator must not label already-imported code with newer file contents.
LOADED_SOURCE_SHA256 = {name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest()
                        for name in SOURCE_FILES}

import mujoco

import course
from climb import ROOT, ROBOT_XML, URDF
from dataset_io import EpisodeRecorder
from dataset_plan import (SPLITS, CASE_KEYS, build_cases, fingerprint, require_release, write_release)
from episode_audit import EpisodeAudit
from verify_dataset import inherited_collision_policy, mesh_hashes, sha256, verify_episode

BASE = Path(os.environ.get("ROVER_DATASET_ROOT", str(Path(__file__).resolve().parents[2] / "datasets/45deg_v3")))

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


def policy_snapshot(params, stairs):
    sources = current_source_hashes()
    if sources != LOADED_SOURCE_SHA256:
        raise RuntimeError("source files changed since imports; start a fresh generator process")
    return {
        "schema": "rover_policy_snapshot_v1",
        "source_sha256": sources,
        "controller_sha256": sources["closed_loop_controller.py"],
        "model_sha256": digest(ROBOT_XML), "urdf_sha256": digest(URDF),
        "mesh_sha256": mesh_hashes(ROBOT_XML), "mujoco_version": mujoco.__version__,
        "control_dt": course.CTRL_DT,
        "simulation_parameters": {key: value for key, value in params.items() if key not in CASE_KEYS},
        "geometry_rule": {"stairs": stairs, "depth_equals_height": True, "slope_deg": 45.0},
    }


def append_attempt(summary):
    compact = {key: value for key, value in summary.items() if key not in {
        "audit", "events", "source_sha256", "source_sha256_end", "source_loaded_sha256",
        "mesh_sha256", "independent_verification", "limitations", "policy_snapshot", "release_snapshot"}}
    with (BASE / "attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(compact, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

def generate(episode_id, height, params, *, stairs=15, split="train", case_spec=None, release=None):
    params = dict(params)
    for key in ("init_x", "init_y", "init_yaw_deg"):
        params.setdefault(key, 0.0)
    if Path(episode_id).name != episode_id or episode_id in {"", ".", ".."}:
        raise ValueError("episode_id must be a nonempty filename component")
    concrete = dict(height=height, stairs=stairs, depth=height,
                    **{key: params[key] for key in ("init_x", "init_y", "init_yaw_deg")})
    candidates = build_cases(split)
    if case_spec is None:
        case_spec = next((case for case in candidates if case["parameters"] == concrete), None)
    if case_spec not in candidates or case_spec["parameters"] != concrete:
        raise ValueError("episode geometry or initial state does not match its fixed split")
    policy = policy_snapshot(params, stairs)
    if split == "test":
        require_release(release, policy)
    elif release is not None:
        require_release(release, policy)
    folder = BASE / "episodes" / episode_id
    if folder.exists():
        raise FileExistsError(f"episode already exists: {folder}")
    folder.mkdir(parents=True)
    source_hashes = freeze_sources(folder)
    model_hash = digest(ROBOT_XML)
    urdf_hash = digest(URDF)
    collision_policy = inherited_collision_policy()
    meshes = mesh_hashes(ROBOT_XML)
    captured_policy = dict(policy, source_sha256=source_hashes,
                           controller_sha256=source_hashes["closed_loop_controller.py"],
                           model_sha256=model_hash, urdf_sha256=urdf_hash, mesh_sha256=meshes)
    if captured_policy != policy:
        raise RuntimeError("policy assets changed before the episode started")
    recorder = EpisodeRecorder(control_dt=course.CTRL_DT, record_physics=True)
    audit_holder = {}
    def on_control(m, d, t, controller):
        if "audit" not in audit_holder:
            audit_holder["audit"] = EpisodeAudit(m)
        recorder.on_control(m, d, t, controller)
    def on_step(m, d, t):
        recorder.on_step(m, d, t)
        audit_holder["audit"].step(m, d, t)
    metadata_path = folder / "metadata.json"
    assignment_fields = {
        "split": split, "case_spec": case_spec, "policy_sha256": fingerprint(policy),
        "policy_snapshot": policy, "release_snapshot": release,
    }
    attempt = {
        "case": episode_id, "dataset": "45deg_v3", "attempt_status": "started",
        "dataset_accepted": False, "simulation_success": False,
        "automated_checks_passed": False, "visual_review_status": "not_eligible",
        "parameters": dict(height=height, stairs=stairs, depth=height, **params),
        "metadata_path": str(metadata_path), "source_sha256": source_hashes,
        "model_sha256": model_hash, "urdf_sha256": urdf_hash, "mesh_sha256": meshes,
        "quality_rejection_reasons": ["attempt_not_completed"], **assignment_fields,
    }
    atomic_json(metadata_path, attempt)
    try:
        result = course.simulate(height, stairs, height, on_control=on_control, on_step=on_step, **params)
    except BaseException as error:
        attempt.update(attempt_status="error",
                       quality_rejection_reasons=[f"simulation_error:{type(error).__name__}:{error}"])
        atomic_json(metadata_path, attempt)
        append_attempt(attempt)
        raise
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
        **result, **record, **assignment_fields,
        "attempt_status": "completed",
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
        "parameters": dict(height=height, stairs=stairs, depth=height, **params),
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
    append_attempt(summary)
    print(json.dumps({k:summary[k] for k in ["case","simulation_success","automated_checks_passed",
                     "quality_rejection_reasons","duration_s","max_roll_deg","max_contact_penetration_m"]}),
          flush=True)
    return summary


def main(argv=None):
    global BASE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=SPLITS, default="train")
    p.add_argument("--heights", help="Assigned heights in cm; default train: 15,20")
    p.add_argument("--nstairs", "--stairs", dest="nstairs", type=int,
                   help="15 for new cases; 12 only for historical 4 cm regression")
    p.add_argument("--prefix", default="dev")
    p.add_argument("--variants", help="Explicit case variants; defaults to base (validation: validation_left)")
    p.add_argument("--contact-timeconst", type=float, default=0.004)
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--dataset-root", type=Path, default=BASE)
    p.add_argument("--release", type=Path, help="Explicit frozen release required for test execution")
    p.add_argument("--freeze-release", type=Path, help="Write a new release snapshot and exit; never runs cases")
    p.add_argument("--dry-run", action="store_true", help="Print case plan without creating files or running simulation")
    a = p.parse_args(argv)
    BASE = a.dataset_root
    try:
        heights = None if a.heights is None else [int(cm.strip()) for cm in a.heights.split(",")]
        variants = (["validation_left"] if a.split == "validation" else ["base"]) if a.variants is None else [name.strip() for name in a.variants.split(",")]
        cases = build_cases(a.split, heights, variants, a.nstairs)
        if not a.prefix or Path(a.prefix).name != a.prefix or a.prefix in {".", ".."}:
            raise ValueError("prefix must be a nonempty filename component")
        params = dict(seconds=90.0, speed=a.speed, friction=4.0, tau_max=50.0,
                      nosing=0.0, front_lift_deg=0.0, middle_lift_deg=0.0,
                      rocker_gain=600.0, rocker_damp=150.0, wheel_gain=10.0,
                      descent_wheel_gain=40.0, steer_damp=50.0,
                      descent_speed=1.0, settle_time=1.2,
                      quality_pitch_limit=60.0, quality_roll_limit=15.0,
                      contact_timeconst=a.contact_timeconst)
        policy = policy_snapshot(params, cases[0]["parameters"]["stairs"])
        release = None if a.release is None else json.loads(a.release.read_text(encoding="utf-8"))
        if release is not None:
            require_release(release, policy)
        if a.dry_run:
            print(json.dumps({
                "dry_run": True, "simulation_started": False, "split": a.split,
                "selected_split_executable": a.split != "test" or release is not None,
                "test_release_valid": release is not None,
                "freeze_release_requested": None if a.freeze_release is None else str(a.freeze_release),
                "policy_sha256": fingerprint(policy), "cases": cases,
            }, ensure_ascii=False, indent=2))
            return 0
        if a.freeze_release is not None:
            if a.split == "regression":
                raise ValueError("freeze a development policy with the new 15-riser geometry")
            document = write_release(a.freeze_release, policy)
            print(json.dumps({"release_frozen": str(a.freeze_release),
                              "policy_sha256": document["policy_sha256"],
                              "simulation_started": False}, indent=2))
            return 0
        if a.split == "test":
            require_release(release, policy)
        for case in cases:
            initial = {key: case["parameters"][key] for key in ("init_x", "init_y", "init_yaw_deg")}
            generate(f"{a.prefix}_{a.split}_{case['case_id']}", case["parameters"]["height"],
                     dict(params, **initial), stairs=case["parameters"]["stairs"],
                     split=a.split, case_spec=case, release=release)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
