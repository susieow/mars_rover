"""Record complete traversals and refuse invalid or unreproducible episodes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import mujoco

from climb import ROOT, ROBOT_XML
from course import simulate
from dataset_io import EpisodeRecorder, replay_check
from episode_audit import EpisodeAudit

BASE = Path("/home/agilex/wzy/rover/datasets/45deg_v3")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def generate(episode_id, height, params):
    folder = BASE / "episodes" / episode_id
    if folder.exists():
        raise FileExistsError(f"episode already exists: {folder}")
    folder.mkdir(parents=True)
    recorder = EpisodeRecorder(control_dt=0.02, record_physics=True)
    audit_holder = {}
    def on_control(m, d, t, controller):
        if "audit" not in audit_holder:
            audit_holder["audit"] = EpisodeAudit(m)
        recorder.on_control(m, d, t, controller)
    def on_step(m, d, t):
        recorder.on_step(m, d, t)
        audit_holder["audit"].step(m, d, t)
    result = simulate(height, 12, height, on_control=on_control, on_step=on_step, **params)
    scene = folder / "scene.xml"
    xml = (ROOT / f"scene_course_{os.getpid()}.xml").read_text()
    xml = xml.replace('file="meshes/', f'file="{ROOT}/meshes/')
    scene.write_text(xml)
    archive = folder / "trajectory.npz"
    record = recorder.save(archive, scene)
    audit = audit_holder["audit"].result()
    replay = replay_check(scene, archive)
    reasons = list(audit["quality_rejection_reasons"])
    if not result["success"]:
        reasons.append("full_route_or_final_pose_failed")
    if result["max_abs_y"] > 0.5:
        reasons.append("lateral_drift_over_0.5m")
    if result["max_yaw_error_deg"] > 30:
        reasons.append("heading_deviation_over_30deg")
    if not replay["replay_ok"]:
        reasons.append("action_replay_mismatch")
    if audit.get("continuous_final_stability_s", 0) < 1.0:
        reasons.append("continuous_stability_less_than_1s")
    reasons = sorted(set(reasons))
    source_files = ["course.py", "closed_loop_controller.py", "episode_audit.py",
                    "dataset_io.py", "generate_dataset.py"]
    source_dir = folder / "sources"
    source_dir.mkdir()
    source_hashes = {}
    for name in source_files:
        source = Path(__file__).parent / name
        shutil.copy2(source, source_dir / name)
        source_hashes[name] = digest(source)
    summary = {
        **result, **record,
        "case": episode_id,
        "dataset": "45deg_v3",
        "dataset_accepted": False,
        "simulation_success": result["success"],
        "physics_valid": not audit["quality_rejection_reasons"],
        "automated_checks_passed": not reasons,
        "visual_review_status": "pending" if not reasons else "not_eligible",
        "quality_rejection_reasons": reasons,
        "final_wheel_contacts": audit["final_ground_contact_by_wheel"],
        "max_contact_penetration_m": audit["max_contact_penetration_m"],
        "measured_actuator_peaks_nm": audit["actuator_force_peak_by_kind_nm"],
        "continuous_final_stability_s": audit.get("continuous_final_stability_s", 0),
        "parameters": dict(height=height, stairs=12, depth=height, **params),
        "metadata_path": str(folder / "metadata.json"),
        "scene_path": str(scene),
        "model_sha256": digest(ROBOT_XML),
        "source_sha256": source_hashes,
        "mujoco_version": mujoco.__version__,
        "audit": audit, "replay": replay,
        "scope": "full_ascent_platform_descent_ground_stop",
        "limitations": [
            "Simulation demonstrations; friction and contact response are not hardware calibrated.",
            "Original robot model retained, including disabled self-collision and collision/visual tire-width differences.",
            "Controller uses privileged simulator contacts and world pose; not a sensor-only deployment controller."
        ],
    }
    (folder / "metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    compact = {k:v for k,v in summary.items() if k not in {"audit","events","source_sha256","limitations"}}
    with (BASE / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(compact, ensure_ascii=False) + "\n")
    print(json.dumps({k:summary[k] for k in ["case","simulation_success","automated_checks_passed",
                     "quality_rejection_reasons","duration_s","max_roll_deg","max_contact_penetration_m"]}),
          flush=True)
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heights", default="4,5,6")
    p.add_argument("--prefix", default="seed")
    p.add_argument("--variants", default="base,yaw_left,yaw_right")
    p.add_argument("--contact-timeconst", type=float, default=0.008)
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
                          nosing=0.0, front_lift_deg=14.0, middle_lift_deg=17.0,
                          rocker_gain=160.0, rocker_damp=8.0, wheel_gain=10.0,
                          descent_speed=1.0, settle_time=1.2,
                          quality_pitch_limit=60.0, quality_roll_limit=15.0,
                          contact_timeconst=a.contact_timeconst, **offsets[variant])
            generate(f"{a.prefix}_h{int(cm):02d}_{variant}", float(cm)/100.0, params)
