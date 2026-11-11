"""Reconstruct already evaluated targets for review media, with provenance checks."""

import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
BACKEND = None


def normalize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    return value


def init_worker(folder):
    global BACKEND
    folder = Path(folder)
    spec = importlib.util.spec_from_file_location("review_backend", folder / "backend.py")
    BACKEND = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BACKEND)
    BACKEND.init_worker(folder / "scene.xml", folder / "seed_prefix.npz")


def evaluate(row):
    return BACKEND.evaluate({"id": row["id"], "sequence": row["sequence"],
                             "horizon": 2., "record_states": True})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = {"scope": "Review media for previously evaluated 20 cm fixed targets; no new search or parameters",
        "stop_requested_after_current_batch": True, "further_search_started": False,
        "scene_path": str(BASE / "scene.xml"), "scene_sha256": sha(BASE / "scene.xml"),
        "backend_path": str(BASE / "backend.py"), "backend_sha256": sha(BASE / "backend.py"),
        "simulation_dt_s": .001, "control_period_s": .001,
        "target_columns": ["front_rad_s", "middle_rad_s", "rear_rad_s", "rocker_left_deg", "rocker_right_deg"],
        "batches": [], "candidates": [],
        "media_provenance": "frozen_backend_reconstruction",
        "provenance_limits": "Original per-step paths were not retained for all candidates. Reconstruction is checked against original terminal state, metrics, audit and route, not claimed pointwise equal to unavailable original paths."}
    for batch, folder in (("near_contact", BASE), ("early_entry", BASE / "early_entry")):
        results = [row for i in range(3) for row in json.loads((folder / f"round{i}_results.json").read_text())]
        unique = {}
        for row in results:
            unique.setdefault(json.dumps(row["sequence"]), row)
        with mp.get_context("spawn").Pool(8, initializer=init_worker, initargs=(str(folder),)) as pool:
            reconstructed = pool.map(evaluate, unique.values())
        by_sequence = dict(zip(unique, reconstructed))
        batch_candidates = []
        for round_index in range(3):
            arrays = {}
            for index, row in enumerate(results[round_index * 64:(round_index + 1) * 64]):
                rebuilt = by_sequence[json.dumps(row["sequence"])]
                keys = ("metrics", "audit", "route", "terminal_qpos", "terminal_qvel", "controller_state",
                        "valid", "reasons", "first_failure_s", "sequence")
                mismatch = [key for key in keys if normalize(rebuilt.get(key)) != row.get(key)]
                stem = f"candidate{index:02d}"
                for key, field in (("controls", "controls"), ("track", "track"), ("support", "support"),
                                   ("qpos", "record_qpos"), ("qvel", "record_qvel")):
                    arrays[f"{stem}_{key}"] = rebuilt[field]
                arrays[f"{stem}_time"] = rebuilt["track"][:, 0]
                metrics = row["metrics"]
                candidate = {
                    "id": f"h20_{batch}_{row['id']}", "kind": "candidate", "height_cm": 20,
                    "batch": batch, "candidate_id": row["id"], "label": row.get("label"),
                    "round": round_index, "candidate_index": index,
                    "seed_time_s": metrics["start_time_s"], "horizon_requested_s": 2.,
                    "actual_physics_steps": len(rebuilt["controls"]),
                    "actual_duration_s": len(rebuilt["controls"]) * .001,
                    "attempt_completed": len(rebuilt["controls"]) == 2000,
                    "process_valid": row["valid"], "process_reasons": row["reasons"],
                    "first_failure_s": row.get("first_failure_s"),
                    "true_both_rear_crossing": False, "full_route_success": False, "dataset_accepted": False,
                    "crossing_evidence": "Worst rear wheel center remains before first riser x=1.6 m; no approximate-bottom interpretation is needed",
                    "max_worst_rear_x_m": float(np.max(rebuilt["track"][:, 2])),
                    "max_approx_rear_bottom_m": float(np.max(rebuilt["track"][:, 4])),
                    "sequence": row["sequence"], "metrics": metrics, "audit": row["audit"], "route": row["route"],
                    "seed_npz": str(folder / "seed_prefix.npz"), "scene_path": str(folder / "scene.xml"),
                    "backend_path": str(folder / "backend.py"), "trace_npz": str(folder / f"round{round_index}_traces.npz"),
                    "controls_key": f"{stem}_controls", "track_key": f"{stem}_track", "support_key": f"{stem}_support",
                    "qpos_key": f"{stem}_qpos", "qvel_key": f"{stem}_qvel", "time_key": f"{stem}_time",
                    "result_json": str(folder / f"round{round_index}_results.json"),
                    "control_sha256": hashlib.sha256(rebuilt["controls"].tobytes()).hexdigest(),
                    "media_provenance": "frozen_backend_reconstruction",
                    "reconstruction_verified": not mismatch, "reconstruction_mismatch_fields": mismatch,
                }
                batch_candidates.append(candidate)
            np.savez_compressed(folder / f"round{round_index}_traces.npz", **arrays)
        summary = {"batch": batch, "planned_attempts": 192, "recorded_attempts": len(results),
            "unique_target_sequences": len(unique), "process_valid": sum(row["valid"] for row in results),
            "true_local_crossing": 0, "full_route_success": 0,
            "seed_time_s": results[0]["metrics"]["start_time_s"], "seed_npz": str(folder / "seed_prefix.npz"),
            "results_dir": str(folder), "search_report": str(folder / "search_report.json"),
            "reconstruction_verified": sum(row["reconstruction_verified"] for row in batch_candidates),
            "reconstruction_mismatches": [row["id"] for row in batch_candidates if not row["reconstruction_verified"]]}
        manifest["batches"].append(summary)
        manifest["candidates"].extend(batch_candidates)
        manifest["candidate_evaluations"] = len(manifest["candidates"])
        (BASE / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    manifest["all_candidates_visible_in_manifest"] = len(manifest["candidates"]) == 384
    (BASE / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
