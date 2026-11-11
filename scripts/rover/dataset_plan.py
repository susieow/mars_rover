"""Fixed rover cases and a release gate for held-out evaluation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "validation", "test", "regression")
HEIGHTS_CM = {"train": (15, 20), "validation": (15, 20), "test": (17, 18), "regression": (4,)}
INITIAL_STATES = {
    "train": {
        "base": {"init_x": 0.0, "init_y": 0.0, "init_yaw_deg": 0.0},
        "yaw_left": {"init_x": -0.02, "init_y": 0.015, "init_yaw_deg": 1.0},
        "yaw_right": {"init_x": 0.02, "init_y": -0.015, "init_yaw_deg": -1.0},
    },
    "validation": {
        "validation_left": {"init_x": -0.035, "init_y": 0.025, "init_yaw_deg": 1.5},
        "validation_right": {"init_x": 0.035, "init_y": -0.025, "init_yaw_deg": -1.5},
    },
    "test": {
        "base": {"init_x": 0.0, "init_y": 0.0, "init_yaw_deg": 0.0},
        "test_left": {"init_x": -0.025, "init_y": 0.02, "init_yaw_deg": 1.25},
        "test_right": {"init_x": 0.025, "init_y": -0.02, "init_yaw_deg": -1.25},
    },
    "regression": {
        "base": {"init_x": 0.0, "init_y": 0.0, "init_yaw_deg": 0.0},
        "yaw_left": {"init_x": -0.02, "init_y": 0.015, "init_yaw_deg": 1.0},
        "yaw_right": {"init_x": 0.02, "init_y": -0.015, "init_yaw_deg": -1.0},
    },
}
CASE_KEYS = ("height", "stairs", "depth", "init_x", "init_y", "init_yaw_deg")
PLAN_VERSION = "equal_stairs_split_v1_15_20"


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cases(split="train", heights=None, variants=None, stairs=None):
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    heights = list(HEIGHTS_CM[split] if heights is None else heights)
    variants = list(INITIAL_STATES[split] if variants is None else variants)
    expected_stairs = 12 if split == "regression" else 15
    stairs = expected_stairs if stairs is None else stairs
    if type(stairs) is not int or stairs != expected_stairs:
        raise ValueError(f"{split} requires {expected_stairs} ascending and descending risers")
    if not heights or len(heights) != len(set(heights)):
        raise ValueError("heights must be nonempty and unique")
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("variants must be nonempty and unique")
    cases = []
    for cm in heights:
        if type(cm) is not int or cm not in HEIGHTS_CM[split]:
            raise ValueError(f"height {cm} cm is not assigned to {split}: {HEIGHTS_CM[split]}")
        for variant in variants:
            if variant not in INITIAL_STATES[split]:
                raise ValueError(f"variant {variant} is not assigned to {split}")
            parameters = dict(height=cm / 100.0, stairs=stairs, depth=cm / 100.0,
                              **INITIAL_STATES[split][variant])
            cases.append({
                "plan_version": PLAN_VERSION, "split": split,
                "case_id": f"h{cm:02d}_n{stairs:02d}_{variant}", "variant": variant,
                "parameters": parameters, "group_sha256": fingerprint(parameters),
                "evaluation_status": "reserved_unvalidated" if split == "test" else "development",
            })
    return cases


def full_plan():
    return {"plan_version": PLAN_VERSION,
            "cases": [case for split in SPLITS for case in build_cases(split)],
            "test_policy": "Reserved until explicit release freeze; retain every attempt including failures.",
            "split_policy": "Assignment depends on geometry and initial state, never outcome."}


def assignment(metadata):
    """Recognize historical 4 cm episodes without relabeling their geometry."""
    parameters = metadata.get("parameters", {})
    split = metadata.get("split")
    legacy_spec = metadata.get("case_spec") or {}
    if split is None or (split == "regression" and legacy_spec.get("plan_version") == "legacy_4cm_12stairs"):
        if parameters.get("height") != 0.04 or parameters.get("stairs") != 12:
            return None
        concrete = {key: parameters.get(key, 0.0) for key in CASE_KEYS}
        if concrete["depth"] != 0.04:
            raise ValueError("legacy regression must retain its equal-height 45-degree geometry")
        return {"split": "regression", "case_id": metadata.get("case"),
                "variant": "legacy", "parameters": concrete,
                "group_sha256": fingerprint(concrete), "plan_version": "legacy_4cm_12stairs"}
    spec = metadata.get("case_spec")
    if not isinstance(spec, dict):
        raise ValueError("split metadata requires case_spec")
    for expected in build_cases(split):
        if spec == expected:
            actual = {key: parameters.get(key) for key in CASE_KEYS}
            if actual != expected["parameters"]:
                raise ValueError("case parameters do not match their assigned split")
            policy = metadata.get("policy_snapshot")
            if not isinstance(policy, dict) or fingerprint(policy) != metadata.get("policy_sha256"):
                raise ValueError("policy snapshot is missing or changed")
            for key in ("source_sha256", "model_sha256", "urdf_sha256", "mesh_sha256"):
                if key in metadata and policy.get(key) != metadata[key]:
                    raise ValueError(f"recorded {key} differs from the policy snapshot")
            expected_parameters = {key: value for key, value in parameters.items() if key not in CASE_KEYS}
            if policy.get("simulation_parameters") != expected_parameters:
                raise ValueError("simulation parameters differ from the policy snapshot")
            if policy.get("geometry_rule") != {"stairs": parameters["stairs"], "depth_equals_height": True, "slope_deg": 45.0}:
                raise ValueError("geometry differs from the policy snapshot")
            if split == "test":
                require_release(metadata.get("release_snapshot"), policy)
            return expected
    raise ValueError("case_spec is not in the fixed split plan")


def release_document(policy):
    return {"schema": "rover_release_v1", "frozen": True,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": fingerprint(full_plan()), "plan": full_plan(),
            "policy_sha256": fingerprint(policy), "policy_snapshot": policy}


def require_release(release, policy):
    if not isinstance(release, dict) or release.get("schema") != "rover_release_v1" or release.get("frozen") is not True:
        raise ValueError("test split requires an explicitly frozen release")
    if release.get("plan_sha256") != fingerprint(full_plan()) or release.get("plan") != full_plan():
        raise ValueError("release split plan differs from the current plan")
    if release.get("policy_sha256") != fingerprint(policy) or release.get("policy_snapshot") != policy:
        raise ValueError("controller, simulation parameters or source hashes changed after release freeze")
    return release


def write_release(path, policy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = release_document(policy)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return document


def split_exports(metadata_rows, accepted_rows):
    """Count attempts independently of acceptance and never redistribute failures."""
    attempts = {split: [] for split in SPLITS}
    accepted = {split: [] for split in SPLITS}
    unknown = []
    groups = {}
    identities = {}
    for path, metadata in metadata_rows:
        spec = assignment(metadata)
        row = {
            "case": metadata.get("case"), "metadata_path": str(path),
            "split": None if spec is None else spec["split"],
            "case_spec": spec, "policy_sha256": metadata.get("policy_sha256"),
            "attempt_status": metadata.get("attempt_status", "completed"),
            "simulation_success": metadata.get("simulation_success") is True,
            "automated_checks_passed": metadata.get("automated_checks_passed") is True,
            "dataset_accepted": metadata.get("dataset_accepted") is True,
            "quality_rejection_reasons": metadata.get("quality_rejection_reasons", []),
        }
        if spec is None:
            unknown.append(row)
            continue
        split = spec["split"]
        group = spec["group_sha256"]
        if group in groups and groups[group] != split:
            raise ValueError(f"case group occurs in more than one split: {group}")
        groups[group] = split
        identities[str(path)] = spec
        attempts[split].append(row)
    for row in accepted_rows:
        spec = identities.get(str(row["metadata_path"]))
        if spec is None:
            raise ValueError("accepted episode has no supported split assignment")
        accepted[spec["split"]].append(dict(row, split=spec["split"], case_spec=spec))
    summary = {
        "plan_version": PLAN_VERSION, "unassigned_legacy_attempts": len(unknown),
        "test_reserved": not attempts["test"], "splits": {},
        "denominator_definition": "Every episode metadata record, including failed, errored and incomplete attempts.",
    }
    for split in SPLITS:
        rows = attempts[split]
        summary["splits"][split] = {
            "attempts": len(rows), "accepted": len(accepted[split]),
            "simulation_successes": sum(row["simulation_success"] for row in rows),
            "automated_passes": sum(row["automated_checks_passed"] for row in rows),
            "incomplete_or_error": sum(row["attempt_status"] != "completed" for row in rows),
            "policy_sha256": sorted({row["policy_sha256"] for row in rows if row["policy_sha256"]}),
        }
    return attempts, accepted, unknown, summary
