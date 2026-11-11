"""Read-only catalog of high-stair diagnostics, separate from accepted datasets."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

BASE = Path.home() / "wzy"
REVIEW = BASE / "rover/review_v4"
MEDIA = BASE / "videos/review_high_stairs"
MEDIA_MANIFEST = MEDIA / "media_manifest.json"
PAGE = Path(__file__).with_name("high_stairs.html")


@lru_cache(maxsize=64)
def _read_cached(path: str, modified: int, size: int):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json(path: Path, default):
    try:
        stat = path.stat()
        return _read_cached(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, TypeError):
        return default


def safe_media(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    path = Path(value)
    try:
        resolved = (path if path.is_absolute() else MEDIA / path).resolve()
        resolved.relative_to(MEDIA.resolve())
        return resolved if resolved.is_file() else None
    except (OSError, ValueError, RuntimeError):
        return None


def _media_rows():
    manifest = read_json(MEDIA_MANIFEST, {})
    if isinstance(manifest, list):
        return manifest
    if not isinstance(manifest, dict):
        return []
    rows = []
    for key in ("candidates", "historical", "additional", "items", "entries", "prefixes"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            rows.extend(value)
        elif isinstance(value, dict):
            rows.extend({"id": identifier, **item} for identifier, item in value.items() if isinstance(item, dict))
    return rows


def registered_media():
    paths = set()
    for row in _media_rows():
        if row.get("render_status") not in ("complete", "ready"):
            continue
        for key in ("video_path", "poster_path"):
            path = safe_media(row.get(key))
            if path:
                paths.add(path)
    return paths


def _attach_media(record, media):
    row = media.get(record["id"], {})
    mismatch = record.get("reconstruction_verified") is False
    ready = row.get("render_status") in ("complete", "ready")
    prefix = media.get(f"h{record['height_cm']}_{record['batch']}_prefix", {})
    prefix_ready = prefix.get("render_status") in ("complete", "ready")
    for name in ("video", "poster", "prefix_video", "prefix_poster"):
        value = row.get(name + "_path")
        if name.startswith("prefix_"):
            value = value or prefix.get(name.removeprefix("prefix_") + "_path")
        path = safe_media(value) if (prefix_ready if name.startswith("prefix_") else ready) else None
        record[name + "_url"] = (
            "/review-media/" + quote(path.relative_to(MEDIA.resolve()).as_posix(), safe="/")
            if path and not mismatch else None
        )
    record["render_status"] = "verification_mismatch" if mismatch else row.get("render_status", "pending")
    if record.get("video_url"):
        record["render_status"] = "ready"
    record["media_provenance"] = record.get("media_provenance") or row.get("media_provenance", "original_recorded_state")
    record["render_error"] = row.get("render_error", row.get("error"))
    record["canonical_media_id"] = row.get("canonical_media_id", record["id"])
    record["media_reused"] = record["canonical_media_id"] != record["id"]
    record["media_reuse_note"] = "Same condition and identical actual control sequence; evaluation record remains separate" if record["media_reused"] else None


def _process_reasons(audit, route):
    return [reason for reason in audit.get("quality_rejection_reasons", []) if not reason.startswith("final_")] + [
        reason for reason in route.get("rejection_reasons", [])
        if not reason.startswith(("route_missing:", "route_final_"))]


def read_catalog():
    media = {row["id"]: row for row in _media_rows() if isinstance(row, dict) and isinstance(row.get("id"), str)}
    candidates, batches = [], []
    for height in (15, 20):
        source = read_json(REVIEW / f"dynamic_search{height}/candidate_manifest.json", {})
        for batch in source.get("batches", []):
            batches.append({**batch, "height_cm": height})
        for item in source.get("candidates", []):
            row = dict(item)
            row.update(kind="candidate", height_cm=height)
            row.setdefault("process_reasons", row.get("reasons", []))
            row.setdefault("true_both_rear_crossing", False)
            row.setdefault("full_route_success", False)
            _attach_media(row, media)
            candidates.append(row)
    history = []
    for path in sorted(REVIEW.glob("*/diagnostic.json")):
        raw = read_json(path, {})
        params = raw.get("parameters", {})
        if params.get("height") not in (.15, .20) or params.get("stairs") != 15:
            continue
        result, audit, route = raw.get("result", {}), raw.get("audit", {}), raw.get("route", {})
        reasons = _process_reasons(audit, route)
        identifier = "historical_" + path.parent.name
        aliases = [identifier, path.parent.name, "diagnostic_" + path.parent.name]
        identifier = next((key for key in aliases if key in media), identifier)
        not_activated = path.parent.name in ("agent_h15_n15_positive12", "agent_h15_n15_positive12_unload") and not raw.get("assist_events")
        row = {
            "id": identifier, "kind": "historical_diagnostic", "height_cm": round(params["height"] * 100),
            "batch": "historical", "candidate_id": path.parent.name, "label": raw.get("mode", path.parent.name),
            "process_valid": not reasons, "process_reasons": reasons, "audit": audit, "route": route,
            "full_route_success": result.get("success") is True,
            "dataset_accepted": raw.get("dataset_accepted") is True,
            "actual_duration_s": result.get("duration_s", raw.get("record", {}).get("duration")),
            "seed_time_s": 0., "first_failure_s": None, "parameters": params,
            "prototype_parameters": raw.get("prototype_parameters", {}),
            "result": result, "abort_reason": result.get("abort_reason"),
            "action_not_activated": not_activated, "assist_events": raw.get("assist_events", raw.get("transfer_events", [])),
            "diagnostic_path": str(path), "scene_path": str(path.parent / "scene.xml"),
            "trajectory_path": str(path.parent / "trajectory.npz"),
        }
        _attach_media(row, media)
        history.append(row)
    additional = []
    for item in media.values():
        if item.get("kind") != "additional_validation":
            continue
        diagnostic = Path(item.get("diagnostic_json") or REVIEW / "dynamic_search20/tail_hold_1s.json")
        try:
            diagnostic.resolve().relative_to(REVIEW.resolve())
        except (ValueError, RuntimeError):
            continue
        raw = read_json(diagnostic, {})
        metrics = raw.get("metrics", {})
        row = {
            "id": item["id"], "kind": "additional_validation", "height_cm": item.get("height_cm", 20),
            "batch": "additional", "candidate_id": raw.get("id", item["id"]),
            "label": "已执行的末段目标延续 1 秒", "seed_time_s": metrics.get("start_time_s"),
            "actual_duration_s": metrics.get("end_time_s", 0.) - metrics.get("start_time_s", 0.),
            "process_valid": raw.get("valid") is True, "process_reasons": raw.get("reasons", []),
            "first_failure_s": raw.get("first_failure_s"), "true_both_rear_crossing": False,
            "full_route_success": False, "dataset_accepted": False, "sequence": raw.get("sequence"),
            "metrics": metrics, "audit": raw.get("audit", {}), "route": raw.get("route", {}),
            "diagnostic_path": str(diagnostic),
            "parent_candidate_id": item.get("parent_candidate_id", "h20_near_contact_round2_candidate13"),
            "independent_initial_state_replay_max_error": raw.get("independent_initial_state_replay_max_error"),
        }
        _attach_media(row, media)
        additional.append(row)
    summaries = {
        "candidate_evaluations": len(candidates),
        "candidate_process_valid": sum(row.get("process_valid") is True for row in candidates),
        "candidate_local_crossings": sum(row.get("true_both_rear_crossing") is True for row in candidates),
        "candidate_full_route_successes_observed": sum(row.get("full_route_success") is True for row in candidates),
        "candidate_media_ready": sum(bool(row.get("video_url")) for row in candidates),
        "historical_attempts": len(history),
        "historical_full_route_successes": sum(row.get("full_route_success") is True for row in history),
        "historical_media_ready": sum(bool(row.get("video_url")) for row in history),
        "additional_validations": len(additional),
        "additional_media_ready": sum(bool(row.get("video_url")) for row in additional),
        "full_route_rate_for_short_horizon_is_defined": False,
    }
    independent = read_json(REVIEW / "batch_review_summary_20261111.json", {})
    return {"summary": summaries, "independent_summary": independent,
            "batches": batches, "candidates": candidates, "historical": history, "additional": additional}


def list_payload():
    catalog = read_catalog()
    fields = ("id", "kind", "height_cm", "batch", "candidate_id", "label", "round", "candidate_index",
              "seed_time_s", "actual_duration_s", "attempt_completed", "process_valid", "process_reasons",
              "first_failure_s", "true_both_rear_crossing", "full_route_success", "sequence", "metrics",
              "video_url", "poster_url", "prefix_video_url", "render_status", "media_provenance",
              "reconstruction_verified", "abort_reason", "action_not_activated", "canonical_media_id",
              "media_reused", "media_reuse_note", "parent_candidate_id")
    return {**catalog,
            "candidates": [{key: row.get(key) for key in fields} for row in catalog["candidates"]],
            "historical": [{key: row.get(key) for key in fields} for row in catalog["historical"]],
            "additional": [{key: row.get(key) for key in fields} for row in catalog["additional"]]}


def find_record(identifier):
    catalog = read_catalog()
    return next((row for row in catalog["candidates"] + catalog["historical"] + catalog["additional"] if row["id"] == identifier), None)
