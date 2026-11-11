"""Publish verified rover episodes only after an external visual review exists."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from verify_dataset import verify_episode
from dataset_plan import SPLITS, assignment, split_exports

DATASET_ROOT = Path(os.environ.get("ROVER_DATASET_ROOT", str(Path(__file__).resolve().parents[2] / "datasets/45deg_v3")))
VIDEOS = Path(os.environ.get("ROVER_VIDEO_ROOT", str(Path(__file__).resolve().parents[2] / "runs/videos")))
VISUAL_SCOPES = {"sampled_pass": "sampled_frames", "full_video_pass": "full_video"}


class PublishRejected(ValueError):
    pass


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublishRejected(f"expected JSON object: {path}")
    return value


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise PublishRejected(f"missing file path under {base}")
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base / candidate).resolve(strict=True)


def _metadata_path(value, dataset_root):
    path = Path(value).resolve(strict=True)
    if path.name != "metadata.json" or path.parent.parent != dataset_root / "episodes":
        raise PublishRejected(f"metadata must be episodes/<case>/metadata.json: {path}")
    return path


def _indexed_paths(index, dataset_root, only_flagged=False):
    if not index.exists():
        return set()
    found = set()
    for number, line in enumerate(index.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise PublishRejected(f"invalid index row: {index}:{number}")
        if only_flagged and row.get("dataset_accepted", row.get("accepted")) is not True:
            continue
        value = row.get("metadata_path", row.get("metadata"))
        if not isinstance(value, str):
            raise PublishRejected(f"accepted index row lacks metadata_path: {index}:{number}")
        path = _metadata_path(value, dataset_root)
        if _read_object(path).get("dataset_accepted") is not True:
            raise PublishRejected(f"accepted index disagrees with metadata: {path}")
        found.add(path)
    return found


def _probe_video(path):
    process = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_type,width,height,nb_frames,duration:format=duration",
        "-of", "json", str(path),
    ], capture_output=True, text=True)
    if process.returncode != 0:
        raise PublishRejected(f"ffprobe failed for {path}: {process.stderr.strip()}")
    result = json.loads(process.stdout)
    streams = result.get("streams", [])
    if len(streams) != 1:
        raise PublishRejected(f"video stream missing: {path}")
    stream = streams[0]
    duration = float(stream.get("duration", result.get("format", {}).get("duration", 0)))
    if stream.get("codec_type") != "video" or int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0 or not 0 < duration < float("inf"):
        raise PublishRejected(f"invalid video stream properties: {path}")
    frames = stream.get("nb_frames")
    if frames is not None and frames != "N/A" and int(frames) <= 0:
        raise PublishRejected(f"video has no frames: {path}")
    return {"ffprobe_ok": True, "duration_s": duration,
            "width": int(stream["width"]), "height": int(stream["height"])}


def _review(metadata_path, metadata, videos_dir):
    episode = metadata_path.parent
    trajectory = _path(metadata.get("trajectory_path"), episode)
    scene = _path(metadata.get("scene_path"), episode)
    if trajectory != episode / "trajectory.npz" or scene != episode / "scene.xml":
        raise PublishRejected(f"metadata paths must match the independently verified episode files: {metadata_path}")
    video = _path(metadata.get("out"), episode)
    try:
        video.relative_to(videos_dir)
    except ValueError as error:
        raise PublishRejected(f"video must be inside {videos_dir}: {video}") from error
    if not video.is_file() or video.stat().st_size == 0:
        raise PublishRejected(f"video missing or empty: {video}")
    review_path = episode / "visual_review.json"
    if not review_path.is_file():
        raise PublishRejected(f"missing external visual review: {review_path}")
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)
    if not isinstance(review, dict):
        raise PublishRejected(f"expected visual review object: {review_path}")
    status = review.get("status")
    if status not in VISUAL_SCOPES:
        raise PublishRejected(f"visual status must be sampled_pass or full_video_pass: {review_path}")
    for key in ("reviewer", "notes"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise PublishRejected(f"visual review requires nonempty {key}: {review_path}")
    requested_images = review.get("reviewed_images")
    if not isinstance(requested_images, list) or not requested_images:
        raise PublishRejected(f"visual review requires reviewed_images: {review_path}")
    images = []
    for value in requested_images:
        image = _path(value, episode)
        try:
            image.relative_to(episode)
        except ValueError as error:
            raise PublishRejected(f"reviewed image must be inside its episode: {image}") from error
        if not image.is_file() or image.stat().st_size == 0 or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise PublishRejected(f"reviewed image is missing, empty or not an image path: {image}")
        images.append(image)
    hashes = {"trajectory": _digest(trajectory), "video": _digest(video),
              "scene": _digest(scene), "visual_review": hashlib.sha256(review_bytes).hexdigest(),
              "reviewed_images": {str(image): _digest(image) for image in images}}
    for key in ("trajectory", "video"):
        if review.get(key + "_sha256") != hashes[key]:
            raise PublishRejected(f"{key}_sha256 mismatch in {review_path}")
    return trajectory, scene, video, review_path, review, hashes


def _stage(path, payload):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _sync_directories(paths):
    for parent in {path.parent for path in paths}:
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _commit(writes):
    """Prepare every new and rollback file before replacing any canonical path."""
    prepared = []
    replaced = []
    keep_backups = False
    try:
        for target, payload in writes:
            backup = new = None
            try:
                backup = _stage(target, target.read_bytes()) if target.exists() else None
                new = _stage(target, payload)
            except BaseException:
                if backup is not None:
                    backup.unlink(missing_ok=True)
                if new is not None:
                    new.unlink(missing_ok=True)
                raise
            prepared.append((target, new, backup))
        try:
            for target, new, backup in prepared:
                os.replace(new, target)
                replaced.append((target, backup))
            _sync_directories([target for target, _, _ in prepared])
        except BaseException as original_error:
            rollback_errors = []
            for target, backup in reversed(replaced):
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target)
                except BaseException as rollback_error:
                    rollback_errors.append(f"{target}: {rollback_error}")
            if rollback_errors:
                keep_backups = True
                raise RuntimeError(
                    "publication failed; rollback backups retained: " + "; ".join(rollback_errors)
                ) from original_error
            _sync_directories([target for target, _ in replaced])
            raise
    finally:
        for _, new, backup in prepared:
            new.unlink(missing_ok=True)
            if backup is not None and not keep_backups:
                backup.unlink(missing_ok=True)


def _split_writes(dataset_root, accepted_rows, replacements=None):
    replacements = {} if replacements is None else replacements
    rows = []
    for path in sorted((dataset_root / "episodes").glob("*/metadata.json")):
        rows.append((path, replacements.get(path, _read_object(path))))
    try:
        attempts, accepted, unknown, summary = split_exports(rows, accepted_rows)
    except ValueError as error:
        raise PublishRejected(str(error)) from error
    writes = []
    for split in SPLITS:
        writes.append((dataset_root / f"{split}.jsonl", _jsonl_bytes(accepted[split])))
        writes.append((dataset_root / f"{split}_attempts.jsonl", _jsonl_bytes(attempts[split])))
    writes.extend([
        (dataset_root / "unassigned_attempts.jsonl", _jsonl_bytes(unknown)),
        (dataset_root / "split_summary.json", _json_bytes(summary)),
    ])
    return writes, summary


def refresh_splits(dataset_root=DATASET_ROOT):
    dataset_root = Path(dataset_root).resolve(strict=True)
    with (dataset_root / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        index = dataset_root / "accepted.jsonl"
        _indexed_paths(index, dataset_root)
        accepted_rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()
                         if line.strip()] if index.exists() else []
        writes, summary = _split_writes(dataset_root, accepted_rows)
        _commit(writes)
    return {"published": False, "split_manifests_refreshed": True, "summary": summary}

def publish(metadata_paths, *, dataset_root=DATASET_ROOT, videos_dir=VIDEOS,
            verifier=None, video_probe=None):
    requested_values = list(metadata_paths)
    if not requested_values:
        raise PublishRejected("no metadata selected; accepted_count=0")
    dataset_root = Path(dataset_root).resolve(strict=True)
    videos_dir = Path(videos_dir).resolve(strict=True)
    requested = {_metadata_path(path, dataset_root) for path in requested_values}
    verifier = verify_episode if verifier is None else verifier
    video_probe = _probe_video if video_probe is None else video_probe
    with (dataset_root / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        accepted_index = dataset_root / "accepted.jsonl"
        display_index = videos_dir / "runs.jsonl"
        existing = set()
        for path in sorted((dataset_root / "episodes").glob("*/metadata.json")):
            if _read_object(path).get("dataset_accepted") is True:
                existing.add(_metadata_path(path, dataset_root))
        existing.update(_indexed_paths(accepted_index, dataset_root))
        existing.update(_indexed_paths(display_index, dataset_root, only_flagged=True))
        candidates = sorted(existing | requested)
        guards = {}
        records = []
        cases = set()
        for path in candidates:
            original = path.read_bytes()
            metadata = _read_object(path)
            if metadata.get("automated_checks_passed") is not True:
                raise PublishRejected(f"automated_checks_passed is not true: {path}")
            verification = verifier(path)
            if not isinstance(verification, dict) or verification.get("verify_ok") is not True:
                reasons = verification.get("rejection_reasons", []) if isinstance(verification, dict) else []
                raise PublishRejected(f"independent verification failed: {path}: {reasons}")
            trajectory, scene, video, review_path, review, hashes = _review(path, metadata, videos_dir)
            video_check = video_probe(video)
            if not isinstance(video_check, dict) or video_check.get("ffprobe_ok") is not True:
                raise PublishRejected(f"video verification failed: {video}")
            case = metadata.get("case")
            if not isinstance(case, str) or not case or case in cases:
                raise PublishRejected(f"case is missing or duplicated: {path}")
            cases.add(case)
            if not isinstance(metadata.get("parameters"), dict) or not metadata["parameters"]:
                raise PublishRejected(f"training parameters missing: {path}")
            try:
                spec = assignment(metadata)
            except ValueError as error:
                raise PublishRejected(f"invalid split assignment: {path}: {error}") from error
            if spec is None:
                raise PublishRejected(f"accepted episode has no supported split assignment: {path}")
            guards[path] = hashlib.sha256(original).hexdigest()
            guards.update({trajectory: hashes["trajectory"], scene: hashes["scene"],
                           video: hashes["video"], review_path: hashes["visual_review"]})
            guards.update({Path(image): digest for image, digest in hashes["reviewed_images"].items()})
            guards.update({path.parent / "sources" / name: digest
                           for name, digest in verification.get("source_snapshot_hashes_verified", {}).items()})
            guards.update({Path(mesh): digest for mesh, digest in verification.get("mesh_hashes_verified", {}).items()})
            updated = dict(metadata, split=spec["split"], case_spec=spec)
            updated.update(
                dataset_accepted=True, visual_review_status=review["status"],
                visual_scope=VISUAL_SCOPES[review["status"]], visual_review_path=str(review_path),
                trajectory_sha256=hashes["trajectory"], video_sha256=hashes["video"],
                visual_review_sha256=hashes["visual_review"], metadata_path=str(path),
                independent_verification=verification,
            )
            payload = _json_bytes(updated)
            hashes["metadata"] = hashlib.sha256(payload).hexdigest()
            hashes["sources"] = metadata.get("source_sha256", {})
            hashes["meshes"] = metadata.get("mesh_sha256", {})
            manifest = {
                "case": case, "split": spec["split"], "case_spec": spec,
                "policy_sha256": metadata.get("policy_sha256"),
                "metadata_path": str(path), "trajectory_path": str(trajectory),
                "scene_path": str(scene), "video_path": str(video),
                "parameters": metadata["parameters"], "sha256": hashes,
                "visual_scope": VISUAL_SCOPES[review["status"]],
                "visual_review_status": review["status"], "reviewer": review["reviewer"],
            }
            display = {key: value for key, value in updated.items() if key not in {
                "events", "audit", "replay", "render", "source_sha256",
                "source_sha256_end", "source_loaded_sha256", "mesh_sha256",
            }}
            display["independent_verification"] = {
                key: verification[key] for key in (
                    "verify_ok", "current_verifier_sha256", "current_audit_sha256"
                ) if key in verification
            }
            records.append((path, payload, manifest, display))
        for path in sorted((dataset_root / "episodes").glob("*/metadata.json")):
            guards.setdefault(path, _digest(path))
        split_writes, split_summary = _split_writes(
            dataset_root, [record[2] for record in records],
            {path: json.loads(payload) for path, payload, _, _ in records})
        for path, expected in guards.items():
            if _digest(path) != expected:
                raise PublishRejected(f"file changed during publication checks: {path}")
        records.sort(key=lambda record: record[2]["case"])
        writes = [(path, payload) for path, payload, _, _ in records]
        writes += [(accepted_index, _jsonl_bytes([record[2] for record in records])),
                   (display_index, _jsonl_bytes([record[3] for record in records]))]
        writes += split_writes
        _commit(writes)
        return {"published": True, "accepted_count": len(records),
                "split_summary": split_summary,
                "newly_accepted_count": len(requested - existing),
                "reverified_existing_count": len(existing),
                "accepted_manifest": str(accepted_index), "display_manifest": str(display_index),
                "cases": [record[2]["case"] for record in records]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", nargs="*", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS)
    parser.add_argument("--refresh-splits", action="store_true",
                        help="Export per-split accepted and all-attempt manifests without accepting new episodes")
    args = parser.parse_args()
    try:
        if args.refresh_splits:
            if args.metadata:
                raise ValueError("--refresh-splits does not accept episode paths")
            result = refresh_splits(args.dataset_root)
        else:
            result = publish(args.metadata, dataset_root=args.dataset_root, videos_dir=args.videos_dir)
    except Exception as error:
        print(json.dumps({"published": False, "newly_accepted_count": 0,
                          "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

