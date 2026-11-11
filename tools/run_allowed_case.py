"""Run an isolated, explicitly scoped 15 cm or 20 cm case."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--height-cm", type=int, choices=(15, 20), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.name):
        parser.error("name must contain only letters, numbers, underscores, and hyphens")
    if not math.isfinite(args.speed) or not 0 < args.speed <= 52.359878:
        parser.error("speed must be finite and within (0, 52.359878] rad/s")
    branch = subprocess.check_output(
        ["git", "-C", str(ROOT), "branch", "--show-current"], text=True
    ).strip()
    if not branch or branch in ("main", "master", "wzy"):
        parser.error("run experiments from a dedicated work branch, never main/master/wzy")
    folder = ROOT / "runs" / args.name
    dataset = folder / "dataset"
    videos = folder / "videos"
    command = [
        sys.executable, "-B", str(ROOT / "scripts/rover/generate_dataset.py"),
        "--split", "train", "--heights", str(args.height_cm), "--nstairs", "15",
        "--variants", "base", "--prefix", args.name,
        "--speed", str(args.speed), "--contact-timeconst", "0.004",
        "--dataset-root", str(dataset),
    ]
    env = os.environ.copy()
    env.update(ROVER_DATASET_ROOT=str(dataset), ROVER_VIDEO_ROOT=str(videos),
               PYTHONDONTWRITEBYTECODE="1")
    plan = {"branch": branch, "height_cm": args.height_cm, "stairs": 15,
            "dataset_root": str(dataset), "videos_root": str(videos),
            "command": command, "dry_run": args.dry_run,
            "note": "The baseline controller has no verified high-stair success."}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return subprocess.run(command + ["--dry-run"], cwd=ROOT, env=env).returncode
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "videos").mkdir()
    (folder / "launch.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return subprocess.run(command, cwd=ROOT, env=env).returncode

if __name__ == "__main__":
    raise SystemExit(main())
