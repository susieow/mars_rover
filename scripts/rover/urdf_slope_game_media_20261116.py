"""Render videos for 3 m game-batch runs whose rear axle reached the top level.

The game300 batch ran headless. This renders every run whose recorded rear axle
reached the last step of its staircase, settled or not, by replaying its saved
trajectory through the same renderer the Phase B/D videos use.

    python scripts/rover/urdf_slope_game_media_20261116.py --only 4_15_SN_18_v4_contact_rise15cm_boost
    python scripts/rover/urdf_slope_game_media_20261116.py --shard 0 --shards 16
    python scripts/rover/urdf_slope_game_media_20261116.py --index
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rover"))
import render_dataset  # noqa: E402
import urdf_slope_media_20261113 as media  # noqa: E402

GAME = ROOT / "runs/urdf_slope_20261112/game300"
INDEX = ROOT / "docs/urdf_slope_game_index.jsonl"
FOLDER = "game300"
# media.py resolves scenes under RUNS/<folder>; keep the shared media/ output dir.
media.RUNS = GAME.parent


def rows() -> list[dict]:
    return [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def stem_of(row: dict) -> str:
    return Path(row["result_path"]).stem


def reached_top(row: dict) -> bool:
    return row["steps_climbed"] >= row["steps_total"]


def targets() -> list[str]:
    out = []
    for row in rows():
        if not reached_top(row):
            continue
        stem = stem_of(row)
        if (GAME / f"{stem}.xml").is_file() and (GAME / f"{stem}_trajectory.npz").is_file():
            out.append(stem)
    return sorted(set(out))


def key(stem: str) -> str:
    return media.media_key(FOLDER, stem)


def render_one(stem: str) -> dict:
    # The render helper writes render_scene.xml next to the scene, so parallel
    # shards would clobber each other; give each process its own scratch dir.
    original = render_dataset._visual_render_model
    scratch = Path(tempfile.mkdtemp(prefix=f"render-{os.getpid()}-", dir=GAME.parent))
    render_dataset._visual_render_model = (
        lambda scene, folder, metadata: original(scene, scratch, metadata))
    try:
        return media.render(FOLDER, stem)
    finally:
        render_dataset._visual_render_model = original
        shutil.rmtree(scratch, ignore_errors=True)


def update_index() -> None:
    data = rows()
    for row in data:
        if reached_top(row) and (media.MEDIA / f"{key(stem_of(row))}.mp4").is_file():
            row["video_key"] = key(stem_of(row))
    INDEX.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data),
                     encoding="utf-8")
    print(f"[media] index updated: {sum(1 for r in data if r['video_key'])} rows with video",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    if args.index:
        return update_index()
    pending = args.only or [
        stem for position, stem in enumerate(targets())
        if position % args.shards == args.shard]
    pending = [stem for stem in pending
               if args.force or not (media.MEDIA / f"{key(stem)}.mp4").exists()]
    print(json.dumps({"shard": args.shard, "shards": args.shards, "pending": len(pending)}),
          flush=True)
    for stem in pending:
        try:
            report = render_one(stem)
            print(json.dumps({"ok": report["media_key"], "frames": report["frames"]}), flush=True)
        except Exception as error:  # one bad run should not cost the rest of the shard
            print(json.dumps({"fail": stem, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
