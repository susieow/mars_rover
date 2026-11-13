"""Build the experiment-two result index that the front end reads.

pb/ holds the 36 runs judged by the original six-wheel rule. pb_v2/ holds the
same configurations rerun under the amended rule, plus the 20 deg gap case.
pd/screen/ and pd/confirm/ hold the control-library phase. Every row carries the
same fields so one filterable table can show all of them; rows whose recorded
states were rendered point at their video.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/urdf_slope_20261112"
MEDIA = RUNS / "media"
OUT = ROOT / "docs/urdf_slope_index.jsonl"


def video_keys() -> dict[tuple[str, str], str]:
    keys = {}
    for path in MEDIA.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        folder = report["phase_folder"]
        stem = report.get("stem") or report["media_key"][len(folder.replace("/", "__")) + 2:]
        keys[(folder, stem)] = report["media_key"]
    return keys


def collect(directory: Path, phase: str, criterion: str,
            videos: dict[tuple[str, str], str], video_folder: str | None = None) -> list[dict]:
    rows = []
    folder = video_folder or directory.relative_to(RUNS).as_posix()
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "result" not in data:
            continue
        geometry = data["geometry"]
        control = data["control_id"]
        result = data["result"]
        settle = data.get("settle_check") or {}
        base = (f'{data["friction"]["stair"]}_{geometry["angle_deg"]:g}'
                f'_{control["order"]}_{control["rocker_amp_deg"]:g}')
        video = (videos.get((folder, base))
                 or videos.get((folder, f'{base}_v{control["speed_rad_s"]:g}_{control["trigger"]}')))
        rows.append({
            "run_id": data["run_id"],
            "phase": phase,
            "criterion": criterion,
            "scope": "single" if geometry["steps"] == 1 else "full",
            "steps_total": geometry["steps"],
            "angle_deg": geometry["angle_deg"],
            "tread_m": geometry["tread_m"],
            "friction": data["friction"]["stair"],
            "order": control["order"],
            "rocker_amp_deg": control["rocker_amp_deg"],
            "speed_rad_s": control["speed_rad_s"],
            "trigger": control["trigger"],
            "ascent_completed": result["ascent_completed"],
            "urdf_compliant": result["urdf_compliant"],
            "numerical_quality_ok": result["numerical_quality_ok"],
            "steps_climbed": result["steps_climbed"],
            "end_reason": result["end_reason"],
            "duration_s": data["duration_s"],
            "penetration_mm": data["penetration"]["max_mm"],
            "loaded_wheels": settle.get("loaded_wheels"),
            "unloaded_gap_mm": settle.get("unloaded_gap_mm"),
            "support_s": (settle.get("support_steps") or 0) * 0.001,
            "settle_s": (settle.get("settled_steps") or 0) * 0.001,
            "violation_counts": data.get("violation_counts") or {},
            "result_path": str(path.relative_to(ROOT)),
            "source_sha256": data["model"]["source_sha256"],
            "video_key": video,
        })
    return rows


def main() -> None:
    videos = video_keys()
    rows = (collect(RUNS / "pb", "B", "v1", videos)
            + collect(RUNS / "pb_v2", "B", "v2", videos)
            + collect(RUNS / "pd/screen", "D-screen", "d", videos, "pd/screen_replay")
            + collect(RUNS / "pd/confirm", "D-confirm", "d", videos))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"[index] {len(rows)} rows, {sum(1 for r in rows if r['video_key'])} with video "
          f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
