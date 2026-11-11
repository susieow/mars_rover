"""Render and register only strict successes on plain 45-degree stairs."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

SCRIPTS = Path("/home/agilex/wzy/rover/scripts/rover")
PY = "/home/agilex/wzy/rover/.venv/bin/python"
OUT = Path("/home/agilex/wzy/videos")

CASES = [
    {"case": "h4_plain_45deg", "height": 0.04, "front": 10.0, "middle": 10.0, "quality": 38.0},
    {"case": "h5_plain_45deg", "height": 0.05, "front": 12.0, "middle": 12.0, "quality": 38.0},
    {"case": "h6_plain_45deg", "height": 0.06, "front": 16.0, "middle": 16.0, "quality": 45.0},
    {"case": "h7_plain_45deg", "height": 0.07, "front": 18.0, "middle": 15.0, "quality": 50.0},
    {"case": "h8_plain_45deg", "height": 0.08, "front": 14.0, "middle": 12.0, "quality": 55.0},
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "archive_old_controller_20261110"
    archive.mkdir(exist_ok=True)
    for old in OUT.glob("45win_h*.mp4"):
        shutil.move(str(old), str(archive / old.name))

    rows = []
    runs = OUT / "runs.jsonl"
    runs.write_text("")

    for case in CASES:
        h = case["height"]
        out = OUT / f"45closed_{round(h * 100):02d}cm.mp4"
        cmd = [
            PY,
            "render_course.py",
            "--height",
            str(h),
            "--stairs",
            "12",
            "--depth",
            str(h),
            "--seconds",
            "150",
            "--speed",
            "6.0",
            "--friction",
            "4.0",
            "--tau-max",
            "50.0",
            "--nosing",
            "0.0",
            "--front-lift-deg",
            str(case["front"]),
            "--middle-lift-deg",
            str(case["middle"]),
            "--quality-pitch-limit",
            str(case["quality"]),
            "--out",
            str(out),
        ]
        proc = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True)
        lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
        if proc.returncode != 0 or not lines:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            print("ERROR", case["case"], tail[-1] if tail else "no JSON result", flush=True)
            continue

        result = json.loads(lines[-1])
        strict = bool(
            result.get("success")
            and result.get("final_shape_ok")
            and result.get("settled")
            and result.get("nosing", 1.0) == 0.0
        )
        if not strict:
            out.unlink(missing_ok=True)
            print(
                "REJECT",
                case["case"],
                "success=",
                result.get("success"),
                "shape=",
                result.get("final_shape_ok"),
                "settled=",
                result.get("settled"),
                "y=",
                result.get("y"),
                "yaw_error=",
                result.get("final_yaw_error_deg"),
                flush=True,
            )
            continue

        result.update(
            case=case["case"],
            dataset="mars_rover_45deg_plain_steps",
            controller_version="closed_loop_v2",
        )
        rows.append(result)
        runs.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n')
        print(
            case["case"],
            "ok=True",
            "x=",
            result["x"],
            "y=",
            result["y"],
            "max_pitch=",
            result["max_pitch_deg"],
            "frames=",
            result["frames"],
            flush=True,
        )

    print("total_strict_successes", len(rows))


if __name__ == "__main__":
    main()
