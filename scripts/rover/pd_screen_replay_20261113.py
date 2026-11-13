"""Replay the Phase D screening cases so their recorded states exist for video review.

The screening folder kept only the judged numbers, not the trajectories. Replaying
the same cases into a separate folder leaves those numbers untouched and lets the
front end show what each control id actually does on one step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pd_phase
import urdf_slope_runtime_20261113 as pb

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/urdf_slope_20261112/pd/screen"
DEST = ROOT / "runs/urdf_slope_20261112/pd/screen_replay"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(pd_phase.screen_cases(), 1):
        stem = f"{index:02d}_{case['angle_deg']:g}_{case['speed']:g}_{case['trigger']}_{case['order']}"
        original = json.loads((SOURCE / f"{stem}.json").read_text(encoding="utf-8"))["result"]
        pb.RUN = DEST
        result = pb.run(case)
        body = result["result"]
        same = all(body[key] == original[key] for key in
                   ("ascent_completed", "urdf_compliant", "numerical_quality_ok",
                    "steps_climbed", "end_reason"))
        print(json.dumps({"stem": stem, "same": same, "steps": body["steps_climbed"],
                          "end": body["end_reason"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
