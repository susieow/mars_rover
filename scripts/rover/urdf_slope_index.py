"""Build the experiment-two result index that the front end reads.

pb/ holds the 36 runs judged by the original six-wheel rule. pb_v2/ holds the
same configurations rerun under the amended rule, plus the 20 deg gap case.
Both sets stay in the index because the criterion change is part of the record.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/urdf_slope_20261112"
OUT = ROOT / "docs/urdf_slope_index.jsonl"


def collect(directory: Path, criterion: str) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "result" not in data:
            continue
        geometry = data["geometry"]
        control = data["control_id"]
        result = data["result"]
        settle = data.get("settle_check") or {}
        rows.append({
            "run_id": data["run_id"],
            "criterion": criterion,
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
        })
    return rows


def main() -> None:
    rows = collect(RUNS / "pb", "v1") + collect(RUNS / "pb_v2", "v2")
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"[index] {len(rows)} rows -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
