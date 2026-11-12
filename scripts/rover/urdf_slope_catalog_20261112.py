"""Build the experiment-two review catalog from recorded cases."""
from __future__ import annotations

import json
from pathlib import Path

import urdf_slope_control_20261112 as exp


def main():
    rows = []
    for result_path in sorted((exp.RUN_ROOT / "cases").glob("*/result.json")):
        folder = result_path.parent
        result = json.loads(result_path.read_text(encoding="utf-8"))
        media_path = folder / "media.json"
        if not media_path.exists():
            continue
        media = json.loads(media_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "id": result["id"],
                "friction": result["parameters"]["stair_friction_slide"],
                "strategy": result["parameters"]["strategy"],
                "direction": result["parameters"]["direction"],
                "angle_deg": result["parameters"]["stair_angle_deg"],
                "height_cm": result["parameters"]["height_cm"],
                "stairs": result["parameters"]["stairs"],
                "duration_s": result["duration_s"],
                "first_contact_s": result["first_contact_s"],
                "intervention_s": result["intervention_s"],
                "termination_reason": result["termination_reason"],
                "ascent_completed": result["ascent_completed"],
                "urdf_compliant_exact": result["urdf_compliant_exact"],
                "numerical_quality_ok": result["numerical_quality_ok"],
                "dataset_accepted": False,
                "max_contact_penetration_mm": result["max_contact_penetration_m"] * 1000.0,
                "violation_names": result["violation_names"],
                "events": result["events"],
                "video_path": media["video_path"],
                "video_sha256": media["video_sha256"],
                "result_path": str(result_path),
                "result_sha256": exp.digest(result_path),
                "trajectory_path": result["trajectory_path"],
                "trajectory_sha256": result["trajectory_sha256"],
            }
        )
    document = {
        "updated": "2026-11-12",
        "experiment": "urdf_slope_control",
        "title": "URDF 约束内的接触恢复控制",
        "summary": (
            "在实验二的 URDF 约束模型中比较原始接触后摇臂周期，以及先卸载、微退、预抬前轮再重接的候选。"
            "本批次为 15°、20 cm、15 级、摩擦 6 的诊断结果。"
        ),
        "completed": len(rows),
        "platform_completed": sum(row["ascent_completed"] for row in rows),
        "urdf_compliant": sum(row["urdf_compliant_exact"] for row in rows),
        "full_success": sum(
            row["ascent_completed"]
            and row["urdf_compliant_exact"]
            and row["numerical_quality_ok"]
            for row in rows
        ),
        "cases": rows,
    }
    exp.write_json(exp.ROOT / "docs/urdf_slope_review.json", document)
    print(
        json.dumps(
            {
                "completed": document["completed"],
                "platform_completed": document["platform_completed"],
                "urdf_compliant": document["urdf_compliant"],
                "full_success": document["full_success"],
            }
        )
    )


if __name__ == "__main__":
    main()
