"""Build the experiment-two review catalog from recorded cases."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import urdf_slope_control_20261112 as exp


def main():
    rows = []
    current_sha = exp.digest(exp.URDF)
    current_limits = exp.get_joint_limits()
    results = sorted((exp.LEGACY_RUN_ROOT / "cases").glob("*/result.json"))
    results += sorted((exp.RUN_ROOT / "cases").glob("*/result.json"))
    for result_path in results:
        folder = result_path.parent
        result = json.loads(result_path.read_text(encoding="utf-8"))
        media_path = folder / "media.json"
        if not media_path.exists():
            continue
        media = json.loads(media_path.read_text(encoding="utf-8"))
        same_model = result.get("urdf_sha256") == current_sha
        audit_path = folder / "model_compliance.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        limits_match = None
        if audit.get("joints"):
            limits_match = all(
                all(audit["joints"].get(name, {}).get("expected", {}).get(key) == value
                    for key, value in limit.items())
                for name, limit in current_limits.items()
            )
        elif same_model and audit.get("passed"):
            limits_match = True
        rows.append(
            {
                "id": result["id"],
                "recorded_model_version": result.get("urdf_sha256", "unknown"),
                "matches_current_urdf": same_model,
                "joint_limits_match_current": limits_match,
                "model_gate_passed": audit.get("passed"),
                "current_model_joint_compliant": (
                    bool(audit.get("passed") and result["urdf_compliant_exact"])
                    if same_model else None
                ),
                "current_urdf_certified": None,
                "assessment_scope": "diagnostic_only; no full-route independent certification",
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
        "updated": datetime.now(timezone.utc).isoformat(),
        "experiment": "urdf_slope_control",
        "title": "URDF 约束内的接触恢复控制",
        "summary": (
            "按记录时的 URDF 版本区分历史与当前模型。旧模型视频不能直接认定为新模型合规成功。"
            "关节限值检查、模型一致性与完整数值认证分别展示。"
        ),
        "current_model": {"version": current_sha, "joint_limits": current_limits,
                          "mass_kg": sum(float(n.get("value")) for n in
                                         exp.ET.parse(exp.URDF).getroot().findall("./link/inertial/mass")),
                          "wheel_friction": 0.6, "wheel_friction_source": "user_requirement"},
        "current_model_records": sum(row["matches_current_urdf"] for row in rows),
        "legacy_model_records": sum(not row["matches_current_urdf"] for row in rows),
        "current_model_joint_passes": sum(row["current_model_joint_compliant"] is True for row in rows),
        "current_urdf_certified_count": 0,
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
