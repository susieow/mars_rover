"""Render eligible saved episodes after an independent verification pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from render_dataset import render_metadata
from verify_dataset import verify_episode

DEFAULT_ROOT = Path("/home/agilex/wzy/rover/datasets/45deg_v3")


def render_candidates(dataset_root):
    reports = []
    eligible = 0
    rendered = 0
    for path in sorted((Path(dataset_root) / "episodes").glob("*/metadata.json")):
        report = {"metadata_path": str(path)}
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            report["case"] = metadata.get("case", path.parent.name)
            if metadata.get("automated_checks_passed") is not True:
                report.update(status="not_eligible", rejection_reasons=["automated_checks_not_passed"])
                reports.append(report)
                continue
            verification = verify_episode(path)
            if verification.get("verify_ok") is not True:
                report.update(status="verification_rejected",
                              rejection_reasons=verification.get("rejection_reasons", ["independent_verification_failed"]))
                reports.append(report)
                continue
            eligible += 1
            report.update(status="rendered", render=render_metadata(path))
            rendered += 1
        except Exception as error:
            report.update(status="error", error=f"{type(error).__name__}: {error}")
        reports.append(report)
    status = "no_eligible_candidates" if eligible == 0 else "complete" if rendered == eligible else "render_failed"
    return {"status": status, "eligible_count": eligible, "rendered_count": rendered,
            "episodes": reports, "published_count": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = render_candidates(args.dataset_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if result["status"] == "complete" else 1)
