"""Diagnostic: rerun a Phase B case with a longer platform.

The 2 m platform is frozen for official runs. This only checks whether a run
that reached six-wheel support and then slid past the far edge can stop at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/agilex/wzy/rover")
sys.path.insert(0, str(ROOT / "scripts/rover"))
import urdf_slope_pb_20261112 as pb  # noqa: E402

NAME = sys.argv[1]
PLATFORM_LEN = float(sys.argv[2])

pb.RUN = pb.RUN / "diag"
case = next(c for c in pb.cases()
            if f'{c["friction"]}_{c["angle_deg"]:g}_{c["order"]}_{c["rocker_amp_deg"]:g}' == NAME)
case["platform_len"] = PLATFORM_LEN
result = pb.run(case)
name = f"{NAME}_pl{PLATFORM_LEN:g}"
(pb.RUN / f"{name}.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n")
r = result["result"]
print(f"[diag] {name}: ascent={r['ascent_completed']} compliant={r['urdf_compliant']} "
      f"steps={r['steps_climbed']} end={r['end_reason']} t={result['duration_s']:.1f}s "
      f"pen={result['penetration']['max_mm']:.1f}mm", flush=True)
