"""One off-angle case for the Phase B boundary gap.

mu4/SP fails at 15 deg on a rocker position overshoot and passes at 25 deg,
which is not monotonic in difficulty. This runs 20 deg to see whether the
25 deg pass is a single point or part of a real interval.
"""
from __future__ import annotations

import json
import math
import time

import urdf_slope_pb_20261112 as pb

ANGLE = 20.0


def main() -> None:
    case = {
        "run_id": f"urdf_slope_20261112/B/{ANGLE:g}deg/mu4/SP/18deg",
        "angle_deg": ANGLE,
        "tread_m": pb.STEP_RISE / math.tan(math.radians(ANGLE)),
        "friction": 4,
        "order": "SP",
        "rocker_amp_deg": 18.0,
        "timeout_s": pb.budget(ANGLE),
    }
    pb.RUN = pb.ROOT / "runs/urdf_slope_20261112/pb_v2"
    started = time.monotonic()
    result = pb.run(case)
    result["wall_seconds"] = time.monotonic() - started
    (pb.RUN / "4_20_SP_18.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    r = result["result"]
    print(f"[B] 4_20_SP_18: ascent={r['ascent_completed']} "
          f"compliant={r['urdf_compliant']} num={r['numerical_quality_ok']} "
          f"steps={r['steps_climbed']} end={r['end_reason']} "
          f"t={result['duration_s']:.1f}s pen={result['penetration']['max_mm']:.1f}mm",
          flush=True)


if __name__ == "__main__":
    main()
