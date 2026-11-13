"""Phase E: timestep convergence at the measured Phase B boundary points.

For every friction/order cell the highest passing angle and the adjacent failing
angle are rerun at 0.5 ms and 0.2 ms. The 1 ms 18 deg rows already exist in
pb_v2 and are referenced, not rerun; so are the 1 ms 20 deg arm rows, which
serve as the other-amplitude control. Control period stays 1 ms.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import urdf_slope_runtime_20261113 as runner

PHASE = "E"
PAIRS = ((4, "SN", 15.0, 25.0), (4, "SP", 25.0, 35.0), (6, "SN", 25.0, 35.0),
         (6, "SP", 25.0, 35.0), (8, "SN", 25.0, 35.0), (8, "SP", 15.0, 25.0))
SPEED = 3.0
AMP = 18.0


def tread(angle_deg: float) -> float:
    return runner.STEP_RISE / math.tan(math.radians(angle_deg))


def case_for(friction: int, order: str, angle: float, amp: float, dt: float) -> dict:
    return {
        "run_id": f"urdf_slope_20261112/{PHASE}/dt{dt * 1e6:g}us/{angle:g}deg/mu{friction}/{order}/{amp:g}deg",
        "phase": PHASE,
        "angle_deg": angle,
        "tread_m": tread(angle),
        "friction": friction,
        "order": order,
        "rocker_amp_deg": amp,
        "speed": SPEED,
        "trigger": "contact",
        "steps": 15,
        "dt": dt,
        "timeout_s": runner.budget(angle),
    }


def cases() -> list[dict]:
    out = []
    for friction, order, passing, failing in PAIRS:
        for angle in (passing, failing):
            for dt in (0.0005, 0.0002):
                out.append(case_for(friction, order, angle, AMP, dt))
    return out


def baseline(root: Path, friction: int, order: str, angle: float, amp: float) -> dict | None:
    path = root / f"runs/urdf_slope_20261112/pb_v2/{friction}_{angle:g}_{order}_{amp:g}.json"
    if not path.is_file():
        return None
    body = json.loads(path.read_text())["result"]
    return {key: body[key] for key in ("ascent_completed", "urdf_compliant",
                                       "numerical_quality_ok", "steps_climbed", "end_reason")}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    run_dir = root / "runs/urdf_slope_20261112/pe"
    runner.RUN = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, case in enumerate(cases(), 1):
        started = time.time()
        result = runner.run(case)
        result["wall_seconds"] = round(time.time() - started, 2)
        (run_dir / f"{index:02d}_mu{case['friction']}_{case['order']}_{case['angle_deg']:g}_{case['rocker_amp_deg']:g}_dt{case['dt'] * 1e6:g}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        rows.append({
            "run_id": case["run_id"],
            "friction": case["friction"],
            "order": case["order"],
            "angle_deg": case["angle_deg"],
            "rocker_amp_deg": case["rocker_amp_deg"],
            "dt_s": case["dt"],
            "result": result["result"],
            "violation_counts": result["violation_counts"],
            "duration_s": result["duration_s"],
            "wall_seconds": result["wall_seconds"],
            "baseline_1ms": baseline(root, case["friction"], case["order"],
                                     case["angle_deg"], case["rocker_amp_deg"]),
            "arm_control_1ms": baseline(root, case["friction"], case["order"],
                                        case["angle_deg"], 20.0),
        })
        body = result["result"]
        print(f"[{index}/{len(cases())}] {case['run_id']} ascent={body['ascent_completed']} "
              f"urdf={body['urdf_compliant']} num={body['numerical_quality_ok']} "
              f"steps={body['steps_climbed']} end={body['end_reason']} wall={result['wall_seconds']}s",
              flush=True)
    checks = {}
    for row in rows:
        key = (row["friction"], row["order"], row["angle_deg"], row["rocker_amp_deg"])
        entry = checks.setdefault(key, {"baseline_1ms": row["baseline_1ms"],
                                        "arm_control_1ms": row["arm_control_1ms"],
                                        "steps": []})
        entry["steps"].append({"dt_s": row["dt_s"], "result": row["result"]})
    for entry in checks.values():
        variants = ([entry["baseline_1ms"]] if entry["baseline_1ms"] else []) + \
                   [step["result"] for step in entry["steps"]]
        entry["consistent"] = len({(v["ascent_completed"], v["urdf_compliant"],
                                    v["numerical_quality_ok"]) for v in variants}) == 1
    summary = {"phase": "E", "control_period_s": 0.001, "pairs": PAIRS,
               "rows": rows,
               "checks": [{"friction": key[0], "order": key[1], "angle_deg": key[2],
                           "rocker_amp_deg": key[3], **value}
                          for key, value in sorted(checks.items())]}
    path = root / "docs/urdf_slope_pe_20261113.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
