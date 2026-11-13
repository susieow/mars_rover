"""Phase D: screen 20 control ids on one step, then confirm the best two on the full staircase.

Screening uses friction 6 at two tread depths: 15 deg (widest tread) and 25 deg
(the measured mu6 boundary angle). A control id is speed x trigger x order.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import urdf_slope_runtime_20261113 as pb

PHASE = "urdf_slope_20261112/D"
SPEEDS = (3.0, 6.0, 9.0, 12.0, 15.0)
TRIGGERS = ("contact", "stall")
ORDERS = ("SP", "SN")
SCREEN_ANGLES = (15.0, 25.0)
FRICTION = 6
AMP = 18.0
CONFIRM_ANGLES = (25.0, 35.0)


def tread(angle_deg: float) -> float:
    return pb.STEP_RISE / math.tan(math.radians(angle_deg))


def timeout_s(case: dict) -> float:
    reach = pb.APPROACH_M + case["steps"] * case["tread_m"] + pb.PLATFORM_LEN
    ideal = reach / (case["speed"] * 0.162697)
    return min(300.0, max(60.0, math.ceil(2.0 * ideal) + 60.0))


def case_for(angle_deg: float, speed: float, trigger: str, order: str,
             steps: int, stage: str) -> dict:
    case = {
        "run_id": f"{PHASE}/{stage}/{angle_deg:g}deg/v{speed:g}/{trigger}/{order}",
        "phase": "D",
        "angle_deg": angle_deg,
        "tread_m": tread(angle_deg),
        "friction": FRICTION,
        "order": order,
        "rocker_amp_deg": AMP,
        "speed": speed,
        "trigger": trigger,
        "steps": steps,
    }
    case["timeout_s"] = pb.budget(angle_deg) if steps > 1 else timeout_s(case)
    return case


def screen_cases() -> list[dict]:
    return [case_for(angle, speed, trigger, order, 1, "screen")
            for angle in SCREEN_ANGLES
            for speed in SPEEDS
            for trigger in TRIGGERS
            for order in ORDERS]


def violations(result: dict) -> int:
    return sum(result.get("violation_counts", {}).values())


def run_all(cases: list[dict], run_dir: Path) -> list[dict]:
    pb.RUN = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, case in enumerate(cases, 1):
        started = time.time()
        result = pb.run(case)
        result["wall_seconds"] = round(time.time() - started, 2)
        (run_dir / f"{index:02d}_{case['angle_deg']:g}_{case['speed']:g}_{case['trigger']}_{case['order']}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        rows.append(result)
        body = result["result"]
        print(f"[{index}/{len(cases)}] {case['run_id']} mount={body['ascent_completed']} "
              f"urdf={body['urdf_compliant']} steps={body['steps_climbed']} "
              f"end={body['end_reason']} wall={result['wall_seconds']}s", flush=True)
    return rows


def rank(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        control = row["control_id"]
        key = (control["speed_rad_s"], control["trigger"], control["order"])
        groups.setdefault(key, []).append(row)
    table = []
    for (speed, trigger, order), group in groups.items():
        mounted = sum(1 for row in group if row["result"]["ascent_completed"])
        table.append({
            "speed_rad_s": speed,
            "trigger": trigger,
            "order": order,
            "mounted": mounted,
            "of": len(group),
            "violation_steps": sum(violations(row) for row in group),
            "duration_s": round(sum(row["duration_s"] for row in group), 3),
        })
    table.sort(key=lambda row: (-row["mounted"], row["violation_steps"], row["duration_s"]))
    return table


def load_screen(run_dir: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(run_dir.glob("*.json"))]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-only", action="store_true",
                        help="reuse the screening JSONs and rerun only the confirmation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    screen_dir = root / "runs/urdf_slope_20261112/pd/screen"
    if args.confirm_only:
        screen = load_screen(screen_dir)
    else:
        screen = run_all(screen_cases(), screen_dir)
    table = rank(screen)
    best = table[:2]
    print("ranking:", json.dumps(table, ensure_ascii=False), flush=True)
    confirm_cases = [case_for(angle, row["speed_rad_s"], row["trigger"], row["order"], 15, "confirm")
                     for row in best
                     for angle in CONFIRM_ANGLES]
    confirm = run_all(confirm_cases, root / "runs/urdf_slope_20261112/pd/confirm")
    summary = {"phase": "D", "friction": FRICTION, "rocker_amp_deg": AMP,
               "screen_angles_deg": list(SCREEN_ANGLES), "confirm_angles_deg": list(CONFIRM_ANGLES),
               "ranking": table, "selected": best,
               "screen": [{"run_id": row["run_id"], "control_id": row["control_id"],
                           "result": row["result"], "violation_counts": row["violation_counts"],
                           "duration_s": row["duration_s"], "wall_seconds": row["wall_seconds"]}
                          for row in screen],
               "confirm": [{"run_id": row["run_id"], "control_id": row["control_id"],
                            "result": row["result"], "violation_counts": row["violation_counts"],
                            "duration_s": row["duration_s"], "wall_seconds": row["wall_seconds"]}
                           for row in confirm]}
    path = root / "docs/urdf_slope_pd_20261113.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
