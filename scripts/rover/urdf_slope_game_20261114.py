"""Rerun every climb-game configuration with the boosted rocker loop.

The game is human-driven, so the batch fixes one scripted policy: full throttle
at 4 rad/s (the game's SPEED_MAX), rocker cycles triggered by front-wheel
contact, 18 deg amplitude for three cycles. Angle, riser height, friction and
rocker order stay exactly as the lobby offers them; only the rocker torque loop
is the new one.

    python scripts/rover/urdf_slope_game_20261114.py --workers 16
    python scripts/rover/urdf_slope_game_20261114.py --index
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rover"))
import urdf_slope_runtime_20261113 as pb  # noqa: E402

RUN = ROOT / "runs/urdf_slope_20261112/game"
OUT = ROOT / "docs/urdf_slope_game_index.jsonl"
ANGLES = tuple(float(a) for a in range(10, 46))
RISES_CM = (5, 10, 15, 20, 25, 30, 35)
FRICTIONS = (4, 6, 8)
ORDERS = ("SP", "SN")
SPEED = 4.0
AMP = 18.0


def cases() -> list[dict]:
    out = []
    for angle in ANGLES:
        for rise_cm in RISES_CM:
            rise = rise_cm / 100.0
            tread = rise / math.tan(math.radians(angle))
            for friction in FRICTIONS:
                for order in ORDERS:
                    out.append({
                        "run_id": (f"urdf_slope_20261112/G/{angle:g}deg/"
                                   f"rise{rise_cm:g}cm/mu{friction}/{order}"),
                        "phase": "G",
                        "angle_deg": angle, "tread_m": tread, "rise_m": rise,
                        "friction": friction, "order": order,
                        "rocker_amp_deg": AMP, "speed": SPEED, "trigger": "contact",
                        "steps": pb.STEP_COUNT, "rocker_mode": "boost", "audit": False,
                        "timeout_s": pb.budget(angle, rise),
                    })
    return out


def run_case(case: dict) -> dict:
    pb.RUN = RUN
    started = time.monotonic()
    result = pb.run(case)
    result["wall_seconds"] = time.monotonic() - started
    target = RUN / f"{pb.stem_for(case)}.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def batch(workers: int, force: bool) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    todo = []
    for case in cases():
        if not force and (RUN / f"{pb.stem_for(case)}.json").exists():
            continue
        todo.append(case)
    print(f"[game] {len(todo)} of {len(cases())} runs pending, {workers} workers", flush=True)
    started = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_case, todo, chunksize=1):
            done += 1
            geo, res = result["geometry"], result["result"]
            print(f"[{done:4d}/{len(todo)}] {result['run_id'].split('urdf_slope_20261112/G/')[1]:34s} "
                  f"rise {geo['rise_cm']:>2.0f}cm climb {res['steps_climbed']:2d} "
                  f"top {res['ascent_completed']!s:5s} {res['end_reason']:16s} "
                  f"{result['duration_s']:6.1f}s sim {time.monotonic() - started:6.0f}s", flush=True)
    print(f"[game] done in {(time.monotonic() - started) / 60:.1f} min", flush=True)


def index() -> None:
    rows = []
    for path in sorted(RUN.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        geo, control, res = data["geometry"], data["control_id"], data["result"]
        settle = data.get("settle_check") or {}
        rows.append({
            "run_id": data["run_id"], "phase": "G", "criterion": "game", "scope": "full",
            "steps_total": geo["steps"], "angle_deg": geo["angle_deg"],
            "rise_cm": geo["rise_cm"], "tread_m": geo["tread_m"],
            "friction": data["friction"]["stair"], "order": control["order"],
            "rocker_amp_deg": control["rocker_amp_deg"],
            "speed_rad_s": control["speed_rad_s"], "trigger": control["trigger"],
            "rocker_mode": control["rocker_mode"],
            "ascent_completed": res["ascent_completed"],
            # The boosted loop runs past the URDF's 100 N.m rocker cap on purpose,
            # so this column says nothing useful here.
            "urdf_compliant": None,
            "numerical_quality_ok": res["numerical_quality_ok"],
            "steps_climbed": res["steps_climbed"],
            "steps_climbed_front": res.get("steps_climbed_front"),
            "end_reason": res["end_reason"], "duration_s": data["duration_s"],
            "penetration_mm": data["penetration"]["max_mm"],
            "loaded_wheels": settle.get("loaded_wheels"),
            "unloaded_gap_mm": settle.get("unloaded_gap_mm"),
            "support_s": (settle.get("support_steps") or 0) * 0.001,
            "settle_s": (settle.get("settled_steps") or 0) * 0.001,
            "violation_counts": data.get("violation_counts") or {},
            "result_path": path.relative_to(ROOT).as_posix(),
            "source_sha256": data["model"]["source_sha256"],
            "video_key": "",
        })
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                   encoding="utf-8")
    passed = sum(1 for row in rows if row["ascent_completed"])
    print(f"[game] index: {len(rows)} rows, {passed} climbs settled on the platform -> {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    if args.index:
        return index()
    batch(args.workers, args.force)


if __name__ == "__main__":
    main()
