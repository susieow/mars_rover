"""Verify the strict plain 45-degree cases without rendering."""

from __future__ import annotations

import json

from course import simulate

CASES = [
    {"case": "h4_plain_45deg", "height": 0.04, "front": 10.0, "middle": 10.0, "quality": 38.0},
    {"case": "h5_plain_45deg", "height": 0.05, "front": 12.0, "middle": 12.0, "quality": 38.0},
    {"case": "h6_plain_45deg", "height": 0.06, "front": 16.0, "middle": 16.0, "quality": 45.0},
    {"case": "h7_plain_45deg", "height": 0.07, "front": 18.0, "middle": 15.0, "quality": 50.0},
    {"case": "h8_plain_45deg", "height": 0.08, "front": 14.0, "middle": 12.0, "quality": 55.0},
]


def main() -> None:
    for case in CASES:
        result = simulate(
            case["height"],
            12,
            case["height"],
            seconds=150.0,
            speed=6.0,
            friction=4.0,
            tau_max=50.0,
            nosing=0.0,
            front_lift_deg=case["front"],
            middle_lift_deg=case["middle"],
            quality_pitch_limit=case["quality"],
        )
        print(
            json.dumps(
                {
                    "case": case["case"],
                    "success": result["success"],
                    "final_shape_ok": result["final_shape_ok"],
                    "settled": result["settled"],
                    "x": result["x"],
                    "y": result["y"],
                    "max_pitch_deg": result["max_pitch_deg"],
                    "max_roll_deg": result["max_roll_deg"],
                    "final_yaw_error_deg": result["final_yaw_error_deg"],
                    "stopped_at_s": result["stopped_at_s"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
