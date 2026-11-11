"""Temporary XML with wheel joint actuatorfrcrange raised to +-120; default model untouched."""
from pathlib import Path

import course
from course import simulate
from climb import ROBOT_XML

tmp = Path("/tmp/rover_actuated_w120.xml")
tmp.write_text(ROBOT_XML.read_text().replace(
    'actuatorfrcrange="-50 50"', 'actuatorfrcrange="-120 120"'))
course.ROBOT_XML = tmp

for h in (0.13, 0.15, 0.20):
    for deg, fl, ml, mt in ((12.0, 0.95, 0.10, 0.70),
                            (20.0, 9.5 * h, h, 7.0 * h)):
        r = simulate(h, 12, h, 150.0, 6.0, 4.0, 120.0, deg,
                     front_lead=fl, mid_lead=ml, mid_trail=mt, pitch_limit=0.0)
        print("h=%.2f deg=%2.0f ok=%-5s top=%-5s down=%-5s max_z=%.3f/%.3f x=%.2f pitch=%.0f" % (
            h, deg, r["success"], r["reached_top"], r["back_down"],
            r["max_z"], r["plat_z"], r["x"], r["max_pitch_deg"]), flush=True)
