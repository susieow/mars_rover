"""h20: larger nosing chamfers and angled approach."""
from course import simulate

print(" nose yaw   ok  top  down  max_z  plat     x   pitch")
for nose in (0.08, 0.10, 0.12, 0.14):
    r = simulate(0.20, 12, 0.20, 150.0, 6.0, 4.0, 120.0, 20.0,
                 front_lead=1.90, mid_lead=0.20, mid_trail=1.40,
                 pitch_limit=0.0, nosing=nose)
    print("%4.2f %3d %5s %5s %5s %6.3f %6.3f %6.2f %6.1f" % (
        nose, 0, r["success"], r["reached_top"], r["back_down"],
        r["max_z"], r["plat_z"], r["x"], r["max_pitch_deg"]), flush=True)
for yaw in (15.0, 30.0):
    r = simulate(0.20, 12, 0.20, 150.0, 6.0, 4.0, 120.0, 20.0,
                 front_lead=1.90, mid_lead=0.20, mid_trail=1.40,
                 pitch_limit=0.0, init_yaw_deg=yaw)
    print("%4.2f %3d %5s %5s %5s %6.3f %6.3f %6.2f %6.1f" % (
        0.0, yaw, r["success"], r["reached_top"], r["back_down"],
        r["max_z"], r["plat_z"], r["x"], r["max_pitch_deg"]), flush=True)
