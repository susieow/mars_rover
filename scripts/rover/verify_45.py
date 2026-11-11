"""Re-confirm the in-phase schedule cases before repopulating the front-end."""
from course import simulate

cases = [ (h, 0.0) for h in (0.04, 0.05, 0.06, 0.08, 0.10) ]
cases += [(0.12, 0.02), (0.12, 0.03), (0.12, 0.04), (0.15, 0.03), (0.20, 0.04)]
print(" h  nose   ok  top  down  max_z  plat     x   pitch")
for h, nose in cases:
    r = simulate(h, 12, h, 150.0, 6.0, 4.0, 120.0, 12.0,
                 mid_lead=0.10, mid_trail=0.70, pitch_limit=0.0,
                 heading_hold=True, nosing=nose)
    print("%4.2f %4.2f %5s %5s %5s %6.3f %6.3f %6.2f %6.1f" % (
        h, nose, r["success"], r["reached_top"], r["back_down"], r["max_z"],
        r["plat_z"], r["x"], r["max_pitch_deg"]), flush=True)
