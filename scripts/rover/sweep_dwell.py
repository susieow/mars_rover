"""One side lifts and holds, then the other side: a walking gait instead of a sine."""
from course import simulate

print(" h  amp hold   ok  top  max_z  plat     x")
for h in (0.10, 0.15, 0.20):
    for amp, hold in ((15, 1.5), (20, 1.5), (20, 3.0), (30, 2.0)):
        r = simulate(h, 12, h, 120.0, 6.0, 4.0, 120.0, 20.0,
                     pitch_limit=0.0, heading_hold=False,
                     dwell_amp=float(amp), dwell_hold=float(hold))
        print("%4.2f %3d %4.1f %5s %5s %6.3f %6.3f %6.2f" % (
            h, amp, hold, r["success"], r["reached_top"], r["max_z"], r["plat_z"], r["x"]), flush=True)
